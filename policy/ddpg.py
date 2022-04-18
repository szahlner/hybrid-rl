import os
import numpy as np
import torch
import torch.nn as nn
import gym
import time
from mpi4py import MPI
from typing import Union
from copy import deepcopy

from utils.mpi.mpi_utils import sync_grads, sync_networks
from utils.mpi.normalizer import Normalizer
from utils.ddpg.replay_buffer import ReplayBuffer
from utils.ddpg.arguments import DdpgNamespace
from utils.logger import EpochLogger

from world_model.deterministic_world_model import DeterministicWorldModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, env_params: dict) -> None:
        """
        The actor network.

        Args:
            env_params (dict): Environment parameters.
        """
        super(Actor, self).__init__()

        self.max_action = env_params["action_max"]

        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params["action"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, env_params: dict) -> None:
        """
        The critic network.

        Args:
            env_params (dict): Environment parameters.
        """
        super(Critic, self).__init__()

        self.max_action = env_params["action_max"]

        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"] + env_params["action"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class DDPG:
    def __init__(self, args: DdpgNamespace, env: Union[gym.Env, gym.GoalEnv], env_params: dict, logger: EpochLogger):
        self.args = args
        self.env = env
        self.env_params = env_params

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.logger = logger
            self.args.save_dir = os.path.join(self.logger.output_dir, self.args.save_dir)

            # Setup logger
            self._setup_logger()

        # Create the networks
        self.actor_network = Actor(env_params).to(device)
        self.critic_network = Critic(env_params).to(device)

        # Sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)

        # Build up the target network
        self.actor_target_network = Actor(env_params).to(device)
        self.critic_target_network = Critic(env_params).to(device)

        # Load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # Create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # Create the replay buffer
        self.buffer = ReplayBuffer(self.env_params, self.args.buffer_size)

        # Create the normalizer
        self.o_norm = Normalizer(size=env_params["obs"], default_clip_range=self.args.clip_range)

        # Create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

        # Model based section
        if self.args.model_based:
            model_dim_chunk = 20  # self.args.model_dim_chunk
            output_dim = env_params["obs"] + env_params["reward"]
            self.model_chunks = [model_dim_chunk for _ in range(output_dim // model_dim_chunk)]
            if output_dim % model_dim_chunk != 0:
                self.model_chunks.append(output_dim % model_dim_chunk)

            self.world_models = []
            for chunk in self.model_chunks:
                self.world_models.append(
                    DeterministicWorldModel(
                        input_dim=env_params["obs"] + env_params["goal"] + env_params["action"],
                        output_dim=chunk,
                        network_dim=1,
                    )
                )
            self.model_chunks.insert(0, 0)
            self.model_chunks = np.cumsum(self.model_chunks).tolist()

            # Buffers
            self.simple_world_model_replay_buffer = ReplayBuffer(self.env_params, self.args.buffer_size)

            # Create the replay buffer
            self.world_model_params = deepcopy(self.env_params)
            self.world_model_params["max_timesteps"] = 5  # change max timesteps to rollout length
            self.world_model_buffer = ReplayBuffer(self.world_model_params, self.args.buffer_size)

    def learn(self):
        """
        train the network

        """
        ts = 0  # timesteps
        start_time = time.time()

        # Start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                for _ in range(self.args.num_rollouts_per_mpi):
                    # Reset the rollouts
                    # Reset the environment
                    obs = self.env.reset()

                    # Start to collect samples
                    for t in range(self.env_params["max_timesteps"]):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)

                        # Feed the actions into the environment
                        obs_new, reward, done, info = self.env.step(action)

                        if MPI.COMM_WORLD.Get_rank() == 0:
                            self.logger.store(Reward=reward)

                        # Append rollouts
                        self.buffer.store(
                            batch=[
                                obs.copy(),
                                obs_new.copy(),
                                np.array([[reward]]).copy(),
                                np.array([[done]]).copy(),
                                action.copy()
                            ]
                        )

                        # Model based section
                        if self.args.model_based:
                            self.simple_world_model_replay_buffer.store(
                                batch=[
                                    obs.copy(),
                                    obs_new.copy(),
                                    np.array([[reward]]).copy(),
                                    np.array([[done]]).copy(),
                                    action.copy()
                                ]
                            )

                            if ts % self.args.model_training_freq == 0 and ts != 0:
                                transitions = self.simple_world_model_replay_buffer.sample(10000)

                                # Train chunked
                                training_inputs = transitions["obs"]
                                training_labels = np.concatenate(
                                    [transitions["obs_next"] - transitions["obs"], transitions["r"]],
                                    axis=-1
                                )

                                training_inputs = np.concatenate([training_inputs, transitions["actions"]], axis=-1)
                                for n in range(len(self.model_chunks) - 1):
                                    tl = training_labels[:, self.model_chunks[n]:self.model_chunks[n+1]]
                                    self.world_models[n].train(training_inputs, tl)

                                # Rollout
                                n_transitions = 10000
                                transitions = self.simple_world_model_replay_buffer.sample(n_transitions)
                                world_model_obs = np.empty(
                                    (
                                        n_transitions,
                                        self.world_model_params["max_timesteps"] + 1,
                                        self.env_params["obs"]
                                    )
                                )
                                world_model_r = np.empty(
                                    (
                                        n_transitions,
                                        self.world_model_params["max_timesteps"],
                                        self.env_params["reward"]
                                    )
                                )
                                world_model_actions = np.empty(
                                    (
                                        n_transitions,
                                        self.world_model_params["max_timesteps"],
                                        self.env_params["action"]
                                    )
                                )
                                world_model_mask = np.empty((n_transitions, self.world_model_params["max_timesteps"]))

                                world_model_obs[:, 0] = transitions["obs"]

                                for n in range(self.world_model_params["max_timesteps"]):
                                    with torch.no_grad():
                                        input_tensor = self._preproc_inputs(world_model_obs[:, n])
                                        actions_tensor = self.actor_network(input_tensor)
                                        world_model_actions[:, n] = self._select_actions(actions_tensor)

                                    diff = np.empty((n_transitions, self.env_params["obs"] + self.env_params["reward"]))
                                    confidence = np.empty(
                                        (n_transitions, self.env_params["obs"] + self.env_params["reward"])
                                    )

                                    for k in range(len(self.model_chunks) - 1):
                                        diff_, confidence_ = self.world_models[k].predict(
                                            inputs=np.concatenate([
                                                world_model_obs[:, n],
                                                world_model_actions[:, n]
                                                ], axis=-1)
                                        )
                                        diff[:, self.model_chunks[k]:self.model_chunks[k+1]] = diff_
                                        confidence[:, self.model_chunks[k]:self.model_chunks[k+1]] = confidence_

                                    world_model_mask[:, n] = np.all(np.where(confidence < 1, True, False), axis=-1)
                                    world_model_obs[:, n + 1] = world_model_obs[:, n] + diff[:, :self.env_params["obs"]]
                                    world_model_r[:, n] = diff[:, self.env_params["obs"]:]

                                # Mark and select good ones
                                mask = np.any(world_model_mask, axis=-1)
                                world_model_obs = world_model_obs[mask]
                                world_model_r = world_model_r[mask]
                                world_model_actions = world_model_actions[mask]

                                if mask.sum() > 0:
                                    for n in range(self.world_model_params["max_timesteps"]):
                                        for k in range(len(world_model_obs)):
                                            self.world_model_buffer.store(
                                                batch=[
                                                    world_model_obs[k, n],
                                                    world_model_obs[k, n + 1],
                                                    world_model_r[k, n],
                                                    False,
                                                    world_model_actions[k, n],
                                                ]
                                            )

                            # Model based section
                            if ts > self.args.model_training_freq and self.world_model_buffer.current_size > 0:
                                self._unreal_update_network()

                                if ts % self.args.n_batches == 0:
                                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                                    self._soft_update_target_network(self.critic_target_network, self.critic_network)

                        ts += 1

                        # Re-assign the observation
                        obs = obs_new

                self._update_normalizer()

                for _ in range(self.args.n_batches):
                    # Train the network
                    self._update_network()

                # Soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # Start to do the evaluation
            success_rate = self._eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                self.logger.log_tabular("Epoch", epoch)
                self.logger.log_tabular("Timesteps", ts)
                self.logger.log_tabular("Time", time.time() - start_time)
                self.logger.log_tabular("SuccessRate", success_rate)
                self.logger.log_tabular("ActorLoss", with_min_and_max=True)
                self.logger.log_tabular("CriticLoss", with_min_and_max=True)
                self.logger.log_tabular("Reward", with_min_and_max=True)

                if self.args.model_based:
                    self.logger.log_tabular("WorldModelReplayBufferSize", self.world_model_buffer.n_transitions_stored)

                self.logger.dump_tabular()

                file_path = os.path.join(self.args.save_dir, "model.pt")
                torch.save(
                    [
                        self.o_norm.mean,
                        self.o_norm.std,
                        self.actor_network.state_dict()
                    ],
                    file_path
                )

                if self.args.model_based:
                    for n, model in enumerate(self.world_models):
                        file_path = os.path.join(self.args.save_dir, f"world_model_{n}.pt")
                        model.save(file_path)

    # pre_process the inputs
    def _preproc_inputs(self, obs: np.ndarray)-> torch.Tensor:
        obs_norm = self.o_norm.normalize(obs)
        inputs = torch.tensor(obs_norm, dtype=torch.float32, device=device)

        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi: torch.Tensor):
        action = pi.cpu().numpy().squeeze()

        # add the gaussian
        action += self.args.noise_eps * self.env_params["action_max"] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params["action_max"], self.env_params["action_max"])

        # random actions...
        random_actions = np.random.uniform(
            low=-self.env_params["action_max"],
            high=self.env_params["action_max"],
            size=self.env_params["action"]
        )

        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)

        return action

    # update the normalizer
    def _update_normalizer(self) -> None:
        last_n_transitions = self.args.num_rollouts_per_mpi * self.env_params["max_timesteps"]
        obs = self.buffer.buffers["obs"][self.buffer.pointer - last_n_transitions:self.buffer.pointer]

        # update
        self.o_norm.update(obs)

        # recompute the stats
        self.o_norm.recompute_stats()

    def _preproc_o(self, o: np.ndarray) -> np.ndarray:
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        return o

    # soft update
    def _soft_update_target_network(self, target, source) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next = transitions["obs"], transitions["obs_next"]
        transitions["obs"] = self._preproc_o(o)
        transitions["obs_next"] = self._preproc_o(o_next)

        # start to do the update
        inputs_norm = self.o_norm.normalize(transitions["obs"])
        inputs_next_norm = self.o_norm.normalize(transitions["obs_next"])

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32, device=device)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32, device=device)
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32, device=device)

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params["action_max"]).pow(2).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.logger.store(
                ActorLoss=actor_loss.item(),
                CriticLoss=critic_loss.item(),
            )

    # update the network
    def _unreal_update_network(self):
        # sample the episodes
        transitions = self.world_model_buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next = transitions["obs"], transitions["obs_next"]
        transitions["obs"] = self._preproc_o(o)
        transitions["obs_next"] = self._preproc_o(o_next)

        # start to do the update
        inputs_norm = self.o_norm.normalize(transitions["obs"])
        inputs_next_norm = self.o_norm.normalize(transitions["obs_next"])

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32, device=device)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32, device=device)
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32, device=device)

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params["action_max"]).pow(2).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    def _setup_logger(self):
        self.logger.store(
            ActorLoss=0,
            CriticLoss=0,
            Reward=0,
        )

    # Do the evaluation
    def _eval_agent(self):
        total_success_rate = []

        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            obs = self.env.reset()

            for _ in range(self.env_params["max_timesteps"]):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs)
                    pi = self.actor_network(input_tensor)

                    # Convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

                obs, _, _, info = self.env.step(actions)
                per_success_rate.append(info["is_success"])

            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()

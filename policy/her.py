import os
import numpy as np
import torch
import torch.nn as nn
import gym
import time
from mpi4py import MPI
from typing import List, Tuple, Union
from copy import deepcopy

from utils.mpi.mpi_utils import sync_grads, sync_networks
from utils.mpi.normalizer import Normalizer
from utils.her.her_sampler import HerSampler
from utils.her.replay_buffer import ReplayBuffer, SimpleReplayBuffer
from utils.her.arguments import HerNamespace
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


class HER:
    def __init__(self, args: HerNamespace, env: Union[gym.Env, gym.GoalEnv], env_params: dict, logger: EpochLogger):
        self.args = args
        self.env = env
        self.env_params = env_params

        # Setup logger
        self.logger = logger

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

        # Her sampler
        self.her_module = HerSampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)

        # Create the replay buffer
        self.buffer = ReplayBuffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        # Create the normalizer
        self.o_norm = Normalizer(size=env_params["obs"], default_clip_range=self.args.clip_range)
        self.g_norm = Normalizer(size=env_params["goal"], default_clip_range=self.args.clip_range)

        # Create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.args.save_dir = os.path.join(self.logger.output_dir, self.args.save_dir)

            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

        # Model based section
        if self.args.model_based:
            model_dim_chunk = 20  # self.args.model_dim_chunk
            output_dim = env_params["obs"] + env_params["goal"]
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
            self.simple_world_model_replay_buffer = SimpleReplayBuffer(self.env_params, self.args.buffer_size)

            # Her sampler
            self.world_model_her_module = HerSampler(
                self.args.replay_strategy,
                self.args.replay_k,
                self.env.compute_reward,
            )

            # Create the replay buffer
            self.world_model_params = deepcopy(self.env_params)
            self.world_model_params["max_timesteps"] = 5  # change max timesteps to rollout length
            self.world_model_buffer = ReplayBuffer(
                self.world_model_params,
                self.args.buffer_size,
                self.world_model_her_module.sample_her_transitions
            )

    def learn(self):
        """
        train the network

        """
        ts = 0  # timesteps
        start_time = time.time()

        # Start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # Reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []

                    # Reset the environment
                    observation = self.env.reset()
                    obs = observation["observation"]
                    ag = observation["achieved_goal"]
                    g = observation["desired_goal"]

                    # Start to collect samples
                    for t in range(self.env_params["max_timesteps"]):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)

                        # Feed the actions into the environment
                        observation_new, reward, _, info = self.env.step(action)
                        obs_new = observation_new["observation"]
                        ag_new = observation_new["achieved_goal"]

                        # if MPI.COMM_WORLD.Get_rank() == 0:
                        self.logger.store(Reward=reward)

                        # Append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())

                        # Model based section
                        if self.args.model_based:
                            self.simple_world_model_replay_buffer.store(
                                batch=[obs.copy(), obs_new.copy(), ag.copy(), ag_new.copy(), g.copy(), action.copy()]
                            )

                            if ts % self.args.model_training_freq == 0 and ts != 0:
                                transitions = self.simple_world_model_replay_buffer.sample(10000)

                                # Train chunked
                                training_inputs = np.concatenate([transitions["obs"], transitions["ag"]], axis=-1)
                                training_outputs = np.concatenate(
                                    [transitions["obs_next"], transitions["ag_next"]],
                                    axis=-1
                                )
                                training_labels = training_outputs - training_inputs
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
                                world_model_ag = np.empty(
                                    (
                                        n_transitions,
                                        self.world_model_params["max_timesteps"] + 1,
                                        self.env_params["goal"]
                                    )
                                )
                                world_model_g = np.empty(
                                    (n_transitions, self.world_model_params["max_timesteps"], self.env_params["goal"])
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
                                world_model_ag[:, 0] = transitions["ag"]
                                world_model_g[:, 0] = transitions["g"]

                                for n in range(self.world_model_params["max_timesteps"]):
                                    world_model_g[:, n] = transitions["g"]

                                    with torch.no_grad():
                                        input_tensor = self._preproc_inputs(world_model_obs[:, n], world_model_g[:, n])
                                        actions_tensor = self.actor_network(input_tensor)
                                        world_model_actions[:, n] = self._select_actions(actions_tensor)

                                    diff = np.empty((n_transitions, self.env_params["obs"] + self.env_params["goal"]))
                                    confidence = np.empty(
                                        (n_transitions, self.env_params["obs"] + self.env_params["goal"])
                                    )

                                    for k in range(len(self.model_chunks) - 1):
                                        diff_, confidence_ = self.world_models[k].predict(
                                            inputs=np.concatenate([
                                                world_model_obs[:, n],
                                                world_model_ag[:, n],
                                                world_model_actions[:, n]
                                                ], axis=-1)
                                        )
                                        diff[:, self.model_chunks[k]:self.model_chunks[k+1]] = diff_
                                        confidence[:, self.model_chunks[k]:self.model_chunks[k+1]] = confidence_

                                    world_model_mask[:, n] = np.all(np.where(confidence < 1, True, False), axis=-1)
                                    world_model_obs[:, n + 1] = world_model_obs[:, n] + diff[:, :self.env_params["obs"]]
                                    world_model_ag[:, n + 1] = world_model_ag[:, n] + diff[:, self.env_params["obs"]:]

                                # Mark and select good ones
                                mask = np.any(world_model_mask, axis=-1)
                                world_model_obs = world_model_obs[mask]
                                world_model_ag = world_model_ag[mask]
                                world_model_g = world_model_g[mask]
                                world_model_actions = world_model_actions[mask]

                                if mask.sum() > 0:
                                    self.world_model_buffer.store_episode(
                                        episode_batch=[
                                            world_model_obs,
                                            world_model_ag,
                                            world_model_g,
                                            world_model_actions,
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
                        ag = ag_new

                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)

                # Convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)

                # Store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                for _ in range(self.args.n_batches):
                    # Train the network
                    self._update_network()

                # Soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # Start to do the evaluation
            success_rate = self._eval_agent()
            self.logger.store(SuccessRate=success_rate)

            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("Timesteps", ts)
            self.logger.log_tabular("Time", time.time() - start_time)
            self.logger.log_tabular("SuccessRate", with_min_and_max=True)
            self.logger.log_tabular("ActorLoss", with_min_and_max=True)
            self.logger.log_tabular("CriticLoss", with_min_and_max=True)
            self.logger.log_tabular("Reward", with_min_and_max=True)

            if self.args.model_based:
                self.logger.store(WorldModelReplayBufferSize=self.world_model_buffer.n_transitions_stored)
                self.logger.log_tabular("WorldModelReplayBufferSize", with_min_and_max=True)

            self.logger.dump_tabular()

            if MPI.COMM_WORLD.Get_rank() == 0:
                file_path = os.path.join(self.args.save_dir, "model.pt")
                torch.save(
                    [
                        self.o_norm.mean,
                        self.o_norm.std,
                        self.g_norm.mean,
                        self.g_norm.std,
                        self.actor_network.state_dict()
                    ],
                    file_path
                )

                if self.args.model_based:
                    for n, model in enumerate(self.world_models):
                        file_path = os.path.join(self.args.save_dir, f"world_model_{n}.pt")
                        model.save(file_path)

    # pre_process the inputs
    def _preproc_inputs(self, obs: np.ndarray, g: np.ndarray) -> torch.Tensor:
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)

        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm], axis=-1)
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device)

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
    def _update_normalizer(self, episode_batch: List[np.ndarray]) -> None:
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]

        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]

        # create the new buffer to store them
        buffer_temp = {
            "obs": mb_obs,
            "ag": mb_ag,
            "g": mb_g,
            "actions": mb_actions,
            "obs_next": mb_obs_next,
            "ag_next": mb_ag_next,
        }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions["obs"], transitions["g"]

        # pre process the obs and g
        transitions["obs"], transitions["g"] = self._preproc_og(obs, g)

        # update
        self.o_norm.update(transitions["obs"])
        self.g_norm.update(transitions["g"])

        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)

        return o, g

    # soft update
    def _soft_update_target_network(self, target, source) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g = transitions["obs"], transitions["obs_next"], transitions["g"]
        transitions["obs"], transitions["g"] = self._preproc_og(o, g)
        transitions["obs_next"], transitions["g_next"] = self._preproc_og(o_next, g)

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions["obs"])
        g_norm = self.g_norm.normalize(transitions["g"])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions["obs_next"])
        g_next_norm = self.g_norm.normalize(transitions["g_next"])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

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

        # if MPI.COMM_WORLD.Get_rank() == 0:
        self.logger.store(
            ActorLoss=actor_loss.item(),
            CriticLoss=critic_loss.item(),
        )

    # update the network
    def _unreal_update_network(self):
        # sample the episodes
        transitions = self.world_model_buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g = transitions["obs"], transitions["obs_next"], transitions["g"]
        transitions["obs"], transitions["g"] = self._preproc_og(o, g)
        transitions["obs_next"], transitions["g_next"] = self._preproc_og(o_next, g)

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions["obs"])
        g_norm = self.g_norm.normalize(transitions["g"])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions["obs_next"])
        g_next_norm = self.g_norm.normalize(transitions["g_next"])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

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

    # Do the evaluation
    def _eval_agent(self):
        total_success_rate = []

        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation["observation"]
            g = observation["desired_goal"]

            for _ in range(self.env_params["max_timesteps"]):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)

                    # Convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new["observation"]
                g = observation_new["desired_goal"]
                per_success_rate.append(info["is_success"])

            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        # global_success_rate /= MPI.COMM_WORLD.Get_size()
        return local_success_rate  # global_success_rate / MPI.COMM_WORLD.Get_size()

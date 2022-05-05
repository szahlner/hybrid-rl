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
from utils.replay_buffer import ReplayBuffer
from utils.ddpg.arguments import DdpgNamespace
from utils.logger import EpochLogger

from world_model.deterministic_world_model import DeterministicWorldModel
from world_model.stochastic_world_model import StochasticWorldModel


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
        # self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params["action"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
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
        # self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class DDPG:
    def __init__(self, args: DdpgNamespace, env: Union[gym.Env, gym.GoalEnv], env_params: dict, logger: EpochLogger):
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

        # Create the replay buffer
        self.buffer = ReplayBuffer(self.env_params, self.args.buffer_size)

        # Create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.args.save_dir = os.path.join(self.logger.output_dir, self.args.save_dir)

            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

        # Model based section
        if self.args.model_based:
            model_dim_chunk = self.args.model_dim_chunk
            output_dim = env_params["obs"] + env_params["reward"]
            self.model_chunks = [model_dim_chunk for _ in range(output_dim // model_dim_chunk)]
            if output_dim % model_dim_chunk != 0:
                self.model_chunks.append(output_dim % model_dim_chunk)

            self.world_models = []
            for chunk in self.model_chunks:
                if self.args.model_type == "deterministic":
                    # Deterministic world model
                    self.world_models.append(
                        DeterministicWorldModel(
                            input_dim=env_params["obs"] + env_params["goal"] + env_params["action"],
                            output_dim=chunk,
                            network_dim=1,
                        )
                    )
                else:
                    # Stochastic world model
                    self.world_models.append(
                        StochasticWorldModel(
                            input_dim=env_params["obs"] + env_params["goal"] + env_params["action"],
                            output_dim=chunk,
                            network_dim=1,
                        )
                    )
            self.model_chunks.insert(0, 0)
            self.model_chunks = np.cumsum(self.model_chunks).tolist()

            # Buffers
            # self.simple_world_model_replay_buffer = ReplayBuffer(self.env_params, self.args.buffer_size)

            # Create the replay buffer
            self.world_model_params = deepcopy(self.env_params)
            self.world_model_params["max_timesteps"] = self.args.model_max_rollout_timesteps  # change max timesteps to rollout length
            self.world_model_buffer = ReplayBuffer(self.world_model_params, self.args.buffer_size)

    def learn(self):
        """
        train the network

        """
        start_time = time.time()
        obs = self.env.reset()

        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0

        self.logger.store(EpisodeReward=episode_reward)

        # Start to collect samples
        for ts in range(int(self.args.max_timesteps)):
            # Select action randomly or according to policy
            if ts < self.args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                pi = self.actor_network(obs_tensor)
                pi = pi.detach().cpu().numpy()
                action = pi + np.random.normal(0, self.env_params["action_max"] * self.args.exploration_noise, size=self.env_params["action"])
                action = action.clip(-self.env_params["action_max"], self.env_params["action_max"])

            # Perform action
            obs_next, reward, done, _ = self.env.step(action)
            done = done if episode_timesteps < self.env_params["max_timesteps"] else True

            # Store data in replay buffer
            self.buffer.store(batch=[obs.copy(), obs_next.copy(), np.array([[reward]]).copy(), np.array([[done]]).copy(), action.copy()])

            obs = obs_next
            episode_reward += reward

            # Model based section
            if self.args.model_based:
                if ts % self.args.model_training_freq == 0 and ts != 0:
                    transitions = self.buffer.sample(self.args.model_n_training_transitions)  # 10000

                    # Train chunked
                    training_inputs = transitions["obs"]
                    training_labels = np.concatenate(
                        [transitions["obs_next"] - transitions["obs"], transitions["r"]],
                        axis=-1
                    )

                    training_inputs = np.concatenate([training_inputs, transitions["actions"]], axis=-1)
                    for n in range(len(self.model_chunks) - 1):
                        tl = training_labels[:, self.model_chunks[n]:self.model_chunks[n + 1]]
                        self.world_models[n].train(training_inputs, tl)

                    # Rollout
                    n_transitions = self.args.model_n_rollout_transitions  # 10000
                    transitions = self.buffer.sample(n_transitions)
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
                            obs_tensor = torch.tensor(world_model_obs[:, n], dtype=torch.float32, device=device)
                            pi = self.actor_network(obs_tensor)
                            pi = pi.detach().cpu().numpy()
                            action = pi + np.random.normal(0, self.env_params["action_max"] * self.args.exploration_noise, size=pi.shape)
                            world_model_actions[:, n] = action.clip(-self.env_params["action_max"], self.env_params["action_max"])

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
                            diff[:, self.model_chunks[k]:self.model_chunks[k + 1]] = diff_
                            confidence[:, self.model_chunks[k]:self.model_chunks[k + 1]] = confidence_

                        if self.args.model_type == "deterministic":
                            world_model_mask[:, n] = np.all(np.where(confidence < 1, True, False), axis=-1)
                        else:
                            world_model_mask[:, n] = np.sum(confidence, axis=-1)

                        world_model_obs[:, n + 1] = world_model_obs[:, n] + diff[:, :self.env_params["obs"]]
                        world_model_r[:, n] = diff[:, self.env_params["obs"]:]

                    # Mark and select good ones
                    if self.args.model_type == "deterministic":
                        mask = np.any(world_model_mask, axis=-1)
                    else:
                        sorted_idx = np.argsort(np.sum(world_model_mask, axis=-1))
                        good_ones = int(len(sorted_idx) * self.args.model_stochastic_percentage)
                        world_model_mask[sorted_idx[:good_ones]] = True
                        world_model_mask[sorted_idx[good_ones:]] = False
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

                    if ts % self.args.policy_freq == 0:
                        self._soft_update_target_network(self.actor_target_network, self.actor_network)
                        self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # Train agent after collecting sufficient data
            if ts >= self.args.start_timesteps:
                self._update_network()

                if ts % self.args.policy_freq == 0:
                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)
            else:
                self.logger.store(ActorLoss=0, CriticLoss=0)

            if done:
                self.logger.store(EpisodeReward=episode_reward)

                # Reset environment
                obs = self.env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if ts % self.args.eval_freq == 0 and ts != 0 and ts >= self.args.start_timesteps:
                reward = self._eval_agent()
                self.logger.store(Reward=reward)

                self.logger.log_tabular("Epoch", episode_num)
                self.logger.log_tabular("Timesteps", ts)
                self.logger.log_tabular("Time", time.time() - start_time)
                self.logger.log_tabular("ActorLoss", with_min_and_max=True)
                self.logger.log_tabular("CriticLoss", with_min_and_max=True)
                self.logger.log_tabular("EpisodeReward", with_min_and_max=True)
                self.logger.log_tabular("Reward", with_min_and_max=True)

                if self.args.model_based:
                    self.logger.store(WorldModelReplayBufferSize=self.world_model_buffer.n_transitions_stored)
                    self.logger.log_tabular("WorldModelReplayBufferSize", with_min_and_max=True)

                self.logger.dump_tabular()

                if MPI.COMM_WORLD.Get_rank() == 0:
                    file_path = os.path.join(self.args.save_dir, "model.pt")
                    torch.save([self.actor_network.state_dict()], file_path)

                    if self.args.model_based:
                        for n, model in enumerate(self.world_models):
                            file_path = os.path.join(self.args.save_dir, f"world_model_{n}.pt")
                            model.save(file_path)

    # soft update
    def _soft_update_target_network(self, target, source) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # transfer them into the tensor
        obs_tensor = torch.tensor(transitions["obs"], dtype=torch.float32, device=device)
        obs_next_tensor = torch.tensor(transitions["obs_next"], dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32, device=device)
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32, device=device)
        d_tensor = torch.tensor(transitions["d"], dtype=torch.float32, device=device)

        # calculate the target Q value function
        with torch.no_grad():
            # concatenate the stuffs
            actions_next = self.actor_target_network(obs_next_tensor)
            q_next_value = self.critic_target_network(obs_next_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + (1 - d_tensor) * self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

        # the q loss
        real_q_value = self.critic_network(obs_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(obs_tensor)
        actor_loss = -self.critic_network(obs_tensor, actions_real).mean()

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

        # transfer them into the tensor
        obs_tensor = torch.tensor(transitions["obs"], dtype=torch.float32, device=device)
        obs_next_tensor = torch.tensor(transitions["obs_next"], dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32, device=device)
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32, device=device)
        d_tensor = torch.tensor(transitions["d"], dtype=torch.float32, device=device)

        # calculate the target Q value function
        with torch.no_grad():
            # concatenate the stuffs
            actions_next = self.actor_target_network(obs_next_tensor)
            q_next_value = self.critic_target_network(obs_next_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + (1 - d_tensor) * self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

        # the q loss
        real_q_value = self.critic_network(obs_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(obs_tensor)
        actor_loss = -self.critic_network(obs_tensor, actions_real).mean()

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
        total_reward = []

        for _ in range(self.args.n_test_rollouts):
            per_reward = []
            obs, done = self.env.reset(), False

            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    pi = self.actor_network(obs_tensor)

                    # Convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

                obs, reward, done, _ = self.env.step(actions)
                per_reward.append(reward)

            total_reward.append(sum(per_reward))

        # total_reward = np.array(total_reward)
        local_reward = np.mean(total_reward)
        return local_reward

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import gym
import time
from mpi4py import MPI
from typing import Tuple, Union
from copy import deepcopy

from utils.mpi.mpi_utils import sync_grads, sync_networks, sync_scalar
from utils.mpi.normalizer import Normalizer
from utils.replay_buffer import ReplayBuffer
from utils.sac.arguments import SacHerNamespace
from utils.logger import EpochLogger

from world_model.deterministic_world_model import DeterministicWorldModel
from world_model.stochastic_world_model import StochasticWorldModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    def __init__(self, env_params: dict) -> None:
        """
        The critic network.

        Args:
            env_params (dict): Environment parameters.
        """
        super(Critic, self).__init__()

        self.max_action = env_params["action_max"]

        # Q1 architecture
        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"] + env_params["action"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out_1 = nn.Linear(256, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(env_params["obs"] + env_params["goal"] + env_params["action"], 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.q_out_2 = nn.Linear(256, 1)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([x, actions / self.max_action], dim=1)

        # Q1 architecture
        x1 = torch.relu(self.fc1(x))
        x1 = torch.relu(self.fc2(x1))
        x1 = torch.relu(self.fc3(x1))
        q_value_1 = self.q_out_1(x1)

        # Q2 architecture
        x2 = torch.relu(self.fc4(x))
        x2 = torch.relu(self.fc5(x2))
        x2 = torch.relu(self.fc6(x2))
        q_value_2 = self.q_out_2(x2)

        return q_value_1, q_value_2


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

        self.mean_linear = nn.Linear(256, env_params["action"])
        self.log_std_linear = nn.Linear(256, env_params["action"])

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)

        # For re-parameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.max_action

        return action, log_prob, mean


class SAC:
    def __init__(self, args: SacHerNamespace, env: Union[gym.Env, gym.GoalEnv], env_params: dict, logger: EpochLogger):
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

        # Entropy tuning
        if self.args.automatic_entropy_tuning:
            if self.args.target_entropy is not None:
                self.target_entropy = self.args.target_entropy
            else:
                self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr_alpha)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            # Alpha
            self.alpha = self.args.alpha

        # Create the replay buffer
        self.buffer = ReplayBuffer(self.env_params, self.args.buffer_size)

        # Create the normalizer
        self.o_norm = Normalizer(size=env_params["obs"], default_clip_range=self.args.clip_range)

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
            self.simple_world_model_replay_buffer = ReplayBuffer(self.env_params, self.args.buffer_size)

            # Create the replay buffer
            self.world_model_params = deepcopy(self.env_params)
            self.world_model_params["max_timesteps"] = self.args.model_max_rollout_timesteps  # change max timesteps to rollout length
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
                            pi, _, _ = self.actor_network.sample(input_tensor)
                            action = pi.cpu().numpy().squeeze()

                        # Feed the actions into the environment
                        obs_new, reward, done, info = self.env.step(action)

                        # if MPI.COMM_WORLD.Get_rank() == 0:
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
                                transitions = self.simple_world_model_replay_buffer.sample(self.args.model_n_training_transitions)  # 10000

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
                                n_transitions = self.args.model_n_rollout_transitions  # 10000
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
                                        actions_tensor, _, _ = self.actor_network.sample(input_tensor)
                                        world_model_actions[:, n] = actions_tensor.cpu().numpy().squeeze()

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
            self.logger.store(SuccessRate=success_rate)

            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("Timesteps", ts)
            self.logger.log_tabular("Time", time.time() - start_time)
            self.logger.log_tabular("SuccessRate", with_min_and_max=True)
            self.logger.log_tabular("ActorLoss", with_min_and_max=True)
            self.logger.log_tabular("CriticLoss", with_min_and_max=True)
            self.logger.log_tabular("AlphaLoss", with_min_and_max=True)
            self.logger.log_tabular("Reward", with_min_and_max=True)

            if self.args.model_based:
                self.logger.store(WorldModelReplayBufferSize=self.world_model_buffer.n_transitions_stored)
                self.logger.log_tabular("WorldModelReplayBufferSize", with_min_and_max=True)

            self.logger.dump_tabular()
            sys.stdout.flush()

            if MPI.COMM_WORLD.Get_rank() == 0:
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
            actions_next, obs_next_log_pi, _ = self.actor_network.sample(inputs_next_norm_tensor)
            q1_next_value, q2_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            min_target_q_value = torch.min(q1_next_value, q2_next_value) - self.alpha * obs_next_log_pi
            target_q_value = r_tensor + self.args.gamma * min_target_q_value
            target_q_value = target_q_value.detach()

            # clip the q value
            if self.args.clip_return:
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        q1, q2 = self.critic_network(inputs_norm_tensor, actions_tensor)
        q1_loss = F.mse_loss(q1, target_q_value)
        q2_loss = F.mse_loss(q2, target_q_value)
        critic_loss = q1_loss + q2_loss

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        pi, log_pi, _ = self.actor_network.sample(inputs_norm_tensor)
        q1_pi, q2_pi = self.critic_network(inputs_norm_tensor, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        if self.args.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha = sync_scalar(self.alpha.detach().cpu().numpy())
            self.alpha.data.copy_(torch.tensor(alpha, dtype=torch.float32, device=device))
        else:
            alpha_loss = torch.tensor(0.).to(device)

        # if MPI.COMM_WORLD.Get_rank() == 0:
        self.logger.store(
            ActorLoss=actor_loss.item(),
            CriticLoss=critic_loss.item(),
            AlphaLoss=alpha_loss.item(),
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
            actions_next, obs_next_log_pi, _ = self.actor_network.sample(inputs_next_norm_tensor)
            q1_next_value, q2_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            min_target_q_value = torch.min(q1_next_value, q2_next_value) - self.alpha * obs_next_log_pi
            target_q_value = r_tensor + self.args.gamma * min_target_q_value
            target_q_value = target_q_value.detach()

            # clip the q value
            if self.args.clip_return:
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        q1, q2 = self.critic_network(inputs_norm_tensor, actions_tensor)
        q1_loss = F.mse_loss(q1, target_q_value)
        q2_loss = F.mse_loss(q2, target_q_value)
        critic_loss = q1_loss + q2_loss

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        pi, log_pi, _ = self.actor_network.sample(inputs_norm_tensor)
        q1_pi, q2_pi = self.critic_network(inputs_norm_tensor, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        if self.args.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha = sync_scalar(self.alpha.detach().cpu().numpy())
            self.alpha.data.copy_(torch.tensor(alpha, dtype=torch.float32, device=device))

    # Do the evaluation
    def _eval_agent(self):
        total_success_rate = []

        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            obs = self.env.reset()

            for _ in range(self.env_params["max_timesteps"]):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs)
                    _, _, pi = self.actor_network.sample(input_tensor)

                    # Convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

                obs, _, _, info = self.env.step(actions)
                per_success_rate.append(info["is_success"])

            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        # global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        # global_success_rate /= MPI.COMM_WORLD.Get_size()
        return local_success_rate  # global_success_rate / MPI.COMM_WORLD.Get_size()

import imageio
import numpy as np
import os
import torch
import torch.nn as nn

from mpi4py import MPI
from termcolor import colored
from colorama import init as colorama_init

from hybridrl.models import RSMPC_DYNAMICS_MODELS
from hybridrl.environments import REWARD_FUNCTIONS
from hybridrl.buffer import ReplayBuffer
from hybridrl.utils.mpi import sync_networks, sync_grads
from hybridrl.utils.logger import Logger


if os.name == 'nt':
    colorama_init()


class RSMPCOptimizer:
    # Not really an optimizer, this is for convenience.
    # At least we are doing random shooting.
    def __init__(self, agent_params, env_params, reward_fn, dynamics):
        self.agent_params = agent_params
        self.env_params = env_params
        self.reward_fn = reward_fn
        self.dynamics = dynamics

    def set_dynamics(self, state_dict):
        self.dynamics.load_state_dict(state_dict)

    def _get_random_actions(self, population_size):
        return np.random.uniform(-1, 1, size=(population_size, self.env_params.action_dim))

    def __call__(self, obs):
        init_actions = self._get_random_actions(self.agent_params.population_size)
        obs = np.tile(obs, (self.agent_params.population_size, 1))
        total_rewards = np.zeros(self.agent_params.population_size)

        for n in range(self.agent_params.horizon):
            if n == 0:
                actions = init_actions
            else:
                actions = self._get_random_actions(self.agent_params.population_size)

            obs = torch.tensor(obs, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            with torch.no_grad():
                obs_next = self.dynamics(obs, actions).detach()

            obs = obs.detach().cpu().numpy()
            obs_next = obs_next.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            rewards = self.reward_fn(obs, actions, obs_next)
            total_rewards += rewards

            obs = obs_next

        idx_best = total_rewards.argmax()
        action = np.array([init_actions[idx_best]])

        return np.clip(action, -1, 1)[0]


class RSMPC:
    """Random shooting MPC (RsMPC) Agent"""

    def __init__(self, experiment_params, agent_params, env_params, env, test_params):
        self.experiment_params = experiment_params
        self.agent_params = agent_params
        self.env_params = env_params
        self.env = env
        self.test_params = test_params

        # tensorboard
        if MPI.COMM_WORLD.Get_rank() == 0 and not test_params.is_test:
            self.logger = Logger(log_dir=experiment_params.log_dir)

        # reward function
        assert env_params.id in REWARD_FUNCTIONS, 'Reward function must be registered.'
        self.reward_fn = REWARD_FUNCTIONS[env_params.id](env_params)

        # dynamics network
        self.dynamics = RSMPC_DYNAMICS_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)

        # sync networks
        sync_networks(self.dynamics)

        # create optimizer
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=agent_params.lr_dynamics)

        # create replay buffers
        self.buffer = ReplayBuffer(agent_params.buffer_size, experiment_params.seed)
        self.buffer_test = ReplayBuffer(agent_params.buffer_size, experiment_params.seed)

        # training / evaluation
        self.is_training = True

        self.optimizer = RSMPCOptimizer(agent_params, env_params, self.reward_fn, self.dynamics)

    def train(self):
        total_steps, update_steps = 0, 0
        all_rewards, all_successes = [], []

        for epoch in range(self.experiment_params.n_epochs):
            rollout_steps = 0
            cycle_dynamics_loss, cycle_dynamics_test_loss = 0, 0

            for cycle in range(self.experiment_params.n_cycles):
                rollout_dynamics_loss, rollout_dynamics_test_loss = 0, 0

                for rollout in range(self.experiment_params.n_rollouts):
                    obs = self.env.reset()

                    rollout_reward, rollout_success = 0, 0

                    for _ in range(self.env_params.max_episode_steps):
                        action = self.optimizer(obs)

                        obs_next, reward, done, info = self.env.step(action)

                        if np.random.uniform() < 0.1:
                            self.buffer_test.add(obs, action, reward, done, obs_next)
                        else:
                            self.buffer.add(obs, action, reward, done, obs_next)

                        obs = obs_next

                        rollout_reward += reward

                        try:
                            rollout_success += float(info['is_success'])
                        except KeyError:
                            rollout_success += 0

                        total_steps += 1
                        rollout_steps += 1

                        if done and self.env_params.end_on_done:
                            break

                    all_rewards.append(rollout_reward)
                    all_successes.append(rollout_success / self.env_params.max_episode_steps)

                if len(self.buffer) > self.agent_params.batch_size:
                    n_batches = rollout_steps if self.experiment_params.n_train_batches is None \
                        else self.experiment_params.n_train_batches
                    for _ in range(n_batches):
                        dynamic_loss, dynamics_test_loss = self._learn()
                        rollout_dynamics_loss += dynamic_loss
                        rollout_dynamics_test_loss += dynamics_test_loss

                        update_steps += 1

                    self.optimizer.set_dynamics(self.dynamics.state_dict())

                cycle_dynamics_loss += rollout_dynamics_loss
                cycle_dynamics_test_loss += rollout_dynamics_test_loss

            avg_reward = float(np.mean(all_rewards[-100:]))
            avg_success = float(np.mean(all_successes[-100:]))

            if MPI.COMM_WORLD.Get_rank() == 0:
                self.is_training = False
                avg_test_reward, avg_test_success = self._test()
                self.is_training = True

                data = {'time_steps': total_steps,
                        'update_steps': update_steps,
                        'epoch': epoch + 1,
                        'cycles': cycle + 1,
                        'rollouts': rollout + 1,
                        'rollout_steps': rollout_steps,
                        'current_reward': all_rewards[-1],
                        'current_success': all_successes[-1],
                        'avg_reward': avg_reward,
                        'avg_success': avg_success,
                        'avg_test_reward': avg_test_reward,
                        'avg_test_success': avg_test_success,
                        'cycle_dynamics_loss': cycle_dynamics_loss,
                        'cycle_dynamics_test_loss': cycle_dynamics_test_loss,
                        'rollout_dynamics_loss': rollout_dynamics_loss,
                        'rollout_dynamics_test_loss': rollout_dynamics_test_loss}
                self.logger.add(data)

        self._save()

    def _learn(self):
        obs, actions, _, _, obs_next = self.buffer.sample_batch(self.agent_params.batch_size)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, dtype=torch.float32)

        obs_predicted = self.dynamics(obs, actions)

        # dynamics gradient
        dynamics_criterion = nn.MSELoss()
        dynamics_loss = dynamics_criterion(obs_predicted, obs_next)
        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        sync_grads(self.dynamics)
        self.dynamics_optimizer.step()

        # dynamics test
        obs, actions, _, _, obs_next = self.buffer_test.sample_batch(int(0.1 * self.agent_params.batch_size))

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, dtype=torch.float32)

        obs_predicted = self.dynamics(obs, actions)

        dynamics_test_criterion = nn.MSELoss()
        dynamics_test_loss = dynamics_test_criterion(obs_predicted, obs_next)

        return dynamics_loss.item(), dynamics_test_loss.item()

    def _save(self):
        torch.save(self.dynamics.state_dict(), '{}/dynamics.pt'.format(self.experiment_params.log_dir))
        self.logger.save()

    def _load(self):
        dynamics_state_dict = torch.load('{}/dynamics.pt'.format(self.experiment_params.log_dir))
        self.dynamics.load_state_dict(dynamics_state_dict)
        self.optimizer.set_dynamics(dynamics_state_dict)

    def test(self):
        self._load()
        self.is_training = False
        self._test()

    def _test(self):
        avg_reward, avg_success = 0, 0
        images = []

        for episode in range(self.test_params.n_episodes):
            obs = self.env.reset()
            episode_reward, episode_steps, episode_success = 0, 0, 0

            for _ in range(self.env_params.max_episode_steps):
                if self.test_params.is_test and self.test_params.gif:
                    image = self.env.render(mode='rgb_array')
                    images.append(image)
                elif self.test_params.is_test and self.test_params.visualize:
                    self.env.render()

                with torch.no_grad():
                    action = self.optimizer(obs)
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward

                try:
                    episode_success += float(info['is_success'])
                except KeyError:
                    episode_success += 0

                episode_steps += 1

                if done and self.env_params.end_on_done:
                    break

            avg_reward += episode_reward
            avg_success += episode_success / self.env_params.max_episode_steps

            if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
                print(colored('[TEST]', 'green') + ' episode: {}, episode_reward: {:.3f}, '
                              'episode_success: {:.2f}'.format(episode + 1, episode_reward,
                                                               episode_success / self.env_params.max_episode_steps))

        avg_reward /= self.test_params.n_episodes
        avg_success /= self.test_params.n_episodes

        if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
            print(colored('[TEST]', 'green') + ' avg reward: {:.3f}'.format(avg_reward))
            print(colored('[TEST]', 'green') + ' avg success: {:.3f}'.format(avg_success))

            if self.test_params.gif:
                imageio.mimsave('{}/{}.gif'.format(self.experiment_params.log_dir, self.env_params.id),
                                [np.array(image) for n, image in enumerate(images) if n % 3 == 0], fps=10)

        return avg_reward, avg_success

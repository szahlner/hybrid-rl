import imageio
import numpy as np
import os
import torch
import torch.nn as nn

from mpi4py import MPI
from termcolor import colored
from colorama import init as colorama_init

from hybridrl.models import CEM_DYNAMICS_MODELS
from hybridrl.environments import REWARD_FUNCTIONS
from hybridrl.buffer import ReplayBuffer
from hybridrl.utils.mpi import sync_networks, sync_grads
from hybridrl.utils.logger import Logger


if os.name == 'nt':
    colorama_init()


class CEMOptimizer:
    # Cross Entropy Method (CEM) optimizer.
    # Estimates the action sequence for a certain initial state by choosing elite sequences and uses their mean.
    def __init__(self, agent_params, env_params, reward_fn, dynamics):
        # safety first
        assert agent_params.n_elite < agent_params.population_size, \
            'Number of elites must not succeed number of population'

        self.agent_params = agent_params
        self.env_params = env_params
        self.reward_fn = reward_fn
        self.dynamics = dynamics

        self.solution_dim = agent_params.horizon * env_params.action_dim
        self.init_mean = np.tile(agent_params.init_mean, self.solution_dim)
        self.init_variance = np.tile(agent_params.init_variance, self.solution_dim)

        self.previous_solution = None

        def sample_truncated_normal(shape, mu, sigma, a, b):
            """Pytorch implementation of truncated normal distribution"""
            uniform = torch.rand(shape)
            normal = torch.distributions.normal.Normal(0, 1)

            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma

            alpha_normal_cdf = normal.cdf(alpha)
            p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

            p = p.numpy()
            one = np.array(1, dtype=p.dtype)
            epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
            v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
            x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))

            return x
        self.sample_trunc_norm = sample_truncated_normal

    def set_dynamics(self, state_dict):
        self.dynamics.load_state_dict(state_dict)

    def reset(self):
        self.previous_solution = self.init_mean

    def __call__(self, obs_start):
        mean, var = self.previous_solution, self.init_variance
        size = [self.agent_params.population_size, self.solution_dim]
        lower_bound = torch.FloatTensor(np.array([-1] * self.solution_dim))
        upper_bound = torch.FloatTensor(np.array([1] * self.solution_dim))

        for _ in range(self.agent_params.max_iter):
            if self.agent_params.sampler == 'truncated_normal':
                mu = torch.FloatTensor(mean)
                sigma = torch.sqrt(torch.FloatTensor(var))
                samples = self.sample_trunc_norm(size, mu, sigma, lower_bound, upper_bound).numpy()
            else:
                samples = np.random.normal(mean, np.sqrt(var), size=size)
                samples = np.clip(samples, a_min=-1, a_max=1)

            samples = samples.reshape((-1, self.agent_params.horizon, self.env_params.action_dim))

            obs = np.tile(obs_start.copy(), (self.agent_params.population_size, 1))
            total_rewards = np.zeros(self.agent_params.population_size)

            for n in range(self.agent_params.horizon):
                actions = samples[:, n, :]

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

            samples = samples.reshape(size)
            idx_best = np.argsort(total_rewards)[::-1]
            elites = samples[idx_best][:self.agent_params.n_elite]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.agent_params.alpha * mean + (1 - self.agent_params.alpha) * new_mean
            var = self.agent_params.alpha * var + (1 - self.agent_params.alpha) * new_var

            sol, solvar = mean, var

            if np.max(var) < self.agent_params.epsilon:
                break

        self.previous_solution = np.concatenate([np.copy(sol)[self.env_params.action_dim:],
                                                 np.zeros(self.env_params.action_dim)])

        action = sol[:self.env_params.action_dim]
        return np.clip(action, -1, 1)


class CEM:
    """Cross Entropy Method (CEM) Agent"""

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
        self.reward_fn = REWARD_FUNCTIONS[env_params.id]()

        # dynamics network
        self.dynamics = CEM_DYNAMICS_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)

        # sync networks
        sync_networks(self.dynamics)

        # create optimizer
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=agent_params.lr_dynamics)

        # create replay buffer
        self.buffer = ReplayBuffer(agent_params.buffer_size, experiment_params.seed)

        # training / evaluation
        self.is_training = True

        self.optimizer = CEMOptimizer(agent_params, env_params, self.reward_fn, self.dynamics)

    def train(self):
        total_steps = 0
        update_steps = 0

        all_rewards = []
        for epoch in range(self.experiment_params.n_epochs):
            rollout_steps = 0

            for cycle in range(self.experiment_params.n_cycles):
                episode_dynamic_loss = 0

                for rollout in range(self.experiment_params.n_rollouts):
                    obs = self.env.reset()
                    self.optimizer.reset()

                    rollout_reward = 0

                    for _ in range(self.env_params.max_episode_steps):
                        action = self.optimizer(obs)
                        #action = self.get_action(obs)[0]

                        obs_next, reward, done, info = self.env.step(action)
                        self.buffer.add(obs, action, reward, done, obs_next)
                        obs = obs_next

                        rollout_reward += reward
                        total_steps += 1
                        rollout_steps += 1

                        if done and self.env_params.end_on_done:
                            break

                    all_rewards.append(rollout_reward)

                if len(self.buffer) > self.agent_params.batch_size:
                    n_batches = rollout_steps if self.experiment_params.n_train_batches is None \
                        else self.experiment_params.n_train_batches
                    for _ in range(n_batches):
                        dynamic_loss = self._learn()
                        episode_dynamic_loss += dynamic_loss

                        update_steps += 1

                    self.optimizer.set_dynamics(self.dynamics.state_dict())

            avg_reward = float(np.mean(all_rewards[-100:]))

            # tensorboard
            if MPI.COMM_WORLD.Get_rank() == 0:
                # test reward = rollout reward, no need to log
                avg_test_reward = float(np.mean(all_rewards[-self.test_params.n_episodes:]))

                data = {'time_steps': total_steps,
                        'update_steps': update_steps,
                        'epoch': epoch + 1,
                        'cycles': cycle + 1,
                        'rollouts': rollout + 1,
                        'rollout_steps': rollout_steps,
                        'current_reward': all_rewards[-1],
                        'avg_reward': avg_reward,
                        'avg_test_reward': avg_test_reward}
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

        return dynamics_loss.item()

    def _save(self):
        torch.save(self.dynamics.state_dict(), '{}/dynamics.pt'.format(self.experiment_params.log_dir))
        self.logger.save()

    def _load(self):
        dynamics_state_dict = torch.load('{}/dynamics.pt'.format(self.experiment_params.log_dir))
        self.dynamics.load_state_dict(dynamics_state_dict)

    def test(self):
        self._load()
        self.is_training = False
        self._test()

    def _test(self):
        avg_reward = 0
        images = []

        for episode in range(self.test_params.n_episodes):
            obs = self.env.reset()
            self.optimizer.reset()

            episode_reward, episode_steps = 0, 0

            for _ in range(self.env_params.max_episode_steps):
                if self.test_params.is_test and self.test_params.gif:
                    image = self.env.render(mode='rgb_array')
                    images.append(image)
                elif self.test_params.is_test and self.test_params.visualize:
                    self.env.render()

                with torch.no_grad():
                    action = self.optimizer(obs)
                    # action = self.get_action(obs)[0]
                obs, reward, _, _ = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

            avg_reward += episode_reward

            if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
                print(colored('[TEST]', 'green') + ' episode: {}, episode_reward: {:.3f}'.format(episode, episode_reward))

        avg_reward /= self.test_params.n_episodes

        if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
            print(colored('[TEST]', 'green') + ' avg reward: {:.3f}'.format(avg_reward))

            if self.test_params.gif:
                imageio.mimsave('{}/{}.gif'.format(self.experiment_params.log_dir, self.env_params.id),
                                [np.array(image) for n, image in enumerate(images) if n % 3 == 0], fps=10)

        return avg_reward

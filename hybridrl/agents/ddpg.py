import imageio
import numpy as np
import os
import torch
import torch.nn as nn

from mpi4py import MPI
from termcolor import colored
from colorama import init as colorama_init

from hybridrl.models import DDPG_ACTOR_MODELS, DDPG_CRITIC_MODELS
from hybridrl.buffer import ReplayBuffer
from hybridrl.noise import OUNoise
from hybridrl.utils.mpi import sync_networks, sync_grads
from hybridrl.utils.logger import Logger


if os.name == 'nt':
    colorama_init()


class DDPG:
    """Deep Deterministic Policy Gradient (DDPG) Agent"""

    def __init__(self, experiment_params, agent_params, env_params, env, test_params):
        self.experiment_params = experiment_params
        self.agent_params = agent_params
        self.env_params = env_params
        self.env = env
        self.test_params = test_params

        # tensorboard
        if MPI.COMM_WORLD.Get_rank() == 0 and not test_params.is_test:
            self.logger = Logger(log_dir=experiment_params.log_dir)

        # create networks
        self.actor = DDPG_ACTOR_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)
        self.critic = DDPG_CRITIC_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)

        # sync networks
        sync_networks(self.actor)
        sync_networks(self.critic)

        # create target networks and load weights
        self.actor_target = DDPG_ACTOR_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)
        self.critic_target = DDPG_CRITIC_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)

        # hard update
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_params.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=agent_params.lr_critic)

        # create replay buffer
        self.buffer = ReplayBuffer(agent_params.buffer_size, experiment_params.seed)

        # create noise
        self.noise = OUNoise(env_params.action_dim)
        self.noise_eps = agent_params.noise_eps

        # training / evaluation
        self.is_training = True

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        action = self.actor(obs).detach()
        action = action.squeeze(0).cpu().numpy()
        action += self.is_training * self.noise_eps * self.noise.noise()
        action = np.clip(action, -1, 1)

        if np.random.uniform() < self.agent_params.random_exploration and self.is_training:
            action = np.random.uniform(-1, 1, size=action.size)

        return action

    def train(self):
        total_steps, update_steps = 0, 0
        all_rewards = []

        for epoch in range(self.experiment_params.n_epochs):
            rollout_steps = 0

            for cycle in range(self.experiment_params.n_cycles):
                cycle_actor_loss, cycle_critic_loss = 0, 0

                for rollout in range(self.experiment_params.n_rollouts):
                    obs = self.env.reset()
                    self._reset()

                    rollout_reward = 0

                    # decay noise
                    self._decay_epsilon()

                    for _ in range(self.env_params.max_episode_steps):
                        action = self.get_action(obs)

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
                        actor_loss, critic_loss = self._learn()
                        self._soft_update(self.actor_target, self.actor)
                        self._soft_update(self.critic_target, self.critic)

                        cycle_actor_loss += actor_loss
                        cycle_critic_loss += critic_loss
                        update_steps += 1

            avg_reward = float(np.mean(all_rewards[-100:]))

            if MPI.COMM_WORLD.Get_rank() == 0:
                self.is_training = False
                avg_test_reward = self._test()
                self.is_training = True

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
        obs, action, reward, done, obs_next = self.buffer.sample_batch(self.agent_params.batch_size)

        done = (done == False) * 1
        obs = torch.tensor(obs, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(np.expand_dims(reward, axis=1), dtype=torch.float32)
        done = torch.tensor(np.expand_dims(done, axis=1), dtype=torch.float32)
        obs_next = torch.tensor(obs_next, dtype=torch.float32)

        with torch.no_grad():
            action_target = self.actor_target(obs_next)
            target_q = self.critic_target(obs_next, action_target).detach()
            #q_expected = reward[:, None] + done[:, None] * self.agent_params.gamma * target_q
            #q_expected = reward + done * self.agent_params.gamma * target_q
            q_expected = reward + self.agent_params.gamma * target_q
        q_predicted = self.critic(obs, action)

        # critic gradient
        critic_criterion = nn.MSELoss()
        critic_loss = critic_criterion(q_predicted, q_expected)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optimizer.step()

        # actor gradient
        action_predicted = self.actor(obs)
        actor_loss = (-self.critic.forward(obs, action_predicted)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def _reset(self):
        self.noise.reset()

    def _decay_epsilon(self):
        self.noise_eps = max(self.noise_eps - self.agent_params.noise_eps_decay, self.agent_params.noise_eps_min)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - self.agent_params.polyak) + param.data * self.agent_params.polyak)

    def _save(self):
        torch.save(self.actor.state_dict(), '{}/actor.pt'.format(self.experiment_params.log_dir))
        torch.save(self.critic.state_dict(), '{}/critic.pt'.format(self.experiment_params.log_dir))
        self.logger.save()

    def _load(self):
        actor_state_dict = torch.load('{}/actor.pt'.format(self.experiment_params.log_dir))
        self.actor.load_state_dict(actor_state_dict)

        critic_state_dict = torch.load('{}/critic.pt'.format(self.experiment_params.log_dir))
        self.critic.load_state_dict(critic_state_dict)

    def test(self):
        self._load()
        self.is_training = False
        self._test()

    def _test(self):
        avg_reward = 0
        avg_success = 0
        images = []

        for episode in range(self.test_params.n_episodes):
            obs = self.env.reset()
            episode_reward, episode_steps, episode_success = 0, 0, 0

            for _ in range(self.env_params.max_episode_steps):
                if self.test_params.is_test and self.test_params.gif:
                    image = self.env.render(mode='rgb_array')
                    images.append(image)

                if self.test_params.is_test and self.test_params.visualize:
                    self.env.render()

                with torch.no_grad():
                    #obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    #action = self.actor(obs).detach()
                    #action = action.squeeze(0).cpu().numpy()
                    #action = np.clip(action, -1., 1.)
                    action = self.get_action(obs)
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_success += 0 #float(info['is_success'])
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

        if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
            print(colored('[TEST]', 'green') + ' avg reward: {:.3f}'.format(avg_reward))

            if self.test_params.gif:
                imageio.mimsave('{}/{}.gif'.format(self.experiment_params.log_dir, self.env_params.id),
                                [np.array(image) for n, image in enumerate(images) if n % 3 == 0], fps=10)

        return avg_reward

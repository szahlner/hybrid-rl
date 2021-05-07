import imageio
import numpy as np
import os
import torch
import torch.nn as nn

from mpi4py import MPI
from termcolor import colored
from colorama import init as colorama_init
from scipy.interpolate import interp1d

from hybridrl.models import RSMPCDDPG_DYNAMICS_MODELS
from hybridrl.models import DDPG_ACTOR_MODELS, DDPG_CRITIC_MODELS
from hybridrl.environments import REWARD_FUNCTIONS
from hybridrl.buffer import ReplaySimilarityBuffer
from hybridrl.utils.mpi import sync_networks, sync_grads, Normalizer
from hybridrl.utils.logger import Logger

if os.name == 'nt':
    colorama_init()


class RSMPCDDPG:
    """Random shooting MPC (RsMPC) Agent"""

    def __init__(self, experiment_params, agent_params, env_params, env, test_params):
        self.experiment_params = experiment_params
        self.agent_params = agent_params
        self.env_params = env_params
        self.env = env
        self.test_params = test_params

        # logger
        if MPI.COMM_WORLD.Get_rank() == 0 and not test_params.is_test:
            self.logger = Logger(log_dir=experiment_params.log_dir)

        # reward function
        assert env_params.id in REWARD_FUNCTIONS, 'Reward function must be registered.'
        self.reward_fn = REWARD_FUNCTIONS[env_params.id](env_params)

        # networks
        self.dynamics = RSMPCDDPG_DYNAMICS_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)

        self.actor = DDPG_ACTOR_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)
        self.critic = DDPG_CRITIC_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)

        self.actor_target = DDPG_ACTOR_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)
        self.critic_target = DDPG_CRITIC_MODELS[env_params.id](env_params.obs_dim, env_params.action_dim)

        # sync networks
        sync_networks(self.dynamics)
        sync_networks(self.actor)
        sync_networks(self.critic)

        # hard update
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=agent_params.lr_dynamics)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_params.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=agent_params.lr_critic)

        # buffers
        self.buffer = ReplaySimilarityBuffer(agent_params.buffer_size, experiment_params.seed, env_params)
        self.buffer_test = ReplaySimilarityBuffer(agent_params.buffer_size, experiment_params.seed, env_params)

        self.buffer_ddpg_all = ReplaySimilarityBuffer(agent_params.buffer_size, experiment_params.seed, env_params)
        self.buffer_ddpg = ReplaySimilarityBuffer(agent_params.buffer_size, experiment_params.seed, env_params)

        # normalizer
        self.normalizer = Normalizer(env_params.obs_dim)

        # training / evaluation
        self.is_training = True

        # steps
        self.steps_rs = []
        self.steps_ddpg = []

    def _get_random_actions(self, population_size):
        return np.random.uniform(-1, 1, size=(population_size, self.env_params.action_dim))

    def act_rs_mpc(self, obs):
        init_actions = self._get_random_actions(self.agent_params.population_size)
        obs = np.tile(obs, (self.agent_params.population_size, 1))
        total_rewards = np.zeros(self.agent_params.population_size)

        # ddpg stuff
        obs_ddpg = obs.copy()
        obs_norm = self.normalizer.normalize(obs_ddpg)
        obs_norm = torch.tensor(obs_norm, dtype=torch.float32)
        init_actions_ddpg = self.actor(obs_norm)
        total_rewards_ddpg = np.zeros(self.agent_params.population_size)

        r_obs = np.empty(shape=(self.agent_params.population_size, self.env_params.obs_dim, self.agent_params.horizon),
                         dtype=object)
        r_actions = np.empty(
            shape=(self.agent_params.population_size, self.env_params.action_dim, self.agent_params.horizon),
            dtype=object)
        r_rewards = np.empty(shape=(self.agent_params.population_size, 1, self.agent_params.horizon), dtype=object)
        r_obs_next = np.empty(
            shape=(self.agent_params.population_size, self.env_params.obs_dim, self.agent_params.horizon), dtype=object)

        for n in range(self.agent_params.horizon):
            obs = torch.tensor(obs, dtype=torch.float32)

            obs_norm = self.normalizer.normalize(obs_ddpg)
            obs_norm = torch.tensor(obs_norm, dtype=torch.float32)
            obs_ddpg = torch.tensor(obs_ddpg, dtype=torch.float32)

            if n == 0:
                actions = init_actions
                actions_ddpg = init_actions_ddpg
            else:
                actions = self._get_random_actions(self.agent_params.population_size)
                actions_ddpg = self.actor(obs_norm)

            actions = torch.tensor(actions, dtype=torch.float32)
            with torch.no_grad():
                obs_next = self.dynamics(obs, actions)
                obs_next_ddpg = self.dynamics(obs_ddpg, actions_ddpg)

            obs = obs.detach().cpu().numpy()
            obs_ddpg = obs_ddpg.detach().cpu().numpy()

            obs_next = obs_next.detach().cpu().numpy()
            obs_next_ddpg = obs_next_ddpg.detach().cpu().numpy()

            actions = actions.detach().cpu().numpy()
            actions_ddpg = actions_ddpg.detach().cpu().numpy()

            rewards = self.reward_fn(obs, actions, obs_next)
            rewards_ddpg = self.reward_fn(obs_ddpg, actions_ddpg, obs_next_ddpg)

            total_rewards += rewards
            total_rewards_ddpg += rewards_ddpg

            r_obs[:, :, n] = obs_ddpg
            r_actions[:, :, n] = actions_ddpg
            r_rewards[:, :, n] = np.expand_dims(rewards_ddpg, axis=1)
            r_obs_next[:, :, n] = obs_next_ddpg

            obs = obs_next
            obs_ddpg = obs_next_ddpg

        idx_best = total_rewards.argmax()
        idx_best_ddpg = total_rewards_ddpg.argmax()

        rollouts_to_take = self.agent_params.horizon
        self.buffer_ddpg_all.add(r_obs[idx_best_ddpg, :, :rollouts_to_take].T,
                                 r_actions[idx_best_ddpg, :, :rollouts_to_take].T,
                                 r_rewards[idx_best_ddpg, :, :rollouts_to_take].T,
                                 np.zeros(shape=(rollouts_to_take, 1), dtype=np.bool),
                                 r_obs_next[idx_best_ddpg, :, :rollouts_to_take].T)
        self.buffer_ddpg.add(r_obs[idx_best_ddpg, :, :1].T,
                             r_actions[idx_best_ddpg, :, :1].T,
                             r_rewards[idx_best_ddpg, :, :1].T,
                             np.zeros(shape=(1, 1), dtype=np.bool),
                             r_obs_next[idx_best_ddpg, :, :1].T)

        if total_rewards[idx_best] > total_rewards_ddpg[idx_best_ddpg]:
            action = np.array([init_actions[idx_best]])

            if self.is_training:
                self.steps_rs.append(1)
                self.steps_ddpg.append(0)
        else:
            init_actions_ddpg = init_actions_ddpg.detach().cpu().numpy()
            action = np.array([init_actions_ddpg[idx_best_ddpg]])

            if self.is_training:
                self.steps_rs.append(0)
                self.steps_ddpg.append(1)

        return np.clip(action, -1, 1)[0]

    def act_rs_mpc_old(self, obs):
        init_actions = self._get_random_actions(self.agent_params.population_size)
        obs = np.tile(obs, (self.agent_params.population_size, 1))
        total_rewards = np.zeros(self.agent_params.population_size)

        for n in range(self.agent_params.horizon):
            obs = torch.tensor(obs, dtype=torch.float32)

            if n == 0:
                actions = init_actions
            else:
                actions = self._get_random_actions(self.agent_params.population_size)

            actions = torch.tensor(actions, dtype=torch.float32)
            with torch.no_grad():
                obs_next = self.dynamics(obs, actions)

            obs = obs.detach().cpu().numpy()
            obs_next = obs_next.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            rewards = self.reward_fn(obs, actions, obs_next)
            total_rewards += rewards

            obs = obs_next

        idx_best = total_rewards.argmax()
        action = np.array([init_actions[idx_best]])

        return np.clip(action, -1, 1)[0]

    def train(self):
        total_steps, update_steps = 0, 0
        all_rewards, all_successes = [], []

        for epoch in range(self.experiment_params.n_epochs):
            rollout_steps = 0
            cycle_dynmaics_loss, cycle_dynamics_test_loss = 0, 0

            for cycle in range(self.experiment_params.n_cycles):
                rollout_dynamics_loss, rollout_dynamics_test_loss = 0, 0

                for rollout in range(self.experiment_params.n_rollouts):
                    obs = self.env.reset()

                    rollout_reward, rollout_success = 0, 0
                    r_obs, r_actions, r_reward, r_terminal, r_obs_next = [], [], [], [], []

                    for _ in range(self.env_params.max_episode_steps):
                        with torch.no_grad():
                            self.dynamics.eval()
                            action = self.act_rs_mpc(obs)
                            self.dynamics.train()

                        # self.env.render()
                        obs_next, reward, done, info = self.env.step(action)

                        # append rollouts
                        r_obs.append(obs.copy())
                        r_actions.append(action.copy())
                        r_reward.append(np.array([reward]).copy())
                        r_terminal.append(np.array([done]).copy())
                        r_obs_next.append(obs_next.copy())

                        # reward and success
                        rollout_reward += reward
                        try:
                            rollout_success += float(info['is_success'])
                        except KeyError:
                            rollout_success += 0

                        # update steps
                        total_steps += 1
                        rollout_steps += 1

                        # end on done?
                        if done and self.env_params.end_on_done:
                            break

                        # re-assign observations
                        obs = obs_next

                    # reward and success
                    all_rewards.append(rollout_reward)
                    all_successes.append(rollout_success / rollout_steps)

                    # add to buffer
                    r_obs = np.array(r_obs)
                    r_actions = np.array(r_actions)
                    r_reward = np.array(r_reward)
                    r_terminal = np.array(r_terminal)
                    r_obs_next = np.array(r_obs_next)

                    # artificially blow up
                    x = np.arange(len(r_obs))
                    obs_fn = interp1d(x, r_obs, axis=0)
                    actions_fn = interp1d(x, r_actions, axis=0)
                    obs_next_fn = interp1d(x, r_obs_next, axis=0)

                    n_artificial = 100
                    x = np.linspace(0, len(r_obs) - 1, n_artificial * len(r_obs))
                    r_obs = obs_fn(x)
                    r_actions = actions_fn(x)
                    r_obs_next = obs_next_fn(x)
                    r_reward = np.expand_dims(self.reward_fn(r_obs, r_actions, r_obs_next), axis=1)
                    r_terminal = np.zeros(shape=(len(r_obs), 1), dtype=np.bool)

                    idx_test = np.random.choice(len(r_obs), int(0.1 * n_artificial * rollout_steps), replace=False)
                    mask = np.array([False] * len(r_obs))
                    mask[idx_test] = True

                    self.buffer_test.add(r_obs[mask], r_actions[mask], r_reward[mask], r_terminal[mask],
                                         r_obs_next[mask])
                    self.buffer.add(r_obs[~mask], r_actions[~mask], r_reward[~mask], r_terminal[~mask],
                                    r_obs_next[~mask])

                    # update normalizer
                    self.normalizer.update(r_obs[~mask])
                    self.normalizer.recompute_stats()

                if len(self.buffer) > self.agent_params.batch_size_dynamics:
                    n_batches = rollout_steps if self.experiment_params.n_train_batches is None \
                        else self.experiment_params.n_train_batches
                    for _ in range(n_batches):
                        dynamic_loss, dynamics_test_loss = self._learn_dynamics()
                        rollout_dynamics_loss += dynamic_loss
                        rollout_dynamics_test_loss += dynamics_test_loss

                        update_steps += 1

                if update_steps > 0:
                    rollout_steps_ddpg = self._uncertainty_buffer()
                    rollout_steps_ddpg = 100 #min(1024, rollout_steps_ddpg)
                    if len(self.buffer_ddpg) > self.agent_params.batch_size_ddpg:
                        n_batches = rollout_steps_ddpg if self.experiment_params.n_train_batches_ddpg is None \
                            else self.experiment_params.n_train_batches_ddpg

                        for _ in range(n_batches):
                            actor_loss, critic_loss = self._learn_ddpg_aux()
                self.buffer_ddpg_all.clear()

                cycle_dynmaics_loss += rollout_dynamics_loss
                cycle_dynamics_test_loss += rollout_dynamics_test_loss

            if MPI.COMM_WORLD.Get_rank() == 0:
                # test
                self.is_training = False
                avg_reward_test, avg_success_test = self._test()
                self.is_training = True

                # test ddpg
                avg_reward_test_ddpg, avg_success_test_ddpg = self._test_ddpg()

                data = {'time_steps': total_steps,
                        'update_steps': update_steps,
                        'epoch': epoch + 1,
                        'cycles': cycle + 1,
                        'rollouts': rollout + 1,
                        'rollout_steps': rollout_steps,
                        'current_reward': all_rewards[-1],
                        'current_success': all_successes[-1],
                        'avg_reward': float(np.mean(all_rewards[-100:])),
                        'avg_success': float(np.mean(all_successes[-100:])),
                        'avg_reward_test': avg_reward_test,
                        'avg_success_test': avg_success_test,
                        'cycle_dynamics_loss': cycle_dynmaics_loss,
                        'cycle_dynamics_test_loss': cycle_dynamics_test_loss,
                        'rollout_dynamics_loss': rollout_dynamics_loss,
                        'rollout_dynamics_test_loss': rollout_dynamics_test_loss,
                        'avg_reward_test_ddpg': avg_reward_test_ddpg,
                        'avg_success_test_ddpg': avg_success_test_ddpg,
                        'rs_steps': sum(self.steps_rs[-rollout_steps:]),
                        'ddpg_steps': sum(self.steps_ddpg[-rollout_steps:])
                        }
                self.logger.add(data)

        self._save()

    def _rollout_ddpg(self):
        obs, _, _, _, _ = self.buffer.sample_batch(self.agent_params.batch_size_ddpg)

        rollout_steps = self.env_params.max_episode_steps

        r_obs = np.empty(shape=(len(obs) * rollout_steps, self.env_params.obs_dim), dtype=np.float32)
        r_actions = np.empty(shape=(len(obs) * rollout_steps, self.env_params.action_dim), dtype=np.float32)
        r_reward = np.empty(shape=(len(obs) * rollout_steps, 1), dtype=np.float32)
        r_obs_next = np.empty(shape=(len(obs) * rollout_steps, self.env_params.obs_dim), dtype=np.float32)

        cur_buf_pos = 0

        for _ in range(rollout_steps):
            obs_norm = self.normalizer.normalize(obs)

            obs = torch.tensor(obs, dtype=torch.float32)
            obs_norm = torch.tensor(obs_norm, dtype=torch.float32)

            self.dynamics.eval()
            with torch.no_grad():
                if np.random.uniform() < 0.1:
                    actions = self._get_random_actions(len(obs))
                    actions = torch.tensor(actions, dtype=torch.float32)
                else:
                    actions = self.actor(obs_norm)
                obs_next = self.dynamics(obs, actions)
            self.dynamics.train()

            obs = obs.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            obs_next = obs_next.detach().cpu().numpy()

            rewards = self.reward_fn(obs, actions, obs_next)
            rewards = np.expand_dims(rewards, axis=1)

            r_obs[cur_buf_pos:cur_buf_pos + len(obs)] = obs
            r_actions[cur_buf_pos:cur_buf_pos + len(obs)] = actions
            r_reward[cur_buf_pos:cur_buf_pos + len(obs)] = rewards
            r_obs_next[cur_buf_pos:cur_buf_pos + len(obs)] = obs_next

            obs = obs_next
            cur_buf_pos += len(obs)

        obs_diff = np.abs(r_obs - r_obs_next)
        idx = self.buffer.is_similar(obs_diff)

        terminal = np.zeros(shape=(len(obs) * rollout_steps, 1), dtype=np.bool)
        self.buffer_ddpg_all.add(r_obs[idx], r_actions[idx], r_reward[idx], terminal[idx], r_obs_next[idx])

        return sum(idx)

    def _learn_dynamics(self):
        obs, actions, _, _, obs_next = self.buffer.sample_batch(self.agent_params.batch_size_dynamics)

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
        self.dynamics.eval()
        obs, actions, _, _, obs_next = self.buffer_test.sample_batch(int(0.1 * self.agent_params.batch_size_dynamics))

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, dtype=torch.float32)

        with torch.no_grad():
            obs_predicted = self.dynamics(obs, actions)

        dynamics_test_criterion = nn.MSELoss()
        dynamics_test_loss = dynamics_test_criterion(obs_predicted, obs_next)
        self.dynamics.train()

        return dynamics_loss.item(), dynamics_test_loss.item()

    def _uncertainty_buffer(self):
        # epistemic uncertainty
        obs, actions, rewards, terminals, obs_next = self.buffer_ddpg_all.sample_batch(
            len(self.buffer_ddpg_all))  # self.agent_params.batch_size_ddpg)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        o = []
        for _ in range(100):
            with torch.no_grad():
                obs_predicted = self.dynamics(obs, actions)

            obs_predicted = obs_predicted.detach().cpu().numpy()
            o += [obs_predicted]

        mean = np.mean(np.array(o), axis=0)
        std = np.std(np.array(o), axis=0)

        upper_bound = np.all(obs_next < mean + 3 * std, axis=1)
        lower_bound = np.all(obs_next > mean - 3 * std, axis=1)
        in_bound = lower_bound & upper_bound

        if sum(in_bound) > 0:
            obs = obs.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()

            self.buffer_ddpg.add(obs[in_bound], actions[in_bound], rewards[in_bound], terminals[in_bound],
                                 obs_next[in_bound])

        return sum(in_bound)

    def _learn_ddpg_aux(self):
        # real buffer
        real_batch_size = int(0.125 * self.agent_params.batch_size_ddpg)
        obs_r, actions_r, _, _, _ = self.buffer.sample_batch(real_batch_size)

        obs_r = self.normalizer.normalize(obs_r)
        obs_r = torch.tensor(obs_r, dtype=torch.float32)
        actions_r = torch.tensor(actions_r, dtype=torch.float32)

        # artificial buffer
        obs, actions, reward, terminal, obs_next = self.buffer_ddpg.sample_batch(
            self.agent_params.batch_size_ddpg - real_batch_size)

        obs = self.normalizer.normalize(obs)
        obs_next = self.normalizer.normalize(obs_next)

        # to tensors
        done = (terminal == False) * 1
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, dtype=torch.float32)

        with torch.no_grad():
            actions_target = self.actor_target(obs_next)
            q_target = self.critic_target(obs_next, actions_target).detach()
            q_expected = reward + done * self.agent_params.gamma_ddpg * q_target

            clip_return = 1 / (1 - self.agent_params.gamma_ddpg)
            q_expected = torch.clamp(q_expected, -clip_return, 0)

        # q loss
        q_predicted = self.critic(obs, actions)
        critic_loss = (q_expected - q_predicted).pow(2).mean()

        # critic network update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optimizer.step()

        # artificial loss
        actions_predicted = self.actor(obs)
        artificial_loss = (-self.critic(obs, actions_predicted)).mean()

        # aux loss
        actions_real = self.actor(obs_r)
        mask = torch.gt(self.critic(obs_r, actions_r), self.critic(obs_r, actions_real))
        mask = mask.float()
        n_mask = int(mask.sum().item())

        if n_mask == 0:
            aux_loss = torch.zeros(1)
        else:
            aux_loss = (torch.mul(actions_real, mask) - torch.mul(actions_r, mask)).pow(2).sum() / n_mask

        # actor loss
        actor_loss = 0.001 * artificial_loss + 1. / real_batch_size * aux_loss

        # actor network update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return actor_loss.item(), critic_loss.item()

    def _learn_ddpg(self):
        obs, actions, reward, terminal, obs_next = self.buffer.sample_batch(self.agent_params.batch_size_ddpg)

        obs = self.normalizer.normalize(obs)
        obs_next = self.normalizer.normalize(obs_next)

        # to tensors
        done = (terminal == False) * 1
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        obs_next = torch.tensor(obs_next, dtype=torch.float32)

        with torch.no_grad():
            actions_target = self.actor_target(obs_next)
            q_target = self.critic_target(obs_next, actions_target).detach()
            q_expected = reward + done * self.agent_params.gamma_ddpg * q_target

        # q loss
        q_predicted = self.critic(obs, actions)
        critic_loss = (q_expected - q_predicted).pow(2).mean()

        # actor loss
        actions_predicted = self.actor(obs)
        actor_loss = (-self.critic(obs, actions_predicted)).mean()

        # actor network update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optimizer.step()

        # critic network update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - self.agent_params.polyak_ddpg) + param.data * self.agent_params.polyak_ddpg)

    def _save(self):
        torch.save(self.dynamics.state_dict(), '{}/dynamics.pt'.format(self.experiment_params.log_dir))

        torch.save(self.actor.state_dict(), '{}/actor.pt'.format(self.experiment_params.log_dir))
        torch.save(self.critic.state_dict(), '{}/critic.pt'.format(self.experiment_params.log_dir))

        torch.save(self.actor_target.state_dict(), '{}/actor_target.pt'.format(self.experiment_params.log_dir))
        torch.save(self.critic_target.state_dict(), '{}/critic_target.pt'.format(self.experiment_params.log_dir))

        self.logger.save()

    def _load(self):
        dynamics_state_dict = torch.load('{}/dynamics.pt'.format(self.experiment_params.log_dir))

        actor_state_dict = torch.load('{}/actor.pt'.format(self.experiment_params.log_dir))
        critic_state_dict = torch.load('{}/critic.pt'.format(self.experiment_params.log_dir))

        actor_target_state_dict = torch.load('{}/actor_target.pt'.format(self.experiment_params.log_dir))
        critic_target_state_dict = torch.load('{}/critic_target.pt'.format(self.experiment_params.log_dir))

        self.dynamics.load_state_dict(dynamics_state_dict)

        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

        self.actor_target.load_state_dict(actor_target_state_dict)
        self.critic_target.load_state_dict(critic_target_state_dict)

    def test(self):
        self._load()
        self.is_training = False
        self._test()

    def _test(self):
        avg_reward, avg_success = 0, 0
        images = []

        self.dynamics.eval()
        for rollout in range(self.test_params.n_episodes):
            obs = self.env.reset()
            r_reward, r_steps, r_success = 0, 0, 0

            for _ in range(self.env_params.max_episode_steps):
                if self.test_params.is_test and self.test_params.gif:
                    image = self.env.render(mode='rgb_array')
                    images.append(image)
                elif self.test_params.is_test and self.test_params.visualize:
                    self.env.render()

                with torch.no_grad():
                    action = self.act_rs_mpc(obs)
                obs, reward, done, info = self.env.step(action)

                r_reward += reward

                try:
                    r_success += float(info['is_success'])
                except KeyError:
                    r_success += 0

                r_steps += 1

                if done and self.env_params.end_on_done:
                    break

            avg_reward += r_reward
            avg_success += r_success / r_steps

            if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
                print(
                    colored('[TEST]', 'green') + ' rollout: {}, reward: {:.3f}, success: {:.2f}'.format(rollout + 1,
                                                                                                        r_reward,
                                                                                                        r_success))

        self.dynamics.train()
        avg_reward /= self.test_params.n_episodes
        avg_success /= self.test_params.n_episodes

        if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
            print(
                colored('[TEST]', 'green') + ' avg_reward: {:.3f}, avg_success: {:.3f}'.format(avg_reward,
                                                                                               avg_success))

        return avg_reward, avg_success

    def _test_ddpg(self):
        avg_reward, avg_success = 0, 0

        for rollout in range(self.test_params.n_episodes):
            obs = self.env.reset()
            r_reward, r_steps, r_success = 0, 0, 0

            for _ in range(self.env_params.max_episode_steps):
                obs = self.normalizer.normalize(obs)
                obs = torch.tensor(obs, dtype=torch.float32)

                with torch.no_grad():
                    action = self.actor(obs)
                    action = action.detach().cpu().numpy()
                    action = np.clip(action, -1, 1)
                obs, reward, done, info = self.env.step(action)

                r_reward += reward

                try:
                    r_success += float(info['is_success'])
                except KeyError:
                    r_success += 0

                r_steps += 1

                if done and self.env_params.end_on_done:
                    break

            avg_reward += r_reward
            avg_success += r_success / r_steps

            if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
                print(colored('[TEST]', 'green') + ' rollout: {}, reward: {:.3f}, success: {:.2f}'.format(rollout + 1,
                                                                                                          r_reward,
                                                                                                          r_success))
        avg_reward /= self.test_params.n_episodes
        avg_success /= self.test_params.n_episodes

        if MPI.COMM_WORLD.Get_rank() == 0 and self.test_params.is_test:
            print(
                colored('[TEST]', 'green') + ' avg_reward: {:.3f}, avg_success: {:.3f}'.format(avg_reward, avg_success))

        return avg_reward, avg_success

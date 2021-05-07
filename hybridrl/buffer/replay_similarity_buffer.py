import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


class ReplaySimilarityBuffer:
    def __init__(self, buffer_size, seed, env_params):
        self._obs_dim = env_params.obs_dim
        self._action_dim = env_params.action_dim
        self._max_episode_steps = env_params.max_episode_steps
        self._last_episode_steps = 0

        self._buffer_size = buffer_size
        self._count = 0
        self._buffers = {
            'obs': np.empty(shape=(buffer_size, self._obs_dim)),
            'actions': np.empty(shape=(buffer_size, self._action_dim)),
            'rewards': np.empty(shape=(buffer_size, 1)),
            'terminal': np.empty(shape=(buffer_size, 1)),
            'obs_next': np.empty(shape=(buffer_size, self._obs_dim)),
            'obs_diff': np.empty(shape=(buffer_size, self._obs_dim))}

        random.seed(seed)

    def add(self, obs, action, reward, terminal, obs_next):
        n_additional = len(obs)

        if self._count + n_additional < self._buffer_size:
            self._buffers['obs'][self._count:self._count + n_additional] = obs
            self._buffers['actions'][self._count:self._count + n_additional] = action
            self._buffers['rewards'][self._count:self._count + n_additional] = reward
            self._buffers['terminal'][self._count:self._count + n_additional] = terminal
            self._buffers['obs_next'][self._count:self._count + n_additional] = obs_next
            self._buffers['obs_diff'][self._count:self._count + n_additional] = np.abs(obs - obs_next)
            self._count += n_additional
        else:
            n_roll = self._buffer_size - (n_additional + self._count)
            self._buffers['obs'] = np.roll(self._buffers['obs'], n_roll, axis=0)
            self._buffers['actions'] = np.roll(self._buffers['actions'], n_roll, axis=0)
            self._buffers['rewards'] = np.roll(self._buffers['rewards'], n_roll, axis=0)
            self._buffers['terminal'] = np.roll(self._buffers['terminal'], n_roll, axis=0)
            self._buffers['obs_next'] = np.roll(self._buffers['obs_next'], n_roll, axis=0)
            self._buffers['obs_diff'] = np.roll(self._buffers['obs_diff'], n_roll, axis=0)

            self._count = self._buffer_size

            self._buffers['obs'][self._count - n_additional:self._count] = obs
            self._buffers['actions'][self._count - n_additional:self._count] = action
            self._buffers['rewards'][self._count - n_additional:self._count] = reward
            self._buffers['terminal'][self._count - n_additional:self._count] = terminal
            self._buffers['obs_next'][self._count - n_additional:self._count] = obs_next
            self._buffers['obs_diff'][self._count - n_additional:self._count] = np.abs(obs - obs_next)

    def __len__(self):
        return self._count

    def sample_batch(self, batch_size):
        if self._count < batch_size:
            batch_idx = np.random.choice(self._count, self._count, replace=False)
        else:
            batch_idx = np.random.choice(self._count, batch_size, replace=False)

        return self._buffers['obs'][batch_idx], \
               self._buffers['actions'][batch_idx], \
               self._buffers['rewards'][batch_idx], \
               self._buffers['terminal'][batch_idx], \
               self._buffers['obs_next'][batch_idx]

    def clear(self):
        self._buffers = {
            'obs': np.empty(shape=(self._buffer_size, self._obs_dim)),
            'actions': np.empty(shape=(self._buffer_size, self._action_dim)),
            'rewards': np.empty(shape=(self._buffer_size, 1)),
            'terminal': np.empty(shape=(self._buffer_size, 1)),
            'obs_next': np.empty(shape=(self._buffer_size, self._obs_dim)),
            'obs_diff': np.empty(shape=(self._buffer_size, self._obs_dim))}
        self._count = 0

    def is_similar(self, obs_diff):
        return [True] * len(obs_diff)

        mean = np.mean(self._buffers['obs_diff'][:self._count], axis=0)

        return np.all(obs_diff < mean, axis=1)

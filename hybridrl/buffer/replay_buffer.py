import numpy as np
import random


class ReplayBuffer:
    def __init__(self, buffer_size, seed):
        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = []
        random.seed(seed)

    def add(self, obs, action, reward, terminal, obs_next):
        transition = (obs, action, reward, terminal, obs_next)

        if self._count < self._buffer_size:
            self._buffer.append(transition)
            self._count += 1
        else:
            self._buffer.pop(0)
            self._buffer.append(transition)

    def __len__(self):
        return self._count

    def sample_batch(self, batch_size):
        if self._count < batch_size:
            batch = random.sample(self._buffer, self._count)
        else:
            batch = random.sample(self._buffer, batch_size)

        obs_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        terminal_batch = np.array([_[3] for _ in batch])
        obs_next_batch = np.array([_[4] for _ in batch])

        return obs_batch, action_batch, reward_batch, terminal_batch, obs_next_batch

    def clear(self):
        self._buffer = []
        self._count = 0

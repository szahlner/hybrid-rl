import threading
import numpy as np
from typing import Callable, List, Optional


class ReplayBuffer:
    def __init__(
        self,
        env_params: dict,
        buffer_size: int,
        sample_func: Callable
    ) -> None:
        self.env_params = env_params
        self.T = env_params["max_timesteps"]
        self.size = buffer_size // self.T

        # Memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func

        # Create the buffer to store info
        self.buffers = {
            "obs": np.empty([self.size, self.T + 1, self.env_params["obs"]]),
            "ag": np.empty([self.size, self.T + 1, self.env_params["goal"]]),
            "g": np.empty([self.size, self.T, self.env_params["goal"]]),
            "actions": np.empty([self.size, self.T, self.env_params["action"]]),
        }

        # Thread lock
        self.lock = threading.Lock()

    # Store the episode
    def store_episode(self, episode_batch: List[np.ndarray]):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            # Store the informations
            self.buffers["obs"][idxs] = mb_obs
            self.buffers["ag"][idxs] = mb_ag
            self.buffers["g"][idxs] = mb_g
            self.buffers["actions"][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # Sample the data from the replay buffer
    def sample(self, batch_size: int):
        temp_buffers = {}

        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]

        temp_buffers["obs_next"] = temp_buffers["obs"][:, 1:, :]
        temp_buffers["ag_next"] = temp_buffers["ag"][:, 1:, :]

        # Sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)

        return transitions

    def _get_storage_idx(self, inc: Optional[int] = None):
        inc = inc or 1

        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        self.current_size = min(self.size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]

        return idx


class SimpleReplayBuffer:
    def __init__(self, env_params: dict, buffer_size: int) -> None:
        self.env_params = env_params

        # Memory management
        self.current_size = 0
        self.pointer = 0
        self.max_size = buffer_size

        # Create the buffer to store info
        self.buffers = {
            "obs": np.empty([buffer_size, self.env_params["obs"]]),
            "obs_next": np.empty([buffer_size, self.env_params["obs"]]),
            "ag": np.empty([buffer_size, self.env_params["goal"]]),
            "ag_next": np.empty([buffer_size, self.env_params["goal"]]),
            "g": np.empty([buffer_size, self.env_params["goal"]]),
            "actions": np.empty([buffer_size, self.env_params["action"]]),
        }

        # Thread lock
        self.lock = threading.Lock()

    # Store the episode
    def store(self, batch: List[np.ndarray]):
        obs, obs_next, ag, ag_next, g, actions = batch

        with self.lock:
            self.buffers["obs"][self.pointer] = obs
            self.buffers["obs_next"][self.pointer] = obs_next
            self.buffers["ag"][self.pointer] = ag
            self.buffers["ag_next"][self.pointer] = ag_next
            self.buffers["g"][self.pointer] = g
            self.buffers["actions"][self.pointer] = actions

        self.pointer = (self.pointer + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    # Sample the data from the replay buffer
    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.current_size, size=batch_size)

        transitions = {
            "obs": self.buffers["obs"][idx],
            "obs_next": self.buffers["obs_next"][idx],
            "ag": self.buffers["ag"][idx],
            "ag_next": self.buffers["ag_next"][idx],
            "g": self.buffers["g"][idx],
            "actions": self.buffers["actions"][idx],
        }

        return transitions

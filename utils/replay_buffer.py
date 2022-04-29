import threading
import numpy as np
from typing import List


class ReplayBuffer:
    def __init__(self, env_params: dict, buffer_size: int) -> None:
        self.env_params = env_params

        # Memory management
        self.current_size = 0
        self.pointer = 0
        self.n_transitions_stored = 0
        self.max_size = buffer_size

        # Create the buffer to store info
        self.buffers = {
            "obs": np.empty([buffer_size, self.env_params["obs"]]),
            "obs_next": np.empty([buffer_size, self.env_params["obs"]]),
            "r": np.empty([buffer_size, 1]),
            "d": np.empty([buffer_size, 1]),
            "actions": np.empty([buffer_size, self.env_params["action"]]),
        }

        # Thread lock
        self.lock = threading.Lock()

    # Store the episode
    def store(self, batch: List[np.ndarray]):
        obs, obs_next, r, d, actions = batch

        with self.lock:
            self.buffers["obs"][self.pointer] = obs
            self.buffers["obs_next"][self.pointer] = obs_next
            self.buffers["r"][self.pointer] = r
            self.buffers["d"][self.pointer] = d
            self.buffers["actions"][self.pointer] = actions

        self.pointer = (self.pointer + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)
        self.n_transitions_stored += 1

    # Sample the data from the replay buffer
    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.current_size, size=batch_size)

        transitions = {
            "obs": self.buffers["obs"][idx],
            "obs_next": self.buffers["obs_next"][idx],
            "r": self.buffers["r"][idx],
            "d": self.buffers["d"][idx],
            "actions": self.buffers["actions"][idx],
        }

        return transitions

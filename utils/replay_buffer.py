import threading
import numpy as np
from typing import Tuple


class ReplayBuffer:
    def __init__(self, env_params: dict, buffer_size: int) -> None:
        self.env_params = env_params

        # memory management
        self.current_size = 0
        self.pointer = 0
        self.max_size = buffer_size

        # create the buffer to store info
        self.buffers = {
            "obs": np.empty([buffer_size, self.env_params["obs"]]),
            "obs_next": np.empty([buffer_size, self.env_params["obs"]]),
            "ag": np.empty([buffer_size, self.env_params["goal"]]),
            "ag_next": np.empty([buffer_size, self.env_params["goal"]]),
            "g": np.empty([buffer_size, self.env_params["goal"]]),
            "actions": np.empty([buffer_size, self.env_params["action"]]),
        }

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store(self, batch: Tuple):
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

    # sample the data from the replay buffer
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

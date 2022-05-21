import threading
import numpy as np
from typing import List, Tuple
from collections import deque

from utils.segment_tree import MinSegmentTree, SumSegmentTree


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


class PrioritizedReplayBuffer:
    """Prioritized Replay buffer with demonstrations."""
    def __init__(
        self,
        env_params: dict,
        buffer_size: int,
        gamma: float = 0.99,
        alpha: float = 0.6,
    ) -> None:
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

        # Priority replay
        self.gamma = gamma
        self.alpha = alpha

        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

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

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr += 1

    # Sample the data from the replay buffer
    def sample(self, batch_size: int, beta: float = 0.4):
        assert beta > 0

        idx = self._sample_proportional(batch_size)
        weights = np.array([self._calculate_weight(i, beta) for i in idx])

        transitions = {
            "obs": self.buffers["obs"][idx],
            "obs_next": self.buffers["obs_next"][idx],
            "r": self.buffers["r"][idx],
            "d": self.buffers["d"][idx],
            "actions": self.buffers["actions"][idx],
            "weights": weights[:, None],
            "idx": idx,
        }

        return transitions

    def update_priorities(self, idx: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(idx) == len(priorities)

        for idx, priority in zip(idx, priorities):
            assert priority > 0
            assert 0 <= idx < self.current_size

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size: int) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, self.current_size)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.current_size) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.current_size) ** (-beta)
        weight = weight / max_weight

        return weight


class NStepReplayBuffer:
    def __init__(self, env_params: dict, buffer_size: int, gamma: float = 0.99, n_step: int = 1) -> None:
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

        # N-step
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

    # Store the episode
    def store(self, batch: List[np.ndarray]):
        self.n_step_buffer.append(batch)

        # Single step transitions not ready
        if len(self.n_step_buffer) < self.n_step:
            return

        obs, obs_next, r, d, actions = self._get_n_step_info()

        with self.lock:
            self.buffers["obs"][self.pointer] = obs
            self.buffers["obs_next"][self.pointer] = obs_next
            self.buffers["r"][self.pointer] = r
            self.buffers["d"][self.pointer] = d
            self.buffers["actions"][self.pointer] = actions

        self.pointer = (self.pointer + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)
        self.n_transitions_stored += 1

    def _get_n_step_info(self) -> Tuple[np.ndarray, np.ndarray, float, bool, np.ndarray]:
        """Return n step rew, next_obs, and done."""
        # Info of the last transition
        _, obs_next, rew, done, _ = self.n_step_buffer[-1]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            _, o_n, r, d, _ = transition

            rew = r + self.gamma * rew * (1 - d)
            obs_next, done = (o_n, d) if d else (obs_next, done)

        obs, _, _, _, action = self.n_step_buffer[0]

        return obs, obs_next, rew, done, action

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

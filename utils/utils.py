import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.confidence = np.zeros((max_size, state_dim + 1))  # +1 for reward

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, confidence=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.confidence[self.ptr] = confidence

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, confidence=False):
        ind = np.random.randint(0, self.size, size=batch_size)

        if confidence:
            conf = self._get_confidence(ind)
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                conf,
            )
        else:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
            )

    def sample_numpy_state(self, idx):
        return self.state[idx]

    def sample_numpy(self, idx, confidence=False):
        if confidence:
            conf = self._get_confidence(idx)

            return (
                self.state[idx],
                self.action[idx],
                self.next_state[idx],
                self.reward[idx],
                self.not_done[idx],
                conf,
            )
        else:
            return (
                self.state[idx],
                self.action[idx],
                self.next_state[idx],
                self.reward[idx],
                self.not_done[idx],
            )

    def clear(self):
        self.ptr = 0
        self.size = 0

    def _get_confidence(self, idx):
        confidence_mean = np.mean(self.confidence[:self.size], axis=0)
        confidence_std = np.std(self.confidence[:self.size], axis=0)
        confidence = np.abs(self.confidence[idx] - confidence_mean) / confidence_std
        confidence = np.where(confidence < 1, 1, confidence)
        confidence = 1.0 / np.mean(confidence)
        return confidence

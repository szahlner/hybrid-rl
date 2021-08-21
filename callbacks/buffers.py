import torch
import numpy as np

from typing import Tuple


class ReplayBuffer:
    def __init__(
        self, n_state: int, n_action: int, size: int, use_cuda: bool = False
    ) -> None:
        self.size = size
        self.n_state = n_state
        self.n_action = n_action
        self.use_cuda = use_cuda

        self.buffer = {
            "state": np.empty(shape=(size, n_state)),
            "action": np.empty(shape=(size, n_action)),
            "reward": np.empty(shape=(size, 1)),
            "mask": np.empty(shape=(size, 1)),
            "next_state": np.empty(shape=(size, n_state)),
        }

        self.count = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        mask: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        n_additional = len(state)

        if self.count + n_additional < self.size:
            self.buffer["state"][self.count : self.count + n_additional] = state
            self.buffer["action"][self.count : self.count + n_additional] = action
            self.buffer["reward"][self.count : self.count + n_additional] = reward
            self.buffer["mask"][self.count : self.count + n_additional] = mask
            self.buffer["next_state"][
                self.count : self.count + n_additional
            ] = next_state
            self.count += n_additional
        else:
            n_roll = self.size - (n_additional + self.count)

            if n_roll < 0:
                n_roll = self.size

            self.buffer["state"] = np.roll(self.buffer["state"], n_roll, axis=0)
            self.buffer["action"] = np.roll(self.buffer["action"], n_roll, axis=0)
            self.buffer["reward"] = np.roll(self.buffer["reward"], n_roll, axis=0)
            self.buffer["mask"] = np.roll(self.buffer["mask"], n_roll, axis=0)
            self.buffer["next_state"] = np.roll(
                self.buffer["next_state"], n_roll, axis=0
            )

            self.count = self.size - n_roll
            self.buffer["state"][self.count : self.count + n_additional] = state
            self.buffer["action"][self.count : self.count + n_additional] = action
            self.buffer["reward"][self.count : self.count + n_additional] = reward
            self.buffer["mask"][self.count : self.count + n_additional] = mask
            self.buffer["next_state"][
                self.count : self.count + n_additional
            ] = next_state
            self.count = self.size

    def __len__(self) -> int:
        return self.count

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.count < batch_size:
            batch_idx = np.random.choice(self.count, self.count, replace=False)
        else:
            batch_idx = np.random.choice(self.count, batch_size, replace=False)

        state = torch.tensor(self.buffer["state"][batch_idx], dtype=torch.float32)
        action = torch.tensor(self.buffer["action"][batch_idx], dtype=torch.float32)
        reward = torch.tensor(self.buffer["reward"][batch_idx], dtype=torch.float32)
        mask = torch.tensor(self.buffer["mask"][batch_idx], dtype=torch.float32)
        next_state = torch.tensor(
            self.buffer["next_state"][batch_idx], dtype=torch.float32
        )

        if self.use_cuda:
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            mask = mask.cuda()
            next_state = next_state.cuda()

        return state, action, reward, mask, next_state

    def clear(self) -> None:
        self.buffer = {
            "state": np.empty(shape=(self.size, self.n_state)),
            "action": np.empty(shape=(self.size, self.n_action)),
            "reward": np.empty(shape=(self.size, 1)),
            "mask": np.empty(shape=(self.size, 1)),
            "next_state": np.empty(shape=(self.size, self.n_state)),
        }
        self.count = 0

    def save(self, save_path: str, verbose: int = 2) -> None:
        np.savez_compressed(
            file=save_path,
            observations=self.buffer["state"],
            actions=self.buffer["action"],
            rewards=self.buffer["reward"],
            dones=self.buffer["mask"],
            next_observations=self.buffer["next_state"],
        )

        if verbose > 0:
            print("Saved buffer to path: '{}'".format(save_path))

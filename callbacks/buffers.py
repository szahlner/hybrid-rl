import warnings

import torch
import numpy as np

from typing import Any, List, Optional, Tuple

try:
    # Check replay buffer fits in memory, if possible
    import psutil
except ImportError:
    psutil = None


class ReplayBuffer:
    def __init__(
        self,
        n_state: int,
        n_action: int,
        size: int,
        n_goal: Optional[int] = None,
        use_cuda: bool = False,
        verbose: int = 2,
    ) -> None:
        self.size = size
        self.n_state = n_state
        self.n_action = n_action
        self.n_goal = n_goal
        self.use_cuda = use_cuda
        self.verbose = verbose

        # Env
        self.buffer = {
            "state": np.empty(shape=(size, n_state), dtype=np.float32),
            "action": np.empty(shape=(size, n_action), dtype=np.float32),
            "reward": np.empty(shape=(size, 1), dtype=np.float32),
            "mask": np.empty(shape=(size, 1), dtype=np.float32),
            "next_state": np.empty(shape=(size, n_state), dtype=np.float32),
        }

        if n_goal is not None:
            # GoalEnv
            self.buffer["achieved_goal"] = np.empty(
                shape=(size, n_goal), dtype=np.float32
            )
            self.buffer["desired_goal"] = np.empty(
                shape=(size, n_goal), dtype=np.float32
            )
            self.buffer["next_achieved_goal"] = np.empty(
                shape=(size, n_goal), dtype=np.float32
            )
            self.buffer["next_desired_goal"] = np.empty(
                shape=(size, n_goal), dtype=np.float32
            )

        # Check that the replay buffer can fit into the memory
        # stable baselines 3
        if psutil is not None:
            memory_available = psutil.virtual_memory().available
            total_memory_usage = 0

            for key in self.buffer.keys():
                total_memory_usage += self.buffer[key].nbytes

            # Convert to GB
            total_memory_usage /= 1e9

            if self.verbose > 0:
                print("Memory usage replay buffer: {}GB".format(total_memory_usage))

            if total_memory_usage > memory_available:
                # Convert to GB
                memory_available /= 1e9
                warnings.warn(
                    "This system does not have enough memory to store the complete replay buffer {}GB > {}GB".format(
                        total_memory_usage, memory_available
                    )
                )

        self.count = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        mask: np.ndarray,
        next_state: np.ndarray,
        achieved_goal: Optional[np.ndarray] = None,
        desired_goal: Optional[np.ndarray] = None,
        next_achieved_goal: Optional[np.ndarray] = None,
        next_desired_goal: Optional[np.ndarray] = None,
    ) -> None:
        n_additional = len(state)

        if self.count + n_additional < self.size:
            # Env
            self.buffer["state"][self.count : self.count + n_additional] = state.copy()
            self.buffer["action"][self.count : self.count + n_additional] = action.copy()
            self.buffer["reward"][self.count : self.count + n_additional] = reward.copy()
            self.buffer["mask"][self.count : self.count + n_additional] = mask.copy()
            self.buffer["next_state"][
                self.count : self.count + n_additional
            ] = next_state.copy()

            if self.n_goal is not None:
                # GoalEnv
                self.buffer["achieved_goal"][
                    self.count : self.count + n_additional
                ] = achieved_goal.copy()
                self.buffer["desired_goal"][
                    self.count : self.count + n_additional
                ] = desired_goal.copy()
                self.buffer["next_achieved_goal"][
                    self.count : self.count + n_additional
                ] = next_achieved_goal.copy()
                self.buffer["next_desired_goal"][
                    self.count : self.count + n_additional
                ] = next_desired_goal.copy()

            self.count += n_additional
        else:
            n_roll = self.size - (n_additional + self.count)

            if n_roll < 0:
                n_roll = self.size

            # Env
            self.buffer["state"] = np.roll(self.buffer["state"], n_roll, axis=0)
            self.buffer["action"] = np.roll(self.buffer["action"], n_roll, axis=0)
            self.buffer["reward"] = np.roll(self.buffer["reward"], n_roll, axis=0)
            self.buffer["mask"] = np.roll(self.buffer["mask"], n_roll, axis=0)
            self.buffer["next_state"] = np.roll(
                self.buffer["next_state"], n_roll, axis=0
            )

            if self.n_goal is not None:
                # GoalEnv
                self.buffer["achieved_goal"] = np.roll(
                    self.buffer["achieved_goal"], n_roll, axis=0
                )
                self.buffer["desired_goal"] = np.roll(
                    self.buffer["achieved_goal"], n_roll, axis=0
                )
                self.buffer["next_achieved_goal"] = np.roll(
                    self.buffer["next_achieved_goal"], n_roll, axis=0
                )
                self.buffer["next_desired_goal"] = np.roll(
                    self.buffer["next_desired_goal"], n_roll, axis=0
                )

            self.count = self.size - n_roll

            # Env
            self.buffer["state"][self.count : self.count + n_additional] = state.copy()
            self.buffer["action"][self.count : self.count + n_additional] = action.copy()
            self.buffer["reward"][self.count : self.count + n_additional] = reward.copy()
            self.buffer["mask"][self.count : self.count + n_additional] = mask.copy()
            self.buffer["next_state"][
                self.count : self.count + n_additional
            ] = next_state.copy()

            if self.n_goal is not None:
                # GoalEnv
                self.buffer["achieved_goal"][
                    self.count : self.count + n_additional
                ] = achieved_goal.copy()
                self.buffer["desired_goal"][
                    self.count : self.count + n_additional
                ] = desired_goal.copy()
                self.buffer["next_achieved_goal"][
                    self.count : self.count + n_additional
                ] = next_achieved_goal.copy()
                self.buffer["next_desired_goal"][
                    self.count : self.count + n_additional
                ] = next_desired_goal.copy()

            self.count = self.size

    def __len__(self) -> int:
        return self.count

    def sample(self, batch_size: int) -> Any:
        if self.count < batch_size:
            batch_idx = np.random.choice(self.count, self.count, replace=False)
        else:
            batch_idx = np.random.choice(self.count, batch_size, replace=False)

        # Env
        state = torch.tensor(self.buffer["state"][batch_idx], dtype=torch.float32)
        action = torch.tensor(self.buffer["action"][batch_idx], dtype=torch.float32)
        reward = torch.tensor(self.buffer["reward"][batch_idx], dtype=torch.float32)
        mask = torch.tensor(self.buffer["mask"][batch_idx], dtype=torch.float32)
        next_state = torch.tensor(
            self.buffer["next_state"][batch_idx], dtype=torch.float32
        )

        if self.n_goal is not None:
            # GoalEnv
            achieved_goal = torch.tensor(
                self.buffer["achieved_goal"][batch_idx], dtype=torch.float32
            )
            desired_goal = torch.tensor(
                self.buffer["desired_goal"][batch_idx], dtype=torch.float32
            )
            next_achieved_goal = torch.tensor(
                self.buffer["next_achieved_goal"][batch_idx], dtype=torch.float32
            )
            next_desired_goal = torch.tensor(
                self.buffer["next_desired_goal"][batch_idx], dtype=torch.float32
            )

        if self.use_cuda:
            # Env
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            mask = mask.cuda()
            next_state = next_state.cuda()

            if self.n_goal is not None:
                # GoalEnv
                achieved_goal = achieved_goal.cuda()
                desired_goal = desired_goal.cuda()
                next_achieved_goal = next_achieved_goal.cuda()
                next_desired_goal = next_desired_goal.cuda()

        if self.n_goal is None:
            # Env
            return state, action, reward, mask, next_state

        # GoalEnv
        return (
            state,
            achieved_goal,
            desired_goal,
            action,
            reward,
            mask,
            next_state,
            next_achieved_goal,
            next_desired_goal,
        )

    def clear(self) -> None:
        # Env
        self.buffer = {
            "state": np.empty(shape=(self.size, self.n_state), dtype=np.float32),
            "action": np.empty(shape=(self.size, self.n_action), dtype=np.float32),
            "reward": np.empty(shape=(self.size, 1), dtype=np.float32),
            "mask": np.empty(shape=(self.size, 1), dtype=np.float32),
            "next_state": np.empty(shape=(self.size, self.n_state), dtype=np.float32),
        }

        if self.n_goal is not None:
            # GoalEnv
            self.buffer["achieved_goal"] = np.empty(shape=(self.size, self.n_goal), dtype=np.float32)
            self.buffer["desired_goal"] = np.empty(shape=(self.size, self.n_goal), dtype=np.float32)
            self.buffer["next_achieved_goal"] = np.empty(shape=(self.size, self.n_goal), dtype=np.float32)
            self.buffer["next_desired_goal"] = np.empty(shape=(self.size, self.n_goal), dtype=np.float32)

        self.count = 0

    def save(self, save_path: str, verbose: int = 2) -> None:
        if self.n_goal is None:
            # Env
            np.savez_compressed(
                file=save_path,
                observations=self.buffer["state"],
                actions=self.buffer["action"],
                rewards=self.buffer["reward"],
                dones=self.buffer["mask"],
                next_observations=self.buffer["next_state"],
            )
        else:
            # GoalEnv
            np.savez_compressed(
                file=save_path,
                observations=self.buffer["state"],
                achieved_goal=self.buffer["achieved_goal"],
                desired_goal=self.buffer["desired_goal"],
                actions=self.buffer["action"],
                rewards=self.buffer["reward"],
                dones=self.buffer["mask"],
                next_observations=self.buffer["next_state"],
                next_achieved_goal=self.buffer["next_achieved_goal"],
                next_desired_goal=self.buffer["next_desired_goal"],
            )

        if verbose > 0:
            print("Saved buffer to path: '{}'".format(save_path))

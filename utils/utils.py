import os
import numpy as np
import torch
import gym
import time
import datetime
from typing import Union

from utils.her.arguments import HerNamespace
from utils.ddpg.arguments import DdpgNamespace
from utils.logger import EpochLogger


def get_env_params(env: Union[gym.Env, gym.GoalEnv]) -> dict:
    obs = env.reset()

    params = {
        "obs": obs["observation"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
        "max_timesteps": env._max_episode_steps,
        "goal": 0,
        "reward": 1,
    }

    if isinstance(obs, dict):
        params["goal"] = obs["desired_goal"].shape[0]
        params["reward"] = 0

    return params


def prepare_logger(args: Union[DdpgNamespace, HerNamespace]) -> EpochLogger:
    log_dir = "./../logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if isinstance(args, HerNamespace):
        agent = "HER"
    elif isinstance(args, DdpgNamespace):
        agent = "DDPG"

    log_dir = os.path.join(
        log_dir,
        f"{agent}_{args.env_name}_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H-%M-%S')}"
    )

    logger_kwargs = {
        "output_dir": log_dir,
        "output_fname": "log.txt",
        "exp_name": f"{agent}",
    }

    logger = EpochLogger(**logger_kwargs)
    config_kwargs = {
        "config": args,
    }

    logger.save_config(config_kwargs)

    return logger






















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

    def sample(self, batch_size, idx=None, confidence=False):
        if idx is None:
            idx = np.random.randint(0, self.size, size=batch_size)

        if confidence:
            conf = self._get_confidence(idx)
            return (
                torch.tensor(self.state[idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.action[idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.reward[idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.not_done[idx], dtype=torch.float32, device=self.device),
                conf,
            )
        else:
            return (
                torch.tensor(self.state[idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.action[idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.reward[idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.not_done[idx], dtype=torch.float32, device=self.device),
            )

    def sample_numpy_state(self, idx):
        return self.state[idx]

    def sample_numpy(self, batch_size, idx=None, confidence=False):
        if idx is None:
            idx = np.random.randint(0, self.size, size=batch_size)

        if confidence:
            conf = self._get_confidence(idx)

            return (
                self.state[idx],
                self.action[idx],
                self.next_state[idx],
                self.reward[idx],
                self.not_done[idx],
                self.confidence[idx],
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

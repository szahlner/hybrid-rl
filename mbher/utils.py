import random

import numpy as np
import math
import torch
import gym

from gym import spaces


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def termination_fn(env_name, obs, act, next_obs):
    if env_name == "Hopper-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                   * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:, None]
        return done


class LocomotiveGoalWrapper(gym.Wrapper):
    def __init__(self, env, goal):
        super().__init__(env)
        self.env = env
        self.goals = goal["goals"]
        self.goal_scaler = goal["scaler"]

        self.achieved_goal = 0
        self.desired_goal = self._get_goal()

        goal_dim = np.prod(self.desired_goal.shape).astype(int)
        observation_space = np.prod(self.env.observation_space.shape)

        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=(observation_space,)),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(goal_dim,)),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(goal_dim,)),
            )
        )

        self._max_episode_steps = env._max_episode_steps

    def step(self, action):
        obs_next, reward, done, info = self.env.step(action)

        self.achieved_goal += reward / self.goal_scaler
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, info)

        obs_next = {
            "observation": obs_next,
            "achieved_goal": self.achieved_goal,
            "desired_goal": self.desired_goal,
        }

        return obs_next, reward, done, info

    def reset(self, **kwargs):
        self.achieved_goal = 0
        self.desired_goal = self._get_goal()

        observation = self.env.reset()

        return {
            "observation": observation,
            "achieved_goal": self.achieved_goal,
            "desired_goal": self.desired_goal,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0.0 if achieved_goal > desired_goal else -1.0

    def _get_goal(self):
        idx = random.randint(0, len(self.goals) - 1)
        return np.array([self.goals[idx]])

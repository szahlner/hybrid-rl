import numpy as np
import gym

from gym import spaces
from collections import OrderedDict


class NormalizedActionsEnvWrapper(gym.ActionWrapper):
    """Wrap action to be in [-1, 1]"""

    def __init__(self, env):
        super(NormalizedActionsEnvWrapper, self).__init__(env)

    def action(self, action):
        a = (self.action_space.high - self.action_space.low) / 2.
        b = (self.action_space.high + self.action_space.low) / 2.
        return a * action + b

    def reverse_action(self, action):
        a = 2. / (self.action_space.high - self.action_space.low)
        b = (self.action_space.high + self.action_space.low) / 2.
        return a * (action - b)


class GoalEnvWrapper:
    """
    Wrapper that allows to use dict observation space (GoalEnv) with 'normal' RL algorithms.
    Assumption: All the spaces of the dict space are of the same type.

    Note: Box spaces only.
    """

    KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']

    def __init__(self, env):
        super(GoalEnvWrapper, self).__init__()

        self.env = env
        self.action_space = env.action_space
        self.spaces = list(env.observation_space.spaces.values())

        space_types = [type(env.observation_space.spaces[key]) for key in self.KEY_ORDER]
        assert len(set(space_types)) == 1, 'The spaces for goal and observation must be of the same type'

        if isinstance(self.spaces[0], spaces.Box):
            goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
            self.obs_dim = env.observation_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert goal_space_shape[1] == 1, 'Only 1D observation spaces are supported yet'
            else:
                assert len(goal_space_shape) == 1, 'Only 1D observation spaces are supported yet'

            lows = np.concatenate([space.low for space in self.spaces])
            highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)
        else:
            raise NotImplementedError('{} space is not supported'.format(type(self.spaces[0])))

    def convert_dict_to_obs(self, obs_dict):
        return np.concatenate([obs_dict[key] for key in self.KEY_ORDER])

    def convert_obs_to_dict(self, observations):
        """Inverse operation of convert_dict_to_obs"""

        return OrderedDict([
            ('observation', observations[:self.obs_dim]),
            ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[self.obs_dim + self.goal_dim:])
        ])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.convert_dict_to_obs(obs), reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        return self.convert_dict_to_obs(self.env.reset())

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

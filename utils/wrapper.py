import numpy as np
import gym


class AntTruncatedV2ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_obs_keep = 27
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.n_obs_keep,))
        self._max_episode_steps = env._max_episode_steps

    def observation(self, obs):
        # Modify obs
        return obs[:self.n_obs_keep]


class HumanoidTruncatedV2ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_obs_keep = 45
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.n_obs_keep,))
        self._max_episode_steps = env._max_episode_steps

    def observation(self, obs):
        # Modify obs
        return obs[:self.n_obs_keep]

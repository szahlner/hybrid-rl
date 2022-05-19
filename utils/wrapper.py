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


class FetchPushTruncatedV1ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_obs_keep = 16
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.n_obs_keep,))
        self._max_episode_steps = env._max_episode_steps

    def observation(self, obs):
        # Modify obs

        grip_pos = obs["observation"][:3] # ok
        object_pos = obs["observation"][3:6] # ok
        # object_rel_pos = obs["observation"][6:9] # remove
        gripper_state = obs["observation"][9:10] # ok
        object_rot = obs["observation"][10:13] # ok
        # object_velp = obs["observation"][13:16] # remove
        # object_velr = obs["observation"][16:19] # remove
        grip_velp = obs["observation"][19:22] # ok
        gripper_vel = obs["observation"][22:] # ok

        obs = {
            "observation": np.concatenate(
                [grip_pos, gripper_state, grip_velp, gripper_vel, object_pos, object_rot],
                axis=-1
            ),
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": obs["desired_goal"],
        }
        return obs

import numpy as np


class ShadowHandReachV0Reward:
    """ShadowHandReach-v0 reward"""

    def __call__(self, obs, actions, obs_next):
        achieved_goal = obs[:, -30:-15]
        desired_goal = obs[:, -15:]

        return -np.linalg.norm(achieved_goal - desired_goal, axis=-1)

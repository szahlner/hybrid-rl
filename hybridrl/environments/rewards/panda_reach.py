import numpy as np


class PandaReachV0Reward:
    """PandaReach-v0 reward"""

    def __call__(self, obs, actions, obs_next):
        achieved_goal = obs_next[:, -6:-3]
        desired_goal = obs_next[:, -3:]

        return -np.linalg.norm(achieved_goal - desired_goal, axis=-1)

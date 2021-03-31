import numpy as np


class MountainCarContinuousV0Reward:
    """MountainCarContinuous-v0 reward"""

    def __init__(self):
        self.goal_pos = 0.45
        self.goal_vel = 0

    def __call__(self, obs, actions, obs_next):
        pos = obs[:, 0]
        vel = obs[:, 1]
        goals = (pos >= self.goal_pos) & (vel >= self.goal_vel)

        costs = np.zeros_like(goals, dtype=np.float32)
        costs += goals * 100.
        costs -= 0.1 * actions[:, 0] ** 2

        return costs

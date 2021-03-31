import numpy as np


class PendulumV0Reward:
    """Pendulum-v0 reward"""

    def __call__(self, obs, actions, obs_next):
        thetas = np.arctan2(obs[:, 1], obs[:, 0])
        theta_dts = obs[:, 2]

        def normalize_angle(x):
            return ((x + np.pi) % (2 * np.pi)) - np.pi

        costs = normalize_angle(thetas) ** 2 + 0.1 * theta_dts ** 2 + 0.001 * actions[:, 0] ** 2

        return -costs

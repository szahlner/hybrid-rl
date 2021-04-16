import numpy as np


class PendulumV0Reward:
    """Pendulum-v0 reward"""

    def __init__(self, env_params):
        self.env_params = env_params

    def _action(self, action):
        a = (self.env_params.action_max - self.env_params.action_min) / 2.
        b = (self.env_params.action_max + self.env_params.action_min) / 2.
        return a * action + b

    def __call__(self, obs, actions, obs_next):
        actions = self._action(actions)

        thetas = np.arctan2(obs[:, 1], obs[:, 0])
        theta_dts = obs[:, 2]

        def normalize_angle(x):
            return ((x + np.pi) % (2 * np.pi)) - np.pi

        costs = normalize_angle(thetas) ** 2 + 0.1 * theta_dts ** 2 + 0.001 * actions[:, 0] ** 2
        return -costs

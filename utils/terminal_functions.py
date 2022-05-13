import numpy as np


def terminal_functions(env_name, obs, action, obs_next):
    if env_name == "Hopper-v2":
        assert len(obs.shape) == len(obs_next.shape) == len(action.shape) == 2

        height = obs_next[:, 0]
        angle = obs_next[:, 1]
        not_done = np.isfinite(obs_next).all(axis=-1) \
                   * np.abs(obs_next[:, 1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "Walker2d-v2":
        assert len(obs.shape) == len(obs_next.shape) == len(action.shape) == 2

        height = obs_next[:, 0]
        angle = obs_next[:, 1]
        not_done = (height > 0.8) \
                   * (height < 2.0) \
                   * (angle > -1.0) \
                   * (angle < 1.0)
        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "Ant-v2":
        assert len(obs.shape) == len(obs_next.shape) == len(action.shape) == 2

        x = obs_next[:, 0]
        not_done = np.isfinite(obs_next).all(axis=-1) \
                   * (x >= 0.2) \
                   * (x <= 1.0)

        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "InvertedPendulum-v2":
        assert len(obs.shape) == len(obs_next.shape) == len(action.shape) == 2

        notdone = np.isfinite(obs_next).all(axis=-1) \
                  * (np.abs(obs_next[:, 1]) <= .2)
        done = ~notdone
        done = done[:, None]
        return done
    elif env_name == "InvertedDoublePendulum-v2":
        assert len(obs.shape) == len(obs_next.shape) == len(action.shape) == 2

        sin1, cos1 = obs_next[:, 1], obs_next[:, 3]
        sin2, cos2 = obs_next[:, 2], obs_next[:, 4]
        theta_1 = np.arctan2(sin1, cos1)
        theta_2 = np.arctan2(sin2, cos2)
        y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))

        done = y <= 1
        done = done[:, None]
        return done
    else:
        # HalfCheetah-v2 goes in here too
        assert len(obs.shape) == len(obs_next.shape) == len(action.shape) == 2
        return np.zeros((len(obs), 1), dtype=np.bool)

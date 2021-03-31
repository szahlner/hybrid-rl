from hybridrl.environments.utils.wrapper import NormalizedActionsEnvWrapper, GoalEnvWrapper

from hybridrl.environments.rewards.pendulum import PendulumV0Reward
from hybridrl.environments.rewards.mountain_car_continuous import MountainCarContinuousV0Reward
from hybridrl.environments.rewards.panda_reach import PandaReachV0Reward
from hybridrl.environments.rewards.panda_push import PandaPushV0Reward
from hybridrl.environments.rewards.shadow_hand_reach import ShadowHandReachV0Reward


REWARD_FUNCTIONS = {
    'Pendulum-v0': PendulumV0Reward,
    'MountainCarContinuous-v0': MountainCarContinuousV0Reward,
    'PandaReach-v0': PandaReachV0Reward,
    'PandaPush-v0': PandaPushV0Reward,
    'ShadowHandReach-v0': ShadowHandReachV0Reward
}

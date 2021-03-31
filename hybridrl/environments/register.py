from gym.envs.registration import register


register(
    id='PandaReach-v0',
    entry_point='hybridrl.environments.robotics.panda:PandaReachGoalEnv'
)

register(
    id='PandaPush-v0',
    entry_point='hybridrl.environments.robotics.panda:PandaPushGoalEnv'
)

register(
    id='ShadowHandReach-v0',
    entry_point='hybridrl.environments.robotics.shadow_hand:ShadowHandReachGoalEnv'
)

register(
    id='ShadowHandReachAllRandom-v0',
    entry_point='hybridrl.environments.robotics.shadow_hand:ShadowHandReachGoalEnvAllRandom'
)


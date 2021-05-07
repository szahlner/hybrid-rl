from hybridrl.models.ddpg import ActorPendulumV0 as DDPG_ActorPendulumV0
from hybridrl.models.ddpg import CriticPendulumV0 as DDPG_CriticPendulumV0
from hybridrl.models.ddpg import ActorMountainCarContinuousV0 as DDPG_ActorMountainCarContinuousV0
from hybridrl.models.ddpg import CriticMountainCarContinuousV0 as DDPG_CriticMountainCarContinuousV0
from hybridrl.models.ddpg import ActorPandaReachV0 as DDPG_ActorPandaReachV0
from hybridrl.models.ddpg import CriticPandaReachV0 as DDPG_CriticPandaReachV0
from hybridrl.models.ddpg import ActorShadowHandReachV0 as DDPG_ActorShadowHandReachV0
from hybridrl.models.ddpg import CriticShadowHandReachV0 as DDPG_CriticShadowHandReachV0

from hybridrl.models.rs_mpc import DynamicsPendulumV0 as RSMPC_DynamicsPendulumV0
from hybridrl.models.rs_mpc import DynamicsMountainCarContinuousV0 as RSMPC_DynamicsMountainCarContinuousV0
from hybridrl.models.rs_mpc import DynamicsPandaReachV0 as RSMPC_DynamicsPandaReachV0
from hybridrl.models.rs_mpc import DynamicsPandaPushV0 as RSMPC_DynamicsPandaPushV0
from hybridrl.models.rs_mpc import DynamicsShadowHandReachV0 as RSMPC_DynamicsShadowHandReachV0

from hybridrl.models.cem import DynamicsPendulumV0 as CEM_DynamicsPendulumV0
from hybridrl.models.cem import DynamicsMountainCarContinuousV0 as CEM_DynamicsMountainCarContinuousV0
from hybridrl.models.cem import DynamicsPandaReachV0 as CEM_DynamicsPandaReachV0
from hybridrl.models.cem import DynamicsPandaPushV0 as CEM_DynamicsPandaPushV0
from hybridrl.models.cem import DynamicsShadowHandReachV0 as CEM_DynamicsShadowHandReachV0

from hybridrl.models.rs_mpc_ddpg import DynamicsPendulumV0 as RSMPCDDPG_DynamicsPendulumV0
from hybridrl.models.rs_mpc_ddpg import DynamicsPandaReachV0 as RSMPCDDPG_DynamicsPandaReachV0


DDPG_ACTOR_MODELS = {
    'Pendulum-v0': DDPG_ActorPendulumV0,
    'MountainCarContinuous-v0': DDPG_ActorMountainCarContinuousV0,
    'PandaReach-v0': DDPG_ActorPandaReachV0,
    'ShadowHandReach-v0': DDPG_ActorShadowHandReachV0
}

DDPG_CRITIC_MODELS = {
    'Pendulum-v0': DDPG_CriticPendulumV0,
    'MountainCarContinuous-v0': DDPG_CriticMountainCarContinuousV0,
    'PandaReach-v0': DDPG_CriticPandaReachV0,
    'ShadowHandReach-v0': DDPG_CriticShadowHandReachV0
}

RSMPC_DYNAMICS_MODELS = {
    'Pendulum-v0': RSMPC_DynamicsPendulumV0,
    'MountainCarContinuous-v0': RSMPC_DynamicsMountainCarContinuousV0,
    'PandaReach-v0': RSMPC_DynamicsPandaReachV0,
    'PandaPush-v0': RSMPC_DynamicsPandaPushV0,
    'ShadowHandReach-v0': RSMPC_DynamicsShadowHandReachV0
}

CEM_DYNAMICS_MODELS = {
    'Pendulum-v0': CEM_DynamicsPendulumV0,
    'MountainCarContinuous-v0': CEM_DynamicsMountainCarContinuousV0,
    'PandaReach-v0': CEM_DynamicsPandaReachV0,
    'PandaPush-v0': CEM_DynamicsPandaPushV0,
    'ShadowHandReach-v0': CEM_DynamicsShadowHandReachV0
}

RSMPCDDPG_DYNAMICS_MODELS = {
    'Pendulum-v0': RSMPCDDPG_DynamicsPendulumV0,
    'PandaReach-v0': RSMPCDDPG_DynamicsPandaReachV0
}

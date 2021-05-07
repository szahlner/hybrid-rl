from schema import Schema, Optional, And


rsmpc_ddpg_schema = {
    Optional('buffer_size', default=int(1e6)): And(int, lambda n: n > 0),
    Optional('lr_dynamics', default=0.001): And(float, lambda n: n > 0),
    Optional('lr_actor', default=0.001): And(float, lambda n: n > 0),
    Optional('lr_critic', default=0.001): And(float, lambda n: n > 0),
    Optional('batch_size_dynamics', default=10): And(int, lambda n: n > 0),
    Optional('batch_size_ddpg', default=10): And(int, lambda n: n > 0),
    Optional('gamma_ddpg', default=0.98): And(float, lambda n: n >= 0),
    Optional('polyak_ddpg', default=0.95): And(float, lambda n: 0 <= n <= 1),
    Optional('horizon', default=20): And(int, lambda n: n > 0),
    Optional('population_size', default=100): And(int, lambda n: n > 0)
}

RSMPC_DDPG_SCHEMA = Schema(rsmpc_ddpg_schema)

from schema import Schema, Optional, And


ddpg_schema = {
    Optional('buffer_size', default=int(1e6)): And(int, lambda n: n > 0),
    Optional('lr_actor', default=0.001): And(float, lambda n: n > 0),
    Optional('lr_critic', default=0.001): And(float, lambda n: n > 0),
    Optional('noise_eps', default=1.0): And(float, lambda n: 0 <= n),
    Optional('noise_eps_min', default=0.001): And(float, lambda n: 0 <= n),
    Optional('noise_eps_decay', default=0.001): And(float, lambda n: 0 <= n),
    Optional('batch_size', default=10): And(int, lambda n: n > 0),
    Optional('gamma', default=0.98): And(float, lambda n: 0 < n <= 1),
    Optional('polyak', default=0.005): And(float, lambda n: 0 <= n <= 1),
    Optional('random_exploration', default=0.3): And(float, lambda n: 0 <= n <= 1)
}

DDPG_SCHEMA = Schema(ddpg_schema)

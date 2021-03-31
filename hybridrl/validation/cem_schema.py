from schema import Schema, Optional, And, Or


cem_schema = {
    Optional('buffer_size', default=int(1e6)): And(int, lambda n: n > 0),
    Optional('lr_dynamics', default=0.001): And(float, lambda n: n > 0),
    Optional('batch_size', default=10): And(int, lambda n: n > 0),
    Optional('horizon', default=10): And(int, lambda n: n > 0),
    Optional('max_iter', default=10): And(int, lambda n: n > 0),
    Optional('population_size', default=100): And(int, lambda n: n > 0),
    Optional('n_elite', default=10): And(int, lambda n: n > 0),
    Optional('alpha', default=0.1): And(float, lambda n: n >= 0),
    Optional('init_mean', default=0.0): float,
    Optional('init_variance', default=1.0): float,
    Optional('epsilon', default=0.0): And(float, lambda n: n >= 0),
    Optional('sampler', default='truncated_normal'): Or(And(str, 'truncated_normal'), And(str, 'numpy_normal'))
}

CEM_SCHEMA = Schema(cem_schema)

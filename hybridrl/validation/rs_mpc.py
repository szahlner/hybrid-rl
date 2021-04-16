from schema import Schema, Optional, And


rsmpc_schema = {
    Optional('buffer_size', default=int(1e6)): And(int, lambda n: n > 0),
    Optional('lr_dynamics', default=0.001): And(float, lambda n: n > 0),
    Optional('batch_size', default=10): And(int, lambda n: n > 0),
    Optional('horizon', default=20): And(int, lambda n: n > 0),
    Optional('population_size', default=200): And(int, lambda n: n > 0)
}

RSMPC_SCHEMA = Schema(rsmpc_schema)
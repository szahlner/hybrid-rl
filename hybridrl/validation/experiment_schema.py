from schema import Schema, Optional, And, Or, Use


###########################
# Experiment settings
###########################
# agent_name    [str]:  Name of the agent (e.g. 'ddpg')
# debug_mode    [bool, optional]: Switch debug mode on/off
# device        [str, optional]:  Name of the device to run the agent on (e.g. 'cpu')
# n_workers     [int, optional]:  Number of workers for training
# seed          [int, optional]:  Random seed
experiment_schema = {
    'agent_name': And(str, Use(str.upper)),
    'log_dir': str,
    Optional('debug_mode', default=False): bool,
    Optional('device', default='cpu'): And(str, Use(str.lower)),
    Optional('seed', default=0): And(int, lambda n: n >= 0),
    Optional('n_epochs', default=10): And(int, lambda n: n > 0),
    Optional('n_cycles', default=10): And(int, lambda n: n > 0),
    Optional('n_rollouts', default=10): And(int, lambda n: n > 0),
    Optional('n_train_batches', default=None): And(int, lambda n: n > 0)
}

EXPERIMENT_SCHEMA = Schema(experiment_schema)

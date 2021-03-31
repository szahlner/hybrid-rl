from schema import Schema, Optional, And


###########################
# Environment settings
###########################
# id                    [str]:  Name/id of the environment (e.g. 'Pendulum-v0')
# max_episode_steps     [int]:  Maximal steps per episode
environment_schema = {
    'id': str,
    Optional('max_episode_steps', default=None): And(int, lambda n: n > 0),
    Optional('end_on_done', default=False): bool,
}

ENVIRONMENT_SCHEMA = Schema(environment_schema)

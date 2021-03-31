from schema import Schema, Optional, And


###########################
# Environment settings
###########################
# id                    [str]:  Name/id of the environment (e.g. 'Pendulum-v0')
# max_episode_steps     [int]:  Maximal steps per episode
test_schema = {
    Optional('n_episodes', default=10): And(int, lambda n: n > 0),
    Optional('visualize', default=False): bool,
    Optional('is_test', default=False): bool,
    Optional('gif', default=False): bool
}

TEST_SCHEMA = Schema(test_schema)

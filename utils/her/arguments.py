import argparse


class HerDdpgNamespace(argparse.Namespace):
    env_name: str
    n_epochs: int
    n_cycles: int
    n_batches: int
    save_interval: int
    seed: int
    num_workers: int
    replay_strategy: str
    clip_return: bool
    save_dir: str
    noise_eps: float
    random_eps: float
    buffer_size: int
    replay_k: int
    clip_obs: float
    batch_size: int
    gamma: float
    action_l2: float
    lr_actor: float
    lr_critic: float
    polyak: float
    n_test_rollouts: int
    clip_range: float
    demo_length: int
    num_rollouts_per_mpi: int

    model_dim_chunk: int
    model_type: str
    model_based: bool
    model_training_freq: int
    model_max_rollout_timesteps: int
    model_n_training_transitions: int
    model_n_rollout_transitions: int
    model_stochastic_percentage: float


class HerSacNamespace(HerDdpgNamespace):

    alpha: float
    lr_alpha: float
    automatic_entropy_tuning: bool
    target_entropy: float


def get_args_her_ddpg():
    parser = argparse.ArgumentParser()

    # the environment setting
    parser.add_argument("--env-name", type=str, default="FetchReach-v1", help="the environment name")
    parser.add_argument("--n-epochs", type=int, default=50, help="the number of epochs to train the agent")
    parser.add_argument("--n-cycles", type=int, default=1, help="the times to collect samples per epoch")
    parser.add_argument("--n-batches", type=int, default=40, help="the times to update the network")
    parser.add_argument("--save-interval", type=int, default=5, help="the interval that save the trajectory")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--num-workers", type=int, default=1, help="the number of cpus to collect samples")
    parser.add_argument("--replay-strategy", type=str, default="future", help="the HER strategy")
    parser.add_argument("--clip-return", action="store_false", help="if clip the returns")
    parser.add_argument("--save-dir", type=str, default="saved_models", help="the path to save the models")
    parser.add_argument("--noise-eps", type=float, default=0.2, help="noise eps")
    parser.add_argument("--random-eps", type=float, default=0.3, help="random eps")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="the size of the buffer")
    parser.add_argument("--replay-k", type=int, default=4, help="ratio to be replace")
    parser.add_argument("--clip-obs", type=float, default=200, help="the clip ratio")
    parser.add_argument("--batch-size", type=int, default=256, help="the sample batch size")
    parser.add_argument("--gamma", type=float, default=0.98, help="the discount factor")
    parser.add_argument("--action-l2", type=float, default=1, help="l2 reg")
    parser.add_argument("--lr-actor", type=float, default=0.001, help="the learning rate of the actor")
    parser.add_argument("--lr-critic", type=float, default=0.001, help="the learning rate of the critic")
    parser.add_argument("--polyak", type=float, default=0.95, help="the average coefficient")
    parser.add_argument("--n-test-rollouts", type=int, default=10, help="the number of tests")
    parser.add_argument("--clip-range", type=float, default=5, help="the clip range")
    parser.add_argument("--demo-length", type=int, default=20, help="the demo length")
    parser.add_argument("--num-rollouts-per-mpi", type=int, default=2, help="the rollouts per mpi")

    # World model
    parser.add_argument("--model-stochastic-percentage", type=float, default=1.0, help="percentage to take from confidence")
    parser.add_argument("--model-n-training-transitions", type=int, default=10000, help="number of training transitions")
    parser.add_argument("--model-n-rollout-transitions", type=int, default=10000, help="number of rollout transitions")
    parser.add_argument("--model-max-rollout-timesteps", type=int, default=5, help="timesteps to perform rollout")
    parser.add_argument("--model-dim-chunk", type=int, default=20, help="model dimension chunk")
    parser.add_argument("--model-type", type=str, choices=["deterministic", "stochastic"], default="deterministic", help="model type")
    parser.add_argument("--model-based", action="store_true", help="if use model based acceleration")
    parser.add_argument("--model-training-freq", type=int, default=100, help="frequency of model training")

    args = parser.parse_args(namespace=HerDdpgNamespace())

    return args


def get_args_her_sac():
    parser = argparse.ArgumentParser()

    # the environment setting
    parser.add_argument("--env-name", type=str, default="FetchReach-v1", help="the environment name")
    parser.add_argument("--n-epochs", type=int, default=50, help="the number of epochs to train the agent")
    parser.add_argument("--n-cycles", type=int, default=1, help="the times to collect samples per epoch")
    parser.add_argument("--n-batches", type=int, default=40, help="the times to update the network")
    parser.add_argument("--save-interval", type=int, default=5, help="the interval that save the trajectory")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--num-workers", type=int, default=1, help="the number of cpus to collect samples")
    parser.add_argument("--replay-strategy", type=str, default="future", help="the HER strategy")
    parser.add_argument("--clip-return", action="store_true", help="if clip the returns")
    parser.add_argument("--save-dir", type=str, default="saved_models", help="the path to save the models")
    parser.add_argument("--noise-eps", type=float, default=0.2, help="noise eps")
    parser.add_argument("--random-eps", type=float, default=0.3, help="random eps")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="the size of the buffer")
    parser.add_argument("--replay-k", type=int, default=4, help="ratio to be replace")
    parser.add_argument("--clip-obs", type=float, default=200, help="the clip ratio")
    parser.add_argument("--batch-size", type=int, default=256, help="the sample batch size")
    parser.add_argument("--gamma", type=float, default=0.98, help="the discount factor")
    parser.add_argument("--action-l2", type=float, default=1, help="l2 reg")
    parser.add_argument("--lr-actor", type=float, default=0.001, help="the learning rate of the actor")
    parser.add_argument("--lr-critic", type=float, default=0.001, help="the learning rate of the critic")
    parser.add_argument("--polyak", type=float, default=0.95, help="the average coefficient")
    parser.add_argument("--n-test-rollouts", type=int, default=10, help="the number of tests")
    parser.add_argument("--clip-range", type=float, default=5, help="the clip range")
    parser.add_argument("--demo-length", type=int, default=20, help="the demo length")
    parser.add_argument("--num-rollouts-per-mpi", type=int, default=2, help="the rollouts per mpi")

    # World model
    parser.add_argument("--model-stochastic-percentage", type=float, default=1.0, help="percentage to take from confidence")
    parser.add_argument("--model-n-training-transitions", type=int, default=10000, help="number of training transitions")
    parser.add_argument("--model-n-rollout-transitions", type=int, default=10000, help="number of rollout transitions")
    parser.add_argument("--model-max-rollout-timesteps", type=int, default=5, help="timesteps to perform rollout")
    parser.add_argument("--model-dim-chunk", type=int, default=20, help="model dimension chunk")
    parser.add_argument("--model-type", type=str, choices=["deterministic", "stochastic"], default="deterministic", help="model type")
    parser.add_argument("--model-based", action="store_true", help="if use model based acceleration")
    parser.add_argument("--model-training-freq", type=int, default=100, help="frequency of model training")

    # Sac
    parser.add_argument("--alpha", type=float, default=0.2, help="Temperature parameter ?? determines the relative importance of the entropy term against the reward (default: 0.2)")
    parser.add_argument("--lr-alpha", type=float, default=0.0003, help="the learning rate of the alpha")
    parser.add_argument("--automatic-entropy-tuning", action="store_true", help="Automaically adjust ?? (default: False)")
    parser.add_argument("--target-entropy", type=float, help="Target entropy")

    args = parser.parse_args(namespace=HerSacNamespace())

    return args

import gym

from policy.spinningup.utils.run_utils import setup_logger_kwargs
from policy.spinningup.algos.pytorch.sac.sac import sac
from policy.spinningup.algos.pytorch.sac import core


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=400)  # -1 means use mbpo epochs
    parser.add_argument('--exp_name', type=str, default='SSAC')
    parser.add_argument('--data_dir', type=str, default='../logs/')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--steps-per-epoch', type=int, default=1000)
    parser.add_argument('--max-ep-len', type=int, default=1000)
    parser.add_argument('--n-evals-per-epoch', type=int, default=10)

    parser.add_argument('--replay-size', type=int, default=1000000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', action='store_false')
    parser.add_argument('--start-steps', type=int, default=5000)
    parser.add_argument("--target-entropy", type=float, default=-3, help="Target entropy")

    # World model
    parser.add_argument("--model-stochastic-percentage", type=float, default=1.0, help="percentage to take from confidence")
    parser.add_argument("--model-n-training-transitions", type=int, default=10000, help="number of training transitions")
    parser.add_argument("--model-n-rollout-transitions", type=int, default=10000, help="number of rollout transitions")
    parser.add_argument("--model-max-rollout-timesteps", type=int, default=5, help="timesteps to perform rollout")
    parser.add_argument("--model-dim-chunk", type=int, default=20, help="model dimension chunk")
    parser.add_argument("--model-type", type=str, choices=["deterministic", "stochastic"], default="deterministic", help="model type")
    parser.add_argument("--model-based", action="store_true", help="if use model based acceleration")
    parser.add_argument("--model-training-freq", type=int, default=1000, help="frequency of model training")

    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)

    args = parser.parse_args()

    if args.model_based:
        if args.model_type == "deterministic":
            args.exp_name = f"{args.exp_name}+DWM"
        else:
            args.exp_name = f"{args.exp_name}+SWM"

    # modify the code here if you want to use a different naming scheme
    exp_name_full = args.exp_name + '_%s' % args.env

    # specify experiment name, seed and data_dir.
    # for example, for seed 0, the progress.txt will be saved under data_dir/exp_name/exp_name_s0
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    sac(lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        steps_per_epoch=args.steps_per_epoch,
        replay_size=args.replay_size,
        polyak=args.polyak,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        start_steps=args.start_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        num_test_episodes=args.n_evals_per_epoch,
        target_entropy=args.target_entropy,
        max_ep_len=args.max_ep_len,
        args=args,
    )

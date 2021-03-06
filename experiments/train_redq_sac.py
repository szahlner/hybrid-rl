import gym
import numpy as np
import torch
import time
import sys
from policy.redq.algos.redq_sac import REDQSACAgent
from policy.redq.algos.core import mbpo_epoches, test_agent
from policy.redq.utils.run_utils import setup_logger_kwargs
from policy.redq.utils.bias_utils import log_bias_evaluation
from policy.redq.utils.logx import EpochLogger

from utils.utils import get_env_params


def redq_sac(env_name, seed=0, epochs='mbpo', steps_per_epoch=1000,
             max_ep_len=1000, n_evals_per_epoch=10,
             logger_kwargs=dict(), debug=False,
             # following are agent related hyperparameters
             hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
             lr=3e-4, gamma=0.99, polyak=0.995,
             alpha=0.2, auto_alpha=True, target_entropy='auto',
             start_steps=5000, delay_update_steps='auto',
             utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
             policy_update_delay=1,
             # following are bias evaluation related
             evaluate_bias=True, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
             args=None,
             ):
    """
    :param env_name: name of the gym environment
    :param seed: random seed
    :param epochs: number of epochs to run
    :param steps_per_epoch: number of timestep (datapoints) for each epoch
    :param max_ep_len: max timestep until an episode terminates
    :param n_evals_per_epoch: number of evaluation runs for each epoch
    :param logger_kwargs: arguments for logger
    :param debug: whether to run in debug mode
    :param hidden_sizes: hidden layer sizes
    :param replay_size: replay buffer size
    :param batch_size: mini-batch size
    :param lr: learning rate for all networks
    :param gamma: discount factor
    :param polyak: hyperparameter for polyak averaged target networks
    :param alpha: SAC entropy hyperparameter
    :param auto_alpha: whether to use adaptive SAC
    :param target_entropy: used for adaptive SAC
    :param start_steps: the number of random data collected in the beginning of training
    :param delay_update_steps: after how many data collected should we start updates
    :param utd_ratio: the update-to-data ratio
    :param num_Q: number of Q networks in the Q ensemble
    :param num_min: number of sampled Q values to take minimal from
    :param q_target_mode: 'min' for minimal, 'ave' for average, 'rem' for random ensemble mixture
    :param policy_update_delay: how many updates until we update policy network
    """
    if debug:  # use --debug for very quick debugging
        hidden_sizes = [2, 2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100

    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set number of epoch
    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: gym.make(args.env)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()

    # Goal-Env
    obs = env.reset()
    if isinstance(obs, dict):
        from gym.wrappers import FilterObservation, FlattenObservation
        keys = ["observation", "desired_goal"]

        # Envs
        env = FlattenObservation(FilterObservation(env, keys))
        test_env = FlattenObservation(FilterObservation(test_env, keys))
        bias_eval_env = FlattenObservation(FilterObservation(bias_eval_env, keys))

    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)

    seed_all(epoch=0)

    env_params = get_env_params(env)

    """prepare to init agent"""
    # get obs and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length
    if isinstance(obs, dict):
        max_ep_len = env.env.env._max_episode_steps if max_ep_len > env.env.env._max_episode_steps else max_ep_len
    else:
        max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # keep track of run time
    start_time = time.time()
    # flush logger (optional)
    sys.stdout.flush()
    #################################################################################################

    """init agent and start training"""
    agent = REDQSACAgent(env_name, obs_dim, act_dim, act_limit, device,
                         hidden_sizes, replay_size, batch_size,
                         lr, gamma, polyak,
                         alpha, auto_alpha, target_entropy,
                         start_steps, delay_update_steps,
                         utd_ratio, num_Q, num_min, q_target_mode,
                         policy_update_delay,
                         args, env_params)

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        # get action from agent
        a = agent.get_exploration_action(o, env)
        # Step the env, get next observation, reward and done signal
        o2, r, d, _ = env.step(a)

        # Very important: before we let agent store this transition,
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        ep_len += 1
        d = False if ep_len == max_ep_len else d

        # give new data to agent
        agent.store_data(o, a, r, o2, d)

        if args.model_based:
            agent.store_data_for_world_model(o, a, r, o2, d)
            agent.do_world_model_stuff(t, logger)

        # let agent update
        agent.train(logger)
        # set obs to next obs
        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            # store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            # reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if (t + 1) >= start_steps and (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent(agent, test_env, max_ep_len, logger, args.n_evals_per_epoch)  # add logging here
            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('SuccessRate', with_min_and_max=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if args.model_based:
                logger.log_tabular("WorldModelReplayBufferSize", with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()

            # flush logged information to disk
            sys.stdout.flush()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=-1)  # -1 means use mbpo epochs
    parser.add_argument('--exp_name', type=str, default='REDQ+SAC')
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
    parser.add_argument('--utd-ratio', type=int, default=1)
    parser.add_argument('--num-Q', type=int, default=2)
    parser.add_argument('--num-min', type=int, default=2)
    parser.add_argument('--policy-update-delay', type=int, default=1)

    parser.add_argument('--evaluate-bias', action='store_false')
    parser.add_argument('--n-mc-eval', type=int, default=1000)
    parser.add_argument('--n-mc-cutoff', type=int, default=350)
    parser.add_argument('--reseed_each_epoch', action='store_false')

    # World model
    parser.add_argument("--model-stochastic-percentage", type=float, default=1.0, help="percentage to take from confidence")
    parser.add_argument("--model-n-training-transitions", type=int, default=10000, help="number of training transitions")
    parser.add_argument("--model-n-rollout-transitions", type=int, default=10000, help="number of rollout transitions")
    parser.add_argument("--model-max-rollout-timesteps", type=int, default=5, help="timesteps to perform rollout")
    parser.add_argument("--model-dim-chunk", type=int, default=20, help="model dimension chunk")
    parser.add_argument("--model-type", type=str, choices=["deterministic", "stochastic"], default="deterministic", help="model type")
    parser.add_argument("--model-based", action="store_true", help="if use model based acceleration")
    parser.add_argument("--model-training-freq", type=int, default=1000, help="frequency of model training")

    args = parser.parse_args()

    # modify the code here if you want to use a different naming scheme
    exp_name_full = args.exp_name + '_%s' % args.env

    # specify experiment name, seed and data_dir.
    # for example, for seed 0, the progress.txt will be saved under data_dir/exp_name/exp_name_s0
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    redq_sac(args.env, seed=args.seed, epochs=args.epochs,
             steps_per_epoch=args.steps_per_epoch,
             max_ep_len=args.max_ep_len,
             n_evals_per_epoch=args.n_evals_per_epoch,
             replay_size=args.replay_size,
             batch_size=args.batch_size,
             lr=args.lr,
             gamma=args.gamma,
             polyak=args.polyak,
             alpha=args.alpha,
             auto_alpha=args.auto_alpha,
             start_steps=args.start_steps,
             utd_ratio=args.utd_ratio,
             num_Q=args.num_Q,
             num_min=args.num_min,
             policy_update_delay=args.policy_update_delay,
             evaluate_bias=args.evaluate_bias,
             n_mc_eval=args.n_mc_eval,
             n_mc_cutoff=args.n_mc_cutoff,
             reseed_each_epoch=args.reseed_each_epoch,
             args=args,
             logger_kwargs=logger_kwargs, debug=args.debug)

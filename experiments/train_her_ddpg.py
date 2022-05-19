import os
import numpy as np
from mpi4py import MPI
import random
import torch

import gym

from policy.her_ddpg import HER
from utils.her.arguments import get_args_her_ddpg, HerDdpgNamespace
from utils.utils import get_env_params, prepare_logger


def train(args: HerDdpgNamespace):
    # Environments imports
    if "ShadowHand" in args.env_name:
        import shadowhand_gym
    if "TriFinger" in args.env_name:
        import trifinger_simulation
    if "parking" in args.env_name:
        import highway_env

    # Create the environment
    if args.env_name == "FetchPushTruncated-v1":
        from utils.wrapper import FetchPushTruncatedV1ObservationWrapper

        env = gym.make("FetchPush-v1")
        env = FetchPushTruncatedV1ObservationWrapper(env)
    else:
        env = gym.make(args.env_name)

    if args.env_name == "parking-v0":
        env._max_episode_steps = 50

    # Set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # Get the environment parameters
    env_params = get_env_params(env)

    # Prepare logger
    logger = prepare_logger(args)

    # Initialize agent / policy
    agent = HER(args, env, env_params, logger)
    agent.learn()


if __name__ == "__main__":
    # MPI configuration
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["IN_MPI"] = "1"

    # Get the params
    args = get_args_her_ddpg()

    # Start loop
    train(args)

import os
import numpy as np
import gym
from gym.wrappers import FilterObservation, FlattenObservation
from mpi4py import MPI
import random
import torch
import shadowhand_gym

from policy.ddpg import DDPG
from utils.ddpg.arguments import get_args_her, DdpgNamespace
from utils.utils import get_env_params, prepare_logger


def train(args: DdpgNamespace):
    # Create the environment
    env = gym.make(args.env_name)

    # Set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # Get the environment parameters
    env_params = get_env_params(env)

    if env_params["goal"] == 0:
        keys = ["observation", "desired_goal"]
        env = FlattenObservation(FilterObservation(env, keys))

    # Prepare logger
    logger = prepare_logger(args)

    # Initialize agent / policy
    agent = DDPG(args, env, env_params, logger)
    agent.learn()


if __name__ == '__main__':
    # MPI configuration
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get the params
    args = get_args_her()

    # Start loop
    train(args)

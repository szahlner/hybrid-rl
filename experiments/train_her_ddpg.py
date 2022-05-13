import os
import numpy as np
from mpi4py import MPI
import random
import torch

import gym

from policy.her_ddpg import HER
from utils.her.arguments import get_args_her_ddpg, HerDdpgNamespace
from utils.utils import get_env_params, prepare_logger


class FetchPushObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # Modify obs

        grip_pos = obs["observation"][:3]
        object_pos = obs["observation"][3:6]
        object_rel_pos = obs["observation"][6:9]
        gripper_state = obs["observation"][9:10]
        object_rot = obs["observation"][10:13]
        object_velp = obs["observation"][13:16]
        object_velr = obs["observation"][16:19]
        grip_velp = obs["observation"][19:22]
        gripper_vel = obs["observation"][22:]

        obs = {
            "observation": np.concatenate([object_pos, object_rel_pos, object_rot, object_velp, object_velr, grip_pos, gripper_state, grip_velp, gripper_vel], axis=-1),
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": obs["desired_goal"],
        }
        return obs


def train(args: HerDdpgNamespace):
    # Environments imports
    if "ShadowHand" in args.env_name:
        import shadowhand_gym
    if "TriFinger" in args.env_name:
        import trifinger_simulation
    if "parking" in args.env_name:
        import highway_env

    # Create the environment
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

    if args.env_name == "FetchPush-v1":
        env = FetchPushObservationWrapper(env)

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

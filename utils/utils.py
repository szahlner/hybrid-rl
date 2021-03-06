import os
import gym
import time
import datetime
from typing import Union
from mpi4py import MPI

from utils.her.arguments import HerDdpgNamespace, HerSacNamespace
from utils.ddpg.arguments import DdpgNamespace, DdpgHerNamespace
from utils.sac.arguments import SacNamespace, SacHerNamespace
from utils.logger import EpochLogger


def get_env_params(env: Union[gym.Env, gym.GoalEnv]) -> dict:
    obs = env.reset()

    params = {
        "obs": 0,
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
        "max_timesteps": env._max_episode_steps,
        "goal": 0,
        "reward": 1,
    }

    if isinstance(obs, dict):
        params["obs"] = obs["observation"].shape[0]
        params["goal"] = obs["desired_goal"].shape[0]
    else:
        params["obs"] = env.observation_space.shape[0]

    return params


def prepare_logger(args: Union[DdpgNamespace, DdpgHerNamespace, SacNamespace, SacHerNamespace, HerDdpgNamespace, HerSacNamespace]) -> EpochLogger:
    log_dir = "./../logs"
    if not os.path.exists(log_dir) and MPI.COMM_WORLD.Get_rank() == 0:
        os.makedirs(log_dir)

    if isinstance(args, HerSacNamespace):
        agent = "HER+SAC"
    elif isinstance(args, HerDdpgNamespace):
        agent = "HER+DDPG"
    elif isinstance(args, DdpgHerNamespace) or isinstance(args, DdpgNamespace):
        agent = "DDPG"
    elif isinstance(args, SacHerNamespace) or isinstance(args, SacNamespace):
        agent = "SAC"

    if args.model_based and args.model_type == "deterministic":
        agent = f"{agent}+DWM"
    elif args.model_based and args.model_type == "stochastic":
        agent = f"{agent}+SWM"

    log_dir = os.path.join(
        log_dir,
        f"{agent}_{args.env_name}_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H-%M-%S')}"
    )

    logger_kwargs = {
        "output_dir": log_dir,
        "output_fname": "log.txt",
        "exp_name": f"{agent}",
    }

    logger = EpochLogger(**logger_kwargs)

    # Sort for world model arguments
    config_args = {}
    world_model_args = {}
    for key, value in vars(args).items():
        if "model" in key:
            world_model_args[key] = value
        else:
            config_args[key] = value

    config_kwargs = {
        "config": config_args,
        "world_model": world_model_args,
    }

    logger.save_config(config_kwargs)

    return logger

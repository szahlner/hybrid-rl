import gym
import numpy as np
import torch
import sys

from dotmap import DotMap
from mpi4py import MPI

from hybridrl.validation import EXPERIMENT_SCHEMA, ENVIRONMENT_SCHEMA, AGENT_SCHEMAS, TEST_SCHEMA
from hybridrl.agents import AGENTS
from hybridrl.environments import NormalizedActionsEnvWrapper, GoalEnvWrapper
from hybridrl.environments.register import *


class Experiment:
    """Experiment class"""

    def __init__(self, params):
        # experiment params validation
        assert params['experiment'] is not None, 'Experiment params must not be none.'
        experiment_params = EXPERIMENT_SCHEMA.validate(params['experiment'])
        experiment_params = DotMap(experiment_params)
        # common
        self.debug_mode = experiment_params.debug_mode

        # debug message
        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('PARAMS VALIDATION')
            print('{}'.format('#' * 60))
            print('Validation of experiment params successful.')

        # test params validation
        assert params['test'] is not None, 'Test params must not be none.'
        test_params = TEST_SCHEMA.validate(params['test'])
        test_params = DotMap(test_params)

        # debug message
        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('Validation of test params successful.')

        # environment params validation
        assert params['env'] is not None, 'Environment params must not be none.'
        env_params = ENVIRONMENT_SCHEMA.validate(params['env'])
        env_params = DotMap(env_params)

        # debug message
        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('Validation of environment params successful.')

        # agent params validation
        assert params['agent'] is not None, 'Agent params must not be none.'
        agent_params = AGENT_SCHEMAS[experiment_params.agent_name].validate(params['agent'])
        agent_params = DotMap(agent_params)

        # debug message
        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('Validation of agent params successful.')
            print('{}'.format('#' * 60))
            print('')
            print('SETTING UP ENVIRONMENT and SEEDING')
            print('{}'.format('#' * 60))

        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('Done setting up environment.')

        # make environment
        if env_params.max_episode_steps:
            try:
                env = gym.make(env_params.id).env
            except AttributeError:
                env = gym.make(env_params.id)
        else:
            env = gym.make(env_params.id)

        if isinstance(env, gym.GoalEnv):
            # goalenv obs to normal obs (dict to vector)
            env = GoalEnvWrapper(env)
        else:
            # normalize actions
            env = NormalizedActionsEnvWrapper(env)

        env_params.obs_dim = env.observation_space.shape[0]
        env_params.action_dim = env.action_space.shape[0]
        env_params.action_max = env.action_space.high[0]

        # set seed
        seed = experiment_params.seed + MPI.COMM_WORLD.Get_rank()
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        experiment_params.seed = seed

        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('Done seeding.')
            print('{}'.format('#' * 60))
            print('')
            print('SETTING UP AGENT')
            print('{}'.format('#' * 60))

        # make agent
        self.agent = AGENTS[experiment_params.agent_name](experiment_params, agent_params, env_params, env, test_params)

        # debug message
        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('Done setting up agent.')
            print('{}'.format('#' * 60))
            sys.stdout.flush()

    def start(self):
        # debug message
        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('')
            print('START LEARNING')
            print('{}'.format('#' * 60))
            sys.stdout.flush()

        self.agent.train()

    def test(self):
        # debug message
        if self.debug_mode and MPI.COMM_WORLD.Get_rank() == 0:
            print('')
            print('START TESTING')
            print('{}'.format('#' * 60))
            sys.stdout.flush()

        self.agent.test()

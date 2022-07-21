import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import random
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from collections import namedtuple


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Hopper-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=125001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--target_entropy', type=float, default=-1, metavar='N',
                    help='Target entropy to use (default: -1)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--her', action="store_true",
                    help='use HER (default: False)')
parser.add_argument('--model_based', action="store_true",
                    help='use Model-based (default: False)')
args = parser.parse_args()

args.cuda = True if torch.cuda.is_available() else False

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# HER
if args.her:
    Experience = namedtuple("Experience", field_names="state action reward next_state done reward_sum")
    future_k = 4
    goal = np.inf  # Unreachable goal

# Model-based
if args.model_based:
    from mbher.world_model import EnsembleDynamicsModel
    from mbher.utils import termination_fn

    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    env_model = EnsembleDynamicsModel(
        network_size=7,
        elite_size=5,
        state_size=state_size,
        action_size=action_size,
        reward_size=1,
        hidden_size=200,
        use_decay=True
    )

    world_model_trained = False

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Tensorboard
writer = SummaryWriter(
    'runs/{}_SAC_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy,
                                  "autotune" if args.automatic_entropy_tuning else "", "her" if args.her else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_trajectory = []
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if args.model_based and args.start_steps < total_numsteps and total_numsteps % 250 == 0:
            # Get all samples from environment
            o, a, r, o_2, _ = memory.sample(batch_size=len(memory))
            inputs = np.concatenate((o, a, o_2), axis=-1)
            labels = np.expand_dims(r, axis=-1)

            # env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

            # world_model_trained = True

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        if args.her:
            # Append to episode trajectory
            episode_trajectory.append(Experience(state, action, reward, next_state, done, episode_reward))
            reward = float(episode_reward > goal)

        memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        state = next_state

    if args.her:
        # Adjust goal
        if i_episode == 1:
            goal = episode_reward
        else:
            gaol = episode_reward if episode_reward > goal else goal

        # Fill up replay memory
        steps_taken = len(episode_trajectory)
        for t in range(steps_taken):
            # Standard experience replay
            o, a, _, o_2, d, r_s = episode_trajectory[t]

            for _ in range(future_k):
                future = random.randint(t, steps_taken - 1)  # Index of future time step
                new_goal = episode_trajectory[future].reward_sum  # Take future reward_sum and set as goal
                new_reward = float(r_s > new_goal)
                memory.push(o, a, new_reward, o_2, float(not d))

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, total_numsteps)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                  episode_steps,
                                                                                  round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar('avg_reward/test', avg_reward, total_numsteps)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()

import numpy as np
import torch
import gym
import argparse
import os
import time
import datetime

import utils.utils as utils
import policy.EnsembleDDPG as EDDPG

from unreal_env.env2 import EnsembleDynamicsModel as UnrealEnvironment
from utils.logger import EpochLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reject_outliers(data, mean, std, m=2.0):
    data_centered = np.abs(data - mean)
    inliers = np.where(data_centered < m * std, True, False)
    return np.all(inliers, axis=1)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    # print("---------------------------------------")
    # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    # print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="EDDPG")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Hopper-v3")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=125e3, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--update_to_data_ratio", default=10, type=int)
    parser.add_argument("--num_q", default=10, type=int)
    parser.add_argument("--num_q_min", default=2, type=int)
    parser.add_argument("--num_pi", default=10, type=int)
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--model_based", action="store_true")

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.update_to_data_ratio}_virtual"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    log_dir = "./../logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(
        log_dir,
        f"{args.policy}_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H-%M-%S')}"
    )

    logger_kwargs = {
        "output_dir": log_dir,
        "output_fname": "progress.txt",
        "exp_name": "virtual",
    }
    logger = EpochLogger(**logger_kwargs)
    config_kwargs = {
        "config": args,
    }
    logger.save_config(config_kwargs)

    if not os.path.exists(os.path.join(log_dir, "results")):
        os.makedirs(os.path.join(log_dir, "results"))

    if args.save_model and not os.path.exists(os.path.join(log_dir, "models")):
        os.makedirs(os.path.join(log_dir, "models"))

    env = gym.make(args.env)

    # Set seeds
    def seed_all():
        env.seed(args.seed)
        env.action_space.np_random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    seed_all()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "num_q": args.num_q,
        "num_q_min": args.num_q_min,
        "num_pi": args.num_pi,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
    }

    # Initialize policy
    policy = EDDPG.EnsembleDDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./../models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    if args.model_based:
        unreal_env = UnrealEnvironment(state_dim, action_dim, 1)

    if False:
        unreal_replay_buffer = utils.ReplayBuffer(state_dim, action_dim,)
        real_ratio = np.linspace(0.5, 1.0, num=int(args.max_timesteps))

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    start_time = time.time()

    def prepare_logger():
        logger.store(
            UnrealBufferSize=0,
            RealBatchSize=0,
            UnrealBatchSize=0,
            RealRatio=0,
            ActorLoss=0,
            CriticLoss=0,
        )
    prepare_logger()

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            if t % 250 == 0 and args.model_based:
                # Train unreal env
                idx = np.arange(replay_buffer.size)
                obs, action, next_obs, reward, _ = replay_buffer.sample_numpy(batch_size=len(idx), idx=idx)
                training_inputs = obs
                training_labels = np.concatenate([next_obs, reward], axis=-1)
                unreal_env.train(training_inputs, action, training_labels)

                if False:
                    # Stats from real replay buffer
                    replay_buffer_state_mean = np.mean(replay_buffer.state[:replay_buffer.size], axis=0)
                    replay_buffer_state_std = np.std(replay_buffer.state[:replay_buffer.size], axis=0)

                    # Update unreal replay buffer
                    if unreal_replay_buffer.size > 0:
                        idxs = np.arange(unreal_replay_buffer.size)
                        obs, action, _, _, _ = unreal_replay_buffer.sample_numpy(idxs)
                        next_obs, confidence = unreal_env.predict(obs, action)
                        # Split observations
                        reward = next_obs[:, state_dim:]
                        next_obs = next_obs[:, :state_dim]
                        # Handle outliers
                        inliers = reject_outliers(next_obs, replay_buffer_state_mean, replay_buffer_state_std, m=1)
                        if inliers.sum() == 0:
                            break
                        # Select inlier observations
                        obs = obs[inliers]
                        action = action[inliers]
                        next_obs = next_obs[inliers]
                        confidence = confidence[inliers]
                        reward = reward[inliers]
                        # Clear buffer
                        unreal_replay_buffer.clear()
                        # Append rollout
                        for k in range(len(obs)):
                            unreal_replay_buffer.add(obs[k], action[k], next_obs[k], reward[k], False, confidence[k])

                    # Rollout unreal env
                    obs, _, _, _, _ = replay_buffer.sample(100000)
                    obs = obs.detach().cpu().numpy()
                    for n in range(5):  # env._max_episode_steps):
                        action = policy.select_action_low_memory(obs)
                        # action = policy.actor(torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
                        action += np.random.normal(0, max_action * args.expl_noise, size=action.shape)
                        action = action.clip(-max_action, max_action)
                        next_obs, confidence = unreal_env.predict(obs, action)
                        # Split observations
                        reward = next_obs[:, state_dim:]
                        next_obs = next_obs[:, :state_dim]
                        # Handle outliers
                        inliers = reject_outliers(next_obs, replay_buffer_state_mean, replay_buffer_state_std, m=1)
                        if inliers.sum() == 0:
                            break
                        # Select inlier observations
                        obs = obs[inliers]
                        action = action[inliers]
                        next_obs = next_obs[inliers]
                        confidence = confidence[inliers]
                        reward = reward[inliers]
                        # Append rollout
                        for k in range(len(obs)):
                            unreal_replay_buffer.add(obs[k], action[k], next_obs[k], reward[k], False, confidence[k])
                        # Re-assign observations
                        obs = next_obs

                    logger.store(
                        UnrealBufferSize=unreal_replay_buffer.size,
                    )

            # actor_loss, critic_loss = 0, 0
            for n_update in range(args.update_to_data_ratio):

                if args.model_based and n_update % 2 == 0:
                    critic_loss = policy.train_critic_virtual(replay_buffer, 256, unreal_env)
                    # logger.store(CriticLoss=critic_loss)
                else:
                    critic_loss = policy.train_critic(replay_buffer, 256)
                    # logger.store(CriticLoss=critic_loss)
                logger.store(CriticLoss=critic_loss)

            actor_loss = policy.train_actor(replay_buffer, 256)
            logger.store(
                ActorLoss=actor_loss,
            )

        if done:
            logger.store(
                EpisodeTimesteps=episode_timesteps,
                EpisodeReward=episode_reward,
            )
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(os.path.join(log_dir, "results", file_name), evaluations)

            if args.save_model:
                policy.save(os.path.join(log_dir, "models", file_name))

                if args.model_based:
                    unreal_env.save(os.path.join(log_dir, "models", file_name))

            logger.log_tabular("Timesteps", t)
            logger.log_tabular("Time", time.time() - start_time)
            logger.log_tabular("EpisodeNum", episode_num)
            logger.log_tabular("EpisodeTimesteps", with_min_and_max=True)
            logger.log_tabular("EpisodeReward", with_min_and_max=True)
            logger.log_tabular("EpisodeEvalReward", evaluations[-1])
            # logger.log_tabular("UnrealBufferSize", with_min_and_max=True)
            # logger.log_tabular("RealBatchSize")
            # logger.log_tabular("UnrealBatchSize")
            # logger.log_tabular("RealRatio")
            logger.log_tabular("ActorLoss", with_min_and_max=True)
            logger.log_tabular("CriticLoss", with_min_and_max=True)
            logger.dump_tabular()

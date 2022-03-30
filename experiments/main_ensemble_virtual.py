import numpy as np
import torch
import gym
import argparse
import os
import time
import datetime

import utils.utils as utils
import policy.EnsembleDDPG as EDDPG

from unreal_env.env2 import EnsembleDynamicsModel as StochasticUnrealEnvironment
from unreal_env.envd import EnsembleDynamicsModel as DeterministicUnrealEnvironment
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
    parser.add_argument("--env", default="Hopper-v2")  # OpenAI gym environment name
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
        policy.load(f"./../{args.load_model}/models/{file_name}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    if args.model_based:
        stochastic_unreal_env = StochasticUnrealEnvironment(state_dim, action_dim, 1, network_size=1)
        deterministic_unreal_env = DeterministicUnrealEnvironment(state_dim, action_dim, 1, network_size=1)
        # unreal_replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

        if args.load_model != "":
            unreal_env_file = file_name if args.load_model == "default" else args.load_model
            policy.load(f"./../{args.load_model}/models/{file_name}")

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
            # UnrealBufferSize=0,
            RealBatchSize=0,
            UnrealBatchSize=0,
            RealRatio=0,
            ActorLoss=0,
            CriticLoss=0,
            # CriticLossVirtual=0,
            ConfMean=0,
            ConfMax=0,
            ConfMin=0,
            ConfStd=0,
            ConfMeanStochastic=0,
            ConfMaxStochastic=0,
            ConfMinStochastic=0,
            ConfStdStochastic=0,
            ConfMeanDeterministic=0,
            ConfMaxDeterministic=0,
            ConfMinDeterministic=0,
            ConfStdDeterministic=0,
            ModellErrorStochastic=0,
            ModellMSEErrorStochastic=0,
            ModellErrorDeterministic=0,
            ModellMSEErrorDeterministic=0,
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
                # idx = np.arange(replay_buffer.size)
                # obs, action, next_obs, reward, _ = replay_buffer.sample_numpy(batch_size=len(idx), idx=idx)
                obs, action, next_obs, reward, _ = replay_buffer.sample_numpy(batch_size=10000)
                training_inputs = obs
                training_labels = np.concatenate([next_obs - obs, reward], axis=-1)

                stochastic_unreal_env.train(training_inputs, action, training_labels)
                deterministic_unreal_env.train(training_inputs, action, training_labels)

                state_numpy, action_numpy, next_obs_numpy, reward_numpy, _ = replay_buffer.sample_numpy(256)
                stochastic_next_obs_numpy, stochastic_confidence_numpy = stochastic_unreal_env.predict(state_numpy, action_numpy)
                deterministic_next_obs_numpy, deterministic_confidence_numpy = deterministic_unreal_env.predict(state_numpy, action_numpy)

                stochastic_next_obs_numpy[:, :state_dim] += state_numpy
                deterministic_next_obs_numpy[:, :state_dim] += state_numpy

                if logger is not None:
                    logger.store(
                        ConfMean=0,
                        ConfMax=0,
                        ConfMin=0,
                        ConfStd=0,
                        ConfMeanStochastic=np.mean(stochastic_confidence_numpy),
                        ConfMaxStochastic=np.max(stochastic_confidence_numpy),
                        ConfMinStochastic=np.min(stochastic_confidence_numpy),
                        ConfStdStochastic=np.std(stochastic_confidence_numpy),
                        ConfMeanDeterministic=np.mean(deterministic_confidence_numpy),
                        ConfMaxDeterministic=np.max(deterministic_confidence_numpy),
                        ConfMinDeterministic=np.min(deterministic_confidence_numpy),
                        ConfStdDeterministic=np.std(deterministic_confidence_numpy),
                        ModellErrorStochastic=(stochastic_next_obs_numpy - np.concatenate([next_obs_numpy, reward_numpy], axis=-1)).sum(),
                        ModellMSEErrorStochastic=np.power(stochastic_next_obs_numpy - np.concatenate([next_obs_numpy, reward_numpy], axis=-1), 2).sum(),
                        ModellErrorDeterministic=(deterministic_next_obs_numpy - np.concatenate([next_obs_numpy, reward_numpy], axis=-1)).sum(),
                        ModellMSEErrorDeterministic=np.power(deterministic_next_obs_numpy - np.concatenate([next_obs_numpy, reward_numpy], axis=-1), 2).sum(),
                    )

                if False:
                    # Clear unreal buffer
                    unreal_replay_buffer.clear()
                    # Rollout unreal env
                    batch_size = min(max(replay_buffer.size, 25000), 100000)
                    obs, _, _, _, _ = replay_buffer.sample(batch_size)
                    obs = obs.detach().cpu().numpy()
                    for n in range(5):  # env._max_episode_steps):
                        action = policy.select_action_low_memory(obs)
                        # action = policy.actor(torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
                        # action += np.random.normal(0, max_action * args.expl_noise, size=action.shape)
                        action = action.clip(-max_action, max_action)
                        next_obs, confidence = unreal_env.predict(obs, action)
                        # Split observations
                        reward = next_obs[:, state_dim:]
                        next_obs = obs + next_obs[:, :state_dim]
                        # Handle outliers
                        inliers = np.all(np.where(confidence < 1, True, False), axis=1)

                        logger.store(
                            ConfMean=np.mean(confidence),
                            ConfMax=np.max(confidence),
                            ConfMin=np.min(confidence),
                            ConfStd=np.std(confidence),
                        )

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

                if args.model_based:
                    # policy.train_critic(unreal_replay_buffer, 256, logger)
                    policy.train_virtual(replay_buffer, 256, deterministic_unreal_env, logger)
                    # policy.train_actor(replay_buffer, 256, logger)

                policy.train(replay_buffer, 256, logger)

            # policy.train_actor(replay_buffer, 256, logger)

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
                    stochastic_unreal_env.save(os.path.join(log_dir, "models", file_name))

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
            # logger.log_tabular("CriticLossVirtual", with_min_and_max=True)
            logger.log_tabular("ConfMean", average_only=True)
            logger.log_tabular("ConfStd", average_only=True)
            logger.log_tabular("ConfMax", average_only=True)
            logger.log_tabular("ConfMin", average_only=True)
            logger.log_tabular("ConfMeanStochastic", average_only=True)
            logger.log_tabular("ConfStdStochastic", average_only=True)
            logger.log_tabular("ConfMaxStochastic", average_only=True)
            logger.log_tabular("ConfMinStochastic", average_only=True)
            logger.log_tabular("ConfMeanDeterministic", average_only=True)
            logger.log_tabular("ConfStdDeterministic", average_only=True)
            logger.log_tabular("ConfMaxDeterministic", average_only=True)
            logger.log_tabular("ConfMinDeterministic", average_only=True)
            logger.log_tabular("ModellErrorStochastic", average_only=True)
            logger.log_tabular("ModellMSEErrorStochastic", average_only=True)
            logger.log_tabular("ModellErrorDeterministic", average_only=True)
            logger.log_tabular("ModellMSEErrorDeterministic", average_only=True)
            logger.dump_tabular()


# Timesteps	Time	EpisodeNum	AverageEpisodeTimesteps	StdEpisodeTimesteps	MaxEpisodeTimesteps	MinEpisodeTimesteps	AverageEpisodeReward	StdEpisodeReward	MaxEpisodeReward	MinEpisodeReward	EpisodeEvalReward	AverageActorLoss	StdActorLoss	MaxActorLoss	MinActorLoss	AverageCriticLoss	StdCriticLoss	MaxCriticLoss	MinCriticLoss	ConfMean	ConfStd	ConfMax	ConfMin
# 4999	1.9844160079956055	233	21.433475	12.47783	95.0	8.0	16.384998	14.255733	132.62253	2.3288743	75.14705162213315	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
# 9999	475.9829981327057	323	53.02222	62.6952	385.0	11.0	94.20781	109.39464	578.5326	5.96232	1012.5141270323213	-98.10534	47.619747	-0.49295887	-149.65329	235.93549	279.31787	2784.9346	0.055417642	0.0033518025	0.0457268	1.2607515	4.540045e-05
# 14999	871.3562071323395	360	140.64865	144.3725	1000.0	42.0	287.43915	140.29736	1066.1262	72.02561	232.14808744384877	-170.70424	16.101175	-138.15327	-199.43457	300.5326	340.9952	3804.2417	36.624634	0.0039485963	0.06069321	1.55393	4.5399942e-05
# 19999	1237.293648481369	400	123.1	38.7155	181.0	28.0	309.35532	118.234924	511.73148	42.75262	503.9874207521606	-155.54266	10.097303	-132.71109	-177.76158	209.53844	207.9545	2443.3474	39.2362	0.001788562	0.031600177	1.1543292	4.5399935e-05
# 24999	1621.7517502307892	431	161.25807	42.90347	213.0	23.0	479.97702	149.4487	673.245	28.594542	686.2197074863642	-159.58847	7.840933	-140.14503	-180.8304	180.20387	190.24559	2721.212	34.83235	0.003278135	0.051692136	1.5067211	4.5399935e-05
# 29999	2065.627806186676	457	192.92308	100.97788	577.0	100.0	563.7235	377.26065	1944.7163	233.19089	606.1759621917815	-180.2905	3.6866693	-164.8422	-193.07138	205.96748	290.78336	4943.7207	40.44147	0.0027875518	0.046668485	1.4812849	4.5399935e-05
# 34999	2536.985804796219	473	284.0625	109.13837	615.0	120.0	934.504	377.28235	2056.8467	286.03107	3161.706522922529	-190.51385	7.5394087	-171.40508	-213.69531	235.69923	335.01285	4343.125	42.999607	0.0019924622	0.038407378	1.380884	4.5399935e-05
# 39999	3027.0080423355103	482	591.2222	282.33817	1000.0	296.0	1820.6174	810.1122	3186.152	1003.46075	3109.381760251229	-212.8069	6.157477	-192.36877	-228.65947	242.75114	363.47726	5609.966	44.291695	0.0015832947	0.034242257	1.3194714	4.5399935e-05
# 44999	4416.471719741821	489	713.8571	264.904	1000.0	297.0	2393.041	861.50745	3342.449	1004.1315	3156.425887157816	-225.37248	5.1053314	-206.35689	-244.03401	224.32938	331.23563	4745.458	44.327293	0.002179626	0.035727724	1.2513363	4.5399935e-05
# 49999	6642.887886762619	501	427.33334	164.44925	744.0	150.0	1419.7086	574.8626	2507.608	416.0803	863.7403663981952	-236.79456	4.8403263	-218.22313	-251.06285	230.9114	336.89398	5238.481	41.82921	1.2071625	0.09236429	1.6254371	1.0991524
# 54999	8391.23149061203	512	440.27274	248.64438	1000.0	84.0	1467.5375	872.7078	3434.5425	182.97842	2963.6213543137233	-237.86047	4.2950864	-222.99834	-252.65218	205.96492	315.76822	4454.5273	39.77314	1.648767	0.0	1.648767	1.648767
# 59999	10175.269238471985	520	554.75	231.02205	1000.0	334.0	1839.2607	738.38763	3254.2085	1134.3594	2212.6602001480446	-241.24358	4.7766366	-227.35869	-259.9271	197.68687	325.9333	5537.5337	36.591133	1.648767	0.0	1.648767	1.648767

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Save path
save_path_images = "./results/images/"
save_path_pdfs = "./results/pdfs/"

# Colors
nice_colors = [
    [0, 0.4470, 0.7410],
    [0.8500, 0.3250, 0.0980],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.3010, 0.7450, 0.9330],
    [0.6350, 0.0780, 0.1840],
    [1, 0, 0],
    [1, 0.5333, 0],
    [0, 0.7333, 0],
    [0.3333, 0, 1],
    [0.6667, 0, 0],
]


# Moving average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# Get all directory names
# Split in envs -> plots -> algos -> experiments
experiments = {}
for env_name in os.listdir("../logs"):
    tmp_dir = os.path.join("../logs", env_name)
    if os.path.isdir(tmp_dir):
        for plots in os.listdir(tmp_dir):
            tmp_dir = os.path.join("../logs", env_name, plots)
            if os.path.isdir(tmp_dir):
                for algo_name in os.listdir(tmp_dir):
                    tmp_dir = os.path.join("../logs", env_name, plots, algo_name)
                    if os.path.isdir(tmp_dir):
                        exp = []
                        for exp_name in os.listdir(tmp_dir):
                            tmp_dir = os.path.join("../logs", env_name, plots, algo_name, exp_name)
                            if os.path.isdir(tmp_dir):
                                exp.append(os.path.join("../logs", env_name, plots, algo_name, exp_name, "log.txt"))
                        if len(exp) > 0:
                            try:
                                experiments[env_name][plots][algo_name] = exp
                            except KeyError:
                                try:
                                    experiments[env_name][plots] = {algo_name: exp}
                                except KeyError:
                                    experiments[env_name] = {plots: {algo_name: exp}}

# Iterate over experiments and plot
for env_name in experiments:
    for plot in experiments[env_name]:

        if any(env in env_name for env in ["Fetch", "ShadowHand"]):
            ##############################
            # SuccessRate
            ##############################
            plot_name = f"{plot} - {env_name} - SuccessRate"
            plt.figure(plot_name)
            for n, algo_name in enumerate(experiments[env_name][plot]):

                data = pd.read_csv(experiments[env_name][plot][algo_name][0], "\t")
                epoch = data["Epoch"].values
                success = np.zeros((len(epoch), len(experiments[env_name][plot][algo_name])))

                for k, exp_name in enumerate(experiments[env_name][plot][algo_name]):
                    data = pd.read_csv(exp_name, "\t")
                    try:
                        success[:, k] = data["SuccessRate"].values
                    except KeyError:
                        success[:, k] = data["AverageSuccessRate"].values

                if "ShadowHand" in env_name:
                    WINDOW_SIZE = 5
                elif "Fetch" in env_name:
                    WINDOW_SIZE = 3
                else:
                    WINDOW_SIZE = 1

                success_median = moving_average(np.median(success, axis=-1), WINDOW_SIZE)
                success_q75, success_q25 = np.percentile(success, [75, 25], axis=-1)
                success_min = moving_average(success_q25, WINDOW_SIZE)
                success_max = moving_average(success_q75, WINDOW_SIZE)
                # success_min = moving_average(np.min(success, axis=-1), WINDOW_SIZE)
                # success_max = moving_average(np.max(success, axis=-1), WINDOW_SIZE)
                epoch = epoch[len(epoch) - len(success_median):]

                plt.plot(epoch, success_median, color=nice_colors[n], label=algo_name)
                plt.fill_between(epoch, success_min, success_max, color=nice_colors[n], alpha=0.3)

            plt.title(env_name)
            plt.xlabel("Epoch")
            plt.ylabel("Median Success Rate")
            plt.ylim([-0.03, 1.03])
            plt.legend()
            plt.savefig(os.path.join(save_path_images, f"{plot_name}.png"))
            plt.savefig(os.path.join(save_path_pdfs, f"{plot_name}.pdf"), format="pdf")
            plt.close()

            ##############################
            # WorldBufferSize
            ##############################
            plot_name = f"{plot} - {env_name} - WorldBufferSize"
            plt.figure(plot_name)
            for n, algo_name in enumerate(experiments[env_name][plot]):

                if "DWM" not in algo_name and "SWM" not in algo_name:
                    continue

                data = pd.read_csv(experiments[env_name][plot][algo_name][0], "\t")
                epoch = data["Epoch"].values
                buffer_size = np.zeros((len(epoch), len(experiments[env_name][plot][algo_name])))

                for k, exp_name in enumerate(experiments[env_name][plot][algo_name]):
                    data = pd.read_csv(exp_name, "\t")
                    try:
                        buffer_size[:, k] = data["WorldModelReplayBufferSize"].values
                    except KeyError:
                        buffer_size[:, k] = data["AverageWorldModelReplayBufferSize"].values

                if "ShadowHand" in env_name:
                    WINDOW_SIZE = 5
                elif "Fetch" in env_name:
                    WINDOW_SIZE = 3
                else:
                    WINDOW_SIZE = 1

                buffer_size_full = np.array([250000 + 10000 * 5 * m for m in range(len(epoch))])
                buffer_size_median = moving_average(np.median(buffer_size, axis=-1), WINDOW_SIZE)
                buffer_size_min = moving_average(np.min(buffer_size, axis=-1), WINDOW_SIZE)
                buffer_size_max = moving_average(np.max(buffer_size, axis=-1), WINDOW_SIZE)
                epoch = epoch[len(epoch) - len(buffer_size_median):]

                buffer_size_full = buffer_size_full[WINDOW_SIZE - 1:]
                buffer_size_median = buffer_size_median / buffer_size_full
                buffer_size_min = buffer_size_min / buffer_size_full
                buffer_size_max = buffer_size_max / buffer_size_full

                plt.plot(epoch, buffer_size_median, color=nice_colors[n], label=algo_name)
                plt.fill_between(epoch, buffer_size_min, buffer_size_max, color=nice_colors[n], alpha=0.3)

            plt.title(env_name)
            plt.xlabel("Epoch")
            plt.ylabel("Median World Buffer Size")
            plt.ylim([-0.03, 1.03])
            plt.legend()
            plt.savefig(os.path.join(save_path_images, f"{plot_name}.png"))
            plt.savefig(os.path.join(save_path_pdfs, f"{plot_name}.pdf"), format="pdf")
            plt.close()
        else:
            ##############################
            # Reward
            ##############################
            plot_name = f"{plot} - {env_name} - Reward"
            plt.figure(plot_name)
            for n, algo_name in enumerate(experiments[env_name][plot]):

                data = pd.read_csv(experiments[env_name][plot][algo_name][0], "\t")
                timesteps = data["Timesteps"].values
                reward = np.zeros((len(timesteps), len(experiments[env_name][plot][algo_name])))

                for k, exp_name in enumerate(experiments[env_name][plot][algo_name]):
                    data = pd.read_csv(exp_name, "\t")
                    try:
                        reward[:, k] = data["Reward"].values
                    except KeyError:
                        reward[:, k] = data["AverageReward"].values

                WINDOW_SIZE = 5

                reward_mean = moving_average(np.mean(reward, axis=-1), WINDOW_SIZE)
                reward_q75, reward_q25 = np.percentile(reward, [75, 25], axis=-1)
                reward_min = moving_average(reward_q25, WINDOW_SIZE)
                reward_max = moving_average(reward_q75, WINDOW_SIZE)
                # reward_min = moving_average(np.min(reward, axis=-1), WINDOW_SIZE)
                # reward_max = moving_average(np.max(reward, axis=-1), WINDOW_SIZE)
                timesteps = timesteps[len(timesteps) - len(reward_mean):]

                plt.plot(timesteps, reward_mean, color=nice_colors[n], label=algo_name)
                plt.fill_between(timesteps, reward_min, reward_max, color=nice_colors[n], alpha=0.3)

            plt.title(env_name)
            plt.xlabel("Timesteps")
            plt.ylabel("Average Reward")
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.xlim(left=0)
            # plt.ylim(bottom=0)
            plt.legend()
            plt.savefig(os.path.join(save_path_images, f"{plot_name}.png"))
            plt.savefig(os.path.join(save_path_pdfs, f"{plot_name}.pdf"), format="pdf")
            plt.close()

            ##############################
            # WorldBufferSize
            ##############################
            plot_name = f"{plot} - {env_name} - WorldBufferSize"
            plt.figure(plot_name)
            for n, algo_name in enumerate(experiments[env_name][plot]):

                if "DWM" not in algo_name and "SWM" not in algo_name:
                    continue

                data = pd.read_csv(experiments[env_name][plot][algo_name][0], "\t")
                timesteps = data["Timesteps"].values
                buffer_size = np.zeros((len(timesteps), len(experiments[env_name][plot][algo_name])))

                for k, exp_name in enumerate(experiments[env_name][plot][algo_name]):
                    data = pd.read_csv(exp_name, "\t")
                    try:
                        buffer_size[:, k] = data["WorldModelReplayBufferSize"].values
                    except KeyError:
                        buffer_size[:, k] = data["AverageWorldModelReplayBufferSize"].values

                if "ShadowHand" in env_name:
                    WINDOW_SIZE = 5
                elif "Fetch" in env_name:
                    WINDOW_SIZE = 3
                else:
                    WINDOW_SIZE = 1

                buffer_size_full = np.array([250000 + 10000 * 5 * m for m in range(len(timesteps))])
                buffer_size_median = moving_average(np.median(buffer_size, axis=-1), WINDOW_SIZE)
                buffer_size_min = moving_average(np.min(buffer_size, axis=-1), WINDOW_SIZE)
                buffer_size_max = moving_average(np.max(buffer_size, axis=-1), WINDOW_SIZE)
                timesteps = timesteps[len(timesteps) - len(buffer_size_median):]

                buffer_size_full = buffer_size_full[WINDOW_SIZE - 1:]
                buffer_size_median = buffer_size_median / buffer_size_full
                buffer_size_min = buffer_size_min / buffer_size_full
                buffer_size_max = buffer_size_max / buffer_size_full

                plt.plot(timesteps, buffer_size_median, color=nice_colors[n], label=algo_name)
                plt.fill_between(timesteps, buffer_size_min, buffer_size_max, color=nice_colors[n], alpha=0.3)

            plt.title(env_name)
            plt.xlabel("Epoch")
            plt.ylabel("Median World Buffer Size")
            plt.ylim([-0.03, 1.03])
            plt.legend()
            plt.savefig(os.path.join(save_path_images, f"{plot_name}.png"))
            plt.savefig(os.path.join(save_path_pdfs, f"{plot_name}.pdf"), format="pdf")
            plt.close()

# plt.show()

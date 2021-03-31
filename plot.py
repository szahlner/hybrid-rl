import os
import pandas as pd
import matplotlib.pyplot as plt

from hybridrl.utils.logger import Logger


env_id = 'ShadowHandReach-v0'
agents = ['ddpg', 'rs_mpc', 'cem']

data = []
for agent in agents:
    log_dir = os.path.join('runs', agent, env_id)
    logger = Logger(log_dir)
    logger.load()
    data.append(logger.get())
df = pd.DataFrame(data)

with plt.style.context('ggplot'):
    color = [(0., 0.4470, 0.7410, 1.),
             (0.8500, 0.3250, 0.0980, 1.),
             (0.4660, 0.6740, 0.1880, 1.)]

    x_label = 'Epoch'
    legend = [x.upper().replace('_', '') for x in agents]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(env_id, fontsize=16)

    for n in range(len(agents)):
        axes[0][0].plot(df['epoch'][n], df['current_reward'][n], color=color[n])
        axes[0][0].set_xlabel(x_label, color="k")
        axes[0][0].set_ylabel('Running Reward', color="k")
        axes[0][0].tick_params(axis="both", colors="k")
        axes[0][0].legend(legend, loc="lower right")

        axes[0][1].plot(df['epoch'][n], df['avg_reward'][n], color=color[n])
        axes[0][1].set_xlabel(x_label, color="k")
        axes[0][1].set_ylabel('Average Reward (100)', color="k")
        axes[0][1].tick_params(axis="both", colors="k")
        axes[0][1].legend(legend, loc="lower right")

        axes[1][0].plot(df['epoch'][n], df['avg_test_reward'][n], color=color[n])
        axes[1][0].set_xlabel(x_label, color="k")
        axes[1][0].set_ylabel('Average Test Reward (10)', color="k")
        axes[1][0].tick_params(axis="both", colors="k")
        axes[1][0].legend(legend, loc="lower right")

    fig.delaxes(axes[1][1])

    plt.show()

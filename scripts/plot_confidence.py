import argparse
import os
import numpy as np
import seaborn

from matplotlib import pyplot as plt


parser = argparse.ArgumentParser("Gather results, plot them and create table")
parser.add_argument("-a", "--algos", help="Algorithms to include", nargs="+", type=str)
parser.add_argument("-e", "--env", help="Environments to include", nargs="+", type=str)
parser.add_argument("-f", "--exp-folders", help="Folders to include", nargs="+", type=str)
parser.add_argument("-l", "--labels", help="Label for each folder", nargs="+", type=str)
parser.add_argument("--no-million", action="store_true", default=False, help="Do not convert x-axis to million")
parser.add_argument("-c", "--cols", help="Number of columns", type=int, default=6)
args = parser.parse_args()

# Activate seaborn
seaborn.set()

args.algos = [algo.upper() for algo in args.algos]

if args.labels is None:
    args.labels = args.exp_folders

for env in args.env:
    for algo in args.algos:
        for folder_idx, exp_folder in enumerate(args.exp_folders):
            log_path = os.path.join(exp_folder, algo.lower())

            if not os.path.isdir(log_path):
                continue

            dirs = [
                os.path.join(log_path, d)
                for d in os.listdir(log_path)
                if (env in d and os.path.isdir(os.path.join(log_path, d)))
            ]

            for _, dir_ in enumerate(dirs):
                try:
                    log = np.load(os.path.join(dir_, "dynamics.npz"), allow_pickle=True)
                except FileNotFoundError:
                    print("Dynamics not found for", dir_)
                    continue

            x_label_suffix = "" if args.no_million else "(in Million)"

            x = log['x'].flatten()
            error_mean = log['error_mean']

            if not args.no_million:
                x = x / 1e6

            n_state = error_mean.shape[1]
            n_cols = args.cols
            n_rows = np.ceil(n_state / n_cols).astype('int')

            # Error
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True)
            y_label = 'Error-Mean'
            fig.canvas.set_window_title(f"{y_label} {env}")
            fig.suptitle(f"{env}", fontsize=14)
            fig.text(0.5, 0.04, f"Timesteps {x_label_suffix}", va='center', ha='center', fontsize=14)
            fig.text(0.04, 0.5, f"Obs-Dims {y_label.lower()}", va='center', ha='center', rotation='vertical', fontsize=14)

            n = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    axs[r, c].plot(x, error_mean[:, n])

                    n += 1
                    if n >= n_state:
                        break

            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True)
            y_label = 'Error-Mse'
            fig.canvas.set_window_title(f"{y_label} {env}")
            fig.suptitle(f"{env}", fontsize=14)
            fig.text(0.5, 0.04, f"Timesteps {x_label_suffix}", va='center', ha='center', fontsize=14)
            fig.text(0.04, 0.5, f"Obs-Dims {y_label.lower()}", va='center', ha='center', rotation='vertical', fontsize=14)

            error_mse = log['error_mse']

            n = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    axs[r, c].plot(x, error_mse[:, n])

                    n += 1
                    if n >= n_state:
                        break

            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True)
            y_label = 'Error-Min-Max'
            fig.canvas.set_window_title(f"{y_label} {env}")
            fig.suptitle(f"{env}", fontsize=14)
            fig.text(0.5, 0.04, f"Timesteps {x_label_suffix}", va='center', ha='center', fontsize=14)
            fig.text(0.04, 0.5, f"Obs-Dims {y_label.lower()}", va='center', ha='center', rotation='vertical', fontsize=14)

            error_min_error, error_max_error = log['error_min_error'], log['error_max_error']

            n = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    color = next(axs[r, c]._get_lines.prop_cycler)['color']
                    axs[r, c].plot(x, error_min_error[:, n], color=color)
                    axs[r, c].plot(x, error_max_error[:, n], color=color)
                    axs[r, c].fill_between(x, error_max_error[:, n], error_min_error[:, n], color=color, alpha=0.3)

                    n += 1
                    if n >= n_state:
                        break

            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True)
            y_label = 'Error-Min-Max-Confidence'
            fig.canvas.set_window_title(f"{y_label} {env}")
            fig.suptitle(f"{env}", fontsize=14)
            fig.text(0.5, 0.04, f"Timesteps {x_label_suffix}", va='center', ha='center', fontsize=14)
            fig.text(0.04, 0.5, f"Obs-Dims {y_label.lower()}", va='center', ha='center', rotation='vertical', fontsize=14)

            error_min_confidence, error_max_confidence = log['error_min_confidence'], log['error_max_confidence']

            n = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    color = next(axs[r, c]._get_lines.prop_cycler)['color']
                    axs[r, c].plot(x, error_min_confidence[:, n], color=color)
                    axs[r, c].plot(x, error_max_confidence[:, n], color=color)
                    axs[r, c].fill_between(x, error_max_confidence[:, n], error_min_confidence[:, n], color=color, alpha=0.3)

                    n += 1
                    if n >= n_state:
                        break

plt.show()

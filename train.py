import argparse
import os
import yaml

from hybridrl.experiments import Experiment


def train():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--experiment-params', help='File for experiment parameters', type=str, required=True)
    args = parser.parse_args()

    # load experiment params
    with open(args.experiment_params, 'r') as experiment_params:
        experiment_params = yaml.safe_load(experiment_params)
    experiment_params['experiment']['log_dir'] = os.path.dirname(args.experiment_params)

    # initialize experiment
    experiment = Experiment(experiment_params)
    experiment.start()


if __name__ == '__main__':
    train()

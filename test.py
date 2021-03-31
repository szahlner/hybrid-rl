import argparse
import os
import yaml

from hybridrl.experiments import Experiment


def test():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--experiment-params', help='File for experiment parameters', type=str, required=True)
    args = parser.parse_args()

    # load experiment params
    with open(args.experiment_params, 'r') as experiment_params:
        experiment_params = yaml.safe_load(experiment_params)
    experiment_params['experiment']['log_dir'] = os.path.dirname(args.experiment_params)
    experiment_params['test']['is_test'] = True

    # initialize experiment
    experiment = Experiment(experiment_params)
    experiment.test()


if __name__ == '__main__':
    test()

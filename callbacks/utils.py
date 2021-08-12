import os
import yaml

from collections import OrderedDict
from pprint import pprint
from typing import Any, Dict


def read_hyperparameters(algo: str,
                         env_id: str,
                         verbose: int = 2) -> Dict[str, Any]:

    # Load hyperparameters from yaml file
    hyperparams_path = os.path.join('hyperparams', 'dynamics', '{}.yml'.format(algo))
    with open(hyperparams_path, 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        else:
            raise ValueError(f"Hyperparameters not found for {algo}-{env_id} dynamics model")

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    if verbose > 0:
        print("Default hyperparameters for dynamics model:")
        pprint(saved_hyperparams)

    return hyperparams


def save_hyperparameters(path: str,
                         env_id: str,
                         params: dict,
                         verbose: int = 2) -> None:
    path = os.path.join(path, env_id, 'dynamics.yml')
    with open(path, 'w') as f:
        ordered_args = OrderedDict(params)
        yaml.dump(ordered_args, f)

    if verbose > 0:
        print('Saved dynamics parameters to: {}'.format(path))


def get_log_path(path, algo, env_id):
    path = os.path.join(path, algo)

    pot_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    pot_dirs = [d for d in pot_dirs if env_id in d]
    latest_id = max([int(d.split('_')[1]) for d in pot_dirs])

    log_path = os.path.join(path, '{}_{}'.format(env_id, latest_id))

    return log_path

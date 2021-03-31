import os
import sys
import numpy as np
import pandas as pd

from termcolor import colored
from colorama import init as colorama_init


if os.name == 'nt':
    colorama_init()


class Logger:
    def __init__(self, log_dir, n_print_header=25, data_schema=None):
        assert os.path.exists(log_dir), 'log dir does not exist'
        self.log_dir = log_dir
        self.n_print_header = n_print_header

        if data_schema is None:
            self.data = None
        else:
            self.data = dict.fromkeys(data_schema)

            for key in self.data.keys():
                self.data[key] = []

        self.size = 0

    def add(self, data):
        if self.data is None:
            self.data = dict.fromkeys(data)

            for key in self.data.keys():
                self.data[key] = []

        assert self.data.keys() == data.keys(), 'dictionaries must have the same keys'

        for key in data.keys():
            self.data[key].append(data[key])

        df = pd.DataFrame(data, index=[0]).round(3)
        if self.size % self.n_print_header == 0:
            df = df.to_string(index=False)
            df = df.split('\n')
            header = df[0]
            data = df[1]
            print(colored('{}'.format(header), 'cyan'))
            print(data)
        else:
            df = df.to_string(index=False)
            df = df.split('\n')[1]
            print(df)
        sys.stdout.flush()

        self.size += 1

    def save(self):
        np.savez_compressed(os.path.join(self.log_dir, 'log.npz'), data=self.data)

    def load(self):
        data = np.load(os.path.join(self.log_dir, 'log.npz'), allow_pickle=True)
        self.data = data['data'].item()

    def get(self):
        return self.data

import numpy as np
import torch
import threading

from mpi4py import MPI


def sync_networks(network):
    flat_params = _get_flat_params_or_grads(network, mode='params')
    MPI.COMM_WORLD.Bcast(flat_params, 0)
    _set_flat_params_or_grads(network, flat_params, mode='params')


def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    global_grads = np.zeros_like(flat_grads)
    MPI.COMM_WORLD.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    global_grads /= MPI.COMM_WORLD.Get_size()
    _set_flat_params_or_grads(network, global_grads, mode='grads')


def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        # local information
        self.local_sum = np.zeros(size, dtype=np.float32)
        self.local_sumsq = np.zeros(size, dtype=np.float32)
        self.local_count = np.zeros(1, dtype=np.float32)

        # total information
        self.total_sum = np.zeros(size, dtype=np.float32)
        self.total_sumsq = np.zeros(size, dtype=np.float32)
        self.total_count = np.ones(1, dtype=np.float32)

        # mean and std
        self.mean = np.zeros(size, dtype=np.float32)
        self.std = np.ones(size, np.float32)

        # thread locker
        self.lock = threading.Lock()

    def update(self, value):
        # update the parameters of the normalizer
        value = value.reshape(-1, self.size)

        # compute
        with self.lock:
            self.local_sum += value.sum(axis=0)
            self.local_sumsq += np.square(value).sum(axis=0)
            self.local_count[0] += len(value)

    def sync(self, local_sum, local_sumsq, local_count):
        # sync parameters across the cpus
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)

        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()

            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0

        # sync stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)

        # update total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count

        # calculate new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))

    def _mpi_average(self, x):
        # average across the cpus data
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()

        return buf

    def normalize(self, value, clip_range=None):
        #return value
        if clip_range is None:
            clip_range = self.default_clip_range

        return np.clip((value - self.mean) / self.std, -clip_range, clip_range)

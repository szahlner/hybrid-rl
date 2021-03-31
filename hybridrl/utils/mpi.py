import numpy as np
import torch

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

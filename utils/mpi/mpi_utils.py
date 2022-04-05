from mpi4py import MPI
import numpy as np
import torch


def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean) ** 2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std


def sync_networks(network):
    """
    Sync networks across the different cores.

    Args:
        network: The network to be synced.
    """
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params')


def sync_grads(network):
    """
    Sync gradients across the different cores.

    Args:
        network: The network to be synced.
    """
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_params_or_grads(network, global_grads, mode='grads')


def sync_bool_and(bool_var):
    """
    Boolean AND operation across different cores.

    Args:
        bool_var: The boolean variable to be synced.
    """
    local_bool_var = np.array([bool_var], dtype=np.int32)
    comm = MPI.COMM_WORLD
    global_bool_var = np.zeros_like(local_bool_var)
    comm.Allreduce(local_bool_var, global_bool_var, op=MPI.SUM)
    return global_bool_var == comm.Get_size()


def sync_bool_or(bool_var):
    """
    Boolean OR operation across different cores.

    Args:
        bool_var: The boolean variable to be synced.
    """
    local_bool_var = np.array([bool_var], dtype=np.int32)
    comm = MPI.COMM_WORLD
    global_bool_var = np.zeros_like(local_bool_var)
    comm.Allreduce(local_bool_var, global_bool_var, op=MPI.SUM)
    return global_bool_var > 0


def _get_flat_params_or_grads(network, mode='params'):
    """
    Get the flattened gradients or parameters from the network.

    Args:
        network: The network to be operated on.
        mode: Chose 'grad' or 'params'. Defaults to 'params'.
    """
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    Set the flattened gradients or parameters of the network.

    Args:
        network: The network to be operated on.
        flat_params: The flattened gradients or parameters to be set.
        mode: Chose 'grad' or 'params'. Defaults to 'params'.
    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()

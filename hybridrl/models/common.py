import numpy as np
import torch


def fanin_init(size, fanin=None):
    """Weight initializer"""

    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

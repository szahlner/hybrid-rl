import numpy as np
import torch
import torch.nn as nn

from typing import Union


class StandardScaler:
    def __init__(self) -> None:
        self.mu = None
        self.std = None

    def fit(self, data: torch.Tensor) -> None:
        self.mu = torch.mean(data, dim=0, keepdim=True)
        self.std = torch.std(data, dim=0, keepdim=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mu) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std + self.mu


class EnsembleLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_ensemble: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_ensemble = n_ensemble
        self.weight_decay = weight_decay
        self.weight = nn.Parameter(torch.Tensor(n_ensemble, in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_ensemble, out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])


def init_weights(
    layer: Union[
        nn.Linear,
        EnsembleLinear,
    ]
) -> None:
    def truncated_normal_init(
        t: torch.Tensor, mean: float = 0.0, std: float = 0.01
    ) -> torch.Tensor:
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t
            )
        return t

    if type(layer) == nn.Linear or isinstance(layer, EnsembleLinear):
        input_dim = layer.in_features
        truncated_normal_init(layer.weight, std=1 / (2 * np.sqrt(input_dim)))
        layer.bias.data.fill_(0.0)

from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from utils.mpi.mpi_utils import sync_grads, sync_networks, sync_bool_and


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StandardScaler:
    def __init__(self) -> None:
        self.mu = None
        self.std = None

    def fit(self, data: np.ndarray) -> None:
        """
        Calculate mean and standard deviation of the given data.

        Args:
            data (np.ndarray): Data to calculate mean and standard deviation from.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the input array using the internal mean and standard deviation to mean = 0, std = 1.
        Args:
            data (np.ndarray):  Data to be transformed.

        Returns:
            np.ndarray: Transformed data.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """
        Undoes the transformation performed by the scaler.

        Args:
            data (np.ndarray):  Data to be re-transformed.

        Returns:
            np.ndarray: Re-transformed data.
        """
        return self.std * data + self.mu


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True
    ) -> None:
        """
        Ensemble fully connected layer.

        Args:
            in_features (int): Amount of input features / input dimension.
            out_features (int): Amount of output features / output dimension.
            ensemble_size (int): Size of the ensemble.
            weight_decay (float): Amount of weight decay to use. Defaults to 0.0.
            bias (bool): Whether to use bias or not. Defaults to True.
        """
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay

        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    @staticmethod
    def forward(x):
        x = x * torch.sigmoid(x)
        return x


class StochasticWorld(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        network_dim: int,
        hidden_dim: Tuple,
        hidden_activation: Any = Swish(),
    ) -> None:
        super(StochasticWorld, self).__init__()

        self.network_dim = network_dim
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim

        self.net = nn.ModuleList()
        in_dim = input_dim

        # Construct network
        for n, out_dim in enumerate(hidden_dim):
            layer = EnsembleFC(in_dim, out_dim, network_dim)
            in_dim = out_dim
            self.net.append(layer)
        self.last_layer = EnsembleFC(in_dim, 2 * self.output_dim, network_dim)
        self.apply(init_weights)

        # self.max_log_var = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        # self.min_log_var = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)

        self.max_log_var = 0.5 * torch.ones((1, self.output_dim), dtype=torch.float, device=device)
        self.min_log_var = -10 * torch.ones((1, self.output_dim), dtype=torch.float, device=device)

    def forward(self, state, ret_log_var=False):
        x = state.repeat([self.network_dim, 1, 1])
        for n, layer in enumerate(self.net):
            x = layer(x)
            x = self.hidden_activation(x)
        out = self.last_layer(x)

        mean = out[:, :, :self.output_dim]

        log_var = self.max_log_var - F.softplus(self.max_log_var - out[:, :, self.output_dim:])
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)

        if ret_log_var:
            return mean, log_var

        return mean, torch.exp(log_var)

    def loss(self, mean, log_var, labels, inc_var_loss=True):
        """
        mean, log_var: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(log_var.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(log_var, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
            total_loss += 0.01 * torch.sum(self.max_log_var) - 0.01 * torch.sum(self.min_log_var)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss


class StochasticWorldModel:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        network_dim: int = 10,
        hidden_dim: Tuple = (200, 200, 200, 200, 200),
        lr: float = 3e-4,
        hidden_activation: Any = Swish(),
    ) -> None:
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_dim = network_dim

        # World model
        self.world_model = StochasticWorld(
            input_dim=input_dim,
            output_dim=output_dim,
            network_dim=network_dim,
            hidden_dim=hidden_dim,
            hidden_activation=hidden_activation,
        ).to(device)

        # Sync
        sync_networks(self.world_model)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)

        # Normalizer / scaler
        self.scaler = StandardScaler()
        self.scaler.mu = np.zeros(input_dim)
        self.scaler.std = np.ones(input_dim)

        self._max_epochs_since_update = None
        self._epochs_since_update = None
        self._state = None
        self._snapshots = None

    def train(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 256,
        train_test_ratio: float = 0.2,
        max_epochs_since_update: int = 5,
    ) -> None:
        # Set internal stuff
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {n: (None, 1e10) for n in range(self.network_dim)}

        # Shuffle + train / test split
        num_test = int(inputs.shape[0] * train_test_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_test:], labels[num_test:]
        test_inputs, test_labels = inputs[:num_test], labels[:num_test]

        # Scale
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        test_inputs = self.scaler.transform(test_inputs)

        # To tensor
        test_inputs = torch.tensor(test_inputs, dtype=torch.float, device=device)
        test_labels = torch.tensor(test_labels, dtype=torch.float, device=device)

        for epoch in itertools.count():
            train_idx = np.random.permutation(train_inputs.shape[0])
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[start_pos: start_pos + batch_size]

                # To tensor + shuffle
                train_input = torch.tensor(train_inputs[idx], dtype=torch.float, device=device)
                train_label = torch.tensor(train_labels[idx], dtype=torch.float, device=device)

                mean, log_var = self.world_model(train_input, ret_log_var=True)
                loss, _ = self.world_model.loss(mean, log_var, train_label.repeat([self.network_dim, 1, 1]))

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                # sync_grads(self.world_model)
                self.optimizer.step()

            with torch.no_grad():
                test_mean, test_log_var = self.world_model(test_inputs, ret_log_var=True)
                _, test_mse_losses = self.world_model.loss(test_mean, test_log_var, test_labels.repeat([self.network_dim, 1, 1]), inc_var_loss=False)
                test_mse_losses = test_mse_losses.detach().cpu().numpy()

                break_train = self._save_best(epoch, test_mse_losses)
                break_train = sync_bool_and(break_train)

                if break_train:
                    break

    def _save_best(self, epoch: int, test_losses: np.ndarray) -> bool:
        """
        Saves best training epoch of current trainings-run.

        Args:
            epoch (int): Current epoch.
            test_losses (np.ndarray): Test losses from the current training-run.

        Returns:
            bool: Whether the conditions to break the current trainings-run are met.
        """
        updated = False

        for n in range(len(test_losses)):
            current = test_losses[n]
            _, best = self._snapshots[n]
            improvement = (best - current) / best

            if improvement > 0.01:
                self._snapshots[n] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(
        self,
        inputs: np.ndarray,
        batch_size: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the next observation and the confidence from the given input.

        Args:
            inputs (np.ndarray): Input to predict the next observation from.
            batch_size (int): Batch size / chunk size to work with.

        Returns:
            Tuple[np.ndarray, ndarray]: Prediction, Confidence.
        """
        # Normalize / scale inputs
        inputs = self.scaler.transform(inputs)

        # Eval mode for deterministic prediction
        prediction_mean = np.empty((self.network_dim, len(inputs), self.output_dim))
        prediction_var = np.empty((self.network_dim, len(inputs), self.output_dim))
        for start_pos in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[start_pos: start_pos + batch_size]).float().to(device)
            mean, var = self.world_model(input, ret_log_var=False)
            prediction_mean[:, start_pos: start_pos + batch_size] = mean.detach().cpu().numpy()
            prediction_var[:, start_pos: start_pos + batch_size] = var.detach().cpu().numpy()
        prediction_mean = np.median(prediction_mean, axis=0)
        prediction_var = np.median(prediction_var, axis=0)

        return prediction_mean, prediction_var

    def save(self, filename: str) -> None:
        """
        Save the stochastic world model.

        Args:
            filename (str): Filename to save the model to.
        """
        torch.save(
            [
                self.scaler.mu,
                self.scaler.std,
                self.world_model.state_dict(),
            ],
            filename
        )

    def load(self, filename: str) -> None:
        """
        Load the stochastic world model.

        Args:
            filename (str): Filename to load the model from.
        """
        mu, std, world_model = torch.load(filename, map_location=lambda storage, loc: storage)

        self.world_model.load_state_dict(world_model)
        self.scaler.mu = mu
        self.scaler.std = std

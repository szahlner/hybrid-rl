from typing import Any, Tuple
import torch
import torch.nn as nn
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


class DeterministicWorld(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        network_dim: int,
        hidden_dim: Tuple,
        dr: float = 0.05,
        hidden_activation: Any = torch.relu,
    ) -> None:
        """
        Deterministic World (MLP).

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            network_dim (int): Network dimension / ensemble size.
            hidden_dim (Tuple): Hidden dimensions, used in the inner fully connected layers.
            dr (float): Dropout rate. Defaults to 0.05.
            hidden_activation (Any): Activation function of the hidden layers. Defaults to RelU.
        """
        super(DeterministicWorld, self).__init__()

        self.network_dim = network_dim
        self.hidden_activation = hidden_activation

        self.net = nn.ModuleList()
        in_dim = input_dim

        # Construct network
        for n, out_dim in enumerate(hidden_dim):
            layer = EnsembleFC(in_dim, out_dim, network_dim)
            in_dim = out_dim
            self.net.append(layer)
            self.net.append(nn.Dropout(dr))
        self.last_layer = EnsembleFC(in_dim, output_dim, network_dim)
        self.apply(init_weights)

    def forward(self, state):
        x = state.repeat([self.network_dim, 1, 1])
        for n, layer in enumerate(self.net):
            x = layer(x)
            x = self.hidden_activation(x)
        out = self.last_layer(x)
        return out


class DeterministicWorldModel:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        network_dim: int = 10,
        hidden_dim: Tuple = (200, 200, 200, 200, 200),
        lr: float = 3e-4,
        dr: float = 0.05,
        hidden_activation: Any = torch.relu,
        num_mc: int = 10,
    ) -> None:
        """
        Deterministic World Model (MLP).

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            network_dim (int): Network dimension / ensemble size.
            hidden_dim (Tuple): Hidden dimensions, used in the inner fully connected layers. Defaults to (200, 200, 200, 200, 200).
            lr (float): Learning rate. Defaults to 3e-4.
            dr (float): Dropout rate. Defaults to 0.05.
            hidden_activation (Any): Activation function of the hidden layers. Defaults to RelU.
            num_mc (int): Amount of runs to make for dropout mc in the predict method. Defaults to 10.
        """
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_dim = network_dim

        # World model
        self.world_model = DeterministicWorld(
            input_dim=input_dim,
            output_dim=output_dim,
            network_dim=network_dim,
            hidden_dim=hidden_dim,
            dr=dr,
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

        self.num_mc = num_mc
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
        """
        Train the deterministic world model.

        Args:
            inputs (np.ndarray): Inputs. Basically observations and actions.
            labels (np.ndarray): Labels. Desired output.
            batch_size (int): Batch size to train on / chunk size. Defaults to 256.
            train_test_ratio(float): Train / test split ratio. Defaults to 0.2.
            max_epochs_since_update (int): Maximal epochs since last update with no sufficient performance increase to abort training. Defaults to 5.
        """
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

                prediction = self.world_model(train_input)
                loss = (prediction - train_label.repeat([self.network_dim, 1, 1])).pow(2).mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                sync_grads(self.world_model)
                self.optimizer.step()

            with torch.no_grad():
                test_prediction = self.world_model(test_inputs)
                test_mse_losses = (test_prediction - test_labels.repeat([self.network_dim, 1, 1])).pow(2).mean(dim=(1, 2))
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
        self.world_model.eval()
        prediction = np.empty((self.network_dim, len(inputs), self.output_dim))
        for start_pos in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[start_pos: start_pos + batch_size]).float().to(device)
            out = self.world_model(input)
            prediction[:, start_pos: start_pos + batch_size] = out.detach().cpu().numpy()
        prediction = np.median(prediction, axis=0)

        # Eval mode for "stochastic" prediction
        self.world_model.train()
        prediction_mc = np.empty((self.num_mc, len(inputs), self.output_dim))
        for n in range(self.num_mc):
            for start_pos in range(0, inputs.shape[0], batch_size):
                input = torch.from_numpy(inputs[start_pos: start_pos + batch_size]).float().to(device)
                out = self.world_model(input)
                out = out.detach().cpu().numpy()
                prediction_mc[n, start_pos: start_pos + batch_size] = np.median(out, axis=0)
        prediction_mc = np.median(prediction_mc, axis=0)

        return prediction, np.abs((prediction_mc - prediction) / prediction)

    def save(self, filename: str) -> None:
        """
        Save the deterministic world model.

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
        Load the deterministic world model.

        Args:
            filename (str): Filename to load the model from.
        """
        mu, std, world_model = torch.load(filename, map_location=lambda storage, loc: storage)

        self.world_model.load_state_dict(world_model)
        self.scaler.mu = mu
        self.scaler.std = std

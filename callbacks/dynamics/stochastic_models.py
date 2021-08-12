import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import count
from typing import Tuple, Union


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
    __constants__ = ['in_features', 'out_features']
    # in_features: int
    # out_features: int
    # n_ensemble: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, n_ensemble: int,
                 weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_ensemble = n_ensemble
        self.weight_decay = weight_decay
        self.weight = nn.Parameter(torch.Tensor(n_ensemble, in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_ensemble, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


class StochasticEnsembleModel(nn.Module):
    def __init__(self, n_state: int, n_action: int, n_ensemble: int, n_reward: int, hidden_dim: int, lr: float,
                 use_decay: bool) -> None:
        super(StochasticEnsembleModel, self).__init__()
        self.use_decay = use_decay
        self.hidden_dim = hidden_dim

        self.nn1 = EnsembleLinear(n_state + n_action, hidden_dim, n_ensemble)
        self.nn2 = EnsembleLinear(hidden_dim, hidden_dim, n_ensemble)
        self.nn3 = EnsembleLinear(hidden_dim, hidden_dim, n_ensemble)
        self.nn4 = EnsembleLinear(hidden_dim, hidden_dim, n_ensemble)

        self.output_dim = n_state + n_reward
        self.out = EnsembleLinear(hidden_dim, 2 * self.output_dim, n_ensemble)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10), requires_grad=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.apply(init_weights)

        self.swish = Swish()

    def forward(self, x: torch.Tensor, ret_log_var: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.out(nn4_output)

        mean = nn5_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar

        return mean, torch.exp(logvar)

    def loss(self, mean: torch.Tensor, logvar: torch.Tensor, labels: torch.Tensor, inc_var_loss: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3

        inv_var = torch.exp(-logvar)

        if inc_var_loss:
            # Average over batch and dim, sum over ensembles
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)

        return total_loss, mse_loss

    def _get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0.0

        for layer in self.children():
            if isinstance(layer, EnsembleLinear):
                decay_loss += layer.weight_decay * torch.sum(torch.square(layer.weight)) / 2.

        return decay_loss

    def train_model(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)

        if self.use_decay:
            loss += self._get_decay_loss()
        loss.backward()

        self.optimizer.step()


class StochasticEnsembleDynamicsModel:
    def __init__(self, use_cuda: bool, n_state: int, n_action: int, n_ensemble: int, n_reward: int, n_elite: int,
                 hidden_dim: int = 256, lr: float = 0.001, dropout_rate: float = 0.05, use_decay: bool = False) -> None:
        self.use_cuda = use_cuda
        self.n_state = n_state
        self.n_action = n_action
        self.n_ensemble = n_ensemble
        self.n_reward = n_reward

        # TODO: remove n_elite
        # self.n_elite = n_elite

        self.hidden_dim = hidden_dim

        self.ensemble_model = StochasticEnsembleModel(n_state=n_state, n_action=n_action, n_ensemble=n_ensemble,
                                                      n_reward=n_reward, hidden_dim=hidden_dim, lr=lr,
                                                      use_decay=use_decay)
        self.scaler = StandardScaler()

        if use_cuda:
            self.ensemble_model.cuda()

        self._max_epochs_since_update_losses = 0
        self._max_epochs_since_update_uncertainty = 0
        self._epochs_since_update_losses = 0
        self._epochs_since_update_uncertainty = 0
        self._state = None
        self._snapshots_losses = None
        self._snapshot_uncertainty = None

    def train(self, state: torch.Tensor, action: torch.Tensor, labels: torch.Tensor,
              batch_size: int = 256, holdout_ratio: float = 0.0, max_epochs_since_update: int = 5) -> bool:
        self._max_epochs_since_update_losses = max_epochs_since_update
        self._max_epochs_since_update_uncertainty = max_epochs_since_update
        self._epochs_since_update_losses = 0
        self._epochs_since_update_uncertainty = 0
        self._state = {}
        self._snapshots_losses = {n: (None, np.inf) for n in range(self.n_ensemble)}
        self._snapshot_uncertainty = (None, np.inf)

        inputs = torch.cat([state, action], dim=1)

        n_holdout = int(len(state) * holdout_ratio)
        permutation = np.random.permutation(len(inputs))
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[n_holdout:], labels[n_holdout:]
        holdout_inputs, holdout_labels = inputs[:n_holdout], labels[:n_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs_tensor = holdout_inputs.repeat([self.n_ensemble, 1, 1])
        holdout_labels_tensor = holdout_labels.repeat([self.n_ensemble, 1, 1])

        train_inputs = train_inputs.detach().cpu().numpy()
        train_labels = train_labels.detach().cpu().numpy()

        if self.use_cuda:
            holdout_inputs_tensor = holdout_inputs_tensor.cuda()
            holdout_labels_tensor = holdout_labels_tensor.cuda()

        for epoch in count():
            train_idx = np.vstack([np.random.permutation(len(train_inputs)) for _ in range(self.n_ensemble)])

            for start_pos in range(0, len(train_inputs), batch_size):
                idx = train_idx[:, start_pos:start_pos + batch_size]
                train_input_tensor = torch.tensor(train_inputs[idx], dtype=torch.float32)
                train_label_tensor = torch.tensor(train_labels[idx], dtype=torch.float32)

                if self.use_cuda:
                    train_input_tensor = train_input_tensor.cuda()
                    train_label_tensor = train_label_tensor.cuda()

                train_mean, train_logvar = self.ensemble_model(train_input_tensor)
                loss, _ = self.ensemble_model.loss(train_mean, train_logvar, train_label_tensor)
                self.ensemble_model.train_model(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs_tensor, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels_tensor,
                                                                 inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()

                # TODO: remove n_elite
                # sorted_loss_idx = np.argsort(holdout_mse_losses)
                # self.elite_model_idx = sorted_loss_idx[:self.n_elite].tolist()

                break_train_losses = self._save_best_losses(epoch, holdout_mse_losses)

                if break_train_losses:
                    break

    def _save_best_losses(self, epoch: int, holdout_losses: np.ndarray) -> bool:
        updated = False

        for n in range(len(holdout_losses)):
            current = holdout_losses[n]
            _, best = self._snapshots_losses[n]

            if np.isinf(best):
                improvement = 1.
            else:
                improvement = (best - current) / best

            if improvement > 0.01:
                self._snapshots_losses[n] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update_losses = 0
        else:
            self._epochs_since_update_losses += 1

        if self._epochs_since_update_losses > self._max_epochs_since_update_losses:
            return True

        return False

    def predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inputs = np.concatenate([state, action], axis=1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        if self.use_cuda:
            inputs_tensor = inputs_tensor.cuda()

        inputs_norm_tensor = self.scaler.transform(inputs_tensor)
        inputs_norm_tensor = inputs_norm_tensor.repeat([self.n_ensemble, 1, 1])

        with torch.no_grad():
            predictions_tensor, predictions_var = self.ensemble_model(inputs_norm_tensor)
            predictions = predictions_tensor.detach().cpu().numpy()
            predictions_var = predictions_var.detach().cpu().numpy()

        return predictions, predictions_var


def init_weights(layer: Union[nn.Linear, EnsembleLinear, StochasticEnsembleModel]) -> None:
    def truncated_normal_init(t: torch.Tensor, mean: float = 0.0, std: float = 0.01) -> torch.Tensor:
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(layer) == nn.Linear or isinstance(layer, EnsembleLinear):
        input_dim = layer.in_features
        truncated_normal_init(layer.weight, std=1 / (2 * np.sqrt(input_dim)))
        layer.bias.data.fill_(0.)

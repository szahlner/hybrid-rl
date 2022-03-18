import torch

torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
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

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0.,
                 bias: bool = True) -> None:
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


class EnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, network_size, hidden_dim, dr=0.05, hidden_activation=torch.relu):
        super(EnsembleModel, self).__init__()

        self.network_size = network_size
        self.hidden_activation = hidden_activation

        self.net = nn.ModuleList()
        in_size = state_size + action_size

        # Construct network
        for n, out_size in enumerate(hidden_dim):
            layer = EnsembleFC(in_size, out_size, network_size)
            in_size = out_size
            self.net.append(layer)
            self.net.append(nn.Dropout(dr))
        self.net.append(EnsembleFC(in_size, state_size + reward_size, network_size))
        self.apply(init_weights)

    def forward(self, state):
        x = state.repeat([self.network_size, 1, 1])
        for n, layer in enumerate(self.net):
            x = layer(x)
            x = self.hidden_activation(x)
        return x


class EnsembleDynamicsModel():
    def __init__(self, state_size, action_size, reward_size, elite_size=7, network_size=10, hidden_dim=(200, 200, 200, 200, 200)):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(
            state_size=state_size,
            action_size=action_size,
            reward_size=reward_size,
            network_size=network_size,
            hidden_dim=hidden_dim
        ).to(device)
        self.optimizer = torch.optim.Adam(self.ensemble_model.parameters(), lr=1e-3)
        self.scaler = StandardScaler()
        self.scaler.mu = np.zeros(state_size + action_size)
        self.scaler.std = np.ones(state_size + action_size)

    def train(self, obs, actions, labels, batch_size=256, holdout_ratio=0.2, max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        inputs = np.concatenate([obs, actions], axis=1)
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        # holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        # holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.random.permutation(train_inputs.shape[0])  # np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)

                prediction = self.ensemble_model(train_input)
                loss = (prediction - train_label.repeat([self.network_size, 1, 1])).pow(2).mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                prediction = self.ensemble_model(holdout_inputs)
                holdout_mse_losses = (prediction - holdout_labels.repeat([self.network_size, 1, 1])).pow(2).mean(dim=(1, 2))
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()

                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, obs, actions, batch_size=1024):
        inputs = np.concatenate([obs, actions], axis=-1)

        inputs = self.scaler.transform(inputs)

        self.ensemble_model.eval()
        prediction = np.empty((self.network_size, len(inputs), self.state_size + self.reward_size))
        for start_pos in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[start_pos : start_pos + batch_size]).float().to(device)
            out = self.ensemble_model(input)
            prediction[:, start_pos : start_pos + batch_size] = out.detach().cpu().numpy()
        self.ensemble_model.train()

        n_mc = 10
        prediction_mc = np.empty((n_mc, len(inputs), self.state_size + self.reward_size))
        for n in range(n_mc):
            for start_pos in range(0, inputs.shape[0], batch_size):
                input = torch.from_numpy(inputs[start_pos : start_pos + batch_size]).float().to(device)
                out = self.ensemble_model(input)
                out = out.detach().cpu().numpy()
                prediction_mc[n, start_pos : start_pos + batch_size] = np.median(out, axis=0)

        return np.median(prediction, axis=0), np.median(prediction_mc, axis=0)

    def save(self, filename):
        torch.save(self.ensemble_model.state_dict(), filename + "_unreal_ensemble.zip")
        torch.save(self.optimizer.state_dict(), filename + "_unreal_ensemble_optimizer.zip")
        torch.save([self.scaler.mu, self.scaler.std], filename + "_unreal_scaler.zip")

    def load(self, filename):
        self.ensemble_model.load_state_dict(torch.load(filename + "_unreal_ensemble.zip", map_location=torch.device(device)))
        self.optimizer.load_state_dict(torch.load(filename + "_unreal_ensemble_optimizer.zip", map_location=torch.device(device)))

        mu, std = torch.load(filename + "_unreal_scaler.zip", map_location=torch.device(device))
        self.scaler.mu = mu
        self.scaler.std = std

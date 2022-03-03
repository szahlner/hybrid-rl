from typing import Tuple, Union

import os

import numpy as np
import torch
import torch as th
import gpytorch as gpth

from itertools import count
from mpi_utils.normalizer import normalizer
from mpi_utils.mpi_utils import sync_grads, sync_networks, sync_bool_and


class UnrealEnvironmentDimension(gpth.models.ExactGP):
    def __init__(self, x, y, likelihood) -> None:
        super().__init__(x, y, likelihood)
        self.mean = gpth.means.ConstantMean()
        self.covar = gpth.kernels.ScaleKernel(gpth.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean(x)
        covar = self.covar(x)
        return gpth.distributions.MultivariateNormal(mean, covar)


class UnrealEnvironment:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.device = device

        self._setup_environment()

        self.normalizer = normalizer(obs_dim+action_dim)

        self._max_epochs_since_update = 0
        self._epochs_since_update = 0
        self._snapshots = None

    def _setup_environment(self):
        # Make env
        likelihoods = [gpth.likelihoods.GaussianLikelihood() for _ in range(self.obs_dim+self.reward_dim)]
        env_dimensions = [
            UnrealEnvironmentDimension(
                x=th.zeros(0, dtype=th.float64, device=self.device),
                y=th.zeros(0, dtype=th.float64, device=self.device),
                likelihood=likelihood,
            ).to(self.device) for likelihood in likelihoods]

        self.env = {
            'model': gpth.models.IndependentModelList(*env_dimensions),
            'likelihood': gpth.likelihoods.LikelihoodList(*likelihoods),
        }
        self.env['mll'] = gpth.mlls.SumMarginalLogLikelihood(self.env['likelihood'], self.env['model'])
        self.env['optimizer'] = th.optim.Adam(self.env['model'].parameters(), lr=0.1)

        sync_networks(self.env['model'])

    def train(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        labels: np.ndarray,
        batch_chunk_size: int = 256,
        test_ratio: float = 0.2,
        max_epochs_since_update: int = 5,
    ) -> None:
        assert len(obs) == len(actions), "Actions and observations must have the same length"
        assert len(obs) == len(labels), "Labels and observations must have the same length"

        # self._setup_environment()

        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._snapshots = [(None, np.inf)]

        inputs = np.concatenate([obs, actions], axis=1)
        test_size = int(len(inputs) * test_ratio)
        perm = np.random.permutation(len(inputs))
        inputs, labels = inputs[perm], labels[perm]

        train_inputs, train_labels = inputs[test_size:], labels[test_size:]
        test_inputs, test_labels = inputs[:test_size], labels[:test_size]

        self.normalizer.update(train_inputs)
        self.normalizer.recompute_stats()

        train_inputs = self.normalizer.normalize(train_inputs)
        test_inputs = self.normalizer.normalize(test_inputs)

        train_inputs = th.tensor(train_inputs, dtype=th.float64, device=self.device)
        train_labels = th.tensor(train_labels, dtype=th.float64, device=self.device)
        test_inputs = th.tensor(test_inputs, dtype=th.float64, device=self.device)
        test_labels = th.tensor(test_labels, dtype=th.float64, device=self.device)

        for epoch in count():
            train_idx = np.random.permutation(len(train_inputs))

            self.env['model'].train()
            self.env['likelihood'].train()

            for start_pos in range(0, len(train_inputs), batch_chunk_size):
                idx = train_idx[start_pos : start_pos + batch_chunk_size]

                train_inputs_chunk = train_inputs[idx]
                train_labels_chunk = train_labels[idx]

                for pos, model in enumerate(self.env['model'].models):
                    model.set_train_data(train_inputs_chunk.clone(), train_labels_chunk[:, pos].clone(), strict=False)

                self.env['optimizer'].zero_grad()
                predictions = self.env['model'](*self.env['model'].train_inputs)
                loss = -self.env['mll'](predictions, self.env['model'].train_targets)
                loss.backward()

                sync_grads(self.env['model'])

                self.env['optimizer'].step()

            with th.no_grad(), gpth.settings.fast_pred_var():
                self.env['model'].eval()
                self.env['likelihood'].eval()

                test_predictions = self.env['likelihood'](*self.env['model'](*[test_inputs for _ in range(self.obs_dim+self.reward_dim)]))

                tp = torch.zeros_like(test_labels)
                for pos, prediction in enumerate(test_predictions):
                    tp[:, pos] = prediction.mean
                tl = (tp - test_labels).pow(2).mean()
                tl = np.expand_dims(tl.detach().cpu().numpy(), -1)

                break_train = self._save_best_losses(epoch, tl)
                break_train = sync_bool_and(break_train)

                if break_train:
                    break

    def _save_best_losses(self, epoch: int, test_loss: np.ndarray) -> bool:
        updated = False

        for n in range(len(test_loss)):
            current = test_loss[n]
            _, best = self._snapshots[n]

            if np.isinf(best):
                improvement = 1.0
            else:
                improvement = (best - current) / best

            # TODO: change 0.01 to variable
            if improvement > 0.01:
                self._snapshots[n] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            return True

        return False

    def predict(self, obs: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, axis=0)

        inputs = np.concatenate([obs, actions], axis=-1)
        inputs = self.normalizer.normalize(inputs)

        inputs = th.tensor(inputs, dtype=th.float64, device=self.device)

        with th.no_grad(), gpth.settings.fast_pred_var():
            self.env['model'].eval()
            self.env['likelihood'].eval()

            shape = (len(obs), self.obs_dim + self.reward_dim)

            predictions = self.env['likelihood'](*self.env['model'](*[inputs for _ in range(self.obs_dim+self.reward_dim)]))
            p, ps = torch.zeros(shape), torch.zeros(shape)
            # cl, cu = torch.zeros(shape), torch.zeros(shape)
            for pos, prediction in enumerate(predictions):
                p[:, pos], ps[:, pos] = prediction.mean, prediction.stddev
                # cl[:, pos], cu[:, pos] = prediction.confidence_region()

        p, ps = p.detach().cpu().numpy(), ps.detach().cpu().numpy()
        # cl, cu = cl.detach().cpu().numpy(), cu.detach().cpu().numpy()
        # c = np.logical_and(p - ps > cl, p + ps < cu)

        return p, ps  # cu - cl

    def predict_low_memory(self, obs: np.ndarray, actions: np.ndarray, batch_chunk_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, axis=0)

        inputs = np.concatenate([obs, actions], axis=-1)
        inputs = self.normalizer.normalize(inputs)

        inputs = th.tensor(inputs, dtype=th.float64, device=self.device)

        shape = (len(obs), self.obs_dim + self.reward_dim)

        p, ps = torch.zeros(shape), torch.zeros(shape)
        # cl, cu = torch.zeros(shape), torch.zeros(shape)

        self.env['model'].eval()
        self.env['likelihood'].eval()

        for start_pos in range(0, len(obs), batch_chunk_size):
            with th.no_grad(), gpth.settings.fast_pred_var():
                predictions = self.env['likelihood'](*self.env['model'](*[inputs[start_pos : start_pos + batch_chunk_size] for _ in range(self.obs_dim+self.reward_dim)]))
                for pos, prediction in enumerate(predictions):
                    p[start_pos : start_pos + batch_chunk_size, pos], ps[start_pos : start_pos + batch_chunk_size, pos] = prediction.mean, prediction.stddev
                    # cl[start_pos : start_pos + batch_chunk_size, pos], cu[start_pos : start_pos + batch_chunk_size, pos] = prediction.confidence_region()

        p, ps = p.detach().cpu().numpy(), ps.detach().cpu().numpy()
        # cl, cu = cl.detach().cpu().numpy(), cu.detach().cpu().numpy()
        # c = np.logical_and(p - ps > cl, p + ps < cu)

        return p, ps  # cu - cl

    def __call__(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        predictions, _ = self.predict(obs, actions)
        return predictions

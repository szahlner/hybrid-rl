import os
import numpy as np
import torch
import gym
import copy

from stable_baselines3.common.callbacks import BaseCallback

from callbacks.dynamics.deterministic_models import DeterministicEnsembleDynamicsModel
from callbacks.dynamics.stochastic_models import StochasticEnsembleDynamicsModel
from callbacks.utils import read_hyperparameters, save_hyperparameters, get_log_path


class HyBridCallback(BaseCallback):
    def __init__(self,
                 log_path: str = 'logs',
                 algo_prefix: str = 'hb-',
                 test_confidence_every: int = 10000,
                 random_exploration: float = 0.,
                 verbose: int = 2) -> None:

        super(HyBridCallback, self).__init__(verbose)
        self.env_model = None
        self.train_freq = np.inf
        # self.real_ratio = 1.
        self.rollout_batch_size = None
        self.replay_buffer = None
        self.batch_size = None
        self.holdout_ratio = None
        self.batches = None
        self.max_epochs_since_update = None
        self.rollout_length = None
        self.buffer_size = None

        self.gradient_steps_policy = 0
        self.train_freq_policy = np.inf
        self.train_now = False
        self.log_path = log_path
        self.algo_prefix = algo_prefix
        self.test_confidence_every = test_confidence_every
        self.random_exploration = random_exploration
        self.vec_normalize_env = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        algo = '{}{}'.format(self.algo_prefix, self.locals['tb_log_name'].lower())
        env_id = self.model.env.envs[0].spec.id
        hyperparams = read_hyperparameters(algo=algo, env_id=env_id, verbose=self.verbose)

        self.train_freq = hyperparams['train_freq']
        # self.real_ratio = hyperparams['real_ratio']
        self.rollout_batch_size = hyperparams['rollout_batch_size']
        self.batch_size = hyperparams['batch_size']
        self.holdout_ratio = hyperparams['holdout_ratio']
        self.batches = hyperparams['batches']
        self.max_epochs_since_update = hyperparams['max_epochs_since_update']
        self.rollout_length = hyperparams['rollout_length']
        self.buffer_size = hyperparams['buffer_size']
        self.policy_batch_size = 2048  # int(self.model.batch_size / hyperparams['real_ratio'])

        self.train_freq_policy = hyperparams['train_freq_policy']
        self.gradient_steps_policy = hyperparams['gradient_steps_policy']  # int(max(0, self.model.gradient_steps))
        self.log_path = get_log_path(self.log_path, algo, env_id)

        assert isinstance(self.model.action_space, gym.spaces.Box), 'Box space is required. No discrete space'
        assert hyperparams['model_type'] in ['deterministic', 'stochastic'], 'model_type must be "deterministic" or "stochastic"'

        # save hyperparams
        save_hyperparameters(self.log_path, env_id, hyperparams)

        # device
        if self.model.device.type == 'cpu':
            self.use_cuda = False
        else:
            self.use_cuda = True

        # number of states / actions
        if isinstance(self.model.observation_space, gym.spaces.dict.Dict):
            assert hyperparams['n_reward'] == 0, 'n_reward must be 0 (zero), it will be computed'
            assert self.rollout_batch_size % self.model.replay_buffer.max_episode_length == 0, 'Modulo operation must be 0.'

            n_state = self.model.observation_space['observation'].shape[0] + \
                      self.model.observation_space['achieved_goal'].shape[0]
        else:
            n_state = self.model.observation_space.shape[0]
        n_action = self.model.action_space.shape[0]

        # setup buffers
        self.vec_normalize_env = self.model.get_vec_normalize_env()
        self.replay_buffer = copy.deepcopy(self.model.replay_buffer)
        self.replay_buffer_clone = copy.deepcopy(self.model.replay_buffer)

        # change buffer size
        actions_shape = self.replay_buffer.actions.shape
        actions_shape = (self.buffer_size, actions_shape[1], actions_shape[2])
        observations_shape = self.replay_buffer.observations.shape
        observations_shape = (self.buffer_size, observations_shape[1], observations_shape[2])

        self.replay_buffer.buffer_size = self.buffer_size
        self.replay_buffer.actions = np.zeros(shape=actions_shape, dtype=np.float32)
        self.replay_buffer.observations = np.zeros(shape=observations_shape, dtype=np.float32)
        self.replay_buffer.next_observations = np.zeros(shape=observations_shape, dtype=np.float32)
        self.replay_buffer.dones = np.zeros(shape=(self.buffer_size, 1), dtype=np.float32)
        self.replay_buffer.rewards = np.zeros(shape=(self.buffer_size, 1), dtype=np.float32)
        self.replay_buffer.timeouts = np.zeros(shape=(self.buffer_size, 1), dtype=np.float32)

        self.replay_buffer_clone.buffer_size = self.buffer_size
        self.replay_buffer_clone.actions = np.zeros(shape=actions_shape, dtype=np.float32)
        self.replay_buffer_clone.observations = np.zeros(shape=observations_shape, dtype=np.float32)
        self.replay_buffer_clone.next_observations = np.zeros(shape=observations_shape, dtype=np.float32)
        self.replay_buffer_clone.dones = np.zeros(shape=(self.buffer_size, 1), dtype=np.float32)
        self.replay_buffer_clone.rewards = np.zeros(shape=(self.buffer_size, 1), dtype=np.float32)
        self.replay_buffer_clone.timeouts = np.zeros(shape=(self.buffer_size, 1), dtype=np.float32)

        if isinstance(self.model.observation_space, gym.spaces.dict.Dict):
            self.replay_buffer.env = self.model.get_env()
            self.replay_buffer_clone.env = self.model.get_env()

        # setup dynamics model
        if hyperparams['model_type'] == 'deterministic':
            self.env_model = DeterministicEnsembleDynamicsModel(use_cuda=self.use_cuda, n_state=n_state,
                                                                n_action=n_action, n_ensemble=hyperparams['n_ensemble'],
                                                                n_elite=hyperparams['n_elite'],
                                                                n_reward=hyperparams['n_reward'],
                                                                hidden_dim=hyperparams['n_hidden'],
                                                                lr=hyperparams['lr'], dropout_rate=hyperparams['dr'],
                                                                use_decay=hyperparams['loss_decay'])
        else:
            self.env_model = StochasticEnsembleDynamicsModel(use_cuda=self.use_cuda, n_state=n_state, n_action=n_action,
                                                             n_ensemble=hyperparams['n_ensemble'],
                                                             n_elite=hyperparams['n_elite'],
                                                             n_reward=hyperparams['n_reward'],
                                                             hidden_dim=hyperparams['n_hidden'],
                                                             lr=hyperparams['lr'], dropout_rate=hyperparams['dr'],
                                                             use_decay=hyperparams['loss_decay'])

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps > self.model.learning_starts and self.num_timesteps % self.train_freq == 0:
            if isinstance(self.model.observation_space, gym.spaces.dict.Dict):
                self._train_dynamics_dict()
                self._rollout_dynamics_dict()
            else:
                self._train_dynamics()
                self._rollout_dynamics()

        if self.num_timesteps > self.model.learning_starts + self.train_freq and \
                self.num_timesteps % self.train_freq_policy == 0:
            self._train_policy()

        if self.num_timesteps > self.model.learning_starts and self.num_timesteps % self.test_confidence_every == 0:
            self._test_confidence()

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def _train_dynamics_dict(self) -> None:
        batches = min(self.model.replay_buffer.pos * self.model.replay_buffer.max_episode_length, self.batches)
        batch = self.model.replay_buffer.sample(batches, env=self.vec_normalize_env)

        observations = torch.cat([batch.observations['observation'],
                                  batch.observations['achieved_goal']], dim=1)

        next_observations = torch.cat([batch.next_observations['observation'],
                                       batch.next_observations['achieved_goal']], dim=1)
        actions = batch.actions
        labels = next_observations

        self.env_model.train(observations, actions, labels,
                             batch_size=self.batch_size, holdout_ratio=self.holdout_ratio,
                             max_epochs_since_update=self.max_epochs_since_update)

    def _train_dynamics(self) -> None:
        batches = min(self.model.replay_buffer.pos, self.batches)
        batch = self.model.replay_buffer.sample(batches, env=self.vec_normalize_env)
        observations, next_observations = batch.observations, batch.next_observations
        rewards, actions = batch.rewards, batch.actions
        labels = torch.cat([rewards, next_observations], dim=1)

        self.env_model.train(observations, actions, labels,
                             batch_size=self.batch_size, holdout_ratio=self.holdout_ratio,
                             max_epochs_since_update=self.max_epochs_since_update)

    def _rollout_dynamics_dict(self) -> None:
        buffer_size = self.model.replay_buffer.pos * self.model.replay_buffer.max_episode_length
        batch_size = min(buffer_size, self.rollout_batch_size)

        batch = self.model.replay_buffer.sample(batch_size, env=self.vec_normalize_env)

        observations = torch.cat([batch.observations['observation'], batch.observations['achieved_goal']], dim=1)

        n_state = self.model.observation_space['observation'].shape[0]
        n_goal = self.model.observation_space['achieved_goal'].shape[0]

        batch_size = batch_size // self.model.replay_buffer.max_episode_length

        for start_pos in range(0, len(observations), batch_size):
            state = observations[start_pos:start_pos + batch_size].detach().cpu().numpy()

            desired_goal = batch.observations['desired_goal'][start_pos:start_pos + batch_size].detach().cpu().numpy()
            next_desired_goal = batch.next_observations['desired_goal'][start_pos:start_pos + batch_size].detach().cpu().numpy()

            obs = {
                'observation': batch.observations['observation'][start_pos:start_pos + batch_size].detach().cpu().numpy(),
                'achieved_goal': batch.observations['achieved_goal'][start_pos:start_pos + batch_size].detach().cpu().numpy(),
                'desired_goal': desired_goal
            }

            for _ in range(self.rollout_length):
                # unscaled action
                unscaled_action, _ = self.model.predict(obs, deterministic=False)

                # Rescale the action from [low, high] to [-1, 1]
                scaled_action = self.model.policy.scale_action(unscaled_action)

                # Add noise to the action (improve exploration)
                if self.model.action_noise is not None:
                    scaled_action = np.clip(scaled_action + self.model.action_noise(), -1, 1)

                if self.random_exploration > 0.:
                    if np.random.normal() < self.random_exploration:
                        scaled_action = np.random.normal(size=scaled_action.shape)

                # We store the scaled action in the buffer
                buffer_action = scaled_action
                action = self.model.policy.unscale_action(scaled_action)

                ensemble_model_means, _ = self.env_model.predict(state, action)

                new_obs = np.mean(ensemble_model_means, axis=0)
                reward = self.replay_buffer.env.envs[0].compute_reward(new_obs[:, n_state:n_state + n_goal],
                                                                       desired_goal, None)
                done = np.zeros_like(reward).astype(np.bool)

                for n in range(len(action)):
                    s = {
                        'observation': state[n, :n_state],
                        'achieved_goal': state[n, n_state:n_state + n_goal],
                        'desired_goal': desired_goal[n]
                    }
                    n_o = {
                        'observation': new_obs[n, :n_state],
                        'achieved_goal': new_obs[n, n_state:n_state + n_goal],
                        'desired_goal': next_desired_goal[n]
                    }

                    self.replay_buffer.add(s, n_o, buffer_action[n], reward[n], done[n], [{}])

                mask = ~done.astype(bool).flatten()
                if mask.sum() == 0:
                    break
                state = new_obs[mask]

    def _rollout_dynamics(self) -> None:
        batch_size = min(self.model.replay_buffer.pos, self.rollout_batch_size)
        batch = self.model.replay_buffer.sample(batch_size, env=self.vec_normalize_env)
        observations = batch.observations

        batch_size = batch_size // 5

        for start_pos in range(0, len(observations), batch_size):
            state = observations[start_pos:start_pos + batch_size].detach().cpu().numpy()

            obs = batch.observations[start_pos:start_pos + batch_size].detach().cpu().numpy()

            for _ in range(self.rollout_length):
                # unscaled action
                unscaled_action, _ = self.model.predict(obs, deterministic=False)

                # Rescale the action from [low, high] to [-1, 1]
                scaled_action = self.model.policy.scale_action(unscaled_action)

                # Add noise to the action (improve exploration)
                if self.model.action_noise is not None:
                    scaled_action = np.clip(scaled_action + self.model.action_noise(), -1, 1)

                if self.random_exploration > 0.:
                    if np.random.normal() < self.random_exploration:
                        scaled_action = np.random.normal(size=scaled_action.shape)

                # We store the scaled action in the buffer
                buffer_action = scaled_action
                action = self.model.policy.unscale_action(scaled_action)

                ensemble_model_means, _ = self.env_model.predict(state, action)

                samples = np.mean(ensemble_model_means, axis=0)

                reward, new_obs = samples[:, :1], samples[:, 1:]
                done = np.zeros_like(reward).astype(np.bool)

                for n in range(len(action)):
                    s = state[n]
                    n_o = new_obs[n]

                    self.replay_buffer.add(s, n_o, buffer_action[n], reward[n], done[n], [{}])

                mask = ~done.astype(bool).flatten()
                if mask.sum() == 0:
                    break
                state = new_obs[mask]

    def _test_confidence(self, batch_size: int = 2000) -> None:
        if isinstance(self.model.observation_space, gym.spaces.dict.Dict):
            batch = self.model.replay_buffer.sample(batch_size, env=self.vec_normalize_env)
            state = torch.cat([batch.observations['observation'],
                               batch.observations['achieved_goal']], dim=1)
            next_state = torch.cat([batch.next_observations['observation'],
                                    batch.next_observations['achieved_goal']], dim=1)

            state = state.detach().cpu().numpy()
            next_state = next_state.detach().cpu().numpy()
        else:
            batch = self.model.replay_buffer.sample(batch_size, env=self.vec_normalize_env)

            state = batch.observations.detach().cpu().numpy()
            next_state = batch.next_observations.detach().cpu().numpy()

        action = batch.actions.detach().cpu().numpy()
        rewards = batch.rewards.detach().cpu().numpy()

        if isinstance(self.model.observation_space, gym.spaces.dict.Dict):
            ground_truth = next_state
        else:
            ground_truth = np.concatenate([rewards, next_state], axis=1)

        ensemble_model_means, ensemble_model_vars = self.env_model.predict(state, action)
        predictions = np.mean(ensemble_model_means, axis=0)

        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        confidence = np.mean(ensemble_model_stds, axis=0)

        error = np.abs(ground_truth - predictions)
        error_min_idx = np.argmin(error, axis=0)
        error_max_idx = np.argmax(error, axis=0)

        error_mean = np.expand_dims(np.mean(error, axis=0), axis=0)
        error_mse = np.expand_dims(np.mean(error ** 2, axis=0), axis=0)
        error_min_error = np.expand_dims(error[error_min_idx, np.arange(len(error_min_idx))], axis=0)
        error_max_error = np.expand_dims(error[error_max_idx, np.arange(len(error_max_idx))], axis=0)

        confidence_min_idx = np.argmin(confidence, axis=0)
        confidence_max_idx = np.argmax(confidence, axis=0)

        confidence_mean = np.expand_dims(np.mean(confidence, axis=0), axis=0)
        confidence_min_confidence = np.expand_dims(
            confidence[confidence_min_idx, np.arange(len(confidence_min_idx))], axis=0)
        confidence_max_confidence = np.expand_dims(
            confidence[confidence_max_idx, np.arange(len(confidence_max_idx))], axis=0)
        confidence_min_error = np.expand_dims(confidence[error_min_idx, np.arange(len(error_min_idx))], axis=0)
        confidence_max_error = np.expand_dims(confidence[error_max_idx, np.arange(len(error_max_idx))], axis=0)

        error_min_confidence = np.expand_dims(error[confidence_min_idx, np.arange(len(confidence_min_idx))], axis=0)
        error_max_confidence = np.expand_dims(error[confidence_max_idx, np.arange(len(confidence_max_idx))], axis=0)

        ground_truth_mean = np.expand_dims(np.mean(ground_truth, axis=0), axis=0)
        ground_truth_min_confidence = np.expand_dims(
            ground_truth[confidence_min_idx, np.arange(len(confidence_min_idx))], axis=0)
        ground_truth_max_confidence = np.expand_dims(
            ground_truth[confidence_max_idx, np.arange(len(confidence_max_idx))], axis=0)
        ground_truth_min_error = np.expand_dims(ground_truth[error_min_idx, np.arange(len(error_min_idx))], axis=0)
        ground_truth_max_error = np.expand_dims(ground_truth[error_max_idx, np.arange(len(error_max_idx))], axis=0)

        predictions_mean = np.expand_dims(np.mean(predictions, axis=0), axis=0)
        predictions_min_confidence = np.expand_dims(
            predictions[confidence_min_idx, np.arange(len(confidence_min_idx))], axis=0)
        predictions_max_confidence = np.expand_dims(
            predictions[confidence_max_idx, np.arange(len(confidence_max_idx))], axis=0)
        predictions_min_error = np.expand_dims(predictions[error_min_idx, np.arange(len(error_min_idx))], axis=0)
        predictions_max_error = np.expand_dims(predictions[error_max_idx, np.arange(len(error_max_idx))], axis=0)

        x = np.expand_dims(np.array([self.num_timesteps]), axis=0)

        path = os.path.join(self.log_path, 'dynamics.npz')
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            x = np.concatenate([data['x'], x], axis=0)

            error_mean = np.concatenate([data['error_mean'], error_mean], axis=0)
            error_mse = np.concatenate([data['error_mse'], error_mse], axis=0)
            error_min_error = np.concatenate([data['error_min_error'], error_min_error], axis=0)
            error_max_error = np.concatenate([data['error_max_error'], error_max_error], axis=0)
            error_min_confidence = np.concatenate([data['error_min_confidence'], error_min_confidence], axis=0)
            error_max_confidence = np.concatenate([data['error_max_confidence'], error_max_confidence], axis=0)

            confidence_mean = np.concatenate([data['confidence_mean'], confidence_mean], axis=0)
            confidence_min_confidence = np.concatenate([data['confidence_min_confidence'],
                                                        confidence_min_confidence], axis=0)
            confidence_max_confidence = np.concatenate([data['confidence_max_confidence'],
                                                        confidence_max_confidence], axis=0)
            confidence_min_error = np.concatenate([data['confidence_min_error'], confidence_min_error], axis=0)
            confidence_max_error = np.concatenate([data['confidence_max_error'], confidence_max_error], axis=0)

            ground_truth_mean = np.concatenate([data['ground_truth_mean'], ground_truth_mean], axis=0)
            ground_truth_min_confidence = np.concatenate([data['ground_truth_min_confidence'],
                                                          ground_truth_min_confidence], axis=0)
            ground_truth_max_confidence = np.concatenate([data['ground_truth_max_confidence'],
                                                          ground_truth_max_confidence], axis=0)
            ground_truth_min_error = np.concatenate([data['ground_truth_min_error'], ground_truth_min_error], axis=0)
            ground_truth_max_error = np.concatenate([data['ground_truth_max_error'], ground_truth_max_error], axis=0)

            predictions_mean = np.concatenate([data['predictions_mean'], predictions_mean], axis=0)
            predictions_min_confidence = np.concatenate([data['predictions_min_confidence'],
                                                         predictions_min_confidence], axis=0)
            predictions_max_confidence = np.concatenate([data['predictions_max_confidence'],
                                                         predictions_max_confidence], axis=0)
            predictions_min_error = np.concatenate([data['predictions_min_error'], predictions_min_error], axis=0)
            predictions_max_error = np.concatenate([data['predictions_max_error'], predictions_max_error], axis=0)

        np.savez_compressed(path, x=x,
                            error_mean=error_mean, error_mse=error_mse,
                            error_min_error=error_min_error, error_max_error=error_max_error,
                            error_min_confidence=error_min_confidence, error_max_confidence=error_max_confidence,
                            confidence_mean=confidence_mean,
                            confidence_min_confidence=confidence_min_confidence,
                            confidence_max_confidence=confidence_max_confidence,
                            confidence_min_error=confidence_min_error, confidence_max_error=confidence_max_error,
                            ground_truth_mean=ground_truth_mean,
                            ground_truth_min_confidence=ground_truth_min_confidence,
                            ground_truth_max_confidence=ground_truth_max_confidence,
                            ground_truth_min_error=ground_truth_min_error,
                            ground_truth_max_error=ground_truth_max_error,
                            predictions_mean=predictions_mean,
                            predictions_min_confidence=predictions_min_confidence,
                            predictions_max_confidence=predictions_max_confidence,
                            predictions_min_error=predictions_min_error,
                            predictions_max_error=predictions_max_error)

    def _train_policy(self) -> None:

        self.replay_buffer_clone = self.model.replay_buffer
        self.model.replay_buffer = self.replay_buffer
        self.model.train(batch_size=self.policy_batch_size, gradient_steps=self.gradient_steps_policy)
        self.model.replay_buffer = self.replay_buffer_clone

        return

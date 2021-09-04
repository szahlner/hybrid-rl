import os
import numpy as np
import torch
import gym
import warnings

from stable_baselines3.common.callbacks import BaseCallback

from callbacks.buffers import ReplayBuffer
from callbacks.dynamics.deterministic_model import EnsembleDynamicsModel
from callbacks.utils import (
    read_hyperparameters,
    save_hyperparameters,
    get_log_path,
    ALGOS,
    MODEL_TYPES,
)

from typing import Optional


class HyBridCallback(BaseCallback):
    def __init__(
        self,
        algo_prefix: str = "hb-",
        log_path: str = "logs",
        random_exploration: float = 0.0,
        test_confidence_every: int = 10000,
        verbose: int = 2,
        recorded_buffer: Optional[str] = None,
    ) -> None:

        super(HyBridCallback, self).__init__(verbose)
        self.env_model = None
        self.train_freq = np.inf
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

        self.recorded_buffer = recorded_buffer
        self.replay_buffer_demo = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.algo = "{}{}".format(self.algo_prefix, self.locals["tb_log_name"].lower())
        self.train_policy = ALGOS[self.algo]

        env_id = self.model.env.envs[0].spec.id
        hyperparams = read_hyperparameters(
            algo=self.algo, env_id=env_id, verbose=self.verbose
        )

        self.train_freq = hyperparams["train_freq"]
        self.rollout_batch_size = hyperparams["rollout_batch_size"]
        self.batch_size = hyperparams["batch_size"]
        self.holdout_ratio = hyperparams["holdout_ratio"]
        self.batches = hyperparams["batches"]
        self.max_epochs_since_update = hyperparams["max_epochs_since_update"]
        self.rollout_length = hyperparams["rollout_length"]
        self.buffer_size = hyperparams["buffer_size"]
        self.batch_size_policy = hyperparams["batch_size_policy"]

        self.train_freq_policy = hyperparams["train_freq_policy"]
        self.gradient_steps_policy = hyperparams["gradient_steps_policy"]
        self.log_path = get_log_path(self.log_path, self.algo, env_id)

        assert isinstance(
            self.model.action_space, gym.spaces.Box
        ), "Box space is required. No discrete space"
        assert (
            hyperparams["model_type"] in MODEL_TYPES
        ), 'model_type must be "deterministic" or "stochastic"'

        # Save hyperparams
        save_hyperparameters(self.log_path, env_id, hyperparams)

        # Set device
        if self.model.device.type == "cpu":
            warnings.warn("Using CPU only will be very slow!", UserWarning)
            self.use_cuda = False
        else:
            self.use_cuda = True

        # Number of states / actions
        n_action = self.model.action_space.shape[0]
        if isinstance(self.model.observation_space, gym.spaces.dict.Dict):
            assert (
                hyperparams["n_reward"] == 0
            ), "n_reward must be 0 (zero), it will be computed"

            n_state = self.model.observation_space["observation"].shape[0]
            n_goal = self.model.observation_space["achieved_goal"].shape[0]

            self.replay_buffer = ReplayBuffer(
                n_state=n_state,
                n_action=n_action,
                n_goal=n_goal,
                size=self.buffer_size,
                use_cuda=self.use_cuda,
            )

            n_state += 2 * n_goal
        else:
            n_state = self.model.observation_space.shape[0]

            self.replay_buffer = ReplayBuffer(
                n_state=n_state,
                n_action=n_action,
                size=self.buffer_size,
                use_cuda=self.use_cuda,
            )

        if self.recorded_buffer is not None:
            # TODO: make dict buffer
            # There are demonstrations available
            data = np.load(self.recorded_buffer, allow_pickle=True)

            states, actions, rewards, dones, next_states = (
                data["observations"],
                data["actions"],
                data["rewards"],
                data["dones"],
                data["next_observations"],
            )

            assert states.shape[-1] == n_state, "Observation dimensions must match"
            assert (
                next_states.shape[-1] == n_state
            ), "Next Observation dimensions must match"
            assert actions.shape[-1] == n_action, "Action dimensions must match"
            assert rewards.shape[-1] == 1, "Rewards must be of length 1"
            assert dones.shape[-1] == 1, "Dones must be of length 1"

            # Fill demonstrations into the real replay buffer
            infos = [{}] * len(states)
            self.model.replay_buffer.extend(
                obs=states,
                next_obs=next_states,
                action=actions,
                reward=rewards,
                mask=dones,
                infos=infos,
            )

        # Setup dynamics model
        if hyperparams["model_type"] == "deterministic":
            self.env_model = EnsembleDynamicsModel(
                use_cuda=self.use_cuda,
                n_state=n_state,
                n_action=n_action,
                n_ensemble=hyperparams["n_ensemble"],
                n_reward=hyperparams["n_reward"],
                hidden_dim=hyperparams["n_hidden"],
                lr=hyperparams["lr"],
                dropout_rate=hyperparams["dr"],
                use_decay=hyperparams["loss_decay"],
            )
        else:
            assert False, "Model type: '{}' not implemented yet".format(
                hyperparams["model_type"]
            )

        if self.verbose > 0:
            print("Done setting up hybrid callback")

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
        if (
            self.num_timesteps > self.model.learning_starts
            and self.num_timesteps % self.train_freq == 0
        ):
            if isinstance(self.model.observation_space, gym.spaces.dict.Dict):
                self._train_dynamics_dict()
                self._rollout_dynamics_dict()
            else:
                self._train_dynamics()
                self._rollout_dynamics()

        if (
            self.num_timesteps > self.model.learning_starts + self.train_freq
            and self.num_timesteps % self.train_freq_policy == 0
        ):
            self._train_policy()

        if (
            self.num_timesteps > self.model.learning_starts + self.train_freq
            and self.num_timesteps % self.test_confidence_every == 0
        ):
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
        batches = min(
            self.model.replay_buffer.pos * self.model.replay_buffer.max_episode_length,
            self.batches,
        )
        batch = self.model.replay_buffer.sample(batches, env=self.vec_normalize_env)

        observations = torch.cat(
            [
                batch.observations["observation"],
                batch.observations["achieved_goal"],
                batch.observations["desired_goal"],
            ],
            dim=1,
        )

        next_observations = torch.cat(
            [
                batch.next_observations["observation"],
                batch.next_observations["achieved_goal"],
                batch.next_observations["desired_goal"],
            ],
            dim=1,
        )

        actions = batch.actions
        labels = next_observations

        self.env_model.train(
            state=observations,
            action=actions,
            labels=labels,
            batch_size=self.batch_size,
            holdout_ratio=self.holdout_ratio,
            max_epochs_since_update=self.max_epochs_since_update,
        )

    def _train_dynamics(self) -> None:
        batches = min(self.model.replay_buffer.pos, self.batches)
        batch = self.model.replay_buffer.sample(batches, env=self.vec_normalize_env)
        observations, next_observations = batch.observations, batch.next_observations
        rewards, actions = batch.rewards, batch.actions
        labels = torch.cat([rewards, next_observations], dim=1)

        self.env_model.train(
            state=observations,
            action=actions,
            labels=labels,
            batch_size=self.batch_size,
            holdout_ratio=self.holdout_ratio,
            max_epochs_since_update=self.max_epochs_since_update,
        )

    def _rollout_dynamics_dict(self) -> None:
        buffer_size = (
            self.model.replay_buffer.pos * self.model.replay_buffer.max_episode_length
        )
        batch_size = min(buffer_size, self.rollout_batch_size)

        batch = self.model.replay_buffer.sample(batch_size, env=self.vec_normalize_env)

        observations = torch.cat(
            [
                batch.observations["observation"],
                batch.observations["achieved_goal"],
                batch.observations["desired_goal"],
            ],
            dim=1,
        )

        n_state = self.model.observation_space["observation"].shape[0]
        n_goal = self.model.observation_space["achieved_goal"].shape[0]

        batch_size = batch_size // 5  # self.model.replay_buffer.max_episode_length

        for start_pos in range(0, len(observations), batch_size):
            state = (
                observations[start_pos : start_pos + batch_size].detach().cpu().numpy()
            )

            desired_goal = (
                batch.observations["desired_goal"][start_pos : start_pos + batch_size]
                .detach()
                .cpu()
                .numpy()
            )
            next_desired_goal = (
                batch.next_observations["desired_goal"][
                    start_pos : start_pos + batch_size
                ]
                .detach()
                .cpu()
                .numpy()
            )

            obs = {
                "observation": batch.observations["observation"][
                    start_pos : start_pos + batch_size
                ]
                .detach()
                .cpu()
                .numpy(),
                "achieved_goal": batch.observations["achieved_goal"][
                    start_pos : start_pos + batch_size
                ]
                .detach()
                .cpu()
                .numpy(),
                "desired_goal": desired_goal,
            }

            for _ in range(self.rollout_length):
                # unscaled action
                unscaled_action, _ = self.model.predict(obs, deterministic=False)

                # Rescale the action from [low, high] to [-1, 1]
                scaled_action = self.model.policy.scale_action(unscaled_action)

                # Add noise to the action (improve exploration)
                if self.model.action_noise is not None:
                    scaled_action = np.clip(
                        scaled_action + self.model.action_noise(), -1, 1
                    )

                # if self.random_exploration > 0.0:
                #     if np.random.uniform() < self.random_exploration:
                #        scaled_action = np.random.uniform(size=scaled_action.shape)

                # We store the scaled action in the buffer
                buffer_action = scaled_action
                action = self.model.policy.unscale_action(scaled_action)

                ensemble_model_means, _ = self.env_model.predict(state, action)

                new_obs = np.mean(ensemble_model_means, axis=0)
                reward = self.model.env.envs[0].compute_reward(
                    new_obs[:, n_state : n_state + n_goal], desired_goal, None
                )
                reward = np.expand_dims(reward, axis=-1)
                done = np.zeros_like(reward).astype(np.bool)

                self.replay_buffer.add(
                    state=state[:, :n_state],
                    achieved_goal=state[:, n_state : n_state + n_goal],
                    desired_goal=desired_goal,
                    action=buffer_action,
                    reward=reward,
                    mask=done,
                    next_state=new_obs[:, :n_state],
                    next_achieved_goal=new_obs[:, n_state : n_state + n_goal],
                    next_desired_goal=next_desired_goal,
                )

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
            state = (
                observations[start_pos : start_pos + batch_size].detach().cpu().numpy()
            )

            obs = (
                batch.observations[start_pos : start_pos + batch_size]
                .detach()
                .cpu()
                .numpy()
            )

            for _ in range(self.rollout_length):
                # unscaled action
                unscaled_action, _ = self.model.predict(obs, deterministic=False)

                # Rescale the action from [low, high] to [-1, 1]
                scaled_action = self.model.policy.scale_action(unscaled_action)

                # Add noise to the action (improve exploration)
                if self.model.action_noise is not None:
                    scaled_action = np.clip(
                        scaled_action + self.model.action_noise(), -1, 1
                    )

                if self.random_exploration > 0.0:
                    if np.random.normal() < self.random_exploration:
                        scaled_action = np.random.normal(size=scaled_action.shape)

                # We store the scaled action in the buffer
                buffer_action = scaled_action
                action = self.model.policy.unscale_action(scaled_action)

                ensemble_model_means, _ = self.env_model.predict(state, action)

                samples = np.mean(ensemble_model_means, axis=0)

                reward, new_obs = samples[:, :1], samples[:, 1:]
                done = np.zeros_like(reward).astype(np.bool)

                self.replay_buffer.add(
                    state=state,
                    action=buffer_action,
                    reward=reward,
                    mask=done,
                    next_state=new_obs,
                )

                mask = ~done.astype(bool).flatten()
                if mask.sum() == 0:
                    break
                state = new_obs[mask]

    def _test_confidence(self, batch_size: int = 100) -> None:
        if isinstance(self.model.observation_space, gym.spaces.dict.Dict):
            return
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

        inputs = np.concatenate([state, action], axis=1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        if self.use_cuda:
            inputs_tensor = inputs_tensor.cuda()

        inputs_norm_tensor = self.env_model.scaler.transform(inputs_tensor)
        inputs_norm_tensor = inputs_norm_tensor.repeat([self.env_model.n_ensemble, 1, 1])

        predictions_tensor_list = []
        with torch.no_grad():
            for _ in range(100):
                predictions_tensor = self.env_model.ensemble_model(inputs_norm_tensor)
                prediction_tensor = torch.mean(predictions_tensor, dim=0)
                predictions_tensor_list += [prediction_tensor]

        predictions_mean = torch.mean(torch.stack(predictions_tensor_list), dim=0).detach().cpu().numpy()
        predictions_std = torch.std(torch.stack(predictions_tensor_list), dim=0).detach().cpu().numpy()
        predictions_median = torch.median(torch.stack(predictions_tensor_list), dim=0).detach().cpu().numpy()
        predictions_q1 = torch.quantile(torch.stack(predictions_tensor_list), q=0.25, dim=0).detach().cpu().numpy()
        predictions_q3 = torch.quantile(torch.stack(predictions_tensor_list), q=0.75, dim=0).detach().cpu().numpy()

        x = np.expand_dims(np.array([self.num_timesteps]), axis=0)
        predictions_mean = np.expand_dims(predictions_mean, axis=0)
        predictions_std = np.expand_dims(predictions_std, axis=0)
        predictions_median = np.expand_dims(predictions_median, axis=0)
        predictions_q1 = np.expand_dims(predictions_q1, axis=0)
        predictions_q3 = np.expand_dims(predictions_q3, axis=0)
        ground_truth = np.expand_dims(ground_truth, axis=0)

        confidence_log = os.path.join(self.log_path, "dynamics.npz")
        if os.path.exists(confidence_log):
            data = np.load(confidence_log, allow_pickle=True)

            x = np.concatenate([data["x"], x], axis=0)
            predictions_mean = np.concatenate([data["predictions_mean"], predictions_mean], axis=0)
            predictions_std = np.concatenate([data["predictions_std"], predictions_std], axis=0)
            predictions_median = np.concatenate([data["predictions_median"], predictions_std], axis=0)
            predictions_q1 = np.concatenate([data["predictions_q1"], predictions_std], axis=0)
            predictions_q3 = np.concatenate([data["predictions_q3"], predictions_std], axis=0)
            ground_truth = np.concatenate([data["ground_truth"], ground_truth], axis=0)

        np.savez_compressed(
            file=confidence_log,
            x=x,
            predictions_mean=predictions_mean,
            predictions_std=predictions_std,
            predictions_median=predictions_median,
            predictions_q1=predictions_q1,
            predictions_q3=predictions_q3,
            ground_truth=ground_truth,
        )

    def _train_policy(self) -> None:

        self.model = self.train_policy(
            buffer=self.replay_buffer,
            model=self.model,
            gradient_steps=self.gradient_steps_policy,
            batch_size=self.batch_size_policy,
        )

import torch

from torch.nn import functional as F

from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.utils import polyak_update

from callbacks.buffers import ReplayBuffer

from typing import Union


def train_policy(
    buffer: ReplayBuffer,
    model: Union[DDPG, TD3],
    gradient_steps: int,
    batch_size: int = 100,
) -> Union[DDPG, TD3]:
    """
    Args:
        buffer (ReplayBuffer): Replay buffer to use.
        model (DDPG, TD3): Algorithm to use. Note: TD3 is the successor of DDGP.
        gradient_steps (int): Gradient steps to make.
        batch_size (int, optional): Batch size. Defaults to 100.

    Returns:
        Union[DDPG, TD3]: Used algorithm.
    """
    # Update learning rate according to lr schedule
    # self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    actor_losses, critic_losses = [], []

    for _ in range(gradient_steps):
        model._n_updates += 1

        # Sample from replay buffer
        if buffer.n_goal is None:
            # Env
            obs, actions, rewards, dones, next_obs = buffer.sample(batch_size)
        else:
            # GoalEnv
            (
                obs,
                achieved_goals,
                desired_goals,
                actions,
                rewards,
                dones,
                next_obs,
                next_achieved_goals,
                next_desired_goals,
            ) = buffer.sample(batch_size)

            obs = {
                "observation": obs,
                "achieved_goal": achieved_goals,
                "desired_goal": desired_goals,
            }

            next_obs = {
                "observation": next_obs,
                "achieved_goal": next_achieved_goals,
                "desired_goal": next_desired_goals,
            }

        with torch.no_grad():
            next_actions = (model.actor_target(next_obs)).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = torch.cat(
                model.critic_target(next_obs, next_actions), dim=1
            )
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            target_q_values = rewards + (1 - dones) * model.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = model.critic(obs, actions)

        # Compute critic loss
        critic_loss = sum(
            [F.mse_loss(current_q, target_q_values) for current_q in current_q_values]
        )
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        model.critic.optimizer.zero_grad()
        critic_loss.backward()
        model.critic.optimizer.step()

        # Delayed policy updates
        if model._n_updates % model.policy_delay == 0:
            # Compute actor loss
            actor_loss = -model.critic.q1_forward(obs, model.actor(obs)).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            model.actor.optimizer.zero_grad()
            actor_loss.backward()
            model.actor.optimizer.step()

            polyak_update(
                model.critic.parameters(), model.critic_target.parameters(), model.tau
            )
            polyak_update(
                model.actor.parameters(), model.actor_target.parameters(), model.tau
            )

    return model

    vec_normalize_env = model.get_vec_normalize_env()

    for _ in range(gradient_steps):
        model._n_updates += 1

        batch = buffer.sample(batch_size, env=vec_normalize_env)

        obs, actions, next_obs = (
            batch.observations,
            batch.actions,
            batch.next_observations,
        )
        rewards, dones = batch.rewards, batch.dones

        if use_cuda:
            obs, actions, next_obs = obs.cuda(), actions.cuda(), next_obs.cuda()
            rewards, dones = rewards.cuda(), dones.cuda()

        with torch.no_grad():
            next_actions = (model.actor_target(next_obs)).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = torch.cat(
                model.critic_target(next_obs, next_actions), dim=1
            )
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            target_q_values = rewards + (1 - dones) * model.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = model.critic_target(obs, actions)

        # Compute critic loss
        critic_loss = sum(
            [F.mse_loss(current_q, target_q_values) for current_q in current_q_values]
        )

        # Optimize the critics
        model.critic.optimizer.zero_grad()
        critic_loss.backward()
        model.critic.optimizer.step()

        # Delay policy updates
        if model._n_updates % model.policy_delay == 0:
            # Compute actor loss
            actor_loss = -model.critic.q1_forward(obs, model.actor(obs)).mean()

            # Optimize the actor
            model.actor.optimizer.zero_grad()
            actor_loss.backward()
            model.actor.optimizer.step()

            polyak_update(
                model.critic.parameters(), model.critic_target.parameters(), model.tau
            )
            polyak_update(
                model.actor.parameters(), model.actor_target.parameters(), model.tau
            )

    return model

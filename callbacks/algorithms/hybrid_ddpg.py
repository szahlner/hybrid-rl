import torch

from torch.nn import functional as F
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update


def train_hybrid_ddpg(buffer: ReplayBuffer,
                      model: DDPG,
                      gradient_steps: int,
                      batch_size: int = 100,
                      use_cuda: bool = False) -> None:
    """
    Args:
        buffer (stable_baselines3.common.buffers.ReplayBuffer): Replay buffer to use.
        model (from stable_baselines3.ddpg.ddpg.DDPG): Algorithm to use.
        gradient_steps (int): Gradient steps to make.
        batch_size (int, optional): Batch size. Defaults to 100.
        use_cuda (bool, optional): Whether to use cuda or not. Defaults to False.
    """
    vec_normalize_env = model.get_vec_normalize_env()

    for _ in range(gradient_steps):
        model._n_updates += 1

        batch = buffer.sample(batch_size, env=vec_normalize_env)

        obs, actions, next_obs = batch.observations, batch.actions, batch.next_observations
        rewards, dones = batch.rewards, batch.dones

        if use_cuda:
            obs, actions, next_obs = obs.cuda(), actions.cuda(), next_obs.cuda()
            rewards, dones = rewards.cuda(), dones.cuda()

        with torch.no_grad():
            next_actions = (model.actor_target(next_obs)).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = torch.cat(model.critic_target(next_obs, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            target_q_values = rewards + (1 - dones) * model.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = model.critic_target(obs, actions)

        # Compute critic loss
        critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])

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

            polyak_update(model.critic.parameters(), model.critic_target.parameters(), model.tau)
            polyak_update(model.actor.parameters(), model.actor_target.parameters(), model.tau)

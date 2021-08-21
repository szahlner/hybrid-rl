import torch

from torch.nn import functional as F
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.utils import polyak_update

from callbacks.buffers import ReplayBuffer


def train_policy(
        buffer: ReplayBuffer,
        model: SAC,
        gradient_steps: int,
        batch_size: int = 100
) -> SAC:
    """
    Args:
        buffer (ReplayBuffer): Replay buffer to use.
        model (SAC): Algorithm to use.
        gradient_steps (int): Gradient steps to make.
        batch_size (int, optional): Batch size. Defaults to 100.

    Returns:
        SAC: Used algorithm.
    """
    # vec_normalize_env = model.get_vec_normalize_env()

    for _ in range(gradient_steps):
        # batch = buffer.sample(batch_size, env=vec_normalize_env)

        # obs, actions, next_obs = batch.observations, batch.actions, batch.next_observations
        # rewards, dones = batch.rewards, batch.dones

        # if use_cuda:
        #     obs, actions, next_obs = obs.cuda(), actions.cuda(), next_obs.cuda()
        #     rewards, dones = rewards.cuda(), dones.cuda()

        # Sample replay buffer
        obs, actions, rewards, dones, next_obs = buffer.sample(batch_size)

        # We need to sample because `log_std` may have changed between two gradient steps
        if model.use_sde:
            model.actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = model.actor.action_log_prob(obs)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if model.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = torch.exp(model.log_ent_coef.detach())
            ent_coef_loss = -(model.log_ent_coef * (log_prob + model.target_entropy).detach()).mean()
        else:
            ent_coef = model.ent_coef_tensor

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            model.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            model.ent_coef_optimizer.step()

        with torch.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = model.actor.action_log_prob(next_obs)
            # Compute the next Q values: min over all critics targets
            next_q_values = torch.cat(model.critic_target(next_obs, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = rewards + (1 - dones) * model.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = model.critic(obs, actions)

        # Compute critic loss
        critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])

        # Optimize the critic
        model.critic.optimizer.zero_grad()
        critic_loss.backward()
        model.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        q_values_pi = torch.cat(model.critic(obs, actions_pi), dim=1)
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

        # Optimize the actor
        model.actor.optimizer.zero_grad()
        actor_loss.backward()
        model.actor.optimizer.step()

        # Update target networks
        if gradient_steps % model.target_update_interval == 0:
            polyak_update(model.critic.parameters(), model.critic_target.parameters(), model.tau)

    model._n_updates += gradient_steps

    return model

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unreal_env.env2 import init_weights, EnsembleFC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class EnsembleActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, network_size, max_action, hidden_activation=torch.relu):
        super(EnsembleActor, self).__init__()

        self.max_action = max_action
        self.network_size = network_size
        self.hidden_activation = hidden_activation

        self.net = nn.ModuleList()
        in_size = input_dim

        # Construct network
        for n, out_size in enumerate(hidden_dim):
            layer = EnsembleFC(in_size, out_size, network_size)
            in_size = out_size
            self.net.append(layer)
        self.last_layer = EnsembleFC(in_size, output_dim, network_size)
        self.apply(init_weights)

    def forward(self, state):
        x = state.repeat([self.network_size, 1, 1])
        for n, layer in enumerate(self.net):
            x = layer(x)
            x = self.hidden_activation(x)
        out = self.max_action * torch.tanh(self.last_layer(x))
        out = out.mean(dim=0)
        return out


class EnsembleCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, network_size, hidden_activation=torch.relu):
        super(EnsembleCritic, self).__init__()

        self.network_size = network_size
        self.hidden_activation = hidden_activation

        self.net = nn.ModuleList()
        in_size = input_dim

        # Construct network
        for n, out_size in enumerate(hidden_dim):
            layer = EnsembleFC(in_size, out_size, network_size)
            in_size = out_size
            self.net.append(layer)
        self.last_layer = EnsembleFC(in_size, output_dim, network_size)
        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = x.repeat([self.network_size, 1, 1])
        for n, layer in enumerate(self.net):
            x = layer(x)
            x = self.hidden_activation(x)
        out = self.last_layer(x)
        return out


class EnsembleDDPG(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        num_Q=10,
        num_Q_min=2,
        num_Pi=10,
        num_Pi_min=2,
    ):
        self.actor_lr = 1e-3
        self.actor = EnsembleActor(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dim=(256, 256),
            network_size=num_Pi,
            max_action=max_action,
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.total_timesteps = 0
        self.policy_freq = 5

        self.num_Pi = num_Pi
        self.num_Pi_min = num_Pi_min

        self.critic = EnsembleCritic(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_dim=(256, 256),
            network_size=num_Q,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.num_Q = num_Q
        self.num_Q_min = num_Q_min

        self.discount = discount
        self.tau = tau

        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_action_low_memory(self, state, batch_chunk_size=1024):
        action = np.empty((len(state), self.action_dim))
        for start_pos in range(0, len(state), batch_chunk_size):
            obs = torch.tensor(state[start_pos : start_pos + batch_chunk_size], dtype=torch.float32, device=device)
            action[start_pos : start_pos + batch_chunk_size] = self.actor(obs).detach().cpu().numpy()
        return action

    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        sample_idx = np.random.choice(self.num_Q, self.num_Q_min, replace=False)
        with torch.no_grad():
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = target_q[sample_idx]
            target_q, _ = torch.min(target_q, dim=0)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimate
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q.repeat([self.num_Q, 1, 1])) * self.num_Q

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        # self.critic.requires_grad_(False)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        # self.critic.requires_grad_(True)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def train_critic_virtual(self, replay_buffer, batch_size=256, real_ratio=0.5, unreal_env=None):
        # Sample replay buffer
        batch_size_real = int(batch_size * real_ratio)
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size_real)
        unreal_state, _, _, _, _ = replay_buffer.sample_numpy(batch_size - batch_size_real)

        unreal_action = self.select_action_low_memory(unreal_state)
        unreal_action += np.random.normal(0, 1 * 0.1, size=unreal_action.shape)
        unreal_action = unreal_action.clip(-1, 1)
        unreal_next_state, _ = unreal_env.predict(unreal_state, unreal_action)
        # Split observations
        unreal_reward = unreal_next_state[:, self.state_dim:]
        unreal_next_state = unreal_next_state[:, :self.state_dim]

        unreal_state = torch.tensor(unreal_state, dtype=torch.float32, device=device)
        unreal_action = torch.tensor(unreal_action, dtype=torch.float32, device=device)
        unreal_next_state = torch.tensor(unreal_next_state, dtype=torch.float32, device=device)
        unreal_reward = torch.tensor(unreal_reward, dtype=torch.float32, device=device)
        unreal_not_done = torch.ones_like(unreal_reward, dtype=torch.bool, device=device)

        # Concatenate
        state = torch.cat([state, unreal_state], dim=0)
        action = torch.cat([action, unreal_action], dim=0)
        next_state = torch.cat([next_state, unreal_next_state], dim=0)
        reward = torch.cat([reward, unreal_reward], dim=0)
        not_done = torch.cat([not_done, unreal_not_done], dim=0)

        # Compute the target Q value
        sample_idx = np.random.choice(self.num_Q, self.num_Q_min, replace=False)
        with torch.no_grad():
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = target_q[sample_idx]
            target_q, _ = torch.min(target_q, dim=0)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimate
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q.repeat([self.num_Q, 1, 1])) * self.num_Q

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen critic target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item()

    def train_actor(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, _, _, _, _ = replay_buffer.sample(batch_size)

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen actor target models
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item()

    def train_virtual(self, replay_buffer, unreal_replay_buffer, batch_size_real=256, batch_size_unreal=256, unreal_env=None):
        self.total_timesteps += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size_real)
        unreal_state, _, _, _, _ = replay_buffer.sample_numpy(batch_size_unreal)

        unreal_action = self.select_action_low_memory(unreal_state)
        unreal_action += np.random.normal(0, 1 * 0.1, size=unreal_action.shape)
        unreal_action = unreal_action.clip(-1, 1)
        unreal_next_state, _ = unreal_env.predict(unreal_state, unreal_action)
        # Split observations
        unreal_reward = unreal_next_state[:, self.state_dim:]
        unreal_next_state = unreal_next_state[:, :self.state_dim]

        unreal_state = torch.tensor(unreal_state, dtype=torch.float32, device=device)
        unreal_action = torch.tensor(unreal_action, dtype=torch.float32, device=device)
        unreal_next_state = torch.tensor(unreal_next_state, dtype=torch.float32, device=device)
        unreal_reward = torch.tensor(unreal_reward, dtype=torch.float32, device=device)
        unreal_not_done = torch.ones_like(unreal_reward, dtype=torch.bool, device=device)

        # for param in self.actor_optimizer.param_groups:
        #    param['lr'] = self.actor_lr * unreal_confidence

        # Concatenate
        state = torch.cat([state, unreal_state], dim=0)
        action = torch.cat([action, unreal_action], dim=0)
        next_state = torch.cat([next_state, unreal_next_state], dim=0)
        reward = torch.cat([reward, unreal_reward], dim=0)
        not_done = torch.cat([not_done, unreal_not_done], dim=0)

        # Compute the target Q value
        sample_idxs = np.random.choice(self.num_Q, self.num_Q_min, replace=False)
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = target_Q[sample_idxs]
            target_Q, _ = torch.min(target_Q, dim=0)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q.repeat([self.num_Q, 1, 1])) * self.num_Q

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.zeros(1, dtype=torch.float32, device=device)

        # Delayed policy updates
        if self.total_timesteps % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen actor target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update the frozen critic target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.critic.state_dict(), filename + f"_critic")
        torch.save(self.critic_target.state_dict(), filename + f"_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + f"_critic_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_target.load_state_dict(torch.load(filename + f"_actor_target"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.critic.load_state_dict(torch.load(filename + f"_critic"))
        self.critic_target.load_state_dict(torch.load(filename + f"_critic_target"))
        self.critic_optimizer.load_state_dict(torch.load(filename + f"_critic_optimizer"))

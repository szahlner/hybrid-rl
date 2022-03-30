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
        num_q=10,
        num_q_min=2,
        num_pi=10,
        actor_lr=1e-3,
        critic_lr=1e-3,
    ):
        self.actor_lr = actor_lr
        self.actor = EnsembleActor(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dim=(256, 256),
            network_size=num_pi,
            max_action=max_action,
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_lr = critic_lr
        self.critic = EnsembleCritic(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_dim=(256, 256),
            network_size=num_q,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.num_q = num_q
        self.num_q_min = num_q_min

        self.discount = discount
        self.tau = tau

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.confidence_mean = []
        self.confidence_std = []

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for param in self.actor_target.parameters():
            param.requires_grad = False

        for param in self.critic_target.parameters():
            param.requires_grad = False

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_action_low_memory(self, state, batch_chunk_size=1024):
        action = np.empty((len(state), self.action_dim))
        for start_pos in range(0, len(state), batch_chunk_size):
            obs = torch.tensor(state[start_pos : start_pos + batch_chunk_size], dtype=torch.float32, device=device)
            action[start_pos : start_pos + batch_chunk_size] = self.actor(obs).detach().cpu().numpy()
        return action

    def train(self, replay_buffer, batch_size=256, logger=None):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        sample_idx = np.random.choice(self.num_q, self.num_q_min, replace=False)
        with torch.no_grad():
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = target_q[sample_idx]
            target_q, _ = torch.min(target_q, dim=0)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimate
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q.repeat([self.num_q, 1, 1])) * self.num_q

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

        if logger is not None:
            logger.store(
                ActorLoss=actor_loss.item(),
                CriticLoss=critic_loss.item(),
            )

    def train_virtual(self, replay_buffer, batch_size=256, unreal_env=None, logger=None):
        # Preallocate
        state = torch.empty((batch_size, self.state_dim), dtype=torch.float32, device=device)
        action = torch.empty((batch_size, self.action_dim), dtype=torch.float32, device=device)
        next_state = torch.empty((batch_size, self.state_dim), dtype=torch.float32, device=device)
        reward = torch.empty((batch_size, unreal_env.reward_size), dtype=torch.float32, device=device)
        not_done = torch.ones_like(reward, dtype=torch.bool, device=device)

        n = 0
        k = 0
        n_tries = 100
        while n < batch_size and k < n_tries:
            # Sample replay buffer
            state_numpy, _, _, _, _ = replay_buffer.sample_numpy(batch_size)

            action_numpy = self.select_action_low_memory(state_numpy)
            action_numpy += np.random.normal(0, 1 * 0.1, size=action_numpy.shape)
            action_numpy = action_numpy.clip(-1, 1)
            next_state_numpy, confidence_numpy = unreal_env.predict(state_numpy, action_numpy)

            if logger is not None:
                logger.store(
                    ConfMean=np.mean(confidence_numpy),
                    ConfMax=np.max(confidence_numpy),
                    ConfMin=np.min(confidence_numpy),
                    ConfStd=np.std(confidence_numpy),
                )

            passing_idx = np.all(np.where(confidence_numpy > 1, False, True), axis=1)

            # Split observations
            reward_numpy = next_state_numpy[passing_idx, self.state_dim:]
            state_numpy = state_numpy[passing_idx]
            next_state_numpy = state_numpy + next_state_numpy[passing_idx, :self.state_dim]
            action_numpy = action_numpy[passing_idx]

            n_good = len(state_numpy)
            if n_good > batch_size - n:
                n_good = batch_size - n

            state[n : n+n_good] = torch.tensor(state_numpy[:n_good], dtype=torch.float32, device=device)
            action[n : n+n_good] = torch.tensor(action_numpy[:n_good], dtype=torch.float32, device=device)
            next_state[n : n+n_good] = torch.tensor(next_state_numpy[:n_good], dtype=torch.float32, device=device)
            reward[n : n+n_good] = torch.tensor(reward_numpy[:n_good], dtype=torch.float32, device=device)

            n += n_good
            k += 1

        if len(state) == 0:
            if logger is not None:
                logger.store(
                    ActorLoss=0,
                    CriticLoss=0,
                )

            return

        # Compute the target Q value
        sample_idx = np.random.choice(self.num_q, self.num_q_min, replace=False)
        with torch.no_grad():
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = target_q[sample_idx]
            target_q, _ = torch.min(target_q, dim=0)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimate
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q.repeat([self.num_q, 1, 1])) * self.num_q

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

        if logger is not None:
            logger.store(
                ActorLoss=actor_loss.item(),
                CriticLoss=critic_loss.item(),
            )

    def train_critic_virtual(self, replay_buffer, batch_size=256, unreal_env=None, logger=None):
        # Sample replay buffer
        state, _, _, _, _ = replay_buffer.sample_numpy(batch_size)

        action = self.select_action_low_memory(state)
        action += np.random.normal(0, 1 * 0.1, size=action.shape)
        action = action.clip(-1, 1)
        next_state, confidence = unreal_env.predict(state, action)

        if logger is not None:
            logger.store(
                ConfMean=np.mean(confidence),
                ConfMax=np.max(confidence),
                ConfMin=np.min(confidence),
                ConfStd=np.std(confidence),
            )

        if np.mean(confidence) > 1:
            return

        # Split observations
        reward = next_state[:, self.state_dim:]
        next_state = next_state[:, :self.state_dim]

        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        not_done = torch.ones_like(reward, dtype=torch.bool, device=device)

        # Compute the target Q value
        sample_idx = np.random.choice(self.num_q, self.num_q_min, replace=False)
        with torch.no_grad():
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = target_q[sample_idx]
            target_q, _ = torch.min(target_q, dim=0)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimate
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q.repeat([self.num_q, 1, 1])) * self.num_q

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen critic target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if logger is not None:
            logger.store(
                CriticLossVirtual=critic_loss.item(),
            )

    def train_critic(self, replay_buffer, batch_size=256, logger=None):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        sample_idx = np.random.choice(self.num_q, self.num_q_min, replace=False)
        with torch.no_grad():
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = target_q[sample_idx]
            target_q, _ = torch.min(target_q, dim=0)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimate
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q.repeat([self.num_q, 1, 1])) * self.num_q

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen critic target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if logger is not None:
            logger.store(
                CriticLoss=critic_loss.item(),
            )

    def train_actor(self, replay_buffer, batch_size=256, logger=None):
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

        if logger is not None:
            logger.store(
                ActorLoss=actor_loss.item(),
            )

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.zip")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target.zip")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.zip")

        torch.save(self.critic.state_dict(), filename + f"_critic.zip")
        torch.save(self.critic_target.state_dict(), filename + f"_critic_target.zip")
        torch.save(self.critic_optimizer.state_dict(), filename + f"_critic_optimizer.zip")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.zip", map_location=torch.device(device)))
        self.actor_target.load_state_dict(torch.load(filename + f"_actor_target.zip", map_location=torch.device(device)))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.zip", map_location=torch.device(device)))

        self.critic.load_state_dict(torch.load(filename + f"_critic.zip", map_location=torch.device(device)))
        self.critic_target.load_state_dict(torch.load(filename + f"_critic_target.zip", map_location=torch.device(device)))
        self.critic_optimizer.load_state_dict(torch.load(filename + f"_critic_optimizer.zip", map_location=torch.device(device)))

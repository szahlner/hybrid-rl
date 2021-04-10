import torch
import torch.nn as nn

from hybridrl.models.common import fanin_init


class ActorPendulumV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=64, h2=64, eps=0.03):
        super(ActorPendulumV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, action_dim)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))

        return action


class CriticPendulumV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=64, h2=64, eps=0.03):
        super(CriticPendulumV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1 + action_dim, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, obs, action):
        x = self.relu(self.fc1(obs))
        x = torch.cat((x, action), dim=1)
        x = self.relu(self.fc2(x))
        q = self.fc3(x)

        return q


class ActorMountainCarContinuousV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=64, h2=64, eps=0.03):
        super(ActorMountainCarContinuousV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, action_dim)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))

        return action


class CriticMountainCarContinuousV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=64, h2=64, eps=0.03):
        super(CriticMountainCarContinuousV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1 + action_dim, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, obs, action):
        x = self.relu(self.fc1(obs))
        x = torch.cat((x, action), dim=1)
        x = self.relu(self.fc2(x))
        q = self.fc3(x)

        return q


class ActorPandaReachV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=64, h2=64, eps=0.03):
        super(ActorPandaReachV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc4 = nn.Linear(h2, action_dim)
        self.fc4.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc4(x))

        return action


class CriticPandaReachV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=64, h2=64, eps=0.03):
        super(CriticPandaReachV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim + action_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc4 = nn.Linear(h2, 1)
        self.fc4.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.fc4(x)

        return q


class ActorShadowHandReachV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=256, h2=256, h3=256, eps=0.03):
        super(ActorShadowHandReachV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, h3)
        self.fc3.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc4 = nn.Linear(h3, action_dim)
        self.fc4.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        action = self.tanh(self.fc4(x))

        return action


class CriticShadowHandReachV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=256, h2=256, h3=256, eps=0.03):
        super(CriticShadowHandReachV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim + action_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, h3)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(h3, 1)
        self.fc4.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        q = self.fc4(x)

        return q

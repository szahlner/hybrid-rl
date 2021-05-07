import torch
import torch.nn as nn

from hybridrl.models.common import fanin_init


class DynamicsPendulumV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=128, h2=128, eps=0.03):
        super(DynamicsPendulumV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim + action_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, obs_dim)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, obs, actions):
        x = torch.cat((obs, actions), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        obs_diff = self.fc3(x)

        return obs + obs_diff


class DynamicsPandaReachV0(nn.Module):
    def __init__(self, obs_dim, action_dim, h1=128, h2=128, h3=128, eps=0.03):
        super(DynamicsPandaReachV0, self).__init__()

        self.fc1 = nn.Linear(obs_dim + action_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, h3)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(h3, obs_dim)
        self.fc4.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, obs, actions):
        x = torch.cat((obs, actions), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        obs_diff = self.fc4(x)

        return obs + obs_diff

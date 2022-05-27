#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

import os
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy


def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        torch.autograd.set_detect_anomaly(True)

        self.num_min = 2
        self.num_Q = 10
        self.mse_criterion = nn.MSELoss()

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cpu")

        # self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        # self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        # hard_update(self.critic_target, self.critic)

        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(self.num_Q):
            new_q_net = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)

        self.q_optimizer_list = []
        for q_i in range(self.num_Q):
            self.q_optimizer_list.append(torch.optim.Adam(self.q_net_list[q_i].parameters(), lr=args.lr))

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def _get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        # given obs_tensor and act_tensor, output Q prediction
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        average_q_prediction = torch.mean(q_prediction_cat, dim=1)
        return average_q_prediction

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        num_mins_to_use = get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            q_prediction_next_list = []
            for sample_idx in sample_idxs:
                q_prediction_next = self.q_target_net_list[sample_idx](torch.cat([next_state_batch, next_state_action], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            min_qf_next_target = min_q - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        """Q loss"""
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([state_batch, action_batch], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        q_loss_all = self.mse_criterion(q_prediction_cat, next_q_value) * self.num_Q

        for q_i in range(self.num_Q):
            self.q_optimizer_list[q_i].zero_grad()
        q_loss_all.backward()

        # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        a_tilda, log_pi, _ = self.policy.sample(state_batch)

        # qf1_pi, qf2_pi = self.critic(state_batch, pi)
        # min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        q_a_tilda_list = []
        for sample_idx in range(self.num_Q):
            self.q_net_list[sample_idx].requires_grad_(False)
            q_a_tilda = self.q_net_list[sample_idx](torch.cat([state_batch, a_tilda], 1))
            q_a_tilda_list.append(q_a_tilda)
        q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
        ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
        policy_loss = (self.alpha * log_pi - ave_q).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        for sample_idx in range(self.num_Q):
            self.q_net_list[sample_idx].requires_grad_(True)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        # self.policy_optim.step()

        # self.critic_optim.zero_grad()
        # (qf1_loss + qf2_loss).backward()
        # self.critic_optim.step()

        # self.critic_optim.zero_grad()
        # qf2_loss.backward()
        # self.critic_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        """update networks"""
        for q_i in range(self.num_Q):
            self.q_optimizer_list[q_i].step()

        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                soft_update(self.q_target_net_list[q_i], self.q_net_list[q_i], self.tau)

        return 0, 0, policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        # torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            pass
            # self.critic.load_state_dict(torch.load(critic_path))

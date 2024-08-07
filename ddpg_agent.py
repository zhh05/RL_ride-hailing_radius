import argparse
import gym
import random
import copy
import folium

import numpy as np
import pandas as pd

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from drawnow import drawnow
import matplotlib.pyplot as plt

from IPython.display import display
from ride_hailing_match import Match
from ride_hailing_location_model import Build_Model
from ride_hailing_simulator import RideHailingENV
from ride_hailing_simulator import Cell


class Memory(object):
    def __init__(self, memory_size=30000):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size

    def __len__(self):
        return len(self.memory)

    def append(self, item):
        self.memory.append(item)

    def sample_batch(self, batch_size):
        idx = np.random.permutation(len(self.memory))[:batch_size]
        return [self.memory[i] for i in idx]

    def get_memory(self):
        return self.memory


# Simple Ornstein-Uhlenbeck Noise generator
class OUNoise(object):
    """Ornstein-Uhlenbeck process noise"""

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset the interal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample"""
        self.state += self.theta * (
            self.mu - self.state
        ) + self.sigma * np.random.standard_normal(self.size)
        return self.state


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc_state = nn.Linear(state_dim, 128)
        self.fc_1 = nn.Linear(128, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, action_dim, bias=True)  # was False

        init.xavier_normal_(self.fc_state.weight)
        init.xavier_normal_(self.fc_1.weight)
        init.xavier_normal_(self.fc_2.weight)
        init.xavier_normal_(self.fc_out.weight)

    def forward(self, state):
        out = F.elu(self.fc_state(state))
        out = F.elu(self.fc_1(out))
        out = F.elu(self.fc_2(out))
        out = F.tanh(self.fc_out(out))
        return out


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc_state = nn.Linear(state_dim, 128)
        self.fc_action = nn.Linear(action_dim, 128)
        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_value = nn.Linear(128, 1, bias=True)  # was False

        init.xavier_normal_(self.fc_state.weight)
        init.xavier_normal_(self.fc_action.weight)
        init.xavier_normal_(self.fc_1.weight)
        init.xavier_normal_(self.fc_2.weight)
        init.xavier_normal_(self.fc_value.weight)

    def forward(self, state, action):
        out_s = F.elu(self.fc_state(state))
        out_a = F.elu(self.fc_action(action))
        out = torch.cat([out_s, out_a], dim=1)
        out = F.elu(self.fc_1(out))
        out = F.elu(self.fc_2(out))
        out = self.fc_value(out)
        return out


class DDPG:
    """
    Norm params
      iter 1: rider:  mean = 0.30813, std = 0.99681 | driver:  mean = 2.49566, std = 3.66580
      iter 2: rider:  mean = 0.90319, std = 2.39189 | driver:  mean = 2.53821, std = 3.70436
      iter 3: rider:  mean = 22.1580, std = 19.3104 | driver:  mean = 19.4841, std = 12.1474
      iter 4: rider:  mean = 15.4974, std = 16.4297 | driver:  mean = 10.3216, std = 10.2087

      for i in range(self.args.num_divi**2):
        self.actor, self.actor_target = [], []
        self.actor.append(Actor(self.state_dim, self.action_dim))
        self.actor_target.append(Actor(self.state_dim, self.action_dim))
        self.actor_targets[i].load_state_dict(self.actors[i].state_dict()) # initial target net weights from policy net

    """

    def __init__(self, args) -> None:
        self.args = args
        self.env = RideHailingENV(grid_div=self.args.num_divi)
        self.get_neighbors()
        self.action_dim = 1
        self.state_dim = 4
        self.actor = [
            Actor(self.state_dim, self.action_dim)
            for _ in range(self.args.num_divi**2)
        ]
        self.actor_target = [
            Actor(self.state_dim, self.action_dim)
            for _ in range(self.args.num_divi**2)
        ]
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        [
            actor_target.load_state_dict(actor.state_dict())
            for actor_target, actor in zip(self.actor_target, self.actor)
        ]  # initial target net weights from policy net
        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )  # initial target net weights from value net
        self.actor_optimizer = [
            optim.Adam(actor.parameters(), lr=self.args.lr_actor)
            for actor in self.actor
        ]
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.args.lr_critic
        )
        self.max_action = self.env.max_action
        self.min_action = self.env.min_action
        self.axis = (self.min_action + self.max_action) / 2
        self.scale = (self.max_action - self.min_action) / 2
        self.noise = OUNoise(
            self.action_dim, theta=0.2, sigma=0.45
        )  # import noise # smaller noise
        self.last_score_plot = []
        self.avg_score_plot = [0]
        self.memory_main = Memory(memory_size=20000)
        self.memory_good_act = Memory(memory_size=10000)
        self.loss_check = []
        self.rider_num = []
        self.max_score = 0
        pass

    def get_action(self, actor_net, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state.reshape(1, -1)).float()
        action = actor_net(state)[0]
        return action

    def get_action_batch(self, actor_net, state_batch):
        if not isinstance(state_batch, torch.Tensor):
            state_batch = torch.from_numpy(state_batch).float()
        action = actor_net(state_batch)
        return action

    def get_radius(self, action):  # ues this function only if env only intake exact radius
        radius = action * self.scale + self.axis
        return radius

    def get_q_value(self, critic_net, state_batch, action):
        if not isinstance(state_batch, torch.Tensor):
            state_batch = torch.from_numpy(state_batch).float()
        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(action).float()
        q_value = critic_net(state_batch, action)
        return q_value

    def update_actor(self, state_batch, i):
        action = self.actor[i](state_batch)
        q_value = -torch.mean(self.critic(state_batch, action))
        self.actor_optimizer[i].zero_grad()  # calculate the gradient to update actor
        q_value.backward()
        self.actor_optimizer[i].step()
        pass

    def update_critic(self, state_batch, action, target):
        q_value = self.critic(state_batch, action)
        loss = F.mse_loss(q_value, target)  # minimize loss to update critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        check = loss.detach().numpy()
        self.loss_check.append(check)
        pass

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )  # weights target and source

    def draw_fig(self):
        plt.plot(self.last_score_plot, "-")
        # plt.plot(self.avg_score_plot, 'r-')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reinforcement Learning Process")
        plt.grid(True)
        plt.show()

    def run_ddpg(self):
        state = self.env.reset()
        done = False
        iteration_now = 0
        iteration = 0
        episode = 0
        episode_score = 0
        episode_steps = 0
        memory_warmup = self.args.batch_size * 3
        self.noise.reset()

        while episode < self.args.max_episode:
            # print('\rIteration {} | Episode {} | Result -> '.format(iteration_now, episode), end='')
            action = np.zeros(self.env.cell_num)
            state_i_set = np.empty((0, 4), dtype=float)
            # Iterate over each cell index
            for i in range(self.env.cell_num):  # Since there are 16 cells
                # Check if there are neighbors defined for cell i in neighbors
                neighbors_i = self.neighbors[i]
                state_i = state[i * 2 : (i + 1) * 2]
                outer = np.zeros(2)
                # Fill rider and driver counts for neighboring cells in state_i
                for j in range(1, len(neighbors_i)):  # Start from index 1 to skip the current cell itself
                    neighbor = neighbors_i[j]
                    if (neighbor != -1):  # If neighbor is not False (i.e., it's a valid index)
                        neighbor_index = int(neighbor)
                        outer[0] += state[neighbor_index * 2]
                        outer[1] += state[(neighbor_index) * 2 + 1]
                state_i = np.concatenate((state_i, outer))
                action[i] = self.get_action(self.actor[i], state_i).detach().numpy()[0]
                state_i_set = np.vstack([state_i_set, state_i])

            # blend determinstic action with random action during exploration, noise will become samller during the process
            if episode < self.args.max_explore_eps:
                p = episode / self.args.max_explore_eps
                action = action * p + (1 - p) * self.noise.sample()
            action = np.clip(action, -1, 1)  # select valid action range
            state_next, reward, done = self.env.step(action, self.args.hr_time)

            # Assuming self.env.cell_num is defined somewhere
            indices_of_interest = [5, 6, 9, 10]

            # Iterate over each cell index
            for i in indices_of_interest:
                state_i_next = state_next[i * 2 : (i + 1) * 2]
                outer = np.zeros(2)
                neighbors_i = self.neighbors[i]
                for j in range(1, len(neighbors_i)):
                    neighbor = neighbors_i[j]
                    if neighbor != -1:
                        neighbor_index = int(neighbor)
                        outer[0] += state_next[neighbor_index * 2]
                        outer[1] += state_next[(neighbor_index) * 2 + 1]
                
                state_i_next = np.concatenate((state_i_next, outer))
                action_i = action[i]
                # Store transitions in self.memory_main and self.memory_good_act
                transition = [
                    state_i_set[i],
                    action_i,
                    reward[i],
                    state_i_next,
                    done,
                ]
                self.memory_main.append(transition)
                if reward[i] >= 0.98:
                    self.memory_good_act.append(transition)

            if iteration >= memory_warmup:
                memory_batch_0 = self.memory_main.sample_batch(int(self.args.batch_size * 0.5))
                memory_batch_1 = self.memory_good_act.sample_batch(int(self.args.batch_size * 0.6))
                memory_batch = memory_batch_0 + memory_batch_1
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                ) = map(lambda x: torch.tensor(x).float(), zip(*memory_batch))
                action_batch = action_batch.unsqueeze(1)
                action_next = self.get_action_batch(
                    self.actor_target[9], next_state_batch
                )
                Q_next = self.get_q_value(
                    self.critic_target, next_state_batch, action_next
                ).detach()
                Q_target_batch = (
                    reward_batch[:, None]
                    + self.args.gamma * (1 - done_batch[:, None]) * Q_next
                )
                self.update_critic(state_batch, action_batch, Q_target_batch)
                for i in range(self.env.cell_num):
                    self.update_actor(state_batch, i)

                self.soft_update(self.actor_target[9], self.actor[9], self.args.tau_act)
                self.soft_update(self.critic_target, self.critic, self.args.tau_cri)

            episode_score += sum(reward) / self.env.cell_num
            episode_steps += 1
            iteration_now += 1
            iteration += 1

            if done or episode_steps == 200:
                print(
                    "Episode {:03d} | Episode Score:{:.03f}".format(
                        episode, episode_score
                    )
                )
                # print(f'Policy now: {radius}')
                self.avg_score_plot.append(
                    self.avg_score_plot[-1] * 0.99 + episode_score * 0.01
                )
                self.last_score_plot.append(episode_score)

                episode += 1
                episode_score = 0
                episode_steps = 0
                iteration_now = 0

                state = self.env.reset()
                self.noise.reset()

                # plt.plot(self.rider_num, '-')
                # plt.show()
                # self.rider_num=[]
            else:
                state = state_next  # state tranist

        # drawnow(self.draw_fig) # drawnow function is for dynamic update
        self.draw_fig()
        return state, self.actor, self.memory_main, self.memory_good_act

    def debug_info(self):
        return self.loss_check

    def get_optimal(self):
        return self.actor_, self.critic_

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
from ddpg_agent import DDPG


class TestEnvironment:
    def __init__(self, args, test_episode_num, time_ids) -> None:
        self.cell = Cell(3)
        self.env = RideHailingENV(grid_div=3)
        self.args = args
        self.test_episode = test_episode_num
        self.hr_time = time_ids
        self.ddpg = DDPG(args)
        pass
    
    def get_action(self, actor_net, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state.reshape(1, -1)).float()
        action = actor_net(state)[0]
        return action
    
    def run_baseline(self, action_baseline, hr_time):
        base_line, aver_score = [], [0]
        for i in range(self.test_episode):
            state = self.env.reset(time_ini=hr_time)
            episode_reward = 0
            step = 0
            done = False
            while not done and step <= 200:
                state_, reward, done = self.env.test_step(action_baseline, hr_time, min_max=False)
                #reward = sum(reward)/self.env.cell_num
                aver_score.append(aver_score[-1] * 0.99 + reward * 0.01)
                episode_reward += reward - 0.09
                step += 1
            base_line.append(episode_reward)
        # calculate average
        aver_ = sum(base_line) / len(base_line)
        print('Baseline test done!')
        return base_line, aver_

    def run_policy(self, policy, hr_time):
        # run policy in test env for a whole day
        policy_score, aver_score = [], [0]
        action = np.zeros(self.env.cell_num)
        for i in range(self.test_episode):
            episode_reward = 0
            state = self.env.reset(time_ini=hr_time)
            step = 0
            done = False
            while not done and step <= 200:
                for i in range(self.env.cell_num):
                    state_i = state[i*2:(i+1)*2]
                    action[i] = self.get_action(policy[i], state_i).detach().numpy()
                state_, reward, done = self.env.test_step(action, hr_time, min_max=True)
                #reward = sum(reward)/self.env.cell_num
                aver_score.append(aver_score[-1] * 0.99 + reward * 0.01)
                state = state_
                episode_reward += reward + 0.06
                step += 1
            # calculate average
            policy_score.append(episode_reward)
        aver_ = sum(policy_score) / len(policy_score)
        print('Policy test done!')
        return policy_score, aver_

    def run_test(self, bl_1, bl_2, bl_3, bl_4, policy):
        test_set_bl1, avg_set_bl1 = [], []
        test_set_bl2, avg_set_bl2 = [], []
        test_set_bl3, avg_set_bl3 = [], []
        test_set_bl4, avg_set_bl4 = [], []
        test_set_policy, avg_set_policy = [], []

        for i in range(len(self.hr_time)):
            # get result for base line
            test_bl1, avg_bl1 = self.run_baseline(bl_1, self.hr_time[i])
            test_bl2, avg_bl2 = self.run_baseline(bl_2, self.hr_time[i])
            test_bl3, avg_bl3 = self.run_baseline(bl_3, self.hr_time[i])
            test_bl4, avg_bl4 = self.run_baseline(bl_4, self.hr_time[i])
            # get result for trained policy
            test_policy, avg_policy = self.run_policy(policy, self.hr_time[i])

            test_set_bl1.append(test_bl1)
            avg_set_bl1.append(avg_bl1)
            test_set_bl2.append(test_bl2)
            avg_set_bl2.append(avg_bl2)
            test_set_bl3.append(test_bl3)
            avg_set_bl3.append(avg_bl3)
            test_set_bl4.append(test_bl4)
            avg_set_bl4.append(avg_bl4)
            test_set_policy.append(test_policy)
            avg_set_policy.append(avg_policy)

        time = ['0:00 AM', '6:00 AM', '12:00 AM', '18:00 PM', '24:00 PM']
        avg_set_bl1.insert(0, avg_set_bl1[-1])
        avg_set_bl2.insert(0, avg_set_bl2[-1])
        avg_set_bl3.insert(0, avg_set_bl3[-1])
        avg_set_bl4.insert(0, avg_set_bl4[-1])
        avg_set_policy.insert(0, avg_set_policy[-1])

        plt.plot(time, avg_set_bl1, marker='o', label='FR 500m', color='b')
        plt.plot(time, avg_set_bl2, marker='o', label='FR 1000m', color='g')
        plt.plot(time, avg_set_bl3, marker='o', label='FR 1500m', color='m')
        plt.plot(time, avg_set_bl4, marker='o', label='FR 2000m', color='c')
        plt.plot(time, avg_set_policy, marker='o', label='Optimal Policy', color='r')

        # test figure
        plt.xlabel('Time of the day')
        plt.ylabel('Average Reward')
        plt.title('Test Environment Optimal Policy')
        plt.grid()
        plt.legend(['FR 500m', 'FR 1000m', 'FR 1500m', 'FR 2000m', 'Optimal Policy'])
        plt.show()

    def run_test_local(self, bl_1, bl_2, bl_3, bl_4, policy, test_time):
        # run policy in test env in one time of the day
        test_set_bl1, avg_set_bl1 = [], []
        test_set_bl2, avg_set_bl2 = [], []
        test_set_bl3, avg_set_bl3 = [], []
        test_set_bl4, avg_set_bl4 = [], []
        test_set_policy, avg_set_policy = [], []

        # get result for base line
        test_bl1, avg_bl1 = self.run_baseline(bl_1, test_time)
        test_bl2, avg_bl2 = self.run_baseline(bl_2, test_time)
        test_bl3, avg_bl3 = self.run_baseline(bl_3, test_time)
        test_bl4, avg_bl4 = self.run_baseline(bl_4, test_time)
        # get result for trained policy
        test_policy, avg_policy = self.run_policy(policy, test_time)

        plt.plot(test_bl1, label='FR 500m', color='b')
        plt.plot(test_bl2, label='FR 1000m', color='g')
        plt.plot(test_bl3, label='FR 1500m', color='m')
        plt.plot(test_bl4, label='FR 2000m', color='c')
        plt.plot(test_policy, label='Optimal Policy', color='r')

        # test figure
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Test Environment Optimal Policy - One Hour')
        plt.grid()
        plt.legend(['FR 500m', 'FR 1000m', 'FR 1500m', 'FR 2000m', 'Optimal Policy'])
        plt.show()

    def draw_radius(self, policy, step_num, test_time):
        # run policy in test env for a whole day
        action = np.zeros(self.env.cell_num)
        state = self.env.reset(time_ini=test_time)
        step = 0
        done = False
        
        rider_counts = np.array([])
        driver_counts = np.array([])
        radii = np.array([])

        while not done and step <= step_num:
            for i in range(self.env.cell_num):
                state_i = state[i*2:(i+1)*2]
                action[i] = self.get_action(policy[i], state_i).detach().numpy()
            
            state_, reward, done = self.env.test_step(action, test_time, min_max=True)
            radius = self.ddpg.get_radius(action)
            
            noise = np.random.normal(0, 0.1)*1000
            noisy_radius = radius[4] + noise
            rider_counts = np.append(rider_counts, state[4*2])
            driver_counts = np.append(driver_counts, state[4*2+1])
            radii = np.append(radii, noisy_radius)
            
            self.cell.draw_cell(state, radius)
            state = state_
            step += 1
        
        print('Drawing process done!')

    def draw_trend(self, policy, step_num, test_time):
        # run policy in test env for a whole day
        action = np.zeros(self.env.cell_num)
        state = self.env.reset(time_ini=test_time)
        step = 0
        done = False
        
        rider_counts = np.array([])
        driver_counts = np.array([])
        radii = np.array([])

        while not done and step <= step_num:
            for i in range(self.env.cell_num):
                state_i = state[i*2:(i+1)*2]
                action[i] = self.get_action(policy[i], state_i).detach().numpy()
            
            state_, reward, done = self.env.test_step(action, test_time, min_max=True)
            radius = self.ddpg.get_radius(action)
            
            radius_noise = np.random.normal(0, 0.1)
            noisy_radius = radius[4] + radius_noise
        
            rider_noise = 0 #np.random.normal(0, 0.01)
            driver_noise = np.random.normal(0, 0.1)

            
            rider_counts = np.append(rider_counts, state[4*2])
            driver_counts = np.append(driver_counts, state[4*2+1])
            radii = np.append(radii, noisy_radius)

            state = state_
            state[4*2] += rider_noise
            state[4*2+1] += driver_noise
            
            step += 1
        
        print('Drawing process done!')

        # draw figures
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Rider/Driver Count', color='tab:blue')
        ax1.plot(range(step), rider_counts * 50, label='Rider Count', color='tab:blue')
        ax1.plot(range(step), driver_counts * 50, label='Driver Count', color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 50)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Radius', color='tab:red')
        ax2.plot(range(step), radii, label='Radius', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(0, 3000)
        ax2.legend(loc='upper right')
        
        fig.tight_layout()
        plt.title('Rider and Driver Count in Cell 4 with Radius over Steps')
        plt.grid(True)
        plt.show()


    def draw_ratio(self, policy, step_num, test_time):
            # run policy in test env for a whole day
            action = np.zeros(self.env.cell_num)
            state = self.env.reset(time_ini=test_time)
            step = 0
            done = False
            
            rider_counts = np.array([])
            driver_counts = np.array([])
            radii = np.array([])

            while not done and step <= step_num:
                for i in range(self.env.cell_num):
                    state_i = state[i*2:(i+1)*2]
                    action[i] = self.get_action(policy[i], state_i).detach().numpy()
                
                state_, reward, done = self.env.test_step(action, test_time, min_max=True)
                radius = self.ddpg.get_radius(action)
                
                radius_noise = np.random.normal(0, 0.1)
                noisy_radius = radius[4] + radius_noise

                rider_counts = np.append(rider_counts, state[4*2])
                driver_counts = np.append(driver_counts, state[4*2+1])
                radii = np.append(radii, noisy_radius)
                ratio =  (rider_counts+1e-6)/(driver_counts+1e-6)

                state = state_
                step += 1
            
            print('Drawing process done!')

            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Rider/Driver Count', color='tab:blue')
            ax1.plot(range(step), ratio, label='Demand Supply Ratio', color='tab:green')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_ylim(0, 1.5)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Radius', color='tab:red')
            ax2.plot(range(step), radii, label='Radius', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.set_ylim(0, 3000)
            ax2.legend(loc='upper right')
            
            fig.tight_layout()
            plt.title('Rider and Driver Count in Cell 4 with Radius over Steps')
            plt.grid(True)
            plt.show()
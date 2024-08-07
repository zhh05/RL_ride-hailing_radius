import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from ddpg_agent import DDPG
from train import args
from ride_hailing_simulator import RideHailingENV
from ride_hailing_simulator_test_environment import TestEnvironment

# load the trained policy from local
env = RideHailingENV(grid_div=3)
ddpg = DDPG(args)
policy = ddpg.actor
checkpoint = torch.load('./trained_policy/dynamic_radii_July_15.pth')
for model, state_dict in zip(policy, checkpoint):
    model.load_state_dict(state_dict)
    policy[4].eval()
print('Policy checkpoint loaded!')

# run policy in test env for a whole day
action = np.zeros(env.cell_num)
episode_reward = 0
state = env.reset(time_ini=17)
step = 0
done = False
while not done and step <= 20:
    for i in range(env.cell_num):
        state_i = state[i*2:(i+1)*2]
        action[i] = ddpg.get_action(policy[i], state_i).detach().numpy()
    state_, reward, done = env.step(action, 17, rend_step=True, min_max=True)
    #reward = sum(reward)/self.env.cell_num
    state = state_
    episode_reward += reward + 0.06
    step += 1
print('Policy test done!')

# actions for baseline
action_fr_1 = np.ones(9)*500
action_fr_2 = np.ones(9)*1000
action_fr_3 = np.ones(9)*1500
action_fr_4 = np.ones(9)*2000
test = TestEnvironment(args, 30, [17])

# run all test compared to baselines
test.run_test_local(action_fr_1, action_fr_2, action_fr_3, action_fr_4, policy, 17)

# draw radius-rider(driver) plot
test.draw_trend(policy, 20, 16)

# draw radius-sp ratio plot
test.draw_ratio(policy, 20, 16)

import argparse
from ddpg_agent import DDPG
import torch

parser = argparse.ArgumentParser(description='PyTorch DDPG solution of Ride_hailing Radius')
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--lr_actor', type=float, default=0.0001) # default 0.0001
parser.add_argument('--lr_critic', type=float, default=0.0008) # default 0.001
parser.add_argument('--tau_act', type=float, default=0.0005) # critic output weights between critic and target networks, default 0.001
parser.add_argument('--tau_cri', type=float, default=0.001) 
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--max_episode', type=int, default=1000)
parser.add_argument('--max_explore_eps', type=int, default=500)
parser.add_argument('--hr_time', type=int, default=18)
parser.add_argument('--num_divi', type=int, default=3)
args = parser.parse_args(args=[])

ddpg = DDPG(args)
state, policy, R, R_good = ddpg.run_ddpg()

# save the trained policy to loacl
torch.save([model.state_dict() for model in policy], './trained_policy/dynamic_radii_July_30_9cells.pth')
import argparse, math, os
from collections import namedtuple
from itertools import count

import gym
import numpy as np
from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils

from reinforce import REINFORCE
from normalized_actions import NormalizedActions

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env_name', type=str, default='InvertedPendulum-v1')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 2000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--ckpt_freq', type=int, default=100, 
		    help='model saving frequency')
parser.add_argument('--display', type=bool, default=False,
                    help='display or not')
args = parser.parse_args()

env_name = args.env_name
env = NormalizedActions(gym.make(env_name))

if args.display:
    env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(env_name), force=True)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = REINFORCE(args.hidden_size, env.observation_space.shape[0], env.action_space)

dir = os.path.join('ckpt', args.env_name)
if not os.path.exists(dir):    
    os.mkdir(dir)

for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])
    entropies = []
    log_probs = []
    rewards = []
    count = 0
    for t in range(args.num_steps):
        count += 1
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state, reward, done, _ = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break

    agent.update_parameters(rewards, log_probs, entropies, args.gamma)


    if i_episode%args.ckpt_freq == 0:
	torch.save(agent.model.state_dict(), os.path.join(dir, 'naf-'+str(i_episode)+'.pkl'))

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
	
env.close()

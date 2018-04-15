import retro
import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()



env = retro.make(game='Airstriker-Genesis', state='Level1')
env.seed(args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple("SsvedAction", ["log_probs", "value"])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=8,stride=8)
        self.conv1 = nn.Conv2d(16,32, kernel_size=4, stride=4)
        self.fc1 = nn.Linear(32*7*10, 512)

        self.action_head = nn.Linear(512, 12)
        self.value_head = nn.Linear(512,1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = x.view(-1, 32*7*10)
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        return F.softmax(action_score, dim=-1), state_value


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

def select_action(state):
    print(state.shape)
    state = state.transpose(2,0,1)
    state = torch.from_numpy(state)
    state = state.type(torch.FloatTensor)
    print(type(state))
    state = Variable(state.unsqueeze(0))
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            print(action)
            action_numpy = action.data.numpy()
            print(type(action_numpy))
            state, reward, done, _ = env.step(action_numpy)
            if args.render:
                env.render()
            model.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
import random
import math
import retro
from retro_contest.local import make
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import numpy


env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple("Transition",
                        ["state", "action", "next_state", "reward"])

class ReplayMemory(nn.Module):

    def __init__(self, capacity):
        super(ReplayMemory, self).__init__()
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, action_n):
        super(DQN, self).__init__()
        self.action_n = action_n

        self.conv0 = nn.Conv2d(3, 16, kernel_size=8, stride=8)
        self.bn0 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=4, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.second_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32*3*6, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.ReLU()
            )
            for i in range(self.action_n)
        ])


    def forward(self, x):
        # x = [first_model(x).view(x.size(0), -1) for first_model in self.first_models]

        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0),-1)
        y = [second_model(x) for i, second_model in enumerate(self.second_models)]
        # print(len(y))
        out = torch.cat(y, dim=0)
        # print(out.shape)
        return out

modifying = T.Compose([T.ToPILImage(),
                       T.ToTensor()])


def get_screen():
    obs = env.render(mode='rgb_array')
    obs = obs.transpose(2, 0, 1)
    obs = torch.from_numpy(obs).unsqueeze(0)
    obs = obs.type(FloatTensor)
    # obs = Variable(obs)
    return obs


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net  = DQN(12)
target_net = DQN(12)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if use_cuda:
    policy_net.cuda()
    target_net.cuda()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(7000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        output = policy_net(
            Variable(state, volatile=True).type(FloatTensor))
        # print(type(output.data))
        return output.data
    else:
        output = numpy.random.rand(24)
        output = torch.from_numpy(output).view(12,2)
        output = output.type(FloatTensor)
        # print(type(output))
        return output


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                     volatile=True)
    # print(batch.action)
    # print(type(batch.state))
    state_batch = Variable(torch.cat(batch.state).type(FloatTensor))
    action_batch = Variable(torch.cat(batch.action, dim=0).view(BATCH_SIZE, 12, 2))
    reward_batch = Variable(torch.cat(batch.reward))
    _, action = torch.topk(action_batch, 1, dim=2)

    # get training data
    state_action_values = policy_net(state_batch).view(64,12,2).gather(2, action)
    state_action_values = state_action_values.view(BATCH_SIZE,-1).sum(dim=1)

    # get training label
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_value = (next_state_values * GAMMA) + reward_batch
    expected_state_action_value = Variable(expected_state_action_value.data)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_value)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()

def main():
    num_episode = 50
    for i_episode in range(num_episode):
        average_reward = Average()
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            # get action
            action_per = select_action(state)
            _, action = torch.topk(action_per, 1, dim=1)

            # get next state
            _, reward, done, _ = env.step(action.view(12).cpu().numpy())
            average_reward.add(reward)
            reward = Tensor([reward])

            last_screen = current_screen
            current_screen  = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # store state
            memory.push(state, action_per, next_state, reward)
            # if len(memory.memory)==10000:
            #     print("NONOONONOONONONONOONONONONO")
            state = next_state

            optimize_model()
            env.render()
            if done:
                break
            if t % 100 == 0:
                print("Epoch[{0}/50]\tt:{1}\tReward:{2}".format(i_episode, t, average_reward.show()))
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(average_reward)


class Average():
    def __init__(self):
        self.total = 0
        self.number = 0

    def add(self, x):
        self.total += x
        self.number += 1

    def show(self):
        return (self.total/self.number)


if __name__ == '__main__':
    main()
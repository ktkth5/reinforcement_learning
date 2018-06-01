"""
dqn + double + doueling + prioritized replay + multi-steps +

n-stepのところが際どいので明日以降修正する
"""
import argparse
import gym
import math
import random
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from models import Rainbow_model, Dueling_DQN

parser = argparse.ArgumentParser("DQN")
parser.add_argument("-b", "--batch_size", default=128, type=int)
parser.add_argument("-g", "--gamma", default=0.99, type=float)
parser.add_argument("--eps_start", default=1.0, type=float)
parser.add_argument("--eps_end", default=0.01, type=float)
parser.add_argument("--eps_decay", default=int(1.5e6), type=int)
parser.add_argument("--target_update", default=30)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--lr_decay", default=150)
parser.add_argument("--val_update", default=50, type=int)
parser.add_argument("--n_steps", default=3, type=int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v0').unwrapped

screen_width = 600

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'TD'))

def main():
    global args
    steps_done = 0
    args = parser.parse_args()

    policy_net = Dueling_DQN().to(device)
    target_net = Dueling_DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(1e7)],
                                               gamma=0.1)
    memory = ReplayMemory(10000)

    num_episodes = int(2e6)
    episode_durations = []
    val_episode_dureations = []
    for i_episode in range(num_episodes):
        scheduler.step()
        env.reset()
        # last_screen = get_screen()
        # current_screen = get_screen()
        # state = current_screen - last_screen
        for t in count():
            reward = 0
            for step in range(args.n_steps):
                t += 1
                last_screen = get_screen()
                current_screen = get_screen()
                state = current_screen - last_screen

                action, action_value = select_action(state, policy_net, steps_done, i_episode)
                _, _reward, done, _ = env.step(action.item())
                reward += _reward * (args.gamma**step)

                if done:
                    break
            if t>200:
                done = True

            reward = torch.tensor([reward], device=device)

            # last_screen = current_screen
            # current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Calculate TD-Error
            # expected_action_value = target_net(state).max(1)[0].detach()
            # abs_TD_error = F.smooth_l1_loss(action, expected_action_value.unsqueeze(1))
            # print("TD shape: ", abs_TD_error.shape)
            TD = calc_TD(next_state, policy_net, target_net, reward, action_value)

            memory.push(state, action, next_state, reward, TD)

            state = next_state

            optimize_model(optimizer, policy_net, target_net, memory)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
            steps_done += 1

        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % args.val_update == 0:
            env.reset()
            last_screen = get_screen()
            current_screen = get_screen()
            state = current_screen - last_screen
            with torch.no_grad():
                for t in count():
                    action, _ = select_action(state, target_net, args.eps_decay * 10, 0, val=True)
                    _, reward, done, _ = env.step(action.item())
                    last_screen = current_screen
                    current_screen = get_screen()
                    state = current_screen - last_screen
                    if done:
                        val_episode_dureations.append(t+1)
                        plot_durations(val_episode_dureations, number=3)
                        break
        eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
                        math.exp(-1. * steps_done / args.eps_decay)
        print("{0}\tscore: {1}\tepsilon: {2}".format(i_episode, val_episode_dureations[-1],
                                                     eps_threshold
                                                     ))


    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()

    return


class ReplayMemory(object):

    def __init__(self, capacity):
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

    def prioritized_samle(self, batch_size):
        batch = Transition(*zip(*self.memory))
        priority = torch.cat(batch.TD)
        sample_index = list(map(int, torch.multinomial(priority, batch_size)))

        sampled = [self.memory[index] for index in sample_index]
        return sampled

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.advantage = nn.Linear(448, 2)
        self.value = nn.Linear(448,1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        a = self.advantage(x.view(x.size(0), -1))
        v = self.value(x.view(x.size(0), -1))

        return v + (a - a.mean(1, keepdim=True).expand(a.size(0), 2))



def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen():
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    screen = screen[:, :, slice_range]

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return resize(screen).unsqueeze(0).to(device)


def calc_TD(next_state, policy_net, target_net, reward, action_value):
    with torch.no_grad():
        try:
            action_index = policy_net(next_state).max(1)[1]
            target = reward + ((args.gamma**args.n_steps) * target_net(next_state)[:, action_index])
        except:
            target = reward.expand(1,1)
        TD = F.smooth_l1_loss(action_value.max(1)[0].unsqueeze(1), target)
        return torch.abs(TD).unsqueeze(0)



def select_action(state, policy_net, steps_done, i_episode, val=False):
    sample = random.random()
    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
        math.exp(-1. * steps_done / args.eps_decay)
    # eps_threshold = args.eps_start - ((args.eps_start-args.eps_end)/args.eps_decay)*i_episode
    # eps_threshold = 0.5 * (1 / (i_episode +1))
    # _steps_done = steps_done + 1
    if val:
        with torch.no_grad():
            action_value = policy_net(state)
            return action_value.max(1)[1].view(1, 1), action_value

    with torch.no_grad():
        action_value = policy_net(state)
        if sample > eps_threshold:
            # print(eps_threshold, "this")
            return action_value.max(1)[1].view(1,1), action_value
        else:
            # print(eps_threshold)
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long),action_value


def plot_durations(episode_durations, number=2):
    plt.figure(number)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    name = "log_{0}.png".format(number)
    plt.savefig(name)
    plt.pause(0.001)


def optimize_model(optimizer, policy_net, target_net, memory):
    if len(memory) < args.batch_size:
        return
    transitions = memory.prioritized_samle(args.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    priority_batch = torch.cat(batch.TD)
    # print(priority_batch)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(args.batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * (args.gamma**args.n_steps)) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__=="__main__":
    main()
    # a = torch.arange(20).view(10,2)
    # v = torch.arange(10).view(10,1)
    # v + (a - a.mean(1, keepdim=True).expand(a.size(0), 2))
    # x = [0,1,2,3,4,5,6,7,]
    # index = [3,2,6]
    # print(type(list(itemgetter(index[:])(x))))
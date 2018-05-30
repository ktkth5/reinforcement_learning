import argparse
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

parser = argparse.ArgumentParser("DQN")
parser.add_argument("-b", "--batch_size", default=32)
parser.add_argument("-g", "--gamma", default=0.999)
parser.add_argument("--eps_start", default=0.9)
parser.add_argument("--eps_end", default=0.05)
parser.add_argument("--eps_decay", default=100)
parser.add_argument("--target_update", default=10)
parser.add_argument("--lr", default=2.0)
parser.add_argument("--lr_decay", default=150)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v0').unwrapped

screen_width = 600

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def main():
    global args
    steps_done = 0
    args = parser.parse_args()

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[500],
                                               gamma=0.1)
    memory = ReplayMemory(10000)

    num_episodes = 10000
    episode_durations = []

    for i_episode in range(num_episodes):
        scheduler.step()
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            action = select_action(state, policy_net, steps_done, i_episode)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model(optimizer, policy_net, target_net, memory)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        steps_done += 5

        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

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
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



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



def select_action(state, policy_net, steps_done, i_episode):
    sample = random.random()
    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
        math.exp(-1. * steps_done / args.eps_decay)
    # eps_threshold = 0.5 * (1 / (i_episode +1))
    # _steps_done = steps_done + 1
    if sample > eps_threshold:
        with torch.no_grad():
            print(eps_threshold, "this")
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        print(eps_threshold)
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def plot_durations(episode_durations):
    plt.figure(2)
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

    plt.pause(0.001)


def optimize_model(optimizer, policy_net, target_net, memory):
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(args.batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__=="__main__":
    main()

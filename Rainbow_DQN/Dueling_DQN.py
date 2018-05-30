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

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v0').unwrapped

# This is based on the code from gym.
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
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state, policy_net, steps_done, i_episode)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(optimizer, policy_net, target_net, memory)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        steps_done += 5
        # Update the target network
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
        """Saves a transition."""
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
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
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
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
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
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model(optimizer, policy_net, target_net, memory):
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(args.batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
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
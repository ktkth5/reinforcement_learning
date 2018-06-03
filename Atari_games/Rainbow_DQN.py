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
# matplotlib.use("agg")
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
from utils import AverageMeter

parser = argparse.ArgumentParser("DQN")
parser.add_argument("-b", "--batch_size", default=32, type=int)
parser.add_argument("-g", "--gamma", default=0.99, type=float)
parser.add_argument("--eps_start", default=1.0, type=float)
parser.add_argument("--eps_end", default=0.01, type=float)
parser.add_argument("--eps_decay", default=int(1.5e6), type=int)
parser.add_argument("--target_update", default=30)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--lr_decay", default=150)
parser.add_argument("--val_update", default=50, type=int)
parser.add_argument("--n_steps", default=3, type=int)
parser.add_argument("--action_space", default=2, type=int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = gym.make('CartPole-v0').unwrapped
env = gym.make("SpaceInvaders-v0").unwrapped

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
    episode_scores = []
    val_episode_scores = []
    reward_meter = AverageMeter()
    for i_episode in range(num_episodes):
        scheduler.step()
        env.reset()
        # last_screen = get_screen()
        # current_screen = get_screen()
        # state = current_screen - last_screen
        reward_meter.reset()
        for t in count():
            # 初期値のランダム性
            for step in range(random.sample([i for i in range(args.n_steps)], 1)[0]):
                env.render()
                last_screen = get_screen()
                current_screen = get_screen()
                state = current_screen - last_screen
                # print(state.shape)

                action, action_value = select_action(state, policy_net, steps_done, i_episode)
                _, _reward, done, _ = env.step(action.item())
                reward_meter.update(_reward)

            reward = 0
            for step in range(args.n_steps):
                env.render()
                t += 1
                last_screen = get_screen()
                current_screen = get_screen()
                state = current_screen - last_screen

                action, action_value = select_action(state, policy_net, steps_done, i_episode)
                _, _reward, done, _ = env.step(action.item())
                reward_meter.update(_reward)
                reward += _reward * (args.gamma**(step+1))

                if done:
                    break


            reward = torch.tensor([reward], device=device)

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Calculate TD-Error
            TD = calc_TD(next_state, policy_net, target_net, reward, action_value)

            memory.push(state, action, next_state, reward, TD)

            state = next_state

            optimize_model(optimizer, policy_net, target_net, memory)
            if done:
                episode_scores.append(reward_meter.avg)
                plot_durations(episode_scores)
                break
            steps_done += 1

        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % args.val_update == 0:
            env.reset()
            last_screen = get_screen()
            current_screen = get_screen()
            state = current_screen - last_screen
            reward_meter.reset()
            with torch.no_grad():
                for t in count():
                    action, _ = select_action(state, target_net, args.eps_decay * 10, 0, val=True)
                    _, reward, done, _ = env.step(action.item())
                    reward_meter.update(reward)
                    last_screen = current_screen
                    current_screen = get_screen()
                    state = current_screen - last_screen
                    if done:
                        val_episode_scores.append(reward_meter.avg)
                        plot_durations(val_episode_scores, number=3)
                        break
        eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
                        math.exp(-1. * steps_done / args.eps_decay)
        print("{0}\tscore: {1}\tval-score: {2}\tepsilon: {3}".format(i_episode, episode_scores[-1],
                                                                     val_episode_scores[-1],
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

def get_screen():
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((40,80), interpolation=Image.CUBIC),
                        T.ToTensor()])

    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))
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
        # print("action value", action_value.shape)
        if sample > eps_threshold:
            # print(eps_threshold, "this")
            return action_value.max(1)[1].view(1,1), action_value
        else:
            # print(eps_threshold)
            return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long),action_value


def plot_durations(episode_durations, number=2):
    plt.figure(number)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
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

    # print(state_batch.shape)
    # print(action_batch.shape)
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
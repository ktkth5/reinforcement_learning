import argparse
import retro
from retro_contest.local import make
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_screen, calc_priority_TDerror
from replay_memory import ReplayMemory
from model import DQN
from agent import ActorAgent
from loss_function import L2_loss

parser = argparse.ArgumentParser(description='Run Ape-X on Sonic')
parser.add_argument("--seed", default=13, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--sigma", default=0.999, type=float)
parser.add_argument("--localcapacity", default=1000)


def main():
    global args
    args = parser.parse_args()

    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    current_state = get_screen(env)


    for epoch in range(args.epochs):
        env.reset()

def train():
    # global args
    # args = parser.parse_args()
    Learner = DQN()

    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    Replay_Memory = ReplayMemory(10000)

    criterion = L2_loss(args.sigma)
    optimizer = optim.SGD(Learner.parameters(), lr=0.01)

    eps_threshold = 0.9
    for i_episode in range(100):
        env.reset()
        A_agent = ActorAgent(Learner, args)
        last_state = get_screen(env)
        current_state = get_screen(env)
        state = current_state - last_state
        for t in count():
            eps_threshold -= 0.000005
            state_var = torch.autograd.Variable(state)
            action_q = A_agent.act(state_var, eps_threshold)
            _, action = action_q.data.max(2)
            action_numpy = action.squeeze(0).numpy()
            # print(list(action_numpy))
            _, reward, done, _ = env.step(action_numpy)
            A_agent.add_to_buffer(reward, action_q, state_var)
            if done:
                break
            if len(A_agent.localbuffer) > 50:
                p, error = calc_priority_TDerror(Learner, criterion, A_agent, 3)
                optimizer.zero_grad()
                error.backward(retain_graph=True)
                optimizer.step()
                print("{0}\t{1}\t{2}".format(i_episode, t, float(error)))
            last_state = current_state
            current_state = get_screen(env)
            state = current_state - last_state

            env.render()

if __name__=="__main__":
    args = parser.parse_args()
    train()
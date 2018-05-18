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
parser.add_argument("--sigma", default=0.99, type=float)
parser.add_argument("--localcapacity", default=400)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


is_cuda = False
if torch.cuda.is_available():
    is_cuda = True


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
    Learner = DQN().to(device)

    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    # env = retro.make(game='Airstriker-Genesis', state='Level1')

    criterion = L2_loss(0.99).to(device)

    if is_cuda:
        Learner = Learner.cuda()
        criterion = criterion.cuda()


    optimizer = optim.SGD(Learner.parameters(), lr=0.01)

    eps_threshold = 0.8
    RM = ReplayMemory(1000)
    A_agent = ActorAgent(Learner, args)
    print("Start Episodes")
    for i_episode in range(50000):
        env.reset()
        A_agent.reset(Learner, args)
        last_state = get_screen(env)
        current_state = get_screen(env)
        state = current_state - last_state
        # state_var = torch.autograd.Variable(state)
        state_var = state.to(device)
        total_reward = 0
        if i_episode % 50 == 0:
            eps_threshold = 0.9
        for t in count():
            if t==0:
                print("episode begin")
            eps_threshold -= 0.000019
            action_q = A_agent.act(state_var, eps_threshold)

            """
            if is_cuda:
                action_q = action_q.cpu()
                _, action = action_q.data.max(2)
            else:
                _, action = action_q.data.max(2)
            """
            _, action = action_q.data.max(2)

            action_numpy = action.squeeze(0).numpy()
            # print(list(action_numpy))
            for i in range(4):
                _, reward, done, _  = env.step(action_numpy)
                total_reward += reward
            last_state = current_state
            current_state = get_screen(env)
            state = current_state - last_state
            # state_var = torch.autograd.Variable(state)
            state_var = state.to(device)
            # 行動語のstateを保存
            A_agent.add_to_buffer(reward, action_q, state_var)

            # ReplayMemoryに状態保存
            if len(A_agent.localbuffer)>10:
                p, error = calc_priority_TDerror(Learner,
                                                 criterion, A_agent, 10)

                RM.push(p,error)

            if done:
                break

            # Optimize Learner model
            # if t%100==0 and len(A_agent.localbuffer)>80 and len(RM)>=30:
        for i in range(4):
            error_batch = RM.priority_sample(30)

            optimizer.zero_grad()
            # error_batch.backward(retain_graph=True)
            error_batch.backward()
            optimizer.step()
            for param in Learner.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            print("{0}\t{1}\tLoss:{2}\tTotal:{3}\tReward:{4}".format(i_episode, t,
                                              float(error_batch),total_reward, reward, ))
        RM.reset()
            # env.render()


        with open("total_reward.txt", "a") as f:
            f.write("{0}\t{1}".format(i_episode, total_reward))
            f.write("\n")

if __name__=="__main__":
    args = parser.parse_args()
    train()
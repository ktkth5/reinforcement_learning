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
    Learner = DQN()

    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    # env = retro.make(game='Airstriker-Genesis', state='Level1')
    Replay_Memory = ReplayMemory(10000)

    criterion = L2_loss(0.5)

    if is_cuda:
        Learner = Learner.cuda()
        criterion = criterion.cuda()

    optimizer = optim.SGD(Learner.parameters(), lr=0.01)

    eps_threshold = 0.9
    RM = ReplayMemory(1000)

    for i_episode in range(100):
        env.reset()
        A_agent = ActorAgent(Learner, args)
        last_state = get_screen(env)
        current_state = get_screen(env)
        state = current_state - last_state
        state_var = torch.autograd.Variable(state)
        for t in count():
            eps_threshold -= 0.000005
            action_q = A_agent.act(state_var, eps_threshold)
            if is_cuda:
                action_q_cpu = action_q.cpu()
                _, action = action_q_cpu.data.max(2)
            else:
                _, action = action_q.data.max(2)
            action_numpy = action.squeeze(0).numpy()
            # print(list(action_numpy))
            _, reward, done, _ = env.step(action_numpy)
            last_state = current_state
            current_state = get_screen(env)
            state = current_state - last_state
            state_var = torch.autograd.Variable(state)
            # 行動語のstateを保存
            A_agent.add_to_buffer(reward, action_q, state_var)

            # ReplayMemoryに状態保存
            if len(A_agent.localbuffer)>10:
                p, error = calc_priority_TDerror(Learner, ActorAgent,
                                                 criterion, A_agent, 10)
                RM.push(p,error)

            if done:
                break

            # Optimize Learner model
            if len(A_agent.localbuffer)%30==0 and len(A_agent.localbuffer)>80:
                error_batch = RM.priority_sample(30)
                optimizer.zero_grad()
                error_batch.backward(retain_graph=True)
                optimizer.step()
                for param in Learner.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                print("{0}\t{1}\t{2}\t{3}".format(i_episode, t,
                                                  float(error_batch), reward))


            env.render()

if __name__=="__main__":
    args = parser.parse_args()
    train()
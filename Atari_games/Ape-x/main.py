import argparse
import gym
from itertools import count
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from utils import get_screen, calc_priority_TDerror, init_param
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
device_cpu = torch.device("cpu")

env = gym.make("SpaceInvaders-v0").unwrapped


def train(Learner):
    # global args
    # args = parser.parse_args()
    # Learner = DQN().to(device)

    # env = retro.make(game='Airstriker-Genesis', state='Level1')

    criterion = L2_loss(0.99).to(device)

    if use_cuda:
        Learner = Learner.cuda()
        criterion = criterion.cuda()


    optimizer = optim.SGD(Learner.parameters(), lr=0.01)

    eps_threshold = 0.3
    RM = ReplayMemory(900)
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
        # if i_episode % 50 == 0:
        #     if not eps_threshold < 0.1:
        #         eps_threshold -= 0.001
        start = time.time()
        for t in count():
            if t==0:
                print("episode begin")
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
            if t==500:
                print("Total time: {0:.2f}".format(time.time()-start))
                # break
            # Optimize Learner model
            # if t%100==0 and len(A_agent.localbuffer)>80 and len(RM)>=30:
            env.render()

        # update Learner part
        for i in range(4):
            if len(RM.memory)>=30:
                error_batch = RM.priority_sample(30)

                optimizer.zero_grad()
                # error_batch.backward(retain_graph=True)
                error_batch.backward()
                optimizer.step()
                # for param in Learner.parameters():
                #     param.grad.data.clamp_(-1, 1)
                optimizer.step()
                print("{0}\t{1}\tLoss:{2:.2f}\tTotal:{3:.2f}\tReward:{4:.2f}".format(
                                                  i_episode, t,
                                                  float(error_batch),total_reward, reward, ))
            else:
                break



        if i_episode % 5==0:
            env.reset()
            last_state = get_screen(env)
            current_state = get_screen(env)
            state = current_state - last_state
            state_var = state.to(device)
            val_reward = 0
            for t in count():
                with torch.no_grad():

                    action_q = Learner(state_var)
                    _, action = action_q.data.max(2)
                    action_numpy = action.squeeze(0).numpy()
                    for i in range(4):
                        _, reward, done, _  = env.step(action_numpy)
                        val_reward += reward
                    last_state = current_state
                    current_state = get_screen(env)
                    state = current_state - last_state
                    state_var = state.to(device)

                    if done:
                        break
                    env.render()

            print("Validation:\tepisode{0}\tReward: {1:.2f}".format(i_episode,val_reward))

            with open("result.txt","a") as f:
                f.write("episode{0}\tReward: {1:.2f}".format(i_episode,val_reward))
                f.write("\n")

        RM.reset()
        # break

        with open("total_reward.txt", "a") as f:
            f.write("{0}\t{1}".format(i_episode, total_reward))
            f.write("\n")

if __name__=="__main__":

    args = parser.parse_args()
    model = DQN().to(device)
    init_param(model)
    train(model)
    # NOTE: this is required for the ``fork`` method to work

    # model = DQN().to(device)
    # num_processes = 2
    # model.share_memory()
    # processes = []
    # for rank in range(num_processes):
    #     p = mp.Process(target=train, args=(model,))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
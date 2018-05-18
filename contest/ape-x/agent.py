import argparse
import random
from collections import namedtuple
import numpy

import torch
from torch.autograd import Variable

from replay_memory import ReplayMemory
from loss_function import L2_loss
from model import DQN

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class ActorAgent():

    def __init__(self, learner, args):
        self.args = args

        self.Learner = learner
        self.Actor = DQN().to(device)
        # if torch.cuda.is_available():
        #     self.Actor = self.Actor.cuda()
        self.copy_parameter()
        self.position = 0
        self.capacity = args.localcapacity

        self.localbuffer = []
        # self.expe = namedtuple("self.expe",
        #                        ["reward","Qvalue","state"])

        # self.criterion = L2_loss(self.args.sigma)

    def act(self, state_var, eps_threshold=None):
        if eps_threshold is None:
            return self.Actor(state_var)
        else:
            sample = random.random()
            # action_q = self.Actor(state_var)

            if sample > eps_threshold:
                with torch.no_grad():
                    action_q = self.Actor(state_var)
                # print(type(output.data))
                return action_q
            else:
                output = numpy.random.rand(24)
                output = torch.from_numpy(output).view(1,12, 2)
                action_q_random = output.type(torch.FloatTensor)
                action_q_random.requires_grad = False
                # return Variable(action_q_random)
                return action_q_random

    def add_to_buffer(self, reward, action_q, state):
        # expe = namedtuple("expe",
        #                         ["reward","Qvalue","state"])
        if len(self.localbuffer)<self.capacity:
            self.localbuffer.append(None)
        reward_torch = torch.tensor([reward], dtype=torch.float)
        self.localbuffer[self.position]= \
            [reward_torch, action_q, state]
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        return random.sample(self.localbuffer, batch_size)

    def copy_parameter(self):
        self.Actor.load_state_dict(self.Learner.state_dict())

    def reset(self, learner, args):
        self.args = args

        self.Learner = learner
        self.copy_parameter()
        self.position = 0
        self.capacity = args.localcapacity

        self.localbuffer = []

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Agent test")
    parser.add_argument("--sigma", type=float, default=0.999)
    args = parser.parse_args()


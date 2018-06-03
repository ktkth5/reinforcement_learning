import retro
from collections import namedtuple

import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.autograd import Variable

from PIL import Image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# clear
def get_screen(env):
    """
    :param env:
    :return: Torch.FloatTensor(3,40,40)
    """
    state = env.render(mode="rgb_array").transpose(2,0,1)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((40,40), interpolation=Image.CUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.2, 0.2, 0.2])
                        ])
    state = resize(state)
    state.unsqueeze_(0)
    if torch.cuda.is_available():
        state = state.cuda()

    return state

def init_param(model):
    for param in model.parameters():
        nn.init.normal_(param.data)

def calc_priority_TDerror(Learner, criterion, A_agent, batch_size):
    epsilon = 0.01
    expe = namedtuple("expe",
                      ["reward", "Qvalue", "state"])

    experience = A_agent.sample(batch_size)
    batch = expe(*zip(*experience))
    state_batch = torch.cat(batch.state)
    Actor_qvalue_batch = torch.cat(batch.Qvalue, dim=0).view(batch_size, 12, 2)
    reward_batch = Variable(torch.cat(batch.reward))

    state = []
    q_value = []
    reward = []
    for p in experience:
        state.append(p[2])
        q_value.append(p[1])
        reward.append(p[0])
    state_batch = torch.cat(state)
    Actor_qvalue_batch = torch.cat(q_value)
    reward_batch = torch.cat(reward)

    # with torch.no_grad():
    Learner_qvalue = Learner(state_batch)
    # print("Learner Q value", Learner_qvalue.requires_grad)
    # Double DQNのロス産出は実装してない
    # print("Actor",Actor_qvalue_batch.requires_grad)
    if use_cuda:
        state_batch = state_batch.cpu()
        Actor_qvalue_batch = Actor_qvalue_batch
        Learner_qvalue = Learner_qvalue.cpu()
    # print("Actor", Actor_qvalue_batch.requires_grad)
    TDerror = criterion(reward_batch, Learner_qvalue, Actor_qvalue_batch)
    # print("TD shape",TDerror.shape)
    priority = torch.abs(TDerror) + epsilon
    # print("prio shape", priority.shape)

    # print(priority.requires_grad)
    # print("TD Error",TDerror.requires_grad)

    return priority, TDerror



if __name__=="__main__":
    x = [(0,1,2),
         (3,4,5),
         (7,8,9),
         (10,11,12)]
    print(x[:][2])

    # env = retro.make(game='Airstriker-Genesis', state='Level1')
    # get_screen(env)
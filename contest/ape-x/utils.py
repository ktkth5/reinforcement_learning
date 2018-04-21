import retro
from collections import namedtuple

import torchvision.transforms as T
import torch
from torch.autograd import Variable

from PIL import Image

# clear
def get_screen(env):
    """
    :param env:
    :return: Torch.FloatTensor(3,40,40)
    """
    state = env.render(mode="rgb_array").transpose(2,0,1)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((40,40), interpolation=Image.CUBIC),
                        T.ToTensor()])
    state = resize(state)
    return state

def calc_priority_TDerror(learner, criterion, A_agent, batch_size):
    epsilon = 0.001
    expe = namedtuple("expe",
                      ["reward", "Qvalue", "state"])

    experience = A_agent.sample(batch_size)
    batch = expe(*zip(*experience))
    state_batch = torch.cat(batch.state).type(torch.FloatTensor)
    Actor_qvalue_batch = torch.cat(batch.Qvalue, dim=0).view(batch_size, 12, 2)
    reward_batch = Variable(torch.cat(batch.reward))

    L_qvalue = learner(state_batch)
    TDerror = criterion(reward_batch, L_qvalue, Actor_qvalue_batch)
    priority = torch.abs(TDerror) + epsilon

    # print(priority)
    # print(TDerror.shape)

    return priority, TDerror



if __name__=="__main__":
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    get_screen(env)
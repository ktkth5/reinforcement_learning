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
    state.unsqueeze_(0)
    if torch.cuda.is_available():
        state = state.cuda()

    return state

def calc_priority_TDerror(Learner, Actor, criterion, A_agent, batch_size):
    epsilon = 0.001
    expe = namedtuple("expe",
                      ["reward", "Qvalue", "state"])

    experience = A_agent.sample(batch_size)
    batch = expe(*zip(*experience))
    state_batch = torch.cat(batch.state)
    Actor_qvalue_batch = torch.cat(batch.Qvalue, dim=0).view(batch_size, 12, 2)
    reward_batch = Variable(torch.cat(batch.reward))

    Learner_qvalue = Learner(state_batch)
    # Double DQNのロス産出は実装してない
    if torch.cuda.is_available():
        state_batch = state_batch.cpu()
        Actor_qvalue_batch = Actor_qvalue_batch
        Learner_qvalue = Learner_qvalue.cpu()
    TDerror = criterion(reward_batch, Learner_qvalue, Actor_qvalue_batch)
    priority = torch.abs(TDerror) + epsilon

    # print(priority)
    # print(TDerror.shape)

    return priority, TDerror



if __name__=="__main__":
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    get_screen(env)
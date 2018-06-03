import torch
import torch.nn as nn
from torch.autograd import Variable

from model import DQN

# clear
class L2_loss(nn.Module):

    def __init__(self, sigma):
        super(L2_loss, self).__init__()
        self.sigma = sigma

    def forward(self, r, x, y):
        """
        :param r: reward, Variable(batch_size,)
        :param x: q action, Variable(batch_size, 12, 2)
        :param y: q target, Variable(batch_size, 12, 2)
        :return: Loss
        """
        y = Variable(y.data)
        batch_size = x.size()[0]
        x = x.view(batch_size, 12, 2)
        y = y.view(batch_size, 12, 2)
        r = r.unsqueeze(1)
        r = torch.cat([r,r],dim=1)
        r = r.unsqueeze(1)

        y = torch.add(r, self.sigma, y)
        td_loss = torch.mean((y-x)**2, dim=2)
        td_loss = torch.mean(td_loss, dim=1)
        td_loss = torch.mean(td_loss, dim=0)
        # print(float(td_loss))
        td_loss.squeeze_(0)
        return td_loss/(2)



if __name__=="__main__":
    d_state = torch.randn(1,3,40,40)
    state = torch.randn(1,3,40,40)
    reward = torch.autograd.Variable(torch.FloatTensor([0.1]))
    target = torch.rand(1,12,2)
    d_state_var = torch.autograd.Variable(d_state)
    state_var = torch.autograd.Variable(state)
    target_var = torch.autograd.Variable(target)
    target_var.unsqueeze_(0)


    import copy
    learner = DQN()
    actor = DQN()

    for param in actor.parameters():
        param.requires_grad = False

    cuda = False
    if torch.cuda.is_available():
        cuda = True
        learner = learner.cuda(0)
        reward = reward.cuda(0)
    optimizer = torch.optim.SGD(learner.parameters(), lr=0.01)
    criterion = L2_loss(0.999)

    learner.train()
    for k in range(100):
        if cuda:
            x = learner(d_state_var.cuda(0))
        else:
            x = learner(d_state_var)
        actor.load_state_dict(learner.state_dict())
        # print(x)
        for i in range(10):
            # state_var = torch.autograd.Variable(torch.randn(1, 3, 40, 40))
            y = actor(state_var)

            if cuda:
                y = y.cuda(0)
            loss = criterion(reward, x, y)
            if i==0:
                print(loss)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            for param in learner.parameters():
                param.grad.data.clamp_(-1,1)
            optimizer.step()
        # print(float(loss))
        # print(x-y)
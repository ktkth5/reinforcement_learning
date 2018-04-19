import torch
import torch.nn as nn

from model import DQN

class L2_loss(nn.Module):

    def __init__(self, GAMMA):
        super(L2_loss, self).__init__()
        self.gamma = GAMMA

    def forward(self, r, x, y):
        """
        :param r: reward, Variable(1,)
        :param x: q action, Variable(batch size, 12, 2)
        :param y: q target, Variable(batch size, 12, 2)
        :return: Loss
        """
        batch_size = x.size()[0]
        x = x.view(batch_size, 12, 2)
        y = y.view(batch_size, 12, 2)

        x = r + x*self.gamma
        td_loss = torch.sum((x-y)**2, dim=2)
        td_loss = torch.sum(td_loss, dim=1)
        # print(float(td_loss))

        return td_loss/2



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

    learner = learner.cuda(0)
    optimizer = torch.optim.SGD(learner.parameters(), lr=0.01)
    criterion = L2_loss(0.999)

    reward = reward.cuda(0)
    for k in range(3):
        x = learner(d_state_var.cuda(0))
        actor.load_state_dict(learner.state_dict())
        print(x)
        for i in range(100):
            # state_var = torch.autograd.Variable(torch.randn(1, 3, 40, 40))
            y = actor(state_var)

            y = y.cuda(0)
            loss = criterion(reward, x, y)
            if i % 30==0:
                print(loss)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print(float(loss))
        print(x-y)
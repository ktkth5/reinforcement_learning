import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=5,stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,256,kernel_size=3,stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256,64,kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.select_actions = nn.ModuleList([
            nn.Sequential(nn.Linear(64*6*6,256),
                          nn.Linear(256,2))
            for i in range(12)
        ])

    def forward(self, state):
        state  = F.relu(self.bn1(self.conv1(state)))
        state  = F.relu(self.bn2(self.conv2(state)))
        state  = F.relu(self.bn3(self.conv3(state)))
        state = state.view(-1,64*6*6)
        action_list = [select_action(state) for select_action in self.select_actions]

        def shaping(action_list):
            action = torch.cat(action_list, dim=1)
            action = action.view(-1,12,2)
            return action

        action = shaping(action_list)

        return action


if __name__=="__main__":
    state = torch.randn(1,3,40,40)
    state_var = torch.autograd.Variable(state)

    model = DQN()
    action = model(state_var)

    import torch.optim as optim

    optimizer = optim.RMSprop(model.parameters())
    target = torch.LongTensor([[0],[0],[0],[0],[1],[1],[1],[1],[0],[0],[0],[0]])
    target_var = torch.autograd.Variable(target)
    target_var = target_var.unsqueeze(0)
    target_var.unsqueeze_(2)
    target_var = torch.cat((target_var, target_var), dim=2)
    target_var.squeeze_(3)

    loss_f = nn.MSELoss()
    loss_f = nn.L1Loss()

    for i in range(10):
        action = model(state_var)
        print(action.size())
        print(target_var.size())
        # _, action = action.data.max(2)
        # print(action.size())
        target_var = target_var.type(torch.FloatTensor)
        loss = loss_f(action, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

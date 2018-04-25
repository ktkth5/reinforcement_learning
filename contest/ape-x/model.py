import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(30)

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
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = state.view(-1,64*6*6)
        action_list = [select_action(state) for select_action
                       in self.select_actions]

        def shaping(action_list):
            action = torch.cat(action_list, dim=1)
            action = action.view(-1,12,2)
            return action

        action = shaping(action_list)
        return F.relu(action)


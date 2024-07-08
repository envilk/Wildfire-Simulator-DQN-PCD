# python libraries

import torch.nn as nn
import torch.nn.functional as F

# own python modules

from common_fixed_variables import *


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(N_OBSERVATIONS, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, N_ACTIONS)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(N_OBSERVATIONS, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, N_ACTIONS)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
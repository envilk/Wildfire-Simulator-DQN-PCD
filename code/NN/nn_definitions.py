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


# for images 28x28 "input_dim = 28, hidden_dim/seq_length = 28"
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        # x -> batch_size, hidden_dim/seq_length, input_dim | for 10 images 28x28 -> (10, 28, 28)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # y -> hidden_dim/seq_length, output_dim (since it consists of a many to many architecture, calling forward
        # method as many times as number of predictions are wanted to be done, is needed)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out)
        return output
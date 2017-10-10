import argparse
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

seed = np.random.randint(1000)
torch.manual_seed(123)

# hyper-parameters
learning_rate = 1e-2


class Policy(nn.module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(10, 128) # change to reflect dimension of input feature
        self.affine2 = nn.Linear(128, 4)  # change to reflect number of actions

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.affine2(x)
        return F.softmax(x)

policy = Policy()
optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate)
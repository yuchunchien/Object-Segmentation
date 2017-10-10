import torch
import torch.nn as nn
import torch.nn.parallel

class CustomNet(nn.Module):
    def __init__(self, ngpu):
        super(CustomNet, self).__init__()
        self.ngpu = ngpu

        # layers
        # main = nn.Sequential()

    def forward (self, input):
        return 0

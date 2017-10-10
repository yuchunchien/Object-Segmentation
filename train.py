from __future__ import print_function
import argparse
import numpy as np
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# flags
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',    required=True, help='specify dataset to use')
parser.add_argument('--dataroot',   required=True, default='./data/', help='path to dataset')
parser.add_argument('--cuda',       action='store_true', help='enable cuda')
parser.add_argument('--workers',    type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize',  type=int, default=64, help='input batch size')
parser.add_argument('--iters',      type=int, default=100, help='epochs to train for')
parser.add_argument('--ngpu',       type=int, default=1, help='number of GPUs to use')
parser.add_argument('--experiment', type=str, default=None, help='where to store samples and models')
args = parser.parse_args()
print("Arguments\n: {0}".format(args))

# create samples folder if it does not exist
if args.experiment is None:
    args.experiment = 'samples'
os.system('mkdir {0}'.format(args.experiment))

# pytorch configurations
torch.manual_seed(np.random.randint(1000))
cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print('Warning: CUDA device available, run with --cuda')

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batchSize,
    shuffle=True,
    num_workers=int(args.workers)
)

# training
for epoch in range(args.iters):
    data_iter = iter(dataloader)


    # checkpoints
    torch.save()
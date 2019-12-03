#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:38:13 2019

@author: tirsgaard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:51:02 2019

@author: tirsgaard
"""

import sys
# Relative paths for some reason do not work :(
sys.path.append('/Users/tirsgaard/Google Drive/Deep_learning_project/model_implementation')
# sys.path.append('/Users/peter/Google Drive/Deep_learning_project/model_implementation')


from go_model import ResNet, ResNetBasicBlock
from go import go_board
from multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
from MCTS import gpu_worker, data_handler, sim_game, sim_games


import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from functools import partial



def resnet40(in_channels, filter_size=128,board_size=9, deepths=[19]):
    return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)

from torchsummary import summary

model = resnet40(17, 128,9)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Make queues for sending data
#m = Manager()
#gpu_Q = Queue()
#data_Q = Queue()
# Also make pipe for receiving v_resign
conn_rec, conn_send = Pipe(False)

model.eval()
with torch.no_grad():
    
    v_resign = sim_games(10, 100, model, 8, float("-inf"), batch_size=8, board_size = 9)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:27:14 2019

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



import re
import os
import glob
import numpy as np




def save_model(model):
    subdirectory = "saved_models/"
    os.makedirs(subdirectory, exist_ok=True)
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory+"*.model"):
        files.append(file)
        
    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "",file)))
        
    if (number_list==[]):
        number_list = [0]
    # Get max number
    latest_new_model = max(number_list)+1
    
    save_name = subdirectory + "model_" + str(latest_new_model) + ".model"
    torch.save(model, save_name)
    

def load_latest_model():
    subdirectory = "saved_models/"
    os.makedirs(subdirectory, exist_ok=True)
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory+"*.model"):
        files.append(file)
        
    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "",file)))
    
    if (number_list==[]):
        number_list = [0]
    # Get max number
    latest_model = max(number_list)
    
    load_name = subdirectory+"model_" + str(latest_model) + ".model"
    model = torch.load(load_name)
    return model


best_model = resnet40(17, 128,9)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

save_model(best_model)

new_model = load_latest_model()
summary(new_model, (17, 9, 9))
#new_model = torch.load("test_model.model")

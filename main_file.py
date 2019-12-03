#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:44:33 2019

@author: tirsgaard
"""
# File for running the entire program
import sys
# Relative paths for some reason do not work :(
sys.path.append('/Users/tirsgaard/Google Drive/Deep_learning_project/model_implementation')
# sys.path.append('/Users/peter/Google Drive/Deep_learning_project/model_implementation')


from go_model import ResNet, ResNetBasicBlock
from go import go_board
from multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
from MCTS import gpu_worker, data_handler, sim_game, sim_games
from training_functions import load_saved_games, save_model, load_latest_model, loss_function
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from updateEloDuel import updateEloDuel
from elo import Implementation

def resnet40(in_channels, filter_size=128,board_size=9, deepths=[19]):
    return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)


## Hyper parameters
number_of_threads = 4
self_play_batch_size = 8
board_size = 9
N_training_games = 2000
N_MCTS_sim = 400
N_duel_games = 100
TRAIN_STEPS = 1000
N_turns = 10000
train_batch_size = 32
num_epochs = int(1000*512/batch_size)
elo_league = Implementation()

# GPU things
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    
## Load model if one exists, else define a new
writer = SummaryWriter()
best_model = load_latest_model()
training_model = load_latest_model()

if (best_model == None):
    best_model = resnet40(17, 128,9)
    save_model(best_model)
    training_model = load_latest_model()

elo_league.addPlayer("model1",rating=0)

## define variables to be used
v_resign = float("-inf")
loop_counter = 0
training_counter = 0
model_iter_counter = 1

skip_self_play = True
skip_training = False


## Running loop
while True:
    if ((not skip_self_play) | (training_counter>0)):
        print("Beginning loop", loop_counter)
        print("Beginning self-play")
        ## Generate new data for training
        best_model.eval()
        with torch.no_grad():
            v_resign = sim_games(N_training_games, 
                                 N_MCTS_sim, 
                                 best_model, 
                                 number_of_threads, 
                                 v_resign, 
                                 batch_size=self_play_batch_size, 
                                 board_size = board_size)
            
        writer.add_scalar('v_resign', v_resign, loop_counter)    
    
    print("Begin training")
    ## Now train model
    # Get learning rate
    if (training_counter<10**5):
        learning_rate = 10**-2
    elif (training_counter<1.5*10**5):
        learning_rate = 10**-3
    else:
        learning_rate = 10**-4
    
    training_model.train()
    ## Get loss function
    criterion = loss_function
    optimizer = optim.SGD(training_model.parameters(),lr=learning_rate, momentum=0.9, weight_decay=10**-4)
    
    # Load newest data
    S, Pi, z, n_points = load_saved_games(N_turns)
    for i in range(num_epochs):
        training_counter += 1
        
        # generate batch
        index = np.random.randint(0, n_points-1, size = train_batch_size)
        Pi_batch = torch.from_numpy(Pi[index]).float()
        z_batch = torch.from_numpy(z[index]).float()
        S_batch = torch.from_numpy(S[index]).float()
        
        # Optimize
        optimizer.zero_grad()
        P_batch, v_batch = training_model.forward(S_batch)
        loss, v_loss, P_loss = criterion(Pi_batch, z_batch, P_batch, v_batch, train_batch_size)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Total_loss/train', loss, training_counter)
        writer.add_scalar('value_loss/train', v_loss, training_counter)
        writer.add_scalar('Policy_loss/train', P_loss, training_counter)
        
        if (i%10 ==0):
            print("Fraction of training done: ", i*train_batch_size/num_epochs)
    
    print("Begin evaluation")
    ## Evaluate training model against best model
    # Below are the needed functions
    best_model.eval()
    training_model.eval()
    with torch.no_grad():
        scores = sim_games(N_duel_games,
                           N_MCTS_sim, 
                           best_model, 
                           number_of_threads, 
                           v_resign, 
                           model2 = training_model, 
                           duel=True ,
                           batch_size=self_play_batch_size, 
                           board_size = board_size)
    
    
    # Find best model
    # Here the model has to win 60% of the 100 games
    if (scores[0]/6>scores[1]/4):
        print("New best model!")
        best_model = training_model
        
        # Update elo
        name_prev_model = "model"+str(model_iter_counter)
        name_best_model = "model"+str(model_iter_counter+1)
        model_iter_counter += 1
        
        prev_best_elo = elo_league.getPlayerRating(name_best_model)
        new_best_elo = elo_league.addPlayer(name_best_model, rating=prev_best_elo)
        
        # Store statistics
        writer.add_scalar('Elo', new_best_elo, model_iter_counter)
    print(scores)
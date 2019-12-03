import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import go_model
from elo import *
import compare_models
from go_model import ResNet, ResNetBasicBlock
from multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
from MCTS import gpu_worker, data_handler, sim_game, sim_games
from training_functions import load_saved_games
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from training_functions import loss_function


def resnet40(in_channels, filter_size=128,board_size=9, deepths=[19]):
    return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)

    
# setting hyperparameters and gettings epoch sizes
batch_size = 81
num_epochs = 100#int(1000*2048/batch_size)
num_iter = 0


cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

writer = SummaryWriter()

best_player = resnet40(17, 128, 9)
player = best_player

UPDATE_GAMES = 500000
TRAIN_STEPS = 1000
N_turns = 10000

## Get loss function
criterion = loss_function
optimizer = optim.SGD(best_player.parameters(),lr=0.1, momentum=0.9, weight_decay=10**-4)
#S, Pi, z, n_points = load_saved_games(N_turns)
n_points = N_turns
S = np.zeros((81,17,9,9))
Pi = np.zeros((81,82))
z = np.zeros((81))
for i in range(81):
    index = np.unravel_index(i, (9,9))
    S = np.random.rand(81,17,9,9)
    Pi[i] = np.random.dirichlet(np.ones(82)*0.3)
    z[i] = 0.5


for i in range(num_epochs):
    # generate batch
    #index = np.random.randint(0, n_points, size = batch_size)
    Pi_batch = torch.from_numpy(Pi).float()#torch.from_numpy(Pi[index]).float()
    z_batch = torch.from_numpy(z).float()#torch.from_numpy(z[index]).float()
    S_batch = torch.from_numpy(S).float()#torch.from_numpy(S[index]).float()
    
    # Optimize
    optimizer.zero_grad()
    P_batch, v_batch = best_player.forward(S_batch)
    loss, v_loss, P_loss = criterion(Pi_batch, z_batch, P_batch, v_batch, batch_size)
    loss.backward()
    optimizer.step()
    writer.add_scalar('Loss/train', loss, i)
    writer.add_scalar('v_loss/train', v_loss, i)
    writer.add_scalar('P_loss/train', P_loss, i)
    print(i)
    


"""
def train_backup():
    criterion = AlphaLoss()
    (state, move, winner) = load_saved_games(10000)
    
    while True:
        for (state, move, winner) in range(len(winner)): 
            if num_iter % TRAIN_STEPS == 0:
                duelllingPlayer = deep_copy(player)
                best_player_wins, duellingPlayer_wins = sim_games(no_processes, 20, best_player, no_processes, float("-inf"), model2 = duelllingPlayer, duel=True ,batch_size=8, board_size = 9)
                # check what it returns
                if duellingPlayer_wins > best_player_wins:
                    best_player = duelllingPlayer
                
            example_game = {
            'state': state,
            'winner': winner,
            'move' : move
            }
            
            optimizer.zero_grad()
            winner, probs = duelllingPlayer(example_game['state'])
            
            loss = criterion(winner, example_game['winner'], probs, example_game['move'])
            
            loss.backward
            optimizer.step
            
            
            if num_iter % UPDATE_GAMES == 0:
                load_saved_games(10000)
                    
                    
            num_iter +=1

train()
"""
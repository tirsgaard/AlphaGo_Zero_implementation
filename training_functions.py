#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:40:57 2019

@author: tirsgaard
"""
import re
import os
import glob
import numpy as np
import torch

def load_saved_games(N_data_points):
    
    

    #data = np.load("games_data/game_data_24.npz")
    #S_array, P_array, z_array
    #print(data['S'])
    
    # Construct large numpy array of data
    S_array = np.empty((N_data_points, 17, 9, 9), dtype=bool)
    Pi_array = np.empty((N_data_points, 82), dtype=float)
    z_array = np.zeros((N_data_points), dtype=float)
    
    subdirectory = "games_data/"
    
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory+"*.npz"):
        files.append(file)
        
    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "",file)))
    # Get max number
    latest_games = max(number_list)
    
    # Counter for keeping track of large array
    data_counter = 0
    while (data_counter<N_data_points):
        # Load data
        file_name = subdirectory+"game_data_"+str(latest_games)+".npz"
        latest_games -= 1
        # Case where not enough data is generated
        if (latest_games<0):
            S_array = S_array[0:data_counter]
            Pi_array = Pi_array[0:data_counter]
            z_array = z_array[0:data_counter]
            
            return S_array, Pi_array, z_array, data_counter+1
            
            
            
        data = np.load(file_name)
        S = data['S']
        Pi = data['P']
        z = data['z']
        for i in range(z.shape[0]):
            # Add data to large arrays
            S_array[data_counter] = S[i]
            Pi_array[data_counter] = Pi[i]
            z_array [data_counter] = z[i] 
            # Increment counter
            data_counter += 1
            # Check if large arrays are filled
            if (data_counter>=N_data_points):
                break
    
    return S_array, Pi_array, z_array, data_counter+1

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
        return None
    # Get max number
    latest_model = max(number_list)
    
    load_name = subdirectory+"model_" + str(latest_model) + ".model"
    model = torch.load(load_name)
    return model

def loss_function(Pi, z, P, v, batch_size):
    v = torch.squeeze(v)
    value_error = torch.mean((v - z)**2)
    inner = torch.log(1e-8 + P)
    policy_error = torch.bmm(Pi.view(batch_size, 1, 82), inner.view(batch_size, 82, 1)).mean()
    total_error = value_error-policy_error
    return total_error, value_error, -policy_error
    

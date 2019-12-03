#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 01:02:17 2019

@author: tirsgaard
"""
import re
import os
import glob
# Make directory if missing
subdirectory = "games_data"
os.makedirs(subdirectory, exist_ok=True)

# Find larges number of games found
# Get files
files = []
os.chdir(subdirectory)
for file in glob.glob("*.npz"):
    files.append(file)
    
# get numbers from files
number_list = []
for file in files:
    number_list.append(int(re.sub("[^0-9]", "",file)))
# Get max number
try:
    new_iter = max(number_list)+1
except:
    new_iter = 1

# Save new data
name = "game_data_" + str(new_iter)
print(name)
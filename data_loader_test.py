#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:34:14 2019

@author: tirsgaard
"""
import re
import os
import glob
import numpy as np

from training_functions import load_saved_games
# Relative paths for some reason do not work :(
#sys.path.append('/Users/tirsgaard/Google Drive/Deep_learning_project/model_implementation')



S, Pi, z = enumerate(load_saved_games(100))
print(S)
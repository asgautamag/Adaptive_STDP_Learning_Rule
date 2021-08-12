#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate .csv file with details of the spike train from the .mat file.
"""

import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
#import random
import pandas as pd
import h5py
#np.random.seed(0)
name=["56"]#, "34","09","10", "11"]#, "05","06","12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30","31", "32", "33", "35", "36", "37", "38", "39", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50"]
#name=["12"]
for n_run in tqdm(range(len(name))):
    file_name="afferent.rand0"+name[n_run]+"_1000_onefourth_225sec.mat"
    f=h5py.File(file_name, 'r')
    PP_t=f.get('spikeList')     #spikeList is the variable(containing timeindex) needed from the file
    PP_t=np.array(PP_t)
    PP_i=f.get('afferentList')
    PP_i=np.array(PP_i)
    PP_i=PP_i-1                 #Python index starts from 0
    PP_t=np.squeeze(PP_t)       #Reduce dims to write csv
    PP_i=np.squeeze(PP_i)
    pd.DataFrame(PP_i, PP_t).to_csv("ihg_matlab_1000_onefourth_225sec_rand_0"+name[n_run]+".csv")

            
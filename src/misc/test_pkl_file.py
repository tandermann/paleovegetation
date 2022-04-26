#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:14:24 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(suppress=True)
np.random.seed(1234)

pkl_file = '/Users/tobias/GitHub/feature_gen_paleoveg/results/testing_multiple_species_weights/BNNMC3_p0_h0_l32_8_s5_b5_5083.pkl'
all_weights = pickle.load(open(pkl_file,'rb'))
all_weights
array_dims = np.array([len(i) for i in all_weights])
np.where(array_dims>4)[0]

all_weights[44]



array_dims[array_dims>4]


a = np.array([-0.098, -0.066,  0.019,  0.181,  0.12 , -0.089, -0.052,  0.053,
          0.003, -0.158,  0.078, -0.213,  0.081, -0.101,  0.235,  0.15 ,
         -0.084,  0.052,  0.022, -0.184, -0.266,  0.215,  0.115, -0.087,
          0.119,  0.073, -0.236,  0.093, -0.093, -0.019, -0.089, -0.071,
         -0.013,  0.202,  0.136, -0.148, -0.105, -0.15 , -0.08 , -0.008,
         -0.065, -0.019, -0.037, -0.118, -0.127, -0.179, -0.244, -0.072,
         -0.123,  0.078, -0.39 , -0.093,  0.042,  0.075, -0.163,  0.468,
          0.013,  0.03 , -0.146, -0.091,  0.098, -0.052, -0.217,  0.114,
          0.162,  0.055, -0.277, -0.049, -0.269, -0.038,  0.05 , -0.145,
          0.274,  0.178,  0.024, -0.012, -0.08 , -0.15 , -0.021, -0.098,
         -0.232, -0.051,  0.113,  0.135,  0.082,  0.084,  0.22 , -0.004,
         -0.107, -0.101,  0.038,  0.19 ,  0.173, -0.052, -0.014,  0.009,
         -0.077,  0.092,  0.155, -0.005, -0.072])

b = np.array([-0.098, -0.066,  0.019,  0.181,  0.12 , -0.089, -0.052,  0.053,
          0.003, -0.158,  0.078, -0.213,  0.081, -0.101,  0.235,  0.15 ,
         -0.084,  0.052,  0.022, -0.184, -0.266,  0.215,  0.115, -0.087,
          0.119,  0.073, -0.236,  0.093, -0.093, -0.019, -0.089, -0.071,
         -0.013,  0.202,  0.136, -0.148, -0.105, -0.15 , -0.08 , -0.008,
         -0.065, -0.019, -0.037, -0.118, -0.127, -0.179, -0.244, -0.072,
         -0.123,  0.078, -0.39 , -0.093,  0.042,  0.075, -0.163,  0.468,
          0.013,  0.03 , -0.146, -0.091,  0.098, -0.052, -0.217,  0.114,
          0.162,  0.055, -0.277, -0.049, -0.269, -0.038,  0.05 , -0.145,
          0.274,  0.178,  0.024, -0.012, -0.08 , -0.15 , -0.021, -0.098,
         -0.232, -0.051,  0.113,  0.135,  0.082,  0.084,  0.22 , -0.004,
         -0.107, -0.101,  0.038,  0.19 ,  0.173, -0.052, -0.014,  0.009,
         -0.077,  0.092,  0.155, -0.005, -0.072])


c = a-b
c[c>0]


all_weights[0]['additional_prm'][1].shape




#pkl_file = '/Users/tobias/GitHub/feature_gen_paleoveg/data/temporal_distances_NN_input.pkl'
#all_weights = pickle.load(open(pkl_file,'rb'))
#all_weights = np.loadtxt('/Users/tobias/GitHub/feature_gen_paleoveg/data/veg_labels.txt').astype(int)
all_weights = np.load('/Users/tobias/GitHub/feature_gen_paleoveg/data/abiotic_features.npy')
selection = list(all_weights[:10])+ list(all_weights[-10:])
np.save('/Users/tobias/GitHub/feature_gen_paleoveg/data/trouble_shooting_data/abiotic_features.npy',selection)
#np.savetxt('/Users/tobias/GitHub/feature_gen_paleoveg/data/trouble_shooting_data/veg_labels.txt',selection,fmt='%i')
#pickle.dump(selection,open('/Users/tobias/GitHub/feature_gen_paleoveg/data/trouble_shooting_data/temporal_distances_NN_input.pkl',"wb"))
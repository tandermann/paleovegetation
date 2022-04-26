#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:31:37 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(suppress=True)
np.random.seed(1234)

pkl_file = '/Users/tobias/GitHub/feature_gen_paleoveg/results/from_cluster/nodes_50_10/BNN_p1_h0_l50_10_s1_binf_167859.pkl'
weights = pickle.load(open(pkl_file, 'rb'))
flattened_weights = np.array([np.concatenate([np.array([i]),array[0].flatten(),array[1].flatten(),array[2].flatten()]) for i,array in enumerate(weights)])
weight_labels = ['it']+['l0_%i'%i for i in range(weights[0][0].shape[0]*weights[0][0].shape[1])] + ['l1_%i'%i for i in range(weights[0][1].shape[0]*weights[0][1].shape[1])] + ['l2_%i'%i for i in range(weights[0][2].shape[0]*weights[0][2].shape[1])]
outdata = pd.DataFrame(data = flattened_weights, columns = weight_labels)
outdata[['it']] = outdata[['it']].astype(int)
outdata.to_csv('/Users/tobias/GitHub/feature_gen_paleoveg/results/from_cluster/nodes_50_10/BNN_p1_h0_l50_10_s1_binf_167859_weights_log_file.txt',sep='\t',index=False,float_format='%.4f')

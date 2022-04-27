#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:37:39 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random, numpy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


def UpdateNormal(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000,9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, i.shape[0],n) # faster than np.random.choice
    z = np.zeros(i.shape) + i
    z[Ix] = z[Ix] + rs.normal(0, d, n)
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    return z, Ix


def select_train_and_test_set(time_stamp_array,seed,wd,testsize=0.1,equal_paleo_and_current_labels=True,current_train_fraction=None):
    np.random.seed(seed)    
    paleo_indices = np.where(time_stamp_array>0)[0]
    current_indices = np.where(time_stamp_array==0)[0]
    paleo_test_indices = np.random.choice(paleo_indices,int(len(paleo_indices)*testsize),replace=False)
    current_test_indices = np.random.choice(current_indices,len(paleo_test_indices),replace=False)
    paleo_train_indices = np.array([i for i in paleo_indices if i not in paleo_test_indices])
    current_train_indices = np.array([i for i in current_indices if i not in current_test_indices])
    if equal_paleo_and_current_labels:
        current_train_indices = np.random.choice(current_train_indices,len(paleo_train_indices),replace=False)
    else:
        current_train_indices = np.random.choice(current_train_indices,int(len(current_indices)*current_train_fraction),replace=False)
    train_indices = np.concatenate([paleo_train_indices,current_train_indices])
    test_indices = np.concatenate([paleo_test_indices,current_test_indices])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    np.savetxt(os.path.join(wd,'train_instance_indices_seed_%i.txt'%seed),train_indices,fmt='%i')
    np.savetxt(os.path.join(wd,'test_instance_indices_seed_%i.txt'%seed),test_indices,fmt='%i')
    return [train_indices,test_indices]


def rescale_abiotic_features(features,feature_set='all'):
    if feature_set=='public': # without precipitation and temperature data
        scale_min = np.array([-180., 25., 0., -4880, 1.06, 297.6])
        scale_max = np.array([-52., 80., 30., 2680, 10.12, 725.83])
    else:
        scale_min = np.array([-180., 25., 0., 0.02, -47.57, -4880, 1.06, 297.6])
        scale_max = np.array([-52., 80., 30., 18.69, 41.03, 2680, 10.12, 725.83])
    scaled_features = (features-scale_min)/(scale_max-scale_min)
    return scaled_features


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


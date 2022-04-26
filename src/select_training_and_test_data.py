#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:00:10 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(1234)

# test set always the same, containing same number of paleo and current labels
size_test = 50
# define test set indices of equal size paleo and current data
additional_features = np.load('data/abiotic_features.npy')
paleo_indices = np.where(additional_features[:,2]>0)[0]
current_indices = np.where(additional_features[:,2]==0)[0]
paleo_test_indices = np.random.choice(paleo_indices,size_test,replace=False)
current_test_indices = np.random.choice(current_indices,size_test,replace=False)
test_indices = np.concatenate([paleo_test_indices,current_test_indices])
np.random.shuffle(test_indices)
np.savetxt('data/train_test_sets/test_instance_indices.txt',test_indices,fmt='%i')


# extract training sets with differing numbers of n_current, but always same paleo labels
n_current = 10998
n_paleo = 281
paleo_train_indices = np.array([i for i in paleo_indices if i not in paleo_test_indices])
current_train_indices = np.array([i for i in current_indices if i not in current_test_indices])
paleo_train_indices = np.random.choice(paleo_train_indices,n_paleo,replace=False)
current_train_indices = np.random.choice(current_train_indices,n_current,replace=False)

train_indices = np.concatenate([paleo_train_indices,current_train_indices])
np.random.shuffle(train_indices)
np.savetxt('data/train_test_sets/train_instance_indices_ncurrent_%i_npaleo_%i.txt'%(n_current,n_paleo),train_indices,fmt='%i')

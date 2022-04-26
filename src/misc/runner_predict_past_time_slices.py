#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:31:14 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import pickle
# load BNN modules
from np_bnn import BNN_lib
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features, make_colormap


np.set_printoptions(suppress=True)
np.random.seed(1234)


# load the weights of the trained bnn
weight_pickle = '/Users/tobias/GitHub/feature_gen_paleoveg/results/testing_lik_temp/lik_temp_10.0/BNNMC3_p1_h0_l32_8_s1_binf_5083.pkl'
# load weights
n_burnin=0
multiple_weights_per_species = True

all_weights = pickle.load(open(weight_pickle,'rb'))
all_weights = all_weights[n_burnin:]
feature_weights = [i['additional_prm'] for i in all_weights]
bnn_weights = [i['weights'] for i in all_weights]
alpha_params = [i['alphas'] for i in all_weights]



plotting_cutoffs = [0.05,0.95]
with PdfPages(os.path.join(os.path.dirname(weight_pickle),'vegetation_prediction_%.2f_%.2f_timeslices.pdf'%(plotting_cutoffs[0],plotting_cutoffs[1]))) as pdf:
    for timepoint in np.arange(31):
        data_folder = '/Users/tobias/GitHub/feature_gen_paleoveg/data/time_slice_features'
        spatial_dists = os.path.join(data_folder,"spatial_distances_%iMA.pkl"%timepoint)
        temporal_dists = os.path.join(data_folder,"temporal_distances_%iMA.pkl"%timepoint)
        additional_features = np.load(os.path.join(data_folder,"abiotic_features_%iMA.npy"%timepoint))
        scaled_additional_features = rescale_abiotic_features(additional_features)
        
        
        # intialize the PredictFeatures object
        pred_features = PredictFeatures(spatial_dists,temporal_dists,scaled_additional_features,multiple_weights_per_species=multiple_weights_per_species)   
        labels = []
        # predict labels with each of the posterior weight samples
        predicted_labels_out_file =  os.path.join(data_folder,"predicted_labels_%iMA.npy"%timepoint)
        for i,feature_weights_rep in enumerate(feature_weights):
            print(i)
            bnn_weights_rep = bnn_weights[i]
            pred_features.update_weigths(feature_weights_rep[0],feature_weights_rep[1])
            features = pred_features.get_features_unseen_data()
            alpha_params_rep = alpha_params[i]
            activation_function = BNN_lib.genReLU(prm=alpha_params_rep, trainable=True) # To use default ReLU: BNN_lib.genReLU()
            pred_label_probs = BNN_lib.RunPredict(features, bnn_weights_rep,actFun=activation_function)
            predicted_labels = np.argmax(pred_label_probs, axis=1)
            labels.append(predicted_labels) 
        labels = np.array(labels)
        np.save(predicted_labels_out_file,labels)
        
        # calculate posterior probs
        posterior_probs_cat_1 = np.sum(labels,axis=0)/len(labels)
        posterior_probs_cat_0 = 1-posterior_probs_cat_1
        label_posterior_probs = np.vstack([posterior_probs_cat_0,posterior_probs_cat_1]).T
        coords = additional_features[:,:2]
        
        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap([c('darkgreen'), plotting_cutoffs[0], c('grey'), plotting_cutoffs[1], c('goldenrod')])
        
        fig = plt.figure(figsize=(10,5))
        plt.scatter(coords[:,0],coords[:,1],c=posterior_probs_cat_1,cmap=rvb,marker=',',s=2)
        plt.title('Predicted vegetation %i MA'%timepoint)
        
        #fig.savefig(os.path.join(os.path.dirname(weight_pickle),'vegetation_prediction_%.2f_%.2f_timeslices.pdf'%(plotting_cutoffs[0],plotting_cutoffs[1])))
        pdf.savefig(fig)
        plt.close()




# # parallelization libraries
# import multiprocessing
# import multiprocessing.pool
# from functools import partial
# CPUs=30
# pool = multiprocessing.Pool(CPUs)
# args = list(enumerate(training_data_space_time_points))
# all_output = np.array(pool.map(partial(extract_features), args))
# pool.close()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:19:55 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle

# load BNN modules
from np_bnn import BNN_lib
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features, make_colormap


# set seed and working directory and get pkl file list
try:
    # this option is useful when parsing a single seed into the script
    # instead of running the loop for many seeds consecutively in this script
    data_folder = str(sys.argv[1])
    weight_folder = str(sys.argv[2])
    all_seeds = [int(sys.argv[3])]
    all_pkl_files = glob.glob(os.path.join(weight_folder,'*_%i_feature_weights.pkl'%all_seeds[0]))
except:
    # alternatively define a weight_folder and loop through all pkl files found in the dir
    weight_folder = '/Users/tobias/GitHub/feature_gen_paleoveg/results/n107_10/prior_0/'
    data_folder = '/Users/tobias/GitHub/feature_gen_paleoveg/data'
    all_pkl_files = glob.glob(os.path.join(weight_folder,'*.pkl'))
    all_seeds = [int(os.path.basename(i).split('.pkl')[0].split('_')[-1]) for i in all_pkl_files]


calc_accuracies_testset = True
make_accuracy_plot = False
predict_current_map = True
load_label_probs = False
n_burnin = 0

# define data files (must be in same weight_folder folder as pkl files with weights)
spatial_dists = os.path.join(data_folder,"spatial_distances_NN_input.pkl")
temporal_dists = os.path.join(data_folder,"temporal_distances_NN_input.pkl")
additional_features = np.load(os.path.join(data_folder,"abiotic_features.npy"))
scaled_additional_features = rescale_abiotic_features(additional_features)
true_labels_file = os.path.join(data_folder,'veg_labels.txt')
true_labels = np.loadtxt(true_labels_file).astype(int)


master_test_accuracy_df = pd.DataFrame(columns=['run_seed','all_mean','all_std','paleo_mean','paleo_std','current_mean','current_std'])
master_acc_df = os.path.join(weight_folder,'test_acc_all_runs.txt')
for seed in all_seeds:
    np.random.seed(seed)  
    # get the pickle files
    weight_pickle = [i for i in all_pkl_files if '_%i.pkl'%seed in i][0]
           
    # load weights
    all_weights = pickle.load(open(weight_pickle,'rb'))
    all_weights = all_weights[n_burnin:]
    feature_weights = [i[-1] for i in all_weights]
    bnn_weights = [i[:-1] for i in all_weights]
    alphas = np.zeros(len(bnn_weights[0]))
    activation_function = BNN_lib.genReLU()#BNN_lib.genReLU(prm=alphas, trainable=True) # To use default ReLU: BNN_lib.genReLU()    

    if calc_accuracies_testset:
        try:
            test_indices_file = os.path.join(weight_folder,'test_instance_indices_seed_%i.txt'%seed)
            test_indices = np.loadtxt(test_indices_file).astype(int)
        except IOError:
            print('Using default test indices')
            # this is in case we are summarizing a run with constant data, where the indices are stored in the data folder
            test_indices_file = os.path.join(data_folder,'test_instance_indices.txt')
            test_indices = np.loadtxt(test_indices_file).astype(int)            
        
        test_labels = true_labels[test_indices]
        
        # select the paleo and current indices to summarize accuracy separatly for these subsets
        paleo_indices = np.where(additional_features[test_indices][:,2]>0)[0]
        current_indices = np.where(additional_features[test_indices][:,2]==0)[0]
        
        # intialize the PredictFeatures object
        pred_features = PredictFeatures(spatial_dists,temporal_dists,scaled_additional_features,instance_index=test_indices)
        # go through posterior
        accuracies = []
        accuracies_paleo = []
        accuracies_current = []
    
        for i,feature_weights_rep in enumerate(feature_weights):
            print(i)
            bnn_weights_rep = bnn_weights[i]
            pred_features.update_weigths(feature_weights_rep[0],feature_weights_rep[1])
            features = pred_features.get_features_unseen_data()
            pred_label_probs = BNN_lib.RunPredict(features, bnn_weights_rep, actFun=activation_function)
            predicted_labels = np.argmax(pred_label_probs, axis=1)
            accuracy = len(predicted_labels[predicted_labels==test_labels])/len(predicted_labels)
            accuracy_paleo = len(predicted_labels[paleo_indices][predicted_labels[paleo_indices]==test_labels[paleo_indices]])/len(predicted_labels[paleo_indices])
            accuracy_current = len(predicted_labels[current_indices][predicted_labels[current_indices]==test_labels[current_indices]])/len(predicted_labels[current_indices])
            accuracies.append(accuracy)
            accuracies_paleo.append(accuracy_paleo)
            accuracies_current.append(accuracy_current)
            
        test_accuracy_scores = np.matrix([seed,np.mean(accuracies),np.std(accuracies),np.mean(accuracies_paleo),np.std(accuracies_paleo),np.mean(accuracies_current),np.std(accuracies_current)])
        test_accuracy_df = pd.DataFrame(test_accuracy_scores,columns=['run_seed','all_mean','all_std','paleo_mean','paleo_std','current_mean','current_std'])
        test_accuracy_df = np.round(test_accuracy_df,3)
        test_accuracy_df = test_accuracy_df.astype({"run_seed": int})
        sample_acc_df = os.path.join(weight_folder,'test_acc_seed_%i.txt'%seed)
        test_accuracy_df.to_csv(sample_acc_df,sep='\t',index=False)
        # update the master df to collect all acc scores in one file
        master_test_accuracy_df = pd.concat([master_test_accuracy_df,test_accuracy_df])
        master_test_accuracy_df.to_csv(master_acc_df,sep='\t',index=False)
        
        if make_accuracy_plot:
            fig = plt.figure(figsize=(15,5))
            bin_size = 0.02
            min_value = np.min(accuracies+accuracies_paleo+accuracies_current)
            max_value = np.max(accuracies+accuracies_paleo+accuracies_current)+bin_size
        
            fig.add_subplot(131)
            plt.title('All')
            plt.hist(accuracies,np.arange(min_value,max_value,bin_size),color='green')
            plt.axvline(np.mean(accuracies),color='green',linestyle='--')
            plt.axvline(np.mean(accuracies_paleo),color='blue',linestyle='--')
            plt.axvline(np.mean(accuracies_current),color='orange',linestyle='--')
        
            fig.add_subplot(132)
            plt.title('Only paleo')
            plt.hist(accuracies_paleo,np.arange(min_value,max_value,bin_size),color='blue')
            plt.axvline(np.mean(accuracies),color='green',linestyle='--')
            plt.axvline(np.mean(accuracies_paleo),color='blue',linestyle='--')
            plt.axvline(np.mean(accuracies_current),color='orange',linestyle='--')
        
            fig.add_subplot(133)
            plt.title('Only current')
            plt.hist(accuracies_current,np.arange(min_value,max_value,bin_size),color='orange')
            plt.axvline(np.mean(accuracies),color='green',linestyle='--')
            plt.axvline(np.mean(accuracies_paleo),color='blue',linestyle='--')
            plt.axvline(np.mean(accuracies_current),color='orange',linestyle='--')
            
            fig.savefig(os.path.join(weight_folder,'test_acc_seed_%i.pdf'%seed))
        

    if predict_current_map:
        # select only current points
        current_indices = np.where(additional_features[:,2]==0)[0]
        current_labels = true_labels[current_indices]
        predicted_labels_out_file = os.path.join(weight_folder,'predicted_labels_current_grid_seed_%i.npy'%seed)
        
        if load_label_probs:
            labels = np.load(predicted_labels_out_file)     
            
        else:
            # intialize the PredictFeatures object
            pred_features = PredictFeatures(spatial_dists,temporal_dists,scaled_additional_features,instance_index=current_indices)   
            accuracies = []
            labels = []
            for i,feature_weights_rep in enumerate(feature_weights):
                print(i)
                bnn_weights_rep = bnn_weights[i]
                pred_features.update_weigths(feature_weights_rep[0],feature_weights_rep[1])
                features = pred_features.get_features_unseen_data()
                pred_label_probs = BNN_lib.RunPredict(features, bnn_weights_rep, actFun=activation_function)
                predicted_labels = np.argmax(pred_label_probs, axis=1)
                labels.append(predicted_labels)
                accuracy = len(predicted_labels[predicted_labels==current_labels])/len(predicted_labels)
                accuracies.append(accuracy)    
            labels = np.array(labels)
            np.save(predicted_labels_out_file,labels)

        posterior_probs_cat_1 = np.sum(labels,axis=0)/len(labels)
        posterior_probs_cat_0 = 1-posterior_probs_cat_1
        label_posterior_probs = np.vstack([posterior_probs_cat_0,posterior_probs_cat_1]).T
        coords = additional_features[current_indices][:,:2]
        
        # load real current vegetation data
        current_veg_empirical_data = pd.read_csv('/Users/tobias/GitHub/feature_gen_paleoveg/data/raw/vegetation_data/current_vegetation_north_america.txt',sep='\t')
        current_veg_labels = current_veg_empirical_data.veg.values        
    
        
        plotting_cutoffs = [0.05,0.95]
        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap([c('darkgreen'), plotting_cutoffs[0], c('grey'), plotting_cutoffs[1], c('goldenrod')])
        
        fig = plt.figure(figsize=(15,5))
        fig.add_subplot(121)
        plt.scatter(coords[:,0],coords[:,1],c=posterior_probs_cat_1,cmap=rvb,marker=',',s=2)
        plt.title('Predicted vegetation')
        fig.add_subplot(122)
        plt.scatter(coords[:,0],coords[:,1],c=current_veg_labels,cmap=rvb,marker=',',s=2)
        plt.title('True vegetation')
        fig.savefig(os.path.join(weight_folder,'current_grid_prediction_%.2f_%.2f_seed_%i.pdf'%(plotting_cutoffs[0],plotting_cutoffs[1],seed)))












        
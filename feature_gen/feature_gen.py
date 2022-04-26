#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:37:39 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

# add feature names/labels to input
# get index of plant and animal features
# sum together values for all mammlas and all plants respectively


import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import scipy
from scipy import ndimage

class FeatureGenerator():
    def __init__(self,
                 spat_dists_file,
                 temp_dists_file,
                 additional_features,
                 labels,
                 feature_group_ids,
                 instance_index = [],
                 testsize=0.1,
                 seed=1234,
                 pickle_file = '',
                 transform_labels = False,
                 priorscale=1,
                 prior_f=1,
                 multiple_weights_per_species=False,
                 sum_faunal_floral_features=False,
                 max_pooling=False,
                 actfun=2): #0 is none, 1 is relu, 2 is swish
        # load the distance data
        spatial_distances = pickle.load(open(spat_dists_file,'rb'))
        temporal_distances = pickle.load(open(temp_dists_file,'rb'))
        label_values = np.loadtxt(labels,dtype=int)
        if transform_labels:
            label_values_numeric, original_label_values = self.turn_labels_to_numeric(label_values)
        else:
            label_values_numeric = label_values
            original_label_values = np.nan
        # select specific instances for training
        if len(instance_index)>0:
            spatial_distances = [spatial_distances[i] for i in instance_index]
            temporal_distances = [temporal_distances[i] for i in instance_index]
            additional_features = np.array([additional_features[i] for i in instance_index])
            label_values_numeric = np.array([label_values_numeric[i] for i in instance_index])
        n_veg_points = len(spatial_distances)
        n_fossil_taxa = len(spatial_distances[0])
        self._spatial_distances = spatial_distances
        self._temporal_distances = temporal_distances
        self._additional_features = additional_features
        self._labels = label_values_numeric
        self._n_veg_points = n_veg_points
        self._n_fossil_taxa = n_fossil_taxa
        self._feature_group_ids = feature_group_ids
        self._sum_faunal_floral_features = sum_faunal_floral_features
        self._max_pooling = max_pooling
        self._actfun = actfun
        input_shapes = [i.shape[0] for i in spatial_distances[0]]
        if seed != -1:
            self._seed = seed
            np.random.seed(self._seed)
        if pickle_file == '':
            self._w1 = np.random.normal(0, 0.1, 2)
            if not multiple_weights_per_species:
                self._w2 = np.random.normal(0, 0.1, n_fossil_taxa)
            else:
                self._w2 = np.random.normal(0, 0.1, sum(input_shapes))
        else:
            feature_weights = pickle.load(open(pickle_file,'rb'))
            self._w1 = feature_weights[-1][0]
            self._w2 = feature_weights[-1][1]
        split_points = np.array([np.cumsum(input_shapes)+(i*sum(input_shapes)) for i in range(n_veg_points)]).flatten()
        taxon_id_array = np.array(list(range(n_fossil_taxa))*n_veg_points)
        # create array with id's of taxon names
        species_index_array = np.zeros(sum(input_shapes)*n_veg_points).astype(int)
        old_index = 0
        for i,index in enumerate(split_points):
            species_index_array[old_index:index] = taxon_id_array[i]
            old_index = index
        index_array_tmp = np.arange(sum(input_shapes))
        index_array = np.tile(index_array_tmp, n_veg_points)
        if multiple_weights_per_species:        
            self._index_array = index_array
        else:
            self._index_array = species_index_array
        index_array_per_instance = species_index_array.reshape(n_veg_points,int(len(species_index_array)/n_veg_points))
        id_temp = np.array([i*(np.max(species_index_array)+1)+row for i,row in enumerate(index_array_per_instance)])
        #print(id_temp.shape,id_temp)
        index_instance_array = id_temp.reshape(id_temp.shape[0]*id_temp.shape[1])
        self._index_instance_array = index_instance_array
        train_test_index_array = np.zeros(n_veg_points).astype(int)
        train_test_index_array[np.random.choice(np.arange(n_veg_points),size=int(n_veg_points*testsize),replace=False)] = 1
        self._train_test_index = train_test_index_array
        self._original_label_values = original_label_values
        self._priorscale = priorscale
        self._priorf = prior_f

    def get_data(self):
        self.update_features()
        training_features = self._features[self._train_test_index==0]
        test_features = self._features[self._train_test_index==1]
        train_labels = self._labels[self._train_test_index==0]
        test_labels = self._labels[self._train_test_index==1]
        data = {'data':training_features,
                'labels':train_labels,
                'label_dict':np.unique(self._labels), # normally
                'test_data':test_features,
                'test_labels':test_labels,
                }
        prior = self.get_prior()
        self._prior = prior
        return data,prior

    def update_weigths(self, w1,w2):
        self._w1 = w1
        self._w2 = w2

    def update_features(self):
        start = time.time()
        # ____________summarize time and space distance to each fossil into one value, using same weight for all occurrences of all species        
        # TODO: check if we can refactor this
        # flatten the distance arrays, since this will be done jointly for all occurrences
        all_spatial_dists = np.concatenate(np.concatenate(np.array(self._spatial_distances,dtype=object)))
        all_temporal_dists = np.concatenate(np.concatenate(np.array(self._temporal_distances,dtype=object)))
        all_dists = np.vstack([all_spatial_dists,all_temporal_dists]).T
        # # format the weights vector
        # w_space_time = np.matrix(self._w1).T
        # # matrix multiplication of features and weights
        # joined_space_time_features = np.einsum('nf, fi->n', all_dists, w_space_time) # nx2, 2x1 = n
        joined_space_time_features = np.einsum('nf, f->n', all_dists, self._w1) # nx2, 2 = n
        if self._actfun == 0: # no activation function
            joined_space_time_features = joined_space_time_features
        elif self._actfun == 1: # relu
            joined_space_time_features[joined_space_time_features<0] = 0
        elif self._actfun == 2: #swish
            joined_space_time_features = joined_space_time_features * (1 + np.exp(-joined_space_time_features)) ** (-1)
        else:
            quit('%s not a valid choice for activation function.'%self._actfun)        # ____________summarize distance features to create one single value for each species
        # group the output values by species
        final_features = self.get_features_by_species(joined_space_time_features)
        final_features = np.hstack([final_features,self._additional_features])
        end = time.time()
        self._time = end-start
        self._features = final_features
        return final_features    


    def get_features_by_species(self,joined_space_time_features):
        # use the index array to repeat the weight values appropiately to use as a factor for the joined_space_time_features
        weight_distance_product = joined_space_time_features * self._w2[self._index_array]    
        # now sum up the distance-weight products for each instance and each species to one value per species
        final_features_tmp = ndimage.sum(weight_distance_product, self._index_instance_array, index=np.unique(self._index_instance_array))
        if self._actfun == 0: # no activation function
            final_features_tmp = final_features_tmp
        elif self._actfun == 1: # relu
            final_features_tmp[final_features_tmp<0] = 0
        elif self._actfun == 2: #swish
            final_features_tmp = final_features_tmp * (1 + np.exp(-final_features_tmp)) ** (-1)
        else:
            quit('%s not a valid choice for activation function.'%self._actfun)
        # reshpae array so that each row is a seaprate instance
        final_features = final_features_tmp.reshape(self._n_veg_points,int(len(final_features_tmp)/self._n_veg_points))
        if self._sum_faunal_floral_features:
            # sum up faunal and floral features into one feature value each
            final_features_merged = []
            for i in np.unique(self._feature_group_ids):
                features_i = final_features[:, self._feature_group_ids == i]
                features_merged_i = np.sum(features_i, axis=1)
                final_features_merged.append(features_merged_i)
            final_features_merged = np.array(final_features_merged).T
            return final_features_merged
        elif self._max_pooling:
            # sum up faunal and floral features into one feature value each
            final_features_merged = []
            for i in np.unique(self._feature_group_ids):
                features_i = final_features[:, self._feature_group_ids == i]
                features_merged_i = np.max(features_i, axis=1)
                final_features_merged.append(features_merged_i)
            final_features_merged = np.array(final_features_merged).T
            return final_features_merged
        else:
            return final_features
    
    def turn_labels_to_numeric(self,labels):
        numerical_labels = np.zeros(len(labels)).astype(int)
        c = 0
        original_label_values = np.unique(labels)
        for i in original_label_values:
            numerical_labels[labels == i] = c
            c += 1
        return numerical_labels,original_label_values

    def get_prior(self):
        w1 = self._w1
        w2 = self._w2
        if self._priorf==0:
            # uniform prior
            log_prior_w1 = np.sum(scipy.stats.uniform.logpdf(w1, loc=0-self._priorscale, scale=(self._priorscale)*2))
            log_prior_w2 = np.sum(scipy.stats.uniform.logpdf(w2, loc=0-self._priorscale, scale=(self._priorscale)*2))            
        elif self._priorf==1:
            # normal prior
            log_prior_w1 = np.sum(scipy.stats.norm.logpdf(w1, 0, scale=self._priorscale))
            log_prior_w2 = np.sum(scipy.stats.norm.logpdf(w2, 0, scale=self._priorscale))
        elif self._priorf==2:
            # cauchy prior
            log_prior_w1 = np.sum(scipy.stats.cauchy.logpdf(w1, 0, scale=self._priorscale))
            log_prior_w2 = np.sum(scipy.stats.cauchy.logpdf(w2, 0, scale=self._priorscale))
        elif self._priorf==3:
            # laplace prior
            log_prior_w1 = np.sum(scipy.stats.laplace.logpdf(w1, 0, scale=self._priorscale))
            log_prior_w2 = np.sum(scipy.stats.laplace.logpdf(w2, 0, scale=self._priorscale)) 
        prior_prob_feature_weights = sum([log_prior_w1,log_prior_w2])
        return prior_prob_feature_weights
    

class PredictFeatures():
    def __init__(self,
                 spat_dists_file,
                 temp_dists_file,
                 additional_features,
                 feature_group_ids,
                 instance_index=[],
                 multiple_weights_per_species=False,
                 sum_faunal_floral_features=False,
                 max_pooling=False,
                 actfun=2): #0 is none, 1 is relu, 2 is swish
        # load the distance data       
        spatial_distances = pickle.load(open(spat_dists_file,'rb'))
        temporal_distances = pickle.load(open(temp_dists_file,'rb'))
        if len(instance_index)>0:
            spatial_distances = [spatial_distances[i] for i in instance_index]
            temporal_distances = [temporal_distances[i] for i in instance_index]
            additional_features = np.array([additional_features[i] for i in instance_index])
        n_fossil_taxa = len(spatial_distances[0])
        n_veg_points = len(spatial_distances)
        input_shapes = [i.shape[0] for i in spatial_distances[0]]
        split_points = np.array([np.cumsum(input_shapes)+(i*sum(input_shapes)) for i in range(n_veg_points)]).flatten()
        taxon_id_array = np.array(list(range(n_fossil_taxa))*n_veg_points)
        # create array with id's of taxon names
        species_index_array = np.zeros(sum(input_shapes)*n_veg_points).astype(int)
        old_index = 0
        for i,index in enumerate(split_points):
            species_index_array[old_index:index] = taxon_id_array[i]
            old_index = index
        index_array_tmp = np.arange(sum(input_shapes))
        index_array = np.tile(index_array_tmp, n_veg_points)
        if multiple_weights_per_species:        
            self._index_array = index_array
        else:
            self._index_array = species_index_array

        index_array_per_instance = species_index_array.reshape(n_veg_points,int(len(species_index_array)/n_veg_points))
        id_temp = np.array([i*(np.max(species_index_array)+1)+row for i,row in enumerate(index_array_per_instance)])
        #print(id_temp.shape,id_temp)
        index_instance_array = id_temp.reshape(id_temp.shape[0]*id_temp.shape[1])
        self._index_instance_array = index_instance_array
        self._w1 = np.random.normal(0, 0.1, 2)
        self._w2 = np.random.normal(0, 0.1, n_fossil_taxa)
        self._spatial_distances = spatial_distances
        self._temporal_distances = temporal_distances
        self._additional_features = additional_features
        self._n_fossil_taxa = n_fossil_taxa
        self._n_veg_points = n_veg_points
        self._feature_group_ids = feature_group_ids
        self._sum_faunal_floral_features = sum_faunal_floral_features
        self._max_pooling = max_pooling
        self._actfun = actfun
        
        
    def update_weigths(self, w1,w2):
        self._w1 = w1
        self._w2 = w2


    def get_features_unseen_data(self):
        start = time.time()
        # ____________summarize time and space distance to each fossil into one value, using same weight for all occurrences of all species        
        # TODO: check if we can refactor this
        # flatten the distance arrays, since this will be done jointly for all occurrences
        all_spatial_dists = np.concatenate(np.concatenate(self._spatial_distances))
        all_temporal_dists = np.concatenate(np.concatenate(self._temporal_distances))
        all_dists = np.vstack([all_spatial_dists,all_temporal_dists]).T
        # # format the weights vector
        # w_space_time = np.matrix(self._w1).T
        # # matrix multiplication of features and weights
        # joined_space_time_features = np.einsum('nf, fi->n', all_dists, w_space_time) # nx2, 2x1 = n
        joined_space_time_features = np.einsum('nf, f->n', all_dists, self._w1) # nx2, 2 = n
        if self._actfun == 0: # no activation function
            joined_space_time_features = joined_space_time_features
        elif self._actfun == 1: # relu
            joined_space_time_features[joined_space_time_features<0] = 0
        elif self._actfun == 2: #swish
            joined_space_time_features = joined_space_time_features * (1 + np.exp(-joined_space_time_features)) ** (-1)
        else:
            quit('%s not a valid choice for activation function.'%self._actfun)
        # ____________summarize distance features to create one single value for each species
        final_features = self.get_features_by_species(joined_space_time_features)
        final_features = np.hstack([final_features,self._additional_features])        
        end = time.time()
        self._time = end-start
        self._features = final_features
        return final_features


    def get_features_by_species(self,joined_space_time_features):
        # use the index array to repeat the weight values appropiately to use as a factor for the joined_space_time_features
        weight_distance_product = joined_space_time_features * self._w2[self._index_array]
        # now sum up the distance-weight products for each instance and each species to one value per species
        final_features_tmp = ndimage.sum(weight_distance_product, self._index_instance_array, index=np.unique(self._index_instance_array))
        if self._actfun == 0: # no activation function
            final_features_tmp = final_features_tmp
        elif self._actfun == 1: # relu
            final_features_tmp[final_features_tmp<0] = 0
        elif self._actfun == 2: #swish
            final_features_tmp = final_features_tmp * (1 + np.exp(-final_features_tmp)) ** (-1)
        else:
            quit('%s not a valid choice for activation function.'%self._actfun)
        # reshpae array so that each row is a seaprate instance
        final_features = final_features_tmp.reshape(self._n_veg_points,int(len(final_features_tmp)/self._n_veg_points))
        if self._sum_faunal_floral_features:
            # sum up faunal and floral features into one feature value each
            final_features_merged = []
            for i in np.unique(self._feature_group_ids):
                features_i = final_features[:, self._feature_group_ids == i]
                features_merged_i = np.sum(features_i, axis=1)
                final_features_merged.append(features_merged_i)
            final_features_merged = np.array(final_features_merged).T
            ## add another relu layer
            #final_features_merged[final_features_merged < 0] = 0
            final_features = final_features_merged
        elif self._max_pooling:
            # sum up faunal and floral features into one feature value each
            final_features_merged = []
            for i in np.unique(self._feature_group_ids):
                features_i = final_features[:, self._feature_group_ids == i]
                features_merged_i = np.max(features_i, axis=1)
                final_features_merged.append(features_merged_i)
            final_features_merged = np.array(final_features_merged).T
            ## add another relu layer
            #final_features_merged[final_features_merged < 0] = 0
            final_features = final_features_merged
        return final_features









# # numba function to parallelize computation    
# @jit(nopython=True, parallel=True)
# def get_features_by_species_numba(joined_space_time_features,final_features,index_array,w2):
#     for i in np.arange(final_features.shape[1]):
#         species_features = joined_space_time_features[index_array==i]
#         species_features_array = species_features.reshape(final_features.shape[0],int(len(species_features)/final_features.shape[0]))
#         # w_taxon = np.matrix(self._w2[i])      
#         # final_features[:,i] = np.einsum('nf, ni->n', species_features_array, w_taxon)
#         final_features[:,i] = np.sum(species_features_array * w2[i], axis=1)
#     return final_features

# joined_space_time_features = np.array([0,1,2,3,4,5,6,7,8,9])
# index_array = np.array([0,0,1,1,2,0,0,1,1,2])
# w2 = np.array([-10,10,-100])

# final_features = np.zeros([2,3])




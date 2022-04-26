#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:27:16 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
np.set_printoptions(suppress=True)
np.random.seed(1234)



# ___________load data
### REAL DATA
# load the empirical data
spatial_distances = pickle.load(open("/Users/tobias/GitHub/feature_gen_paleoveg/data/spatial_distances_NN_input.pkl",'rb'))
temporal_distances = pickle.load(open("/Users/tobias/GitHub/feature_gen_paleoveg/data/temporal_distances_NN_input.pkl",'rb'))
n_veg_points = len(spatial_distances)
n_fossil_taxa = len(spatial_distances[0])

input_shapes = [i.shape[0] for i in spatial_distances[0]]
split_points = np.array([np.cumsum(input_shapes)+(i*sum(input_shapes)) for i in range(n_veg_points)]).flatten()
taxon_id_array = np.array(list(range(n_fossil_taxa))*n_veg_points)
# create array with id's of taxon names
index_array = np.zeros(sum(input_shapes)*n_veg_points)
old_index = 0
for i,index in enumerate(split_points):
    index_array[old_index:index] = taxon_id_array[i]
    old_index = index


start = time.time()
# ____________summarize time and space distance to each fossil into one value, using same weight for all occurrences of all species
# also keep track of the input shapes, since we will have different numbers of occs for each taxon. the shapes should be the same for all veg-points, so we only extract the first one here
w1 = np.random.random(2)
# flatten the distance arrays, since this will be done jointly for all occurrences
all_spatial_dists = np.concatenate(np.concatenate(spatial_distances))
all_temporal_dists = np.concatenate(np.concatenate(temporal_distances))
all_dists = np.vstack([all_spatial_dists,all_temporal_dists]).T
# format the weights vector
w_space_time = np.matrix(w1).T
# matrix multiplication of features and weights
joined_space_time_features = np.einsum('nf, fi->n', all_dists, w_space_time) # nx2, 2x1 = n


# ____________summarize distance features to create one single value for each species
w2 = np.random.random(n_fossil_taxa)
final_features = np.zeros([n_veg_points,n_fossil_taxa])
# group the output values by species
for i in range(n_fossil_taxa):
    species_features = joined_space_time_features[index_array==i]
    species_features_array = species_features.reshape(n_veg_points,int(len(species_features)/n_veg_points))
    w_taxon = np.matrix(w2[i])      
    final_features[:,i] = np.einsum('nf, ni->n', species_features_array, w_taxon)

end = time.time()
print(end-start)



















# shared_species_weights = True

# # ___________load data
# ### REAL DATA
# # load the empirical data
# spatial_distances = pickle.load(open("/Users/tobias/GitHub/feature_gen_paleoveg/data/spatial_distances_NN_input.pkl",'rb'))
# temporal_distances = pickle.load(open("/Users/tobias/GitHub/feature_gen_paleoveg/data/temporal_distances_NN_input.pkl",'rb'))
# n_veg_points = len(spatial_distances)
# n_fossil_taxa = len(spatial_distances[0])


# ### RANDOM FAKE DATA
# # # generate fake data
# # n_veg_points = 3
# # n_fossil_taxa = 5
# # fossils_per_taxon = 10
# # spatial_distances = np.array([np.array([np.random.random(fossils_per_taxon) for i in range(n_fossil_taxa)]) for i in range(n_veg_points)])
# # temporal_distances =  np.array([np.array([np.random.random(fossils_per_taxon) for i in range(n_fossil_taxa)]) for i in range(n_veg_points)])

# ### MANUALLY CREATED FAKE DATA
# # manually create data with different numbers of occurrences for each species
# # spatial_distances = np.array([
# #                      np.array([np.array([0.2,0.3,0.5,0.4]),np.array([0.2,0.3]),np.array([0.2,0.5,0.4]),np.array([0.5,0.4])]),
# #                      np.array([np.array([0.2,0.3,0.5,0.4]),np.array([0.2,0.3]),np.array([0.2,0.5,0.4]),np.array([0.5,0.4])]),
# #                      np.array([np.array([0.2,0.3,0.5,0.4]),np.array([0.2,0.3]),np.array([0.2,0.5,0.4]),np.array([0.5,0.4])]) 
# #                      ])
# # temporal_distances = spatial_distances
# # n_veg_points = len(spatial_distances)
# # n_fossil_taxa = len(spatial_distances[0])


# start = time.time()
# # ____________summarize time and space distance to each fossil into one value, using same weight for all occurrences of all species
# # also keep track of the input shapes, since we will have different numbers of occs for each taxon. the shapes should be the same for all veg-points, so we only extract the first one here
# input_shapes = [i.shape[0] for i in spatial_distances[0]]
# w1 = np.random.random(2)
# # flatten the distance arrays, since this will be done jointly for all occurrences
# all_spatial_dists = np.concatenate(np.concatenate(spatial_distances))
# all_temporal_dists = np.concatenate(np.concatenate(temporal_distances))
# all_dists = np.vstack([all_spatial_dists,all_temporal_dists]).T
# # format the weights vector
# w_space_time = np.matrix(w1).T
# # matrix multiplication of features and weights
# joined_space_time_features = np.einsum('nf, fi->n', all_dists, w_space_time) # nx2, 2x1 = n
# # group the output values again by species
# space_time_features_veg_points = np.array_split(joined_space_time_features, n_veg_points, axis=0) # split by veg point
# space_time_features_by_veg_points_and_taxon = np.array([np.array(np.array_split(i, np.cumsum(input_shapes))[:-1]) for i in space_time_features_veg_points])
# # create array with id's of taxon names


# # split_points = np.array([np.cumsum(input_shapes)+(i*sum(input_shapes)) for i in range(n_veg_points)]).flatten()
# # features_by_species = np.array(np.array(np.array_split(joined_space_time_features, split_points)[:-1]))
# # features_by_species_joined = [np.vstack(features_by_species[i::n_fossil_taxa][:]).astype(float) for i in range(n_fossil_taxa)]
# # len(features_by_species_joined[9][1])



# # ____________summarize distance features to create one single value for each species
# if shared_species_weights:
#     w2 = np.random.random(n_fossil_taxa)
#     features_by_taxon = space_time_features_by_veg_points_and_taxon.T
#     #final_features = np.array([np.einsum('nf, ni->n', np.vstack(species_features[:]).astype(float), np.matrix(w2[i])) for i,species_features in enumerate(features_by_taxon)]).T
#     final_features = []
#     for i,species_features in enumerate(features_by_taxon):
#         species_features_array = np.vstack(species_features[:]).astype(float)
#         w_taxon = np.matrix(w2[i])      
#         final_features.append(np.einsum('nf, ni->n', species_features_array, w_taxon)) # 1xn, 1x1 = 1  

#     # this was a previous approach, but much slower
#     # w2 = np.random.random(n_fossil_taxa) 
#     # final_features = np.array([np.array([np.einsum('nf, ni->n', np.matrix(features_vegpoint[taxon]), np.matrix(w2[taxon]).T) for taxon in range(n_fossil_taxa)]).flatten() for i,features_vegpoint in enumerate(space_time_features_by_veg_points_and_taxon)])
#     # # # for each vegetation point
#     # # for i,features_vegpoint in enumerate(space_time_features_by_veg_points_and_taxon):
#     # #     # for each taxon
#     # #     for taxon in range(n_fossil_taxa):
#     # #         w_taxon = np.matrix(w2[taxon]).T
#     # #         features_taxon = np.matrix(features_vegpoint[taxon])       
#     # #         np.einsum('nf, ni->n', features_taxon, w_taxon) # 1xn, 1x1 = 1    
# else:
#     w2 = np.array([np.random.random(n_fossil_taxa) for i in range(n_veg_points)])  
#     final_features = np.array([np.array([np.einsum('nf, ni->n', np.matrix(features_vegpoint[taxon]), np.matrix(w2[i][taxon]).T) for taxon in range(n_fossil_taxa)]).flatten() for i,features_vegpoint in enumerate(space_time_features_by_veg_points_and_taxon)])
#     # # for each vegetation point
#     # for i,features_vegpoint in enumerate(space_time_features_by_veg_points_and_taxon):
#     #     # for each taxon
#     #     for taxon in range(n_fossil_taxa):
#     #         w_taxon = np.matrix(w2[i][taxon]).T
#     #         features_taxon = np.matrix(features_vegpoint[taxon])       
#     #         np.einsum('nf, ni->n', features_taxon, w_taxon) # 1xn, 1x1 = 1

# end = time.time()
# print(end-start)






















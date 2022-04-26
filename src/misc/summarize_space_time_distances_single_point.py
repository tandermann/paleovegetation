#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:05:39 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.random.seed(1234)

n_fossil_taxa = 5
fossils_per_taxon = 10

# ___________generate data, mimicking the empirical temporal and spatial distances to each occurrence for several taxa for one given vegetation point
spatial_distances = np.array([np.random.random(fossils_per_taxon) for i in range(n_fossil_taxa)])
temporal_distances =  np.array([np.random.random(fossils_per_taxon) for i in range(n_fossil_taxa)])


# ____________summarize time and space distance to each fossil into one value, using same weight for all occurrences of all species
w1 = np.random.random(2)
# flatten the distance arrays, since this will be done jointly for all occurrences
all_spatial_dists = spatial_distances.flatten()
all_temporal_dists = temporal_distances.flatten()
all_dists = np.vstack([all_spatial_dists,all_temporal_dists]).T
# also keep track of the input shapes, since we will have different numbers of occs for each taxon
input_shapes = [i.shape[0] for i in spatial_distances]
# format the weights vector
w_space_time = np.matrix(w1).T
# matrix multiplication of features and weights
joined_space_time_features = np.einsum('nf, fi->n', all_dists, w_space_time) # nx2, 2x1 = n
# group the output values again by species
space_time_features_by_taxon = np.array(np.array_split(joined_space_time_features, np.cumsum(input_shapes))[:-1])


# ____________summarize distance features to create one single value for each species
w2 = np.random.random(space_time_features_by_taxon.shape)    
final_features = np.array([np.einsum('nf, fi->n', np.matrix(space_time_features_by_taxon[taxon]), np.matrix(w2[taxon]).T) for taxon in range(n_fossil_taxa)])
# for taxon in range(n_fossil_taxa):
#     w_taxon = np.matrix(w2[taxon]).T
#     features_taxon = np.matrix(space_time_features_by_taxon[taxon])
#     np.einsum('nf, fi->n', features_taxon, w_taxon) # 1xn, nx1 = 1

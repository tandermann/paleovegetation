#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:00:29 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from feature_gen.utils import rescale_abiotic_features, make_colormap

np.set_printoptions(suppress=True)
np.random.seed(1234)


additional_features = np.load('data/abiotic_features.npy')
current_veg_empirical_data = pd.read_csv('data/raw/vegetation_data/current_vegetation_north_america.txt',sep='\t')
current_veg_labels = current_veg_empirical_data.veg.values        
outdir = 'plots'
outfile_name = 'current_veg_map.pdf'


current_indices = np.where(additional_features[:,2]==0)[0]
coords = additional_features[current_indices][:,:2]
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('darkgreen'), c('goldenrod')])

fig = plt.figure(figsize=(10,5))
plt.scatter(coords[:,0],coords[:,1],c=current_veg_labels,cmap=rvb,marker=',',s=2)
plt.title('Current potential vegetation (SYNMAP)')
fig.savefig(os.path.join(outdir,outfile_name))




# get training indices
n_current = 281
n_paleo = 281
training_instance_file = 'data/train_test_sets/train_instance_indices_ncurrent_%i_npaleo_%i.txt'%(n_current,n_paleo)
training_instances = np.loadtxt(training_instance_file,dtype=int)
# extract labels for these instances
veg_labels_file = 'data/veg_labels.txt'
veg_labels = np.loadtxt(veg_labels_file,dtype=int)
selected_labels = veg_labels[training_instances]
# extract coords for these instances from abiotic features
abiotic_features_file = 'data/abiotic_features.npy'
abiotic_features = np.load(abiotic_features_file)
selected_coords = abiotic_features[training_instances,0:2]
current_indices = np.where(abiotic_features[training_instances,2]==0)
# plot on map
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('darkgreen'), c('goldenrod')])

fig = plt.figure(figsize=(10,5))
plt.scatter(coords[:,0],coords[:,1],c='grey',cmap=rvb,marker=',',s=2)
plt.scatter(selected_coords[current_indices,0],selected_coords[current_indices,1],c=selected_labels[current_indices],cmap=rvb,marker=',',s=2)
#plt.title('Training coords')
plt.tight_layout()
fig.savefig('plots/current_training_labels_ncurrent_%i_npaleo_%i.pdf'%(n_current,n_paleo))


c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('darkgreen'), c('goldenrod')])
colors = [
    '#f7fcb9'
    '#d9f0a3'
    '#addd8e'
    '#78c679'
    '#41ab5d'
    '#238443'
    '#006837'
    '#004529'
]


'''
# get training indices
n_current = 281
n_paleo = 281
training_instance_file = 'cluster_content/feature_gen_mc3/data/train_test_sets/train_instance_indices_ncurrent_%i_npaleo_%i.txt'%(n_current,n_paleo)
training_instances = np.loadtxt(training_instance_file,dtype=int)
# extract labels for these instances
veg_labels_file = 'cluster_content/feature_gen_mc3/data/veg_labels.txt'
veg_labels = np.loadtxt(veg_labels_file,dtype=int)
selected_labels = veg_labels[training_instances]
# extract coords for these instances from abiotic features
abiotic_features_file = 'cluster_content/feature_gen_mc3/data/abiotic_features.npy'
abiotic_features = np.load(abiotic_features_file)
selected_coords = abiotic_features[training_instances,0:2]
current_indices = np.where(abiotic_features[training_instances,2]==0)
# get actual predictions
predicted_labels_out_file = 'cluster_content/feature_gen_mc3/runs_2021/current_%i_paleo_%i/time_slice_predictions/predicted_labels/predicted_labels_0MA.npy'%(n_current,n_paleo)
post_prob = np.load(predicted_labels_out_file)
# get coordinates
additional_features = np.load("cluster_content/feature_gen_mc3/data/abiotic_features_%iMA.npy" % timepoint)
coords = additional_features[:,:2]
rvb2 = make_colormap([c('darkgreen'),0.5, c('goldenrod')])
# plot on map
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('darkgreen'), c('goldenrod')])
fig = plt.figure(figsize=(10,5))
#plt.scatter(coords[:,0],coords[:,1],c='grey',cmap=rvb,marker=',',s=2,alpha=0.2)
plt.scatter(coords[:,0],coords[:,1],c=post_prob[:,1],cmap=rvb2,marker=',',s=2,alpha=0.3)
plt.scatter(selected_coords[current_indices,0],selected_coords[current_indices,1],c=selected_labels[current_indices],cmap=rvb,marker=',',s=2)
plt.title('Training coords')
fig.savefig('plots/current_training_labels_ncurrent_%i_npaleo_%i_with_pred.pdf'%(n_current,n_paleo))



predicted_labels_out_file = 'cluster_content/feature_gen_mc3/runs_2021/current_281_paleo_281/time_slice_predictions/predicted_labels/predicted_labels_0MA.npy'
labels = np.load(predicted_labels_out_file)
# get coordinates
abiotic_features_file = 'cluster_content/feature_gen_mc3/data/time_slice_features/abiotic_features_0MA.npy'
additional_features = np.load(abiotic_features_file)

# calculate posterior probs
posterior_probs_cat_1 = np.sum(labels,axis=0)/len(labels)
posterior_probs_cat_0 = 1-posterior_probs_cat_1
label_posterior_probs = np.vstack([posterior_probs_cat_0,posterior_probs_cat_1]).T
coords = additional_features[:,:2]

colors=['darkgreen','grey',Z'goldenrod']
plotting_cutoffs = [0.5,0.5000001]
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c(colors[0]), plotting_cutoffs[0], c(colors[1]), plotting_cutoffs[1], c(colors[2])])

fig = plt.figure(figsize=(10,5))
plt.scatter(coords[:,0],coords[:,1],c=posterior_probs_cat_1,cmap=rvb,marker=',',s=2,alpha=0.3)
plt.scatter(selected_coords[current_indices,0],selected_coords[current_indices,1],c=selected_labels[current_indices],cmap=rvb,marker=',',s=2)
plt.title('Predicted vegetation %i MA'%timepoint)

'''






#from shapely.geometry.polygon import LinearRing, Polygon
#plt.savefig('world.png',dpi=75)


#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as plt
## setup Lambert Conformal basemap.
## set resolution=None to skip processing of boundary datasets.
#m = Basemap(width=12000000,height=9000000,projection='lcc',
#            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
#
#m.bluemarble()
#plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:29:20 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import os

np.set_printoptions(suppress=True)
np.random.seed(1234)


def bin_ages_in_geo_stages_get_mean(age_array):
    geological_stages = np.array([0,0.0042,0.0082,0.0117,0.129,0.774,1.8,2.58,3.6,5.333,7.246,11.63,13.82,15.97,20.44,23.03,27.82,33.9])
    if max(age_array) >= max(geological_stages):
        print('Error in "bin_ages_in_geo_stages_get_mean_stage_age()": No conversion possible. Max age must not exceed oldest geological stage (%.2f)'%max(geological_stages))
    else:
        geological_stages_mid_ages = np.array([(geological_stages[i]+geological_stages[i+1])/2 for i,_ in enumerate(geological_stages[:-1])])
        new_age_array = geological_stages_mid_ages[np.digitize(age_array,geological_stages,right=False)-1]
        return new_age_array
    
def distance_flat(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    delta_x = abs(x1 - x2)
    delta_y = abs(y1 - y2)
    d = (delta_x ** 2 + delta_y ** 2) ** 0.5
    return(d)

# def distance(origin, destination):
#     lon1,lat1 = origin
#     lon2,lat2 = destination
#     radius = 6371  # km
#     dlat = np.radians(lat2 - lat1)
#     dlon = np.radians(lon2 - lon1)
#     a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
#          np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
#          np.sin(dlon / 2) * np.sin(dlon / 2))
#     a=np.amin([a,np.ones(len(a))],axis=0)
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     d = radius * c
#     return d
#
#
# def distance_two_points(origin, destination):
#     lon1,lat1 = origin
#     lon2,lat2 = destination
#     radius = 6371  # km
#     dlat = math.radians(lat2 - lat1)
#     dlon = math.radians(lon2 - lon1)
#     a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
#          math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
#          math.sin(dlon / 2) * math.sin(dlon / 2))
#     a=np.min([a,1.])
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     d = radius * c
#     return d


def round_coordinates_to_quarter_degree(coordinate_pair):
    rounded_coordinates = [np.round(coordinate_pair[0]*2)/2,np.round(coordinate_pair[1]*2)/2]
    if rounded_coordinates[0] < coordinate_pair[0]:
        rounded_coordinates[0] = rounded_coordinates[0]+0.25
    else:
        rounded_coordinates[0] = rounded_coordinates[0]-0.25
    if rounded_coordinates[1] < coordinate_pair[1]:
        rounded_coordinates[1] = rounded_coordinates[1]+0.25
    else:
        rounded_coordinates[1] = rounded_coordinates[1]-0.25
    return rounded_coordinates


def extract_raw_distance_features(coords,binned_age,occ_coords,temp_occ_df,target_species,n_age_bins,max_time):
    # get taxon distances (provide coords and occ_cords in albers projection!!)
    spatial_distances = np.round(distance_flat(coords, occ_coords))
#    spatial_distances = np.round(distance(coords,occ_coords),2)
    temp_occ_df['dists'] = spatial_distances
    # create empty array to fill with final shortest distances
    final_spatial_distances = np.zeros([len(target_species),n_age_bins])
    final_spatial_distances[final_spatial_distances==0] = np.nan
    final_temporal_distances = []
    # for each age bin find the closest spatial distance for each taxon and save those values
    for age_bin_i, age_dfs in enumerate(list(temp_occ_df.groupby('rounded_ages'))):
        age = np.round(age_dfs[0],5)
        temporal_dist = np.abs(binned_age-age)/max_time
        final_temporal_distances.append(temporal_dist)
        group_by_species = age_dfs[1].groupby('species').dists
        min_dists_df = group_by_species.min()
        distances = min_dists_df.values.astype(float)
        species = min_dists_df.index.values.astype(str)
        species_indices = [np.where(target_species==spec)[0][0] for spec in species]
        final_spatial_distances[species_indices,age_bin_i] = distances/max_dist
    return (final_spatial_distances,final_temporal_distances)


def extract_raw_abiotic_features(coords,age,binned_age,paleocoords,elev_files_dict,prec_files_dict,temp_files_dict,paleotemp_global,co2_global):
    # round the coordinates to the nearest .25 or .75 (no .00 or .50!) to match with the grid points
    rounded_paleocoordinates = round_coordinates_to_quarter_degree(paleocoords)
    # also round to full degree for elevation data
    rounded_paleocoordinates_full = np.round(paleocoords)
    # get closest elevation value for the spacetime point
    closest_t_with_elev_value = list(elev_files_dict.keys())[np.abs(np.array(list(elev_files_dict.keys())) - age).argmin()]
    elev_data = elev_files_dict[closest_t_with_elev_value]
    elev = float(elev_data[(elev_data[['lon','lat']] == rounded_paleocoordinates_full).all(axis=1)]['elev'])
    # get closest precipitation value for the spacetime point
    closest_t_with_prec_value = list(prec_files_dict.keys())[np.abs(np.array(list(prec_files_dict.keys())) - age).argmin()]
    prec_data = prec_files_dict[closest_t_with_prec_value]
    prec = float(prec_data[(prec_data[['x','y']] == rounded_paleocoordinates).all(axis=1)]['prec'])
    # get closest temperature value for the spacetime point
    closest_t_with_temp_value = list(temp_files_dict.keys())[np.abs(np.array(list(temp_files_dict.keys())) - age).argmin()]
    temp_data = temp_files_dict[closest_t_with_temp_value]
    temp = float(temp_data[(temp_data[['x','y']] == rounded_paleocoordinates).all(axis=1)]['temp'])
    # get global average temperature
    closest_t_global_with_temp_value = paleotemp_global['time'].values.flat[np.abs(paleotemp_global['time'].values - age).argmin()]
    temperature_global = paleotemp_global['temperature'].values[paleotemp_global['time'].values==closest_t_global_with_temp_value][0]
    # get global average CO2
    closest_t_global_with_co2_value = co2_global['time'].values.flat[np.abs(co2_global['time'].values - age).argmin()]
    co2_value = co2_global['co2'].values[co2_global['time'].values==closest_t_global_with_co2_value][0]
    return (elev,prec,temp,temperature_global,co2_value)


def clean_spatial_and_temporal_dist_arrays(spatial_distances,temporal_distances):
    # reduce species arrays to only real values (remove nans)
    final_spatial_distances = []
    final_temporal_distances = []
    for i,veg_point_dists in enumerate(np.array(spatial_distances)):
        time_distance_array_stages = temporal_distances[i]
        # for each species get the list of spatial and temporal distances
        spatial_distances_by_species = []
        temporal_distances_by_species = []
        for species_values in veg_point_dists:
            spatial_dists = species_values[~np.isnan(species_values)]
            temporal_dists = time_distance_array_stages[~np.isnan(species_values)]
            spatial_distances_by_species.append(spatial_dists)
            temporal_distances_by_species.append(temporal_dists)
        final_spatial_distances.append(spatial_distances_by_species)
        final_temporal_distances.append(temporal_distances_by_species)
    return final_spatial_distances, final_temporal_distances


def clean_spatial_and_temporal_dist_arrays_single(spatial_distances,temporal_distances):
    # for each species get the list of spatial and temporal distances
    spatial_distances_by_species = []
    temporal_distances_by_species = []
    for species_values in spatial_distances:
        spatial_dists = species_values[~np.isnan(species_values)]
        temporal_dists = temporal_distances[~np.isnan(species_values)]
        spatial_distances_by_species.append(spatial_dists)
        temporal_distances_by_species.append(temporal_dists)
    return spatial_distances_by_species, temporal_distances_by_species



n_epochs_for_taxon_selection = 9
#cropping_window = [-180, -52, 25, 80]
max_dist = distance_flat([-7508588,3770928],[1916419,6089847]) # in albers projection, expressed in m
max_time = 30.0



#__________________LOAD AND FORMAT FOSSIL AND VEGETATION DATA__________________
# read the data files
all_occurrences_file = 'data/data_for_distance_extraction/final_occs_present_past_albers.txt'
all_occurrences = pd.read_csv(all_occurrences_file,sep='\t')
# # the below steps are already done previously in the format_fossil_occurrence_dataframes.py script
# all_occurrences = all_occurrences.drop(index=np.where(all_occurrences.lon<cropping_window[0])[0])
# all_occurrences = all_occurrences.drop(index=np.where(all_occurrences.lon>cropping_window[1])[0])
# all_occurrences = all_occurrences.drop(index=np.where(all_occurrences.lat<cropping_window[2])[0])
# all_occurrences = all_occurrences.drop(index=np.where(all_occurrences.lat>cropping_window[3])[0])
# all_occurrences.to_csv('data/raw/fossil_data/all_fossil_data_cleaned.txt',sep='\t',index=False)

#plt.plot(all_occurrences.lon,all_occurrences.lat,'.')

paleo_veg_labels_file = 'data/raw/vegetation_data/paleo_vegetation_north_america_albers.txt'
paleo_veg_labels_raw = pd.read_csv(paleo_veg_labels_file,sep='\t')
paleo_veg_labels = paleo_veg_labels_raw[['Longitude','Latitude','mean_age','Openness','paleolon','paleolat','x_albers','y_albers']]
paleo_veg_labels.columns = ['lon','lat','age','label','paleolon','paleolat','x_albers','y_albers']

paleo_veg_labels.label[paleo_veg_labels.label=='Closed']=0
paleo_veg_labels.label[paleo_veg_labels.label=='Open']=1
# round ages to geological stages (midpoints)
veg_label_ages = bin_ages_in_geo_stages_get_mean(paleo_veg_labels.age.values)
paleo_veg_labels = paleo_veg_labels.assign(rounded_ages=veg_label_ages)
# remove duplicates
paleo_veg_labels['joined_values'] = paleo_veg_labels["lon"].map(str) + paleo_veg_labels["lat"].map(str) + paleo_veg_labels["rounded_ages"].map(str)
#[i for i in np.unique(paleo_veg_labels.joined_values) if np.unique(paleo_veg_labels[paleo_veg_labels.joined_values == i]['label']).shape[0] == 2]
#paleo_veg_labels[paleo_veg_labels.joined_values=='-117.10335.06112.725000000000001'].label
unique_target_indeces = [np.random.choice(paleo_veg_labels[paleo_veg_labels.joined_values == i].index) for i in np.unique(paleo_veg_labels.joined_values)]
paleo_veg_labels_raw = paleo_veg_labels[paleo_veg_labels.index.isin(unique_target_indeces)]
paleo_veg_labels = paleo_veg_labels_raw.drop(['joined_values'], axis=1)
paleo_veg_labels.to_csv('data/raw/vegetation_data/selected_paleo_vegetation_north_america_albers.txt',sep='\t',index=False)
n_paleo_data = len(paleo_veg_labels)

current_veg_labels_file = 'data/raw/vegetation_data/current_vegetation_north_america_albers.txt'
current_veg_labels_raw = pd.read_csv(current_veg_labels_file,sep='\t')
current_veg_labels_raw = current_veg_labels_raw.assign(age=np.zeros(len(current_veg_labels_raw)))
current_veg_labels = current_veg_labels_raw[['x','y','age','veg','x','y','x_albers','y_albers']]
current_veg_labels.columns = ['lon','lat','age','label','paleolon','paleolat','x_albers','y_albers']
# round ages to geological stages (midpoints)
veg_label_ages = bin_ages_in_geo_stages_get_mean(current_veg_labels.age.values)
current_veg_labels = current_veg_labels.assign(rounded_ages=veg_label_ages)

# merge paleo and current data
veg_labels = pd.concat([paleo_veg_labels,current_veg_labels])
veg_labels = veg_labels.reset_index().iloc[:,1:]


# this filter has already been applied in the format_fossil_occurrence_dataframes.py script
target_species = np.unique(all_occurrences.species.values)
# # only work with taxa that occur in a min number of epochs
# species, counts = np.unique(all_occurrences.species.values,return_counts=True)
# pa_df_file = 'data/raw/fossil_data/presence_absence_by_epoch.txt'
# # load the presence absence in geostage df
# pa_df = pd.read_csv(pa_df_file,sep='\t')
# target_species = []
# for i in species:
#     n_epochs = pa_df[pa_df.iloc[:,0]==i].iloc[:,1:].values.sum()
#     if n_epochs >= n_epochs_for_taxon_selection:
#         target_species.append(i)
# target_species = np.array(target_species)
# #target_species = species[counts>100]
# all_occurrences_selected = all_occurrences[all_occurrences.species.isin(target_species)]
# round ages to geological stages (midpoints)
occurrence_ages = bin_ages_in_geo_stages_get_mean(all_occurrences.mean_age.values)
all_occurrences_selected = all_occurrences.assign(rounded_ages=np.round(occurrence_ages,4))
n_age_bins = len(all_occurrences_selected.groupby('rounded_ages'))
all_occurrences_selected.to_csv('data/data_for_distance_extraction/final_occs_present_past_binned_ages_albers.txt',sep='\t',index=False)


# round veg data points into bins as well
occ_coords_albers = np.array([all_occurrences_selected.x_albers.values,all_occurrences_selected.y_albers.values])
temp_occ_df = all_occurrences_selected.copy()
veg_label_coords = veg_labels[['lon','lat']].values.astype(float)
veg_label_coords_albers = veg_labels[['x_albers','y_albers']].values.astype(float)

veg_label_paleocoords = veg_labels[['paleolon','paleolat']].values.astype(float)
veg_label_ages = veg_labels['age'].values.astype(float)
veg_label_ages_rounded = bin_ages_in_geo_stages_get_mean(veg_label_ages)
veg_label_labels = veg_labels['label'].values
final_labels = veg_label_labels.astype(int)

write_occ_dfs_to_file = False
if write_occ_dfs_to_file:
    past_occ_data = all_occurrences_selected[all_occurrences_selected.mean_age>0]
    taxon_fossil_occs_data = np.array([np.array([i[0],len(i[1])]) for i in past_occ_data.groupby('species')])
    taxon_fossil_occs_df = pd.DataFrame(data = taxon_fossil_occs_data,columns=['taxon','n_fossil_occs'])
    taxon_fossil_occs_df.to_csv('/Users/tobias/GitHub/paleovegetation/doc/suppl_material/data_s2.txt',sep='\t',header=True, index=False)
    for i in past_occ_data.groupby('species'):
        i[1].to_csv('/Users/tobias/GitHub/feature_gen_paleoveg/data/raw/fossil_data/occurrences_by_taxon/fossil/%s_fossil_occs.txt'%i[0],sep='\t',header=True, index=False)
        
    gbif_occ_data = all_occurrences_selected[all_occurrences_selected.mean_age==0]
    taxon_current_occs_data = np.array([np.array([i[0],len(i[1])]) for i in gbif_occ_data.groupby('species')])
    taxon_current_occs_df = pd.DataFrame(data = taxon_current_occs_data,columns=['taxon','n_observations'])
    taxon_current_occs_df.to_csv('/Users/tobias/GitHub/paleovegetation/doc/suppl_material/data_s3.txt',sep='\t',header=True, index=False)
    for i in gbif_occ_data.groupby('species'):
        i[1].to_csv('/Users/tobias/GitHub/feature_gen_paleoveg/data/raw/fossil_data/occurrences_by_taxon/current/%s_current_occs.txt'%i[0],sep='\t',header=True, index=False)

#______________________________________________________________________________



#_____________________ABIOTIC DATA FOR FEATURE EXTRACTION______________________
# get global temperature data
paleotemp_file = 'data/raw/climatic_data/global_average_temperature.txt'
paleotemp_global = pd.read_csv(paleotemp_file,sep='\t')
paleotemp_global = paleotemp_global[paleotemp_global.time <= max_time]
# get CO2 data
co2_data = np.array([297.6, 301.36, 304.84, 307.86, 310.36, 312.53, 314.48, 316.31, 317.42, 317.63,
                     317.74, 318.51, 318.29, 316.5, 315.49, 317.64, 318.61, 316.6, 317.77, 328.27,
                     351.12, 381.87, 415.47, 446.86, 478.31, 513.77, 550.74, 586.68, 631.48, 684.13,
                     725.83, 757.81, 789.39, 813.79, 824.25, 812.6, 784.79, 755.25, 738.41, 727.53,
                     710.48, 693.55, 683.04, 683.99, 690.93, 694.44, 701.62, 718.05, 731.95, 731.56,
                     717.76]) # data from Luis Palazzesi et al, 2021, Nature Communications
co2_data = co2_data[:int(max_time)+1] # only keep the last 30 million years
co2_df_data = np.vstack([np.arange(0,max_time+1),co2_data])
co2_global = pd.DataFrame(co2_df_data.T,columns=['time','co2'])

# get climate and elevation rasters
elev_folder = 'data/raw/elevation/*.txt'
prec_folder = 'data/raw/climatic_data/precipitation/*.txt'
temp_folder = 'data/raw/climatic_data/temperature/*.txt'


# load elevation data grid
elev_files = np.array(sorted(glob.glob(elev_folder)))
time_stamps = np.array([int(os.path.basename(i).split('_')[-1].split('.')[0].replace('ma','') ) for i in elev_files])
# select only files within timeframe
elev_files_selected = elev_files[time_stamps <= max_time]
time_stamps_selected = time_stamps[time_stamps <= max_time]
all_elev_data = [pd.read_csv(i,sep='\t') for i in elev_files_selected]
# rescale the prec values to values between 0 and 1
elev_files_dict = dict(zip(time_stamps_selected,all_elev_data))

# load precipitation data grid
prec_files = np.array(sorted(glob.glob(prec_folder)))
time_stamps = np.array([int(i.split('/')[-1].split('_')[-1].replace('ma.txt','')) for i in prec_files])
# select only files within timeframe
prec_files_selected = prec_files[time_stamps <= max_time]
time_stamps_selected = time_stamps[time_stamps <= max_time]
all_prec_data = [pd.read_csv(i,sep='\t') for i in prec_files_selected]
# rescale the prec values to values between 0 and 1
prec_files_dict = dict(zip(time_stamps_selected,all_prec_data))

# load temperature data grid
temp_files = np.array(sorted(glob.glob(temp_folder)))
time_stamps_temp = np.array([int(i.split('/')[-1].split('_')[-1].replace('ma.txt','')) for i in temp_files])
# select only files within timeframe
temp_files_selected = temp_files[time_stamps_temp <= max_time]
time_stamps_temp_selected = time_stamps_temp[time_stamps_temp <= max_time]
all_temp_data = [pd.read_csv(i,sep='\t') for i in temp_files_selected]
temp_files_dict = dict(zip(time_stamps_temp_selected,all_temp_data))
#______________________________________________________________________________



# get distances and abiotic features for each vegetation point
spatial_distances_all_points = []
temporal_distances_all_points = []
abiotic_features = []
# iterate through vegetation labels
for i,coords in enumerate(veg_label_coords):
    coords_albers = veg_label_coords_albers[i]
    print(i)
    
    # get abiotic features
    age = veg_label_ages[i]
    binned_age = veg_label_ages_rounded[i]
    paleocoords = veg_label_paleocoords[i]

    elev,prec,temp,temperature_global,co2_value = extract_raw_abiotic_features(coords,age,binned_age,paleocoords,elev_files_dict,prec_files_dict,temp_files_dict,paleotemp_global,co2_global)
    final_spatial_distances,final_temporal_distances = extract_raw_distance_features(coords_albers,binned_age,occ_coords_albers,temp_occ_df,target_species,n_age_bins,max_time)
    
    abiotic_features.append([paleocoords[0],paleocoords[1],age,prec,temp,elev,temperature_global,co2_value])
    spatial_distances_all_points.append(final_spatial_distances)
    temporal_distances_all_points.append(np.array(final_temporal_distances))
    
final_spatial_distances_all_points,final_temporal_distances_all_points = clean_spatial_and_temporal_dist_arrays(spatial_distances_all_points,temporal_distances_all_points)



# pickle.dump(final_spatial_distances_all_points,open("/Users/tobias/GitHub/feature_gen_paleoveg/data/spatial_distances_NN_input_current_sample.pkl","wb"))
# pickle.dump(final_temporal_distances_all_points,open("/Users/tobias/GitHub/feature_gen_paleoveg/data/temporal_distances_NN_input_current_sample.pkl","wb"))
# np.savetxt('/Users/tobias/GitHub/feature_gen_paleoveg/data/veg_labels_current_sample.txt',final_labels,fmt='%i')
# np.savetxt('/Users/tobias/GitHub/feature_gen_paleoveg/data/selected_taxa.txt',target_species,fmt='%s')
# abiotic_features = np.array(abiotic_features)
# np.save('/Users/tobias/GitHub/feature_gen_paleoveg/data/abiotic_features_current_sample.npy',abiotic_features)

spatial_out = "data/spatial_distances_NN_input.pkl"
temporal_out = "data/temporal_distances_NN_input.pkl"
pickle.dump(final_spatial_distances_all_points,open(spatial_out,"wb"))
pickle.dump(final_temporal_distances_all_points,open(temporal_out,"wb"))
np.savetxt('data/veg_labels.txt',final_labels,fmt='%i')
np.savetxt('data/selected_taxa.txt',target_species,fmt='%s')
abiotic_features = np.array(abiotic_features)
abiotic_features_out = 'data/abiotic_features.npy'
np.save(abiotic_features_out,abiotic_features)


print('Finished.')
quit()

# # define instance-indices for training and test set
# np.random.seed(1234)
# additional_features = np.load('/Users/tobias/GitHub/feature_gen_paleoveg/data/abiotic_features.npy')
# size_test = 0.1
# paleo_indices = np.where(additional_features[:,2]>0)[0]
# current_indices = np.where(additional_features[:,2]==0)[0]
# paleo_test_indices = np.random.choice(paleo_indices,int(len(paleo_indices)*size_test),replace=False)
# current_test_indices = np.random.choice(current_indices,int(len(current_indices)*size_test),replace=False)
# paleo_train_indices = np.array([i for i in paleo_indices if i not in paleo_test_indices])
# current_train_indices = np.array([i for i in current_indices if i not in current_test_indices])
# current_train_indices = np.random.choice(current_train_indices,len(paleo_train_indices),replace=False)

# test_indices = np.concatenate([paleo_test_indices,current_test_indices])
# train_indices = np.concatenate([paleo_train_indices,current_train_indices])

# np.savetxt('/Users/tobias/GitHub/feature_gen_paleoveg/data/test_instance_indices.txt',test_indices,fmt='%i')
# np.savetxt('/Users/tobias/GitHub/feature_gen_paleoveg/data/train_instance_indices.txt',train_indices,fmt='%i')



# prepare data for past time slice predictions_________________________________
# load already compiled raw feature data
precompiled_spatial_dists = pickle.load(open(spatial_out,'rb'))
precompiled_temp_dists = pickle.load(open(temporal_out,'rb'))
precompiled_abiotic_features = np.load(abiotic_features_out)
# remove the paleoveg data points to only get the na map grid data
precompiled_spatial_dists_current = precompiled_spatial_dists[n_paleo_data:]
precompiled_temp_dists_current = precompiled_temp_dists[n_paleo_data:]
precompiled_abiotic_features_current = precompiled_abiotic_features[n_paleo_data:,:]
# get list of coordinates
precompiled_grid_coords_current = precompiled_abiotic_features_current[:,:2]
list_precompiled_coords = np.array([str(i) for i in precompiled_grid_coords_current])

# now load the coordinates for past map grid cells, including their paleocoords
paleocoord_folder = 'data/raw/current_grid_with_paleocoords'
paleocoord_files = sorted(glob.glob(os.path.join(paleocoord_folder,'*.txt')))


# first match the current grid with precompiled data, isnc ehtis one contains lakes that need to be matched differently than for other time slices
for age in np.arange(31).astype(float):
    print(age)
    binned_age = bin_ages_in_geo_stages_get_mean(np.array([age]))[0]
    current_grid_file = [i for i in paleocoord_files if '_%iMA'%age in i][0]
    current_grid_data = pd.read_csv(current_grid_file,sep='\t')
    current_grid_coords = current_grid_data[['lng','lat']].values
    current_grid_coords_albers = current_grid_data[['x_present_albers','y_present_albers']].values
    current_grid_paleocoords = current_grid_data[['paleolng','paleolat']].values
    # extract the raw features for all grid cells that are present in the precompiled data
    age_bin_spatial_dists = []
    age_bin_temporal_dists = []
    age_bin_abiotic_features = []
    for i,coords in enumerate(current_grid_coords):
        coords_albers = current_grid_coords_albers[i]
        if i%1000 == 0:
            print(i)
        #print(i)
        # temporal distances are always the same for a given time slice, so let's calculate them once at the beginning and then recycle
        if i==0:
            spatial_distances,temporal_distances = extract_raw_distance_features(coords_albers,binned_age,occ_coords_albers,temp_occ_df,target_species,n_age_bins,max_time)
            __,final_temporal_dists = clean_spatial_and_temporal_dist_arrays_single(spatial_distances,np.array(temporal_distances))
        paleocoords = current_grid_paleocoords[i]
        elev, prec,temp,temperature_global, co2_value = extract_raw_abiotic_features(coords,age,binned_age,paleocoords,elev_files_dict,prec_files_dict,temp_files_dict,paleotemp_global,co2_global)
        final_abiotic_features = np.array([paleocoords[0],paleocoords[1],age,prec,temp,elev,temperature_global,co2_value])

        # if feature data already present, use that data
        if str(coords) in list_precompiled_coords:
            target_index = np.where(list_precompiled_coords==str(coords))[0][0]
            final_spatial_dists = precompiled_spatial_dists_current[target_index]
            #final_temporal_dists_old = precompiled_temp_dists_current[target_index]
            #final_abiotic_features_old = precompiled_abiotic_features_current[target_index]
            #print(coords,final_abiotic_features,final_abiotic_features_old)
        # otherwise recalculate feature data
        else:
            spatial_distances,temporal_distances = extract_raw_distance_features(coords_albers,binned_age,occ_coords_albers,temp_occ_df,target_species,n_age_bins,max_time)
            final_spatial_dists,__ = clean_spatial_and_temporal_dist_arrays_single(spatial_distances,np.array(temporal_distances))
            #final_abiotic_features = np.array([paleocoords[0],paleocoords[1],age,prec,temp,temperature_global])
            #print('N',coords,final_abiotic_features)
        age_bin_spatial_dists.append(final_spatial_dists)
        age_bin_temporal_dists.append(final_temporal_dists)
        age_bin_abiotic_features.append(final_abiotic_features)
    
    
    time_slice_spatial_out = "data/time_slice_features/spatial_distances_%iMA.pkl"%age
    time_slice_temporal_out = "data/time_slice_features/temporal_distances_%iMA.pkl"%age
    time_slice_abiotic_features_out = 'data/time_slice_features/abiotic_features_%iMA.npy'%age
    
    pickle.dump(age_bin_spatial_dists,open(time_slice_spatial_out,"wb"))
    pickle.dump(age_bin_temporal_dists,open(time_slice_temporal_out,"wb"))
    abiotic_features = np.array(age_bin_abiotic_features)
    np.save(time_slice_abiotic_features_out,abiotic_features)







# target_indices = [i for i, val in enumerate(precompiled_grid_coords_current) if val in current_grid_coords]

# # get discrepancies between current grid and the coords that we already have the raw feature data compiled for
# discrepancies = list(set([tuple(i) for i in current_grid_coords])-set([tuple(i) for i in precompiled_grid_coords_current]))
# discrepancies_paleocoords = discrepancies

# # get distances and abiotic features for each vegetation point
# spatial_distances_discrepancies = []
# temporal_distances_discrepancies = []
# abiotic_features_discrepancies = []
# # iterate through vegetation labels
# age=0
# binned_age = bin_ages_in_geo_stages_get_mean(np.array([age]))
# for i,coords in enumerate(discrepancies):
#     print(i)
    
#     coords=list(coords)
#     paleocoords = list(discrepancies_paleocoords[i])
    
#     prec,temp,temperature_global,final_spatial_distances,final_temporal_distances = extract_raw_features(coords,age,binned_age,paleocoords,prec_files_dict,temp_files_dict,paleotemp_global,occ_coords,temp_occ_df,target_species,n_age_bins,max_time)

#     abiotic_features_discrepancies.append([paleocoords[0],paleocoords[1],age,prec,temp,temperature_global])
#     spatial_distances_discrepancies.append(final_spatial_distances)
#     temporal_distances_discrepancies.append(np.array(final_temporal_distances))

# final_spatial_distances_discrepancies,final_temporal_distances_discrepancies = clean_spatial_and_temporal_dist_arrays(spatial_distances_discrepancies,temporal_distances_discrepancies)







# past_grid_file = [i for i in paleocoord_files if '_1MA' in i][0]





# time_slices = np.arange(31)

# xcoord = [i[0] for i in discrepancies]
# ycoord = [i[1] for i in discrepancies]

# plt.scatter(coords[:,0],coords[:,1],c=posterior_probs_cat_1,cmap=rvb,marker=',',s=2)
# plt.scatter(xcoord,ycoord,c='black',marker=',',s=2)

# for timepoint in time_slices:
#     # get the precipitation and temp values for the given time and place
#     break    




































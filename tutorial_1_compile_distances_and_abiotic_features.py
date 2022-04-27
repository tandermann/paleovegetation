#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:24:21 2022

@author: Tobias Andermann (tobias.andermann@ebc.uu.se)
"""

import sys,os,glob
import numpy as np
import pandas as pd
import pickle


def bin_ages_in_geo_stages_get_mean(age_array,geological_stages):
    if max(age_array) >= max(geological_stages):
        print(
            'Error in "bin_ages_in_geo_stages_get_mean_stage_age()": No conversion possible. Max age must not exceed oldest geological stage (%.2f)' % max(
                geological_stages))
    else:
        geological_stages_mid_ages = np.array(
            [(geological_stages[i] + geological_stages[i + 1]) / 2 for i, _ in enumerate(geological_stages[:-1])])
        new_age_array = geological_stages_mid_ages[np.digitize(age_array, geological_stages, right=False) - 1]
        return new_age_array


def distance_flat(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    delta_x = abs(x1 - x2)
    delta_y = abs(y1 - y2)
    d = (delta_x ** 2 + delta_y ** 2) ** 0.5
    return (d)


def round_coordinates_to_quarter_degree(coordinate_pair):
    rounded_coordinates = [np.round(coordinate_pair[0] * 2) / 2, np.round(coordinate_pair[1] * 2) / 2]
    if rounded_coordinates[0] < coordinate_pair[0]:
        rounded_coordinates[0] = rounded_coordinates[0] + 0.25
    else:
        rounded_coordinates[0] = rounded_coordinates[0] - 0.25
    if rounded_coordinates[1] < coordinate_pair[1]:
        rounded_coordinates[1] = rounded_coordinates[1] + 0.25
    else:
        rounded_coordinates[1] = rounded_coordinates[1] - 0.25
    return rounded_coordinates


def extract_raw_distance_features(coords, binned_age, occ_coords, temp_occ_df, target_species, n_age_bins, max_time):
    # get taxon distances (provide coords and occ_cords in albers projection!!)
    spatial_distances = np.round(distance_flat(coords, occ_coords))
    #    spatial_distances = np.round(distance(coords,occ_coords),2)
    temp_occ_df['dists'] = spatial_distances
    # create empty array to fill with final shortest distances
    final_spatial_distances = np.zeros([len(target_species), n_age_bins])
    final_spatial_distances[final_spatial_distances == 0] = np.nan
    final_temporal_distances = []
    # for each age bin find the closest spatial distance for each taxon and save those values
    for age_bin_i, age_dfs in enumerate(list(temp_occ_df.groupby('rounded_ages'))):
        age = np.round(age_dfs[0], 5)
        temporal_dist = np.abs(binned_age - age) / max_time
        final_temporal_distances.append(temporal_dist)
        group_by_species = age_dfs[1].groupby('species').dists
        min_dists_df = group_by_species.min()
        distances = min_dists_df.values.astype(float)
        species = min_dists_df.index.values.astype(str)
        species_indices = [np.where(target_species == spec)[0][0] for spec in species]
        final_spatial_distances[species_indices, age_bin_i] = distances / max_dist
    return (final_spatial_distances, final_temporal_distances)


def extract_raw_abiotic_features(coords, age, binned_age, paleocoords, elev_files_dict, paleotemp_global, co2_global):
    # round the coordinates to the nearest .25 or .75 (no .00 or .50!) to match with the grid points
    rounded_paleocoordinates = round_coordinates_to_quarter_degree(paleocoords)
    # also round to full degree for elevation data
    rounded_paleocoordinates_full = np.round(paleocoords)
    # get closest elevation value for the spacetime point
    closest_t_with_elev_value = list(elev_files_dict.keys())[
        np.abs(np.array(list(elev_files_dict.keys())) - age).argmin()]
    elev_data = elev_files_dict[closest_t_with_elev_value]
    elev = float(elev_data[(elev_data[['lon', 'lat']] == rounded_paleocoordinates_full).all(axis=1)]['elev'])
    # # get closest precipitation value for the spacetime point
    # closest_t_with_prec_value = list(prec_files_dict.keys())[
    #     np.abs(np.array(list(prec_files_dict.keys())) - age).argmin()]
    # prec_data = prec_files_dict[closest_t_with_prec_value]
    # prec = float(prec_data[(prec_data[['x', 'y']] == rounded_paleocoordinates).all(axis=1)]['prec'])
    # # get closest temperature value for the spacetime point
    # closest_t_with_temp_value = list(temp_files_dict.keys())[
    #     np.abs(np.array(list(temp_files_dict.keys())) - age).argmin()]
    # temp_data = temp_files_dict[closest_t_with_temp_value]
    # temp = float(temp_data[(temp_data[['x', 'y']] == rounded_paleocoordinates).all(axis=1)]['temp'])
    # get global average temperature
    closest_t_global_with_temp_value = paleotemp_global['time'].values.flat[
        np.abs(paleotemp_global['time'].values - age).argmin()]
    temperature_global = \
    paleotemp_global['temperature'].values[paleotemp_global['time'].values == closest_t_global_with_temp_value][0]
    # get global average CO2
    closest_t_global_with_co2_value = co2_global['time'].values.flat[np.abs(co2_global['time'].values - age).argmin()]
    co2_value = co2_global['co2'].values[co2_global['time'].values == closest_t_global_with_co2_value][0]
    return (elev, temperature_global, co2_value)


def clean_spatial_and_temporal_dist_arrays(spatial_distances, temporal_distances):
    # reduce species arrays to only real values (remove nans)
    final_spatial_distances = []
    final_temporal_distances = []
    for i, veg_point_dists in enumerate(np.array(spatial_distances)):
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


def clean_spatial_and_temporal_dist_arrays_single(spatial_distances, temporal_distances):
    # for each species get the list of spatial and temporal distances
    spatial_distances_by_species = []
    temporal_distances_by_species = []
    for species_values in spatial_distances:
        spatial_dists = species_values[~np.isnan(species_values)]
        temporal_dists = temporal_distances[~np.isnan(species_values)]
        spatial_distances_by_species.append(spatial_dists)
        temporal_distances_by_species.append(temporal_dists)
    return spatial_distances_by_species, temporal_distances_by_species



#__________________SETTINGS__________________
np.set_printoptions(suppress=True)
np.random.seed(1234) # set seed for reproducibility
geological_stages_boundaries = np.array([0,
                                         0.0042,
                                         0.0082,
                                         0.0117,
                                         0.129,
                                         0.774,
                                         1.8,
                                         2.58,
                                         3.6,
                                         5.333,
                                         7.246,
                                         11.63,
                                         13.82,
                                         15.97,
                                         20.44,
                                         23.03,
                                         27.82,
                                         33.9]) # in Ma (dates derived from International Chronostratigraphic Chart, v2020/03)
max_dist = distance_flat([-7508588,3770928],
                         [1916419,6089847]) # in Albers projection, expressed in m
max_time = 30.0 # in million years
outdir = 'tutorial/training_data'
if not os.path.exists(outdir):
    os.makedirs(outdir)
#______________________________________________________



#__________________LOAD AND FORMAT FOSSIL AND VEGETATION DATA__________________
# read the occurrence data
all_occurrences_file = 'tutorial/raw_data/fossil_and_current_occurrences.txt'
all_occurrences = pd.read_csv(all_occurrences_file,sep='\t')
target_species = np.unique(all_occurrences.species.values)
# round to midpoint of geo-stage
occurrence_ages = bin_ages_in_geo_stages_get_mean(all_occurrences.mean_age.values,geological_stages_boundaries)
all_occurrences_selected = all_occurrences.assign(rounded_ages=np.round(occurrence_ages,4))
n_age_bins = len(all_occurrences_selected.groupby('rounded_ages')) # determine the number of stages with fossil data

# read the paleovegetation data
paleo_veg_labels_file = 'tutorial/raw_data/paleo_vegetation.txt'
paleo_veg_labels = pd.read_csv(paleo_veg_labels_file,sep='\t') # 0 = closed, 1 = open vegetation
n_paleo_data = len(paleo_veg_labels)
# round to midpoint of geo-stage
paleoveg_ages = bin_ages_in_geo_stages_get_mean(paleo_veg_labels.age.values,geological_stages_boundaries)
paleo_veg_labels = paleo_veg_labels.assign(rounded_ages=np.round(paleoveg_ages,4))

# read the current vegetation
current_veg_labels_file = 'tutorial/raw_data/current_vegetation.txt'
current_veg_labels_raw = pd.read_csv(current_veg_labels_file,sep='\t')
# for the purpose of this tutorial, let us only select a subset of the current vegetation data, in size equivalent to our paleovegetation data
current_veg_labels_selected_ids = np.random.choice(np.arange(len(current_veg_labels_raw)),n_paleo_data,replace=False)
current_veg_labels_raw = current_veg_labels_raw.iloc[current_veg_labels_selected_ids].copy()
current_veg_labels_raw = current_veg_labels_raw.assign(age=np.zeros(len(current_veg_labels_raw)))
current_veg_labels = current_veg_labels_raw[['x','y','age','veg','x','y','x_albers','y_albers']]
current_veg_labels.columns = ['lon','lat','age','label','paleolon','paleolat','x_albers','y_albers']
# round to midpoint of geo-stage
veg_label_ages = bin_ages_in_geo_stages_get_mean(current_veg_labels.age.values,geological_stages_boundaries)
current_veg_labels = current_veg_labels.assign(rounded_ages=veg_label_ages)

# merge paleo and current data
veg_labels = pd.concat([paleo_veg_labels,current_veg_labels])
veg_labels = veg_labels.reset_index().iloc[:,1:]

# store the coordinate, age, and label arrays for occurrences and vegetation as separate objects
occ_coords_albers = np.array([all_occurrences_selected.x_albers.values,all_occurrences_selected.y_albers.values])
veg_label_coords = veg_labels[['lon','lat']].values.astype(float)
veg_label_coords_albers = veg_labels[['x_albers','y_albers']].values.astype(float)
veg_label_paleocoords = veg_labels[['paleolon','paleolat']].values.astype(float)
veg_label_ages = veg_labels['age'].values.astype(float)
veg_label_ages_rounded = veg_labels['rounded_ages'].values.astype(float)
veg_label_labels = veg_labels['label'].values
final_labels = veg_label_labels.astype(int)

# store a temporary copy of the occurrences
temp_occ_df = all_occurrences_selected.copy()
#______________________________________________________



#_____________________ABIOTIC DATA FOR FEATURE EXTRACTION______________________
# get global temperature data
paleotemp_file = 'tutorial/raw_data/global_average_temperature.txt'
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
elev_folder = 'tutorial/raw_data/elevation/*.txt'
#prec_folder = 'tutorial/raw_data/precipitation/*.txt' # We don't have the permission to publicly share data for precipitation and temperature.
#temp_folder = 'tutorial/raw_data/temperature/*.txt' # If needed, please request data from Tobias Andermann (tobias.andermann@ebc.uu.se), with permission from Christopher Scotese.

# load elevation data grid
elev_files = np.array(sorted(glob.glob(elev_folder)))
time_stamps = np.array([int(os.path.basename(i).split('_')[-1].split('.')[0].replace('ma','') ) for i in elev_files])
# select only files within timeframe
elev_files_selected = elev_files[time_stamps <= max_time]
time_stamps_selected = time_stamps[time_stamps <= max_time]
all_elev_data = [pd.read_csv(i,sep='\t') for i in elev_files_selected]
# rescale the prec values to values between 0 and 1
elev_files_dict = dict(zip(time_stamps_selected,all_elev_data))
#______________________________________________________



# _____________________EXTRACT DISTANCES AND FEATURES_____________________
spatial_distances_all_points = []
temporal_distances_all_points = []
abiotic_features = []
# iterate through vegetation labels
for i, coords in enumerate(veg_label_coords):
    coords_albers = veg_label_coords_albers[i]
    print(i)

    # get abiotic features
    age = veg_label_ages[i]
    binned_age = veg_label_ages_rounded[i]
    paleocoords = veg_label_paleocoords[i]

    elev, temperature_global, co2_value = extract_raw_abiotic_features(coords,
                                                                       age,
                                                                       binned_age,
                                                                       paleocoords,
                                                                       elev_files_dict,
                                                                       paleotemp_global,
                                                                       co2_global)
    final_spatial_distances, final_temporal_distances = extract_raw_distance_features(coords_albers,
                                                                                      binned_age,
                                                                                      occ_coords_albers,
                                                                                      temp_occ_df,
                                                                                      target_species,
                                                                                      n_age_bins,
                                                                                      max_time)

    abiotic_features.append([paleocoords[0], paleocoords[1], age, elev, temperature_global, co2_value])
    spatial_distances_all_points.append(final_spatial_distances)
    temporal_distances_all_points.append(np.array(final_temporal_distances))
final_spatial_distances_all_points, final_temporal_distances_all_points = clean_spatial_and_temporal_dist_arrays(spatial_distances_all_points, temporal_distances_all_points)
#______________________________________________________



# _____________________WRITE OUTPUT FILES_____________________
spatial_out = os.path.join(outdir,"spatial_distances_NN_input.pkl")
temporal_out = os.path.join(outdir,"temporal_distances_NN_input.pkl")
pickle.dump(final_spatial_distances_all_points,open(spatial_out,"wb"))
pickle.dump(final_temporal_distances_all_points,open(temporal_out,"wb"))
np.savetxt(os.path.join(outdir,'veg_labels.txt'),final_labels,fmt='%i')
np.savetxt(os.path.join(outdir,'selected_taxa.txt'),target_species,fmt='%s')
abiotic_features = np.array(abiotic_features)
abiotic_features_out = os.path.join(outdir,'abiotic_features.npy')
np.save(abiotic_features_out,abiotic_features)
#______________________________________________________



# _____________________SELECT TRAIN AND TEST SET_____________________
# make sure test and train set have same number of current and paleo instances
paleo_indices = np.where(abiotic_features[:,2]>0)[0]
current_indices = np.where(abiotic_features[:,2]==0)[0]
# extract train and test indices
testsize = 0.2
n_paleo = int(len(paleo_indices)-np.round(len(paleo_indices)*testsize))
n_current = int(len(current_indices)-np.round(len(current_indices)*testsize))
# randomly select these numbers of indices as train data
paleo_indices_train = np.random.choice(paleo_indices,n_paleo,replace=False)
current_indices_train = np.random.choice(current_indices,n_current,replace=False)
selected_indices_train = np.concatenate([paleo_indices_train,current_indices_train])
# define rest as test set
paleo_indices_test = np.array(list(set(paleo_indices)-set(paleo_indices_train)))
current_indices_test = np.array(list(set(current_indices)-set(current_indices_train)))
selected_indices_test = np.concatenate([paleo_indices_test,current_indices_test])
# shuffle the instances
np.random.shuffle(selected_indices_train)
np.random.shuffle(selected_indices_test)
# save to file
np.savetxt(os.path.join(outdir,'train_instances.txt'),selected_indices_train,fmt='%i')
np.savetxt(os.path.join(outdir,'test_instances.txt'),selected_indices_test,fmt='%i')

#______________________________________________________

"""
Created on Tue Aug  6 14:53:38 2019
Updated on Thu Jan 20 19:18:21 2022

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import numpy as np
import pandas as pd
import os


#__________________________________SETTINGS____________________________________
cropping_window = [-180, -52, 25, 80] # north america
#cropping_window = [-180,180,-90,90] # global
# chosing a continent is required for the sample evenly in space-time function to work properly (unless it's global, then chose None!
selected_continent = 'north_america'
out_path = 'data/data_for_distance_extraction'
mammals=True
plants=True
reduce_mammals_to_genus = True
max_age_MA = 30
mammal_fossil_path = 'data/raw/fossil_data/mammal_fossils_joined_databases_cleaned.txt'
plant_macrofossil_path = 'data/raw/fossil_data/angiosperm_cenozoic_database.txt'
aquatic_mammals_file = 'data/raw/fossil_data/aquatic_mammals.txt'

# #____
select_genera_evenly_in_space_time = False
# # if select_genera_evenly_in_space_time == False define how many occurrences a taxon should at least have to be included
# least_occurrences = 3 #200
# # if select_genera_evenly_in_space_time == True select the path to the geographic info and set how many occurrences a taxon should at least have per space-time slice
# occurrence_count_file = 'data/spatial_data/mammal_occurrence_count_continents.txt'
# sample_per_continent_timeslice = 20
# least_occurrences_plants = 3


#____
# define the minimum number of epochs we want selected species to be in
n_epochs = 9

# load the pa in geostage df
pa_df_file = 'data/raw/fossil_data/presence_absence_by_epoch.txt'
pa_df = pd.read_csv(pa_df_file,sep='\t')
tmp = pa_df.columns.values
tmp[0] = 'species'
pa_df.columns = tmp
#______________________________________________________________________________

     
     
     

#____________________________CUSTOM FUNCTIONS__________________________________
def format_mammal_data(mammal_fossil_data,max_age_MA,sample_random_age=False,seed=1234):
    np.random.seed(seed)
    mammal_fossil_data.columns = ['species','max_age','min_age','lat', 'lon','family','order','source','citation']
    # convert columns to numeric
    numeric_mammal_fossil_data = mammal_fossil_data[['max_age','min_age','lat','lon']].apply(pd.to_numeric, errors='coerce').copy()
    numeric_mammal_fossil_data[['species','family','order','source','citation']] = mammal_fossil_data[['species','family','order','source','citation']]
    numeric_mammal_fossil_data = numeric_mammal_fossil_data[['species','max_age','min_age','lat', 'lon','family','order','source','citation']]
    # remove NaNs
    final_mammal_data = numeric_mammal_fossil_data[~numeric_mammal_fossil_data.isnull().any(axis=1)].copy()
    # turn occasional negative min_ages to 0
    final_mammal_data.loc[final_mammal_data.min_age < 0,'min_age'] = 0.
    # remove low precision records
    final_mammal_data_deltaT = final_mammal_data['max_age'].values-final_mammal_data['min_age'].values
    final_mammal_data = final_mammal_data[final_mammal_data_deltaT<=5]
    # calculate mean age
    if sample_random_age:
        final_mammal_data['mean_age'] = np.random.uniform(final_mammal_data['min_age'],final_mammal_data['max_age'])
    else:
        final_mammal_data['mean_age'] = np.mean(final_mammal_data[['min_age','max_age']],axis=1)
    # remove old indices
    old_indices = final_mammal_data[final_mammal_data['mean_age']>max_age_MA].index
    final_mammal_data = final_mammal_data.drop(old_indices)
    # sort and reindex the df
    final_mammal_data = final_mammal_data.sort_values('species')
    final_mammal_data = final_mammal_data.reset_index(drop=True).copy()
    return final_mammal_data

def format_plant_data(plant_fossil_data,max_age_MA,sample_random_age=False,seed=1234):
    np.random.seed(seed)
    plant_fossil_data.columns = ['species','max_age','min_age','lat', 'lon','family']
    # convert columns to numeric
    numeric_plant_fossil_data = plant_fossil_data[['max_age','min_age','lat','lon']].apply(pd.to_numeric, errors='coerce').copy()
    numeric_plant_fossil_data[['species','family']] = plant_fossil_data[['species','family']]
    numeric_plant_fossil_data = numeric_plant_fossil_data[['species','max_age','min_age','lat', 'lon','family']]
    # remove NaNs
    final_plant_data = numeric_plant_fossil_data[~numeric_plant_fossil_data.isnull().any(axis=1)].copy()
    # remove low precision records
    final_plant_data_deltaT = final_plant_data['max_age'].values-final_plant_data['min_age'].values
    final_plant_data = final_plant_data[final_plant_data_deltaT<=5]
    # calculate mean age
    if sample_random_age:
        final_plant_data['mean_age'] = np.random.uniform(final_plant_data['min_age'],final_plant_data['max_age'])
    else:
        final_plant_data['mean_age'] = np.mean(final_plant_data[['min_age','max_age']],axis=1)
    # remove old indeces
    old_indices = final_plant_data[final_plant_data['mean_age']>max_age_MA].index
    final_plant_data = final_plant_data.drop(old_indices)
    # sort and reindex the df
    final_plant_data = final_plant_data.sort_values('species')
    final_plant_data = final_plant_data.reset_index(drop=True).copy()
    return final_plant_data

def select_same_number_of_data_rows_per_label(annotated_data):
    # draw the same number of data points from both labels
    label_counts = np.unique(annotated_data.Openness_guessed,return_counts=True)
    minority_label = label_counts[0][label_counts[1] == min(label_counts[1])][0]
    minority_count = label_counts[1][label_counts[1] == min(label_counts[1])][0]
    # draw random indeces from both labels
    open_indeces = annotated_data[annotated_data.Openness_guessed=='Open'].index.values
    closed_indeces = annotated_data[annotated_data.Openness_guessed=='Closed'].index.values
    chosen_indeces_open = np.random.choice(open_indeces,size=minority_count,replace=False)
    chosen_indeces_closed = np.random.choice(closed_indeces,size=minority_count,replace=False)
    chosen_indeces = list(chosen_indeces_open) + list(chosen_indeces_closed)
    # get the data for the drawn indeces
    selection = annotated_data[annotated_data.index.isin(chosen_indeces)].copy()
    return selection

def get_feature_indices_of_best_taxa(pa_df,min_epochs=9):
    # turn values into array
    pa_array = pa_df.iloc[:,1:].values
    # turn array into 0 and 1
    sum_array = np.sum(pa_array,axis=1)
    # get species with min x epochs
    best_species = pa_df.iloc[sum_array>=min_epochs,:].species.values
    return best_species

#______________________________________________________________________________





#___________________________________MAIN_______________________________________

# define output folder, given the coordinates of defined region
out_folder = out_path
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# get list of accepted taxa
all_selected_taxa = get_feature_indices_of_best_taxa(pa_df,n_epochs)
all_selected_taxa = np.delete(all_selected_taxa, np.where(all_selected_taxa == 'Kogia')) #remove this whale genus


if mammals:
    # get list of mammal species
    mammal_fossil_data = pd.read_csv(mammal_fossil_path,sep='\t',header=None)
    mammal_data = format_mammal_data(mammal_fossil_data,max_age_MA)
    # use defined region as cropping window to only select fossils within this region
    na_mammal_data = mammal_data[(mammal_data.lon>=cropping_window[0]) & (mammal_data.lon<=cropping_window[1]) & (mammal_data.lat>=cropping_window[2]) & (mammal_data.lat<=cropping_window[3])].copy()
    # reduce mammal names to genus, if selected
    if reduce_mammals_to_genus:
        na_mammal_data['species'] = [i.split(' ')[0] for i in na_mammal_data.species]
    # remove marine species
    aquatic_mammals = pd.read_csv(aquatic_mammals_file,sep='\t')
    exclude_orders = ['Sirenia']
    aquatic_families = np.unique(aquatic_mammals.family)
    exclude_families = [i.lower().capitalize() for i in aquatic_families]
    selected_mammal_data_na_non_aquatic = na_mammal_data[~na_mammal_data.family.isin(exclude_families)].copy()
    final_non_aquatic_mammal_data = selected_mammal_data_na_non_aquatic[~selected_mammal_data_na_non_aquatic.order.isin(exclude_orders)].copy()

    ## side track: ________________________________________________________________
    #    # create separate fossil file for each genus for R-script 'code_continent_for_mammal_genera.r'
    #    # round the mammal fossil occurrence into 5Ma bins
    #    na_mammal_data_rounded_ages = final_non_aquatic_mammal_data.copy()
    #    na_mammal_data_rounded_ages['age_rounded'] = np.round(na_mammal_data_rounded_ages.mean_age*0.2)/0.2
    #    # for every genus bin data into time bins and write separate file    
    #    for i in na_mammal_data_rounded_ages.groupby(['species']):
    #        for j in i[1].groupby(['age_rounded']):
    #            j[1].to_csv('/Users/tobias/GitHub/paleovegetation_mammals/data/raw/fossil_data/by_genus/%s_%iMa_occurrences.txt'%(i[0],int(j[0])),sep='\t')
    ## _____________________________________________________________________________

    # # select fossil taxa to work with
    # if select_genera_evenly_in_space_time:
    #     # select the genera based on how many occurrences they have in each continent and time slice (n best per cont/time-slice)
    #     occurrence_count_data = pd.read_csv(occurrence_count_file,sep='\t')
    #     all_selected_taxa = []
    #     for time_bin_data in occurrence_count_data.groupby(['age_bin']):
    #         indeces = []
    #         for continent in time_bin_data[1].columns[2:7]:
    #             if selected_continent:
    #                 if continent == selected_continent:
    #                     # check how many datapoints we have for each genus/timeslice, if not enough take all there is, otherwise extract the n best
    #                     if len(time_bin_data[1][time_bin_data[1][continent]>=3]) < sample_per_continent_timeslice:
    #                         # still assure though that each selected taxon has at least 3 records, which is required for feature selection
    #                         best_records = time_bin_data[1][time_bin_data[1][continent]>=3][continent].sort_values(ascending=False).index.values
    #                     else:
    #                         best_records = time_bin_data[1][continent].sort_values(ascending=False).iloc[:sample_per_continent_timeslice].index.values
    #                     indeces.append(best_records)
    #             else:
    #                 # check how many datapoints we have for each genus/timeslice, if not enough take all there is, otherwise extract the n best
    #                 if len(time_bin_data[1][time_bin_data[1][continent]>=3]) < sample_per_continent_timeslice:
    #                     # still assure though that each selected taxon has at least 3 records, which is required for feature selection
    #                     best_records = time_bin_data[1][time_bin_data[1][continent]>=3][continent].sort_values(ascending=False).index.values
    #                 else:
    #                     best_records = time_bin_data[1][continent].sort_values(ascending=False).iloc[:sample_per_continent_timeslice].index.values
    #                 indeces.append(best_records)
    #         indeces_time_slice = [item for sublist in indeces for item in sublist]
    #         selected_taxa_time_slice = np.unique(time_bin_data[1][time_bin_data[1].index.isin(indeces_time_slice)].genus.values)
    #         all_selected_taxa.append(selected_taxa_time_slice)
    #     all_selected_taxa = np.unique(np.array(sorted([item for sublist in all_selected_taxa for item in sublist])))
    #     # check how many genera were selected
    #     #len(all_selected_taxa)
    #     # check how many occurrences the selected genera have
    #     #np.array([len(na_mammal_data[na_mammal_data.species==i]) for i in all_selected_taxa])
    # else:
    #     # set least_occurrences, species with fewer occurrences will not be included in the list
    #     all_selected_taxa = np.unique(final_non_aquatic_mammal_data.species,return_counts=True)[0][np.unique(final_non_aquatic_mammal_data.species,return_counts=True)[1]>=least_occurrences]

    # only select fossils from selected taxa
    final_mammal_data = final_non_aquatic_mammal_data[final_non_aquatic_mammal_data.species.isin(all_selected_taxa)].copy()
    final_mammal_data['kingdom'] = 'Animalia'


if plants:
    # get plant macrofossils
    macrofossil_data = pd.read_csv(plant_macrofossil_path,sep='\t',low_memory=False)
    # select columns
    target_columns = [' aff etc)"\tSpecific epithet"','Age max-Ma','Age min-Ma','Latitude','Longitude','Family']
    macrofossil_data = macrofossil_data[target_columns]
    macrofossil_data = format_plant_data(macrofossil_data,max_age_MA)
    # use defined region as cropping window to only select fossils within this region
    na_plant_data = macrofossil_data[(macrofossil_data.lon>=cropping_window[0]) & (macrofossil_data.lon<=cropping_window[1]) & (macrofossil_data.lat>=cropping_window[2]) & (macrofossil_data.lat<=cropping_window[3])].copy()
    # # set least_occurrences for plant fossils, species with fewer occurrences will not be included in the list
    # selected_plant_species = np.unique(na_plant_data.species,return_counts=True)[0][np.unique(na_plant_data.species,return_counts=True)[1]>=least_occurrences_plants]
    final_plant_data = na_plant_data[na_plant_data.species.isin(all_selected_taxa)].copy()
    final_plant_data['source'] = 'angiosperm_db'
    final_plant_data['kingdom'] = 'Plantae'
    final_plant_data['citation'] = 'Xing, Y. et al. Testing the biases in the rich Cenozoic angiosperm macrofossil record. International Journal of Plant Sciences 177, 371–388 (2016).'


if mammals and plants:
    # join mammal and plant fossils into one df
    final_fossil_data = pd.concat([final_mammal_data[final_plant_data.columns], final_plant_data], ignore_index=True)
    # make sure that in the final species list we first have all mammals and then all plants
    final_species_list = list(all_selected_taxa)
elif mammals and not plants:
    final_fossil_data = final_mammal_data
    final_species_list = list(all_selected_taxa)
elif plants and not mammals:
    final_fossil_data = final_plant_data
    final_species_list = list(all_selected_taxa)
else:
    final_fossil_data = pd.DataFrame()
    final_species_list = []

# save species list as npy and txt file
np.save(os.path.join(out_folder,'species_list_order.npy'), final_species_list)
with open(os.path.join(out_folder,'species_list_order.txt'), 'w') as out_file:
     [out_file.write('%s\n'%i) for i in final_species_list]
# save final fossil data to csv
final_fossil_data.to_csv(os.path.join(out_folder,'final_fossil_data.txt'),sep='\t',header=True,index=False)





#______________________WRITE SETTINGS TO FILE__________________________________
#create a settings file with all the chosen flags etc
with open(os.path.join(out_folder,'settings.txt'), 'w') as out_file:
     out_file.write('cropping_window: %s\n'%str(cropping_window))
     out_file.write('out_path: %s\n'%str(out_path))     
     out_file.write('reduce_mammals_to_genus: %s\n'%str(reduce_mammals_to_genus))
     out_file.write('max_age_MA: %s\n'%str(max_age_MA))
     out_file.write('mammal_fossil_path: %s\n'%str(mammal_fossil_path))
     out_file.write('plant_macrofossil_path: %s\n'%str(plant_macrofossil_path))
     out_file.write('aquatic_mammals_file: %s\n'%str(aquatic_mammals_file))
     out_file.write('selected_continent: %s\n'%str(selected_continent))
     out_file.write('select_genera_evenly_in_space_time: %s\n'%str(select_genera_evenly_in_space_time))
     # out_file.write('least_occurrences: %s\n'%str(least_occurrences))
     # out_file.write('occurrence_count_file: %s\n'%str(occurrence_count_file))
     # out_file.write('sample_per_continent_timeslice: %s\n'%str(sample_per_continent_timeslice))
     # out_file.write('least_occurrences_plants: %s\n'%str(least_occurrences_plants))
     out_file.write('mammals: %s\n'%str(mammals))
     out_file.write('plants: %s\n'%str(plants))
     out_file.write('n_mammal_taxa: %i\n'%len(np.unique(final_mammal_data.species)))
     out_file.write('n_plant_taxa: %i\n'%len(np.unique(final_plant_data.species)))
     
     
#______________________________________________________________________________


#______________________APPEND CURRENT OCCS TO FOSSIL DF________________________
current_occs_file = 'data/raw/current_occurrences/all_gbif_occurrences.txt'
current_occs = pd.read_csv(current_occs_file,sep='\t')
current_occs.columns = ['species','lon','lat']
current_occs['mean_age'] = 0.0
current_occs = current_occs[(current_occs.lon >= cropping_window[0]) & (current_occs.lon <= cropping_window[1]) & (
            current_occs.lat >= cropping_window[2]) & (current_occs.lat <= cropping_window[3])].copy()
final_current_occs = current_occs[current_occs.species.isin(all_selected_taxa)].copy()
final_occs = pd.concat([final_fossil_data[final_current_occs.columns], final_current_occs], ignore_index=True)
final_occs.to_csv(os.path.join(out_folder,'final_occs_present_past.txt'),sep='\t',header=True,index=False)

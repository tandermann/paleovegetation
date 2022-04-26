import pandas as pd
import glob, os
import numpy as np

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


# read occs
all_occs_file = 'data/data_for_distance_extraction/final_occs_present_past.txt'
all_occs_df = pd.read_csv(all_occs_file,sep='\t')
all_occs_df['rounded_ages'] = np.round(all_occs_df.mean_age).astype(int)
coord_pairs = np.array([all_occs_df.lon.values,all_occs_df.lat.values]).T
rounded_coord_pairs = np.array([round_coordinates_to_quarter_degree(i) for i in coord_pairs])
all_occs_df['rounded_lon'] = rounded_coord_pairs[:,0]
all_occs_df['rounded_lat'] = rounded_coord_pairs[:,1]

# get current to paleo coords translation info
paleocoord_folder = 'data/raw/current_grid_with_paleocoords'
paleocoord_files = glob.glob(os.path.join(paleocoord_folder,'*.txt'))
paleocoord_dict = {}
__ = [paleocoord_dict.setdefault(int(i.split('paleocoords_albers_')[-1].replace('MA.txt','')),pd.read_csv(i,sep='\t')) for i in paleocoord_files]


# read time slice predictions
timeslice_pred_path = 'results/production_model_best_model/time_slice_predictions/predicted_labels'
timeslice_pred_files = glob.glob(os.path.join(timeslice_pred_path,'*.npy'))
year_stamps = [int(i.split('predicted_labels_')[-1].replace('MA.npy','')) for i in timeslice_pred_files]
timeslice_pred_files_sorted = np.array(timeslice_pred_files)[np.argsort(year_stamps)]
master_taxon_open_veg_dict = {}
master_taxon_timestamp_dict = {}
master_taxon_n_occs_dict = {}
for i in timeslice_pred_files_sorted:
    time_stamp = int(i.split('predicted_labels_')[-1].replace('MA.npy',''))
    print(time_stamp)
    veg_pred_probs = np.load(i)
    prob_1 = np.mean(np.array([np.argmax(i, axis=1) for i in veg_pred_probs]), axis=0)
    prob_0 = 1 - prob_1
    post_prob = np.array([prob_0, prob_1]).T
    veg_pred = np.argmax(post_prob,axis=1)

    selected_paleocoord_df = paleocoord_dict[time_stamp]
    current_coords = np.array([selected_paleocoord_df.lng.values, selected_paleocoord_df.lat.values]).T
    paleo_coords = np.array([selected_paleocoord_df.paleolng.values, selected_paleocoord_df.paleolat.values]).T

    # select occurrences in this time slice
    row_ids = np.where(all_occs_df.rounded_ages.values == time_stamp)[0]
    selected_records = all_occs_df.loc[row_ids,:].copy()
    # export paleocoords of these records
    occ_coords = np.array([selected_records.rounded_lon.values,selected_records.rounded_lat.values]).T
    taxon_list = selected_records.species.values
    taxon_veg_value_list_timeslice = {}
    for id,j in enumerate(occ_coords):
        try:
            index = np.where(np.all(current_coords == j, axis=1))[0][0]
        except:
            #print('Couldn\'t find ',j)
            pass
        paleocoords_occ = paleo_coords[index,:]
        veg_prediction = veg_pred[index]
        taxon = taxon_list[id]
        taxon_veg_value_list_timeslice.setdefault(taxon,[])
        taxon_veg_value_list_timeslice[taxon].append(veg_prediction)
    for taxon in taxon_veg_value_list_timeslice.keys():
        grassland_fraction = np.mean((taxon_veg_value_list_timeslice[taxon]))
        master_taxon_open_veg_dict.setdefault(taxon,[])
        master_taxon_open_veg_dict[taxon].append(grassland_fraction)
        master_taxon_timestamp_dict.setdefault(taxon,[])
        master_taxon_timestamp_dict[taxon].append(time_stamp)
        master_taxon_n_occs_dict.setdefault(taxon,[])
        master_taxon_n_occs_dict[taxon].append(len((taxon_veg_value_list_timeslice[taxon])))


master_taxon_open_veg_dict['Odocoileus']
master_taxon_timestamp_dict['Odocoileus']
master_taxon_n_occs_dict['Odocoileus']

openness_scores_all_taxa = np.array([[i,np.mean(master_taxon_open_veg_dict[i])] for i in master_taxon_open_veg_dict.keys()])
openness_scores_df = pd.DataFrame(openness_scores_all_taxa,columns=['taxon','openness_score'])
openness_scores_df_sorted = openness_scores_df.sort_values('openness_score',ascending=False)
openness_scores_df_sorted.to_csv('results/habitat_preference_of_taxa.txt',sep='\t',index=False)


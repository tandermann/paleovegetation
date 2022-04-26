import pandas as pd
import glob, os
import numpy as np
import matplotlib.pyplot as plt


geological_stages = np.array([0, 0.0042, 0.0082, 0.0117, 0.129, 0.774, 1.8, 2.58, 3.6, 5.333, 7.246, 11.63, 13.82, 15.97, 20.44, 23.03, 27.82, 33.9])

additional_features = np.load( "data/abiotic_features.npy")
# get current and paleo instance ids
current_ids = np.where(additional_features[:,2]==0)[0]
paleo_ids = np.where(additional_features[:,2]>0)[0]
# get labels to plot open and closed histograms separately
veg_labels_file = 'data/veg_labels.txt'
veg_labels = np.loadtxt(veg_labels_file)
closed_veg_paleo_ids = paleo_ids[np.where(veg_labels[paleo_ids]==0)]
open_veg_paleo_ids = paleo_ids[np.where(veg_labels[paleo_ids]==1)]
# extract coords for plotting
current_map_coords = additional_features[current_ids,:2]
paleo_map_coords = additional_features[paleo_ids,:2]


grass_col = (210/255, 167/255, 65/255)
forest_col = (41/255, 98/255, 24/255)

fig = plt.figure(figsize=[10,10])
subplot = fig.add_subplot(221)
subplot.hist(additional_features[paleo_ids, 2],bins=geological_stages,edgecolor='black',color='grey')
plt.title('Paleovegetation training points - All')
plt.xlabel('Time (Mya)')
plt.ylabel('Count (per geological stage)')
subplot = fig.add_subplot(222)
subplot.hist(additional_features[open_veg_paleo_ids, 2],bins=geological_stages,edgecolor='black',color=grass_col)
plt.title('Paleovegetation training points - Open')
plt.xlabel('Time (Mya)')
plt.ylabel('Count (per geological stage)')
subplot = fig.add_subplot(223)
subplot.hist(additional_features[closed_veg_paleo_ids, 2],bins=geological_stages,edgecolor='black',color=forest_col)
plt.title('Paleovegetation training points - Closed')
plt.xlabel('Time (Mya)')
plt.ylabel('Count (per geological stage)')
plt.tight_layout()
fig.savefig('plots/paleoveg_points_through_time.pdf')




all_occs_file = 'data/raw/fossil_data/all_fossil_data_selected.txt'
all_occs_df = pd.read_csv(all_occs_file,sep='\t')


fig = plt.figure(figsize=(10, 5))
#3=prec, 4=temp
plt.scatter(additional_features[current_ids, 0], additional_features[current_ids,1], c='lightgrey', marker=',', s=3)
plt.scatter(-100, 42, c='goldenrod', marker=',', s=30)
plt.tight_layout()
plt.axis('off')
fig.savefig('plots/single_vegetation_point.pdf')
plt.close()


# make precipitation and temperature plot
fig = plt.figure(figsize=(10, 5))
#3=prec, 4=temp
plt.scatter(additional_features[:, 0], additional_features[:,1], c=additional_features[:,3], marker=',', s=4,cmap='plasma')
plt.tight_layout()
plt.axis('off')
fig.savefig('plots/precipitation.pdf')
fig.savefig('plots/precipitation.png')
plt.close()
fig = plt.figure(figsize=(10, 5))
#3=prec, 4=temp
plt.scatter(additional_features[:, 0], additional_features[:,1], c=additional_features[:,4], marker=',', s=4,cmap='plasma')
plt.tight_layout()
plt.axis('off')
fig.savefig('plots/temperature.pdf')
fig.savefig('plots/temperature.png')
plt.close()


outdir = 'plots/taxon_occs_plots'
for taxon_df in all_occs_df.groupby('species'):
    print(taxon_df[0])
    fig = plt.figure(figsize=[6,6])
    all_but_current_indices = np.where(taxon_df[1].rounded_ages.values>0.0021)[0]
    current_indices = np.where(taxon_df[1].rounded_ages.values==0.0021)[0]
    plt.hist(taxon_df[1].rounded_ages.values[all_but_current_indices],bins=geological_stages,edgecolor='black',color='grey')
    plt.xlabel('Time (Mya)')
    plt.ylabel('Count (per geological stage)')
    plt.title('%s, current occs: %i, paleo occs: %i'%(taxon_df[0],len(current_indices),len(all_but_current_indices)))
    plt.tight_layout()
    fig.savefig(os.path.join(outdir,'occ_time_histograms/%s_occs_hist.pdf'%taxon_df[0]))
    plt.close()

    taxon_dir = os.path.join(outdir,'taxon_occurrence_maps/%s'%taxon_df[0])
    if not os.path.exists(taxon_dir):
        os.makedirs(taxon_dir)

    if taxon_df[0] == 'Equus':
        for timestamp in np.unique(taxon_df[1].rounded_ages.values):
            timeslice_indices = np.where(taxon_df[1].rounded_ages.values == timestamp)[0]
            fig = plt.figure(figsize=(10,5))
            plt.scatter(current_map_coords[:, 0], current_map_coords[:, 1], c='lightgrey', marker=',', s=3)
            plt.scatter(taxon_df[1].lon.values[timeslice_indices], taxon_df[1].lat.values[timeslice_indices], c='red', marker=',', s=15)
            plt.tight_layout()
            plt.axis('off')
            fig.savefig(os.path.join(taxon_dir,'%s_%.4f_occs.pdf'%(taxon_df[0],timestamp)))
            plt.close()



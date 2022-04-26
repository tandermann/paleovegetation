import numpy as np
import pandas as pd
import glob,os, sys
import matplotlib.pyplot as plt
from PIL._imaging import font


def calcHPD(data, level):
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        sys.exit('\n\nToo little data to calculate marginal parameters.')
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)):
            rk = d[k+nIn-1] - d[k]
            if rk < r :
                r = rk
                i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])

selected_model = 'results/production_model_best_model'
#model_1 = 'cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_281_n_paleo_281_regular'
#model_9 = 'cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_281_n_paleo_281_regular_max_pooling'

indir = os.path.join(selected_model,'time_slice_predictions/predicted_labels')
time_slice_pred_files = glob.glob(os.path.join(indir,'*.npy'))


# compile hpd interval for each timepoint
area_coords_folder = 'data/raw/ecoregion_cells_paleocoordinates/cells_by_ecoregion'
#target_areas = [os.path.basename(i).replace('.txt','') for i in glob.glob(os.path.join(area_coords_folder,'*.txt'))]
#for target_area in target_areas:
target_area = 0#'great_plains'
# get the coords of all points for which we are predicting (the current layer does not contain points that fall in lakes)
all_coords_file_no_lakes = 'data/raw/current_grid_with_paleocoords/grid_points_paleocoords_albers_0MA.txt'
all_coords_no_lakes = pd.read_csv(all_coords_file_no_lakes, sep='\t').values[:, :2]
all_coords_file = 'data/raw/current_grid_with_paleocoords/grid_points_paleocoords_albers_1MA.txt'
all_coords = pd.read_csv(all_coords_file, sep='\t').values[:, :2]
if target_area:
    # get the coords of the target area
    target_coords = pd.read_csv(os.path.join(area_coords_folder,'%s.txt'%target_area)).values
    # get the ids of the target indices
    target_ids_no_lakes = []
    target_ids= []
    for i,coords in enumerate(target_coords):
        try:
            target_ids_no_lakes.append(np.where(np.all(all_coords_no_lakes == coords, axis=1))[0][0])
        except:
            pass
        try:
            target_ids.append(np.where(np.all(all_coords == coords, axis=1))[0][0])
        except:
            pass
else:
    target_ids_no_lakes = np.arange(len(all_coords_no_lakes))
    target_ids = np.arange(len(all_coords))
hpd_cutoff = 0.99
year_list = []
counts_list = []
hpd_boundaries_list = []
for file in time_slice_pred_files:
    year = int(os.path.basename(file).replace('predicted_labels_','').replace('MA.npy',''))
    if year == 0:
        target_instances = target_ids_no_lakes
    else:
        target_instances = target_ids
    time_slice_predictions = np.load(file)
    time_slice_predictions_selected = time_slice_predictions[:,target_instances,:]
    n_mcmc_samples,n_veg_points,n_cats = time_slice_predictions_selected.shape
    counts_forest = []
    counts_grass = []
    for i in np.arange(n_mcmc_samples):
        veg_calls_probs = time_slice_predictions_selected[i,:]
        veg_calls = np.argmax(veg_calls_probs,axis=1)
        # only veg calls above posterior threshold
        #softmax_threshold = 0.0
        #confidence_veg_calls = veg_calls[(veg_calls_probs > softmax_threshold).any(axis=1)]
        confidence_veg_calls = veg_calls
        n_grass = np.count_nonzero(confidence_veg_calls)/len(confidence_veg_calls)
        n_forest = (len(confidence_veg_calls)-np.count_nonzero(confidence_veg_calls))/len(confidence_veg_calls)
        #n_forest, n_grass = np.unique(veg_calls, return_counts=True)[1]
        counts_forest.append(n_forest)
        counts_grass.append(n_grass)
    lower_hpd_grass, upper_hpd_grass = calcHPD(counts_grass, hpd_cutoff)
    #lower_hpd_grass = min(counts_grass)
    #upper_hpd_grass = max(counts_grass)
    year_list.append(year)
    counts_list.append(counts_grass)
    hpd_boundaries_list.append([lower_hpd_grass, upper_hpd_grass])
    # lower_hpd_grass, upper_hpd_grass = calcHPD(counts_grass,hpd_cutoff)
    # year_grass_counts_dict[year] = np.array([np.mean(counts_grass),lower_hpd_grass,upper_hpd_grass])
    # lower_hpd_forest, upper_hpd_forest = calcHPD(counts_forest,hpd_cutoff)
    # year_forest_counts_dict[year] = np.array([np.mean(counts_forest),lower_hpd_forest,upper_hpd_forest])
    # lower_hpd_grass2, upper_hpd_grass2 = calcHPD(counts_grass,hpd_cutoff2)
    # year_grass_counts_dict2[year] = np.array([np.mean(counts_grass),lower_hpd_grass2,upper_hpd_grass2])
    # lower_hpd_forest2, upper_hpd_forest2 = calcHPD(counts_forest,hpd_cutoff2)
    # year_forest_counts_dict2[year] = np.array([np.mean(counts_forest),lower_hpd_forest2,upper_hpd_forest2])
# time_array = sorted(year_grass_counts_dict)
# counts_array_grass = np.array([year_grass_counts_dict[i] for i in time_array])
# counts_array_forest = np.array([year_forest_counts_dict[i] for i in time_array])

sorted_years = np.array(year_list)[np.argsort(year_list)]
sorted_counts_list = np.array(counts_list)[np.argsort(year_list)]
sorted_hpd_boundaries_list = np.array(hpd_boundaries_list)[np.argsort(year_list)]
np.savetxt('source_data/Fig_3_posterior_sample_open_veg_through_time.txt',sorted_counts_list,fmt='%.4f')



#
# # get overview of posterior probability of fraction 'open' over threshold
# threshold_value_open_fraction = 0.1
# plt.plot([sum(i>threshold_value_open_fraction)/len(i) for i in sorted_counts_list])
# plt.title('Grassland fraction above %.2f'%threshold_value_open_fraction)
# plt.xlabel('Time (Ma)')
# plt.ylabel('Posterior prob.')
# plt.show()
# plt.close()
#
#
# plt.plot(sorted_counts_list)
#
#
# times_of_crossing_threshold = [np.max(np.where(i>threshold_value_open_fraction)) for i in sorted_counts_list.T]
# plt.hist(times_of_crossing_threshold,bins=np.arange(0,31,1))
#
# times_above_threshold = [np.where(i>threshold_value_open_fraction)[0] for i in sorted_counts_list.T]
# concat = np.concatenate(times_above_threshold)
# plt.hist(concat,bins=np.arange(0,32,1))
#
# plt.boxplot(np.array(sorted_counts_list).T)
# plt.close()

#
# n_rows = 5
# fig = plt.figure(figsize=(15,10))
# for i,count_array in enumerate(sorted_counts_list):
#     year = sorted_years[i]
#     #print(int(year/n_rows)+1, year%n_rows+1)
#     subplot = fig.add_subplot(n_rows, int(np.ceil(len(sorted_years)/n_rows)),year+1, zorder=1)
#     plt.hist(count_array,bins=np.arange(0,1,0.02),color=grass_col)
#     plt.axvline(threshold_value_open_fraction, color='red', zorder=2)
#     plt.axvspan(0,threshold_value_open_fraction, color='red',alpha=0.2, zorder=2)
#     plt.title('%i Ma' % year)
#     plt.xlim(-0.1,1.1)
#     plt.ylim(0,300)
#     plt.grid(axis='both', linestyle='dashed', which='major', zorder=0)
#     plt.tight_layout()
# fig.savefig('plots/proportion_open_habitat_over_threshold_%.2f.pdf'%threshold_value_open_fraction)
# plt.close()
#

# determine slope in each point
slope_list = np.vstack([np.zeros(sorted_counts_list.shape[1]),[sorted_counts_list[::-1][i]-sorted_counts_list[::-1][i-1] for i,value in enumerate(sorted_counts_list[::-1]) if i > 0 ]]).T
#mean_values = np.mean(sorted_counts_list,axis=1)[::-1]
#slope_list = [0] + [mean_values[i]-mean_values[i-1] for i,value in enumerate(mean_values) if i > 0 ]
sorted_slope_list = np.array(slope_list[:,::-1]) * 100
sorted_slope_list_mean = np.mean(sorted_slope_list,axis=0)
t_max_increase = sorted_years[np.where(sorted_slope_list_mean==np.max(sorted_slope_list_mean))[0][0]]
#plt.plot(sorted_slope_list)
#plt.show()

# grassland through time plot with cutoffs
grass_col = (210/255, 167/255, 65/255)
forest_col = (41/255, 98/255, 24/255)
thresholds = [0.,0.1,0.5]
try:
    point1 = np.max(np.where(sorted_hpd_boundaries_list[:,0]>thresholds[0]))
except:
    point1 = 0
try:
    point2 = t_max_increase#np.max(np.where(sorted_hpd_boundaries_list[:,0]>thresholds[1]))
except:
    point2 = 0
try:
    point3 = np.max(np.where(sorted_hpd_boundaries_list[:,0]>thresholds[2]))
except:
    point3 = 0


fig = plt.figure(figsize=(11,6))
plt.plot(sorted_years[:point1+1], np.mean(sorted_counts_list,axis=1)[:point1+1], color=grass_col)
plt.plot(sorted_years[point1:], np.mean(sorted_counts_list,axis=1)[point1:], color='grey')
# if not target_area:
#     plt.axvspan(5, 0, color='blue', alpha=0.1, zorder=1)
# plt.plot(time_array,counts_array_forest[:,0],color=forest_col)
# plt.fill_between(time_array, counts_array_forest[:,1], counts_array_forest[:,2], color=forest_col, linewidth=0, alpha=0.4,label='Closed vegetation',zorder=1)
plt.fill_between(sorted_years[:point1+1], sorted_hpd_boundaries_list[:, 0][:point1+1], sorted_hpd_boundaries_list[:, 1][:point1+1], color=grass_col, linewidth=0,
                 alpha=0.4, label='Open vegetation', zorder=2)
plt.fill_between(sorted_years[point1:], sorted_hpd_boundaries_list[:, 0][point1:], sorted_hpd_boundaries_list[:, 1][point1:], color='grey', linewidth=0,
                 alpha=0.4, label='Open vegetation', zorder=2)
plt.axvline(point1, color='grey', linestyle='dashed',zorder=2)
plt.axvline(point2, color='blue', linestyle='dashed',zorder=2,alpha=0.5)
#plt.axvline(point3, color=grass_col, linestyle='dashed',zorder=2)
# plt.fill_between(time_array, counts_array_forest2[:,1], counts_array_forest2[:,2], color=forest_col, linewidth=0,alpha=0.2)
# plt.fill_between(time_array, counts_array_grass2[:,1], counts_array_grass2[:,2], color=grass_col, linewidth=0, alpha=0.2)
# plt.title('Vegetation North America (%.2f HPD)'%hpd_cutoff)
plt.xlim(-0.2,30.2)
plt.ylim(-0.05, 1.05)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=14)
for tick in ax.get_xticklabels():
    tick.set_fontname("Sans Serif")
ax.yaxis.set_ticks([0.,0.2,0.4,0.6,0.8,1.0])
ax.yaxis.set_ticklabels(['0%','20%','40%','60%','80%','100%'], fontname="Sans Serif")
ax.set_xticks(sorted_years, minor=True)
ax.invert_xaxis()
plt.grid(axis='both', linestyle='dashed', which='major', zorder=3)
ax2 = plt.gca().twinx()
# for i in sorted_slope_list:
#     ax2.plot(sorted_years, i, 'b-',alpha=0.01)
ax2.plot(sorted_years, sorted_slope_list_mean, 'b-',alpha=0.5)
#ax2.set_ylabel('Y2 data', color='b')
ax2.tick_params(axis='y', colors='blue')
ax2.yaxis.set_ticks([0,2,4,6,8,10,12,14])
ax2.yaxis.set_ticklabels(['+0%','+2%','+4%','+6%','+8%','+10%','+12%','+14%'], fontname="Sans Serif")
ax2.tick_params(axis='y', which='major', labelsize=14)
#ax2.set_ylim(-1,15)
#plt.axhline(threshold_value_open_fraction, color='red', linestyle='dashed',zorder=2)
#plt.gca().xaxis.tick_top()
#plt.xticks(rotation = -90)
#plt.yticks(rotation = -90)
#plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
#fig.savefig('plots/proportion_open_habitat_through_time_threshold_%.2f.pdf'%threshold_value_open_fraction,transparent=True)
if target_area:
    fig.savefig('plots/proportion_open_habitat_through_time_%s.pdf'%target_area,transparent=True)
else:
    fig.savefig('plots/proportion_open_habitat_through_time.pdf', transparent=True)



threshold_value_open_fraction = 0.5
area_color = grass_col#'#1f77b4'
for i,count_array in enumerate(sorted_counts_list):
    fig = plt.figure(figsize=(2, 2))
    year = sorted_years[i]
    lower_hpd,upper_hpd = sorted_hpd_boundaries_list[i]
    count_array = count_array[count_array>lower_hpd]
    count_array = count_array[count_array<upper_hpd]
    #print(int(year/n_rows)+1, year%n_rows+1)
    #subplot = fig.add_subplot(n_rows, int(np.ceil(len(sorted_years)/n_rows)),year+1, zorder=1)
    plt.hist(count_array,bins=np.arange(0,1,0.02),color=grass_col)
#    plt.axvline(0, linestyle='dashed', color=area_color, zorder=2)
    plt.axvline(threshold_value_open_fraction, linestyle='dashed', color='black', zorder=2)
    plt.axvspan(0,threshold_value_open_fraction, color=area_color,alpha=0.2, zorder=2)
    plt.title('%i Ma' % year)
    plt.xlim(-0.1,1.1)
    plt.ylim(0,200)
    #plt.ylabel('Density')
    plt.grid(axis='both', linestyle='dashed', which='major', zorder=0)
    #plt.gca().invert_xaxis()
    plt.gca().xaxis.set_ticklabels(['','0%','50%','100%'])
    plt.gca().yaxis.set_ticklabels([])
    plt.tight_layout()
    fig.savefig('plots/hists_per_timepoint/open_habitat_fraction_posterior_%i_no_labs_thres_%.1f.pdf'%(year,threshold_value_open_fraction),transparent=True)
    plt.close()


from functools import reduce
def add_num(a,b):
    return a+b

a = [1,2,3,10]
map(add_num,a,a)

# get info from coords files, which indices are which ecoregion
# extract only those timeslice predictions
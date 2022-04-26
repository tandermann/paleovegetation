import numpy as np
import glob,os
import matplotlib.pyplot as plt

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


hpd_cutoff = 0.95
hpd_cutoff2 = 0.95
best_models = [
    'cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_281_n_paleo_281_no_abiotic_nodes_32_8',
    'cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_281_n_paleo_281_regular',
    'cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_281_n_paleo_281_regular_max_pooling'
                ]

master_counts_array_grass = []
master_counts_array_forest = []
for model_dir in best_models:
    indir = os.path.join(model_dir,'time_slice_predictions/predicted_labels')
    time_slice_pred_files = glob.glob(os.path.join(indir,'*.npy'))
    year_grass_counts_dict = {}
    year_grass_counts_dict2 = {}
    year_forest_counts_dict = {}
    year_forest_counts_dict2 = {}

    for file in time_slice_pred_files:
        year = int(os.path.basename(file).replace('predicted_labels_','').replace('MA.npy',''))
        time_slice_predictions = np.load(file)
        n_mcmc_samples,n_veg_points,n_cats = time_slice_predictions.shape
        counts_forest = []
        counts_grass = []
        for i in np.arange(n_mcmc_samples):
            veg_calls_probs = time_slice_predictions[i,:]
            veg_calls = np.argmax(veg_calls_probs,axis=1)
            n_grass = np.count_nonzero(veg_calls)/n_veg_points
            n_forest = (n_veg_points-np.count_nonzero(veg_calls))/n_veg_points
            #n_forest, n_grass = np.unique(veg_calls, return_counts=True)[1]
            counts_forest.append(n_forest)
            counts_grass.append(n_grass)
        lower_hpd_grass, upper_hpd_grass = calcHPD(counts_grass,hpd_cutoff)
        year_grass_counts_dict[year] = np.array([np.mean(counts_grass),lower_hpd_grass,upper_hpd_grass])
        lower_hpd_forest, upper_hpd_forest = calcHPD(counts_forest,hpd_cutoff)
        year_forest_counts_dict[year] = np.array([np.mean(counts_forest),lower_hpd_forest,upper_hpd_forest])
        lower_hpd_grass2, upper_hpd_grass2 = calcHPD(counts_grass,hpd_cutoff2)
        year_grass_counts_dict2[year] = np.array([np.mean(counts_grass),lower_hpd_grass2,upper_hpd_grass2])
        lower_hpd_forest2, upper_hpd_forest2 = calcHPD(counts_forest,hpd_cutoff2)
        year_forest_counts_dict2[year] = np.array([np.mean(counts_forest),lower_hpd_forest2,upper_hpd_forest2])

    time_array = sorted(year_grass_counts_dict)
    counts_array_grass = np.array([year_grass_counts_dict[i] for i in time_array])
    counts_array_forest = np.array([year_forest_counts_dict[i] for i in time_array])
    master_counts_array_grass.append(counts_array_grass)
    master_counts_array_forest.append(counts_array_forest)
#counts_array_grass2 = np.array([year_grass_counts_dict2[i] for i in time_array])
#counts_array_forest2 = np.array([year_forest_counts_dict2[i] for i in time_array])

grass_col = (210/255, 167/255, 65/255)
forest_col = (41/255, 98/255, 24/255)

model_names = ['Model 2', 'Model 1', 'Model 9']

fig = plt.figure(figsize=(8,8))
for i,counts_array_grass in enumerate(master_counts_array_grass):
    subplot = fig.add_subplot(3,1,i+1)
    plt.plot(time_array,counts_array_grass[:,0],color=grass_col)
    #plt.plot(time_array,counts_array_forest[:,0],color=forest_col)
    #plt.fill_between(time_array, counts_array_forest[:,1], counts_array_forest[:,2], color=forest_col, linewidth=0, alpha=0.4,label='Closed vegetation',zorder=1)
    plt.fill_between(time_array, counts_array_grass[:,1], counts_array_grass[:,2], color=grass_col, linewidth=0, alpha=0.4,label='Open vegetation',zorder=2)
    #plt.fill_between(time_array, counts_array_forest2[:,1], counts_array_forest2[:,2], color=forest_col, linewidth=0,alpha=0.2)
    #plt.fill_between(time_array, counts_array_grass2[:,1], counts_array_grass2[:,2], color=grass_col, linewidth=0, alpha=0.2)
    plt.gca().invert_xaxis()
    #plt.title('Vegetation North America (%.2f HPD)'%hpd_cutoff)
    if i==2:
        plt.xlabel('Time (Mya)')
    if i ==1:
        plt.ylabel('Fraction of total vegetation cover')
    plt.ylim(-0.05,1.05)
    plt.title(model_names[i])
    plt.grid(axis='both', linestyle='dashed', which='major',zorder=0)
    plt.tight_layout()
#plt.legend()
fig.savefig('plots/veg_through_time_plot.pdf')

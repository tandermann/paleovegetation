import numpy as np
import pandas as pd
import np_bnn as bn
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features, make_colormap
import matplotlib.pyplot as plt

# load feature and label data
abiotic_features_file = 'cluster_content/feature_gen_mc3/data/abiotic_features.npy'
spatial_distances_file = 'cluster_content/feature_gen_mc3/data/spatial_distances_NN_input.pkl'
temporal_distances_file = 'cluster_content/feature_gen_mc3/data/temporal_distances_NN_input.pkl'
labels_file = 'cluster_content/feature_gen_mc3/data/veg_labels.txt'
test_indices_file = 'cluster_content/feature_gen_mc3/data/train_test_sets/test_instance_indices.txt'
taxon_names_file = 'data/selected_taxa.txt'
abiotic_features = np.load(abiotic_features_file)
scaled_additional_features = rescale_abiotic_features(abiotic_features)
test_indices = np.loadtxt(test_indices_file).astype(int)
labels = np.loadtxt(labels_file).astype(int)
test_labels = labels[test_indices]

paleo_test_indices = test_indices[np.where(abiotic_features[test_indices][:, 2] > 0)[0]]
paleo_test_labels = test_labels[np.where(abiotic_features[test_indices][:, 2] > 0)[0]]

taxon_names = np.loadtxt(taxon_names_file, dtype=str)
feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])
multiple_weights_per_species = True
sum_faunal_floral_features = False
max_pooling = False
feature_actfun = 1

pred_features_obj = PredictFeatures(spatial_distances_file,
                                    temporal_distances_file,
                                    scaled_additional_features,
                                    feature_group_ids,
                                    instance_index=paleo_test_indices,
                                    multiple_weights_per_species = multiple_weights_per_species,
                                    sum_faunal_floral_features = sum_faunal_floral_features,
                                    max_pooling = max_pooling,
                                    actfun = feature_actfun
                                    )


abiotic_feature_names = ['Longitude','Latitude','Time','Precipitation','Temperature','Global_mean_temp']
feature_names = list(taxon_names)+abiotic_feature_names

# load pkl file
pkl_file = 'cluster_content/feature_gen_mc3/new_runs_2021/testing_backup/n_current_281_n_paleo_281_regular/current_281_paleo_281_p1_h0_l32_8_s1_binf_1234_1.pkl'
bnn_obj, mcmc_obj, logger_obj, feature_gen_obj = bn.load_obj(pkl_file)
posterior_samples = logger_obj._post_weight_samples
n_posterior_samples = 100
selected_ids = np.random.choice(np.arange(len(posterior_samples)),n_posterior_samples,replace=False)
posterior_samples = [posterior_samples[i] for i in selected_ids]
feature_imp_dict = {}
for i, weights_dict in enumerate(posterior_samples):
    print(i)
    # read fwature weights and apply to raw featrue values
    feature_weights_rep = weights_dict['additional_prm']
    pred_features_obj.update_weigths(feature_weights_rep[0],feature_weights_rep[1])
    # extract final features for this rep
    test_features = pred_features_obj.get_features_unseen_data()
    # predict test set using feature importance
    feature_importance_out = bn.feature_importance(test_features,
                                                   weights_pkl=None,
                                                   weights_posterior=[weights_dict],
                                                   true_labels=paleo_test_labels,
                                                   fname_stem='',
                                                   feature_names=feature_names,
                                                   verbose=False,
                                                   post_summary_mode=0,
                                                   n_permutations=100,
                                                   feature_blocks=dict(),
                                                   write_to_file=False,
                                                   predictions_outdir='results/feature_importance',
                                                   unlink_features_within_block=True,
                                                   actFun=bnn_obj._act_fun,
                                                   output_act_fun=bnn_obj._output_act_fun)
    tmp = [feature_imp_dict.setdefault(i,[]) for i in feature_importance_out.feature_name]
    tmp2 = [feature_imp_dict[i].append(feature_importance_out.iloc[np.where(feature_importance_out.feature_name.values==i)[0][0],2]) for i in feature_importance_out.feature_name]


first_col = np.array(list(feature_imp_dict.keys()))
remaining_cols = np.array([feature_imp_dict[i] for i in feature_imp_dict.keys()])
data_array = np.vstack([first_col.T,remaining_cols.T]).T
feature_imp_df = pd.DataFrame(data_array)
feature_imp_df.to_csv('results/feature_importance/feature_importance_df_100_posterior_samples.txt',sep='\t',index=False,header=False)

final_feat_imp_list = []
for i in feature_imp_dict.keys():
    feature_imp_res = [i,np.mean(feature_imp_dict[i]),np.std(feature_imp_dict[i])]
    final_feat_imp_list.append(feature_imp_res)
final_feature_imp_df = pd.DataFrame(final_feat_imp_list)
final_feature_imp_df.columns = ['feature_name','delta_acc_mean','delta_acc_std']
sorted_final_feature_imp_df = final_feature_imp_df.sort_values('delta_acc_mean',ascending=False)
feature_importance_out_file = 'results/feature_importance/feature_importance_final_sorted.txt'
sorted_final_feature_imp_df = sorted_final_feature_imp_df[sorted_final_feature_imp_df.feature_name != 'Kogia']
sorted_final_feature_imp_df.to_csv(feature_importance_out_file,sep='\t',index=False,header=True)


feature_imp_data = pd.read_csv(feature_importance_out_file,sep='\t')
x = feature_imp_data.index.values
y = feature_imp_data.delta_acc_mean.values
e = feature_imp_data.delta_acc_std.values
selected_indices = np.where((y-e)>0)[0]

type_id = np.zeros(len(x)).astype(int)
plant_bolean = ['eae' in i for i in feature_imp_data.feature_name.values]
type_id[plant_bolean] = 1
abiotic_bolean = [i in ['Time','Temperature','Global_mean_temp','Latitude','Longitude','Precipitation'] for i in feature_imp_data.feature_name.values]
type_id[abiotic_bolean] = 2


fig = plt.figure(figsize=[10, 5])

subplot = fig.add_subplot(121)
subplot.errorbar(x[np.where(type_id==0)[0]], y[np.where(type_id==0)[0]], e[np.where(type_id==0)[0]], marker='.',linestyle='None', label='Mammal taxa',color='brown')
subplot.errorbar(x[np.where(type_id==1)[0]], y[np.where(type_id==1)[0]], e[np.where(type_id==1)[0]], marker='.',linestyle='None', label='Plant taxa',color='green')
subplot.errorbar(x[np.where(type_id==2)[0]], y[np.where(type_id==2)[0]], e[np.where(type_id==2)[0]], marker='.',linestyle='None', label='Abiotic features',color='black')
#subplot.errorbar(x, y, e, marker='.',linestyle='None', label='All features',color='grey')
#subplot.errorbar(x[selected_indices], y[selected_indices], e[selected_indices], marker='.',linestyle='None', label='Significantly positive features',color='blue')
plt.gca().axhline(0,linestyle='--',color='black')
plt.ylim(-0.04,0.16)
plt.xlabel('Feature importance rank')
plt.ylabel('Delta prediction accuracy')
plt.legend()
plt.tight_layout()

subplot = fig.add_subplot(122)
new_x = np.arange(len(selected_indices))
new_type_id = type_id[selected_indices]
subplot.errorbar(new_x[np.where(new_type_id==0)[0]], y[selected_indices][np.where(new_type_id==0)[0]], e[selected_indices][np.where(new_type_id==0)[0]], marker='.',linestyle='None', label='Mammal taxa',color='brown')
subplot.errorbar(new_x[np.where(new_type_id==1)[0]], y[selected_indices][np.where(new_type_id==1)[0]], e[selected_indices][np.where(new_type_id==1)[0]], marker='.',linestyle='None', label='Plant taxa',color='green')
subplot.errorbar(new_x[np.where(new_type_id==2)[0]], y[selected_indices][np.where(new_type_id==2)[0]], e[selected_indices][np.where(new_type_id==2)[0]], marker='.',linestyle='None', label='Abiotic features',color='black')
#subplot.errorbar(new_x, y[selected_indices], e[selected_indices], marker='.',linestyle='None', label='Significantly positive features',color='blue')
plt.xticks(new_x, feature_imp_data.feature_name.values[selected_indices], rotation=45)
plt.ylim(-0.04,0.16)
plt.gca().axhline(0,linestyle='--',color='black')
plt.tight_layout()

fig.savefig('plots/feature_importance_final_sorted.pdf')

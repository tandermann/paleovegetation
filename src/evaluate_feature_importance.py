import numpy as np
import pandas as pd
import np_bnn as bn
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features, make_colormap
import matplotlib.pyplot as plt
import glob,os


run_prediction = True
only_paleo_instances = False
plot_error = False
feature_importance_out_file = 'results/feature_importance/feature_importance_final_sorted.txt'
abiotic_feature_names = ['Longitude',
                         'Latitude',
                         'Time',
                         'Precipitation',
                         'Temperature',
                         'Elevation',
                         'Global Temp.',
                         'Global CO2']
pkl_file = 'results/production_model_best_model/continued_final_model_p1_h0_l8_s1_binf_1234.pkl'
if run_prediction:
    # load feature and label data
    abiotic_features_file = 'data/abiotic_features.npy'
    spatial_distances_file = 'data/spatial_distances_NN_input.pkl'
    temporal_distances_file = 'data/temporal_distances_NN_input.pkl'
    labels_file = 'data/veg_labels.txt'
    taxon_names_file = 'data/selected_taxa.txt'
    abiotic_features = np.load(abiotic_features_file)
    scaled_additional_features = rescale_abiotic_features(abiotic_features)
    all_veg_labels = np.loadtxt(labels_file).astype(int)
    # train_indices_file = os.path.join('data/train_test_sets/train_instance_indices_ncurrent_281_npaleo_281.txt')
    # train_instances = np.loadtxt(train_indices_file).astype(int)
    # all_non_train_instances = np.array([i for i in np.arange(len(abiotic_features)) if i not in train_instances])
    # current_non_train_instances = all_non_train_instances[np.where(abiotic_features[all_non_train_instances][:, 2] == 0)[0]]
    # current_non_train_labels = all_veg_labels[current_non_train_instances]
    taxon_names = np.loadtxt(taxon_names_file, dtype=str)
    feature_names = list(taxon_names)+abiotic_feature_names
    feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])
    multiple_weights_per_species = True
    sum_faunal_floral_features = False
    max_pooling = False
    feature_actfun = 2
    # get posterior weight samples from logger object
    bnn_obj, mcmc_obj, logger_obj, feature_gen_obj = bn.load_obj(pkl_file)
    posterior_samples = logger_obj._post_weight_samples
    # select n posterior samples
    total_posterior_samples = len(posterior_samples)
    select_n_posterior_samples = 1000
    selected_ids = np.random.choice(np.arange(total_posterior_samples),select_n_posterior_samples,replace=False)
    selected_posterior_samples = [posterior_samples[i] for i in selected_ids]

    train_indices_file = 'data/instance_selection_for_training/selected_instances_paleo_331_current_331.txt'
    train_indices = np.loadtxt(train_indices_file).astype(int)
    # select only paleo instances if setting is activate
    if only_paleo_instances:
        selected_test_indices = train_indices[abiotic_features[train_indices, 2] > 0]
    else:
        selected_test_indices = train_indices

    # get train features and labels
    train_labels = all_veg_labels[selected_test_indices]
    pred_features_obj = PredictFeatures(spatial_distances_file,
                                        temporal_distances_file,
                                        scaled_additional_features,
                                        feature_group_ids,
                                        instance_index=selected_test_indices,
                                        multiple_weights_per_species=multiple_weights_per_species,
                                        sum_faunal_floral_features=sum_faunal_floral_features,
                                        max_pooling=max_pooling,
                                        actfun=feature_actfun
                                        )
    ref_acc_list = []
    pred_acc_list = []
    for i, weights_dict in enumerate(selected_posterior_samples):
        print(i)
        feature_weights_rep = weights_dict['additional_prm']
        pred_features_obj.update_weigths(feature_weights_rep[0], feature_weights_rep[1])
        # extract final features for this rep
        predict_features = pred_features_obj.get_features_unseen_data()

        post_softmax_probs, post_prob_predictions = bn.get_posterior_cat_prob(predict_features,
                                                                              [weights_dict],
                                                                              post_summary_mode=1,
                                                                              actFun=bnn_obj._act_fun,
                                                                              output_act_fun=bnn_obj._output_act_fun)
        ref_accuracy = bn.CalcAccuracy(post_prob_predictions, train_labels)
        ref_acc_list.append(ref_accuracy)
        # get all feature column ids
        pred_acc_all_features_rep = []
        for feature_column_id in np.arange(predict_features.shape[1]):
            #print(feature_column_id)
            post_softmax_probs, post_prob_predictions = bn.get_posterior_cat_prob(predict_features,
                                                                                  [weights_dict],
                                                                                  feature_index_to_shuffle=[
                                                                                      feature_column_id],
                                                                                  post_summary_mode=1,
                                                                                  unlink_features_within_block=True,
                                                                                  actFun=bnn_obj._act_fun,
                                                                                  output_act_fun=bnn_obj._output_act_fun)
            accuracy = bn.CalcAccuracy(post_prob_predictions, train_labels)
            # delta_acc = ref_accuracy-accuracy
            pred_acc_all_features_rep.append(accuracy)
        pred_acc_list.append(pred_acc_all_features_rep)

    ref_accs = np.array(ref_acc_list)
    feat_perm_accs = np.array(pred_acc_list)

    delta_accs = np.array([ref_accs-i for i in feat_perm_accs.T])
    mean_delta_acc_features = np.mean(delta_accs,axis=1)
    std_delta_acc_features = np.std(delta_accs,axis=1)

    feature_imp_data = np.vstack([feature_names,mean_delta_acc_features,std_delta_acc_features])
    final_feature_imp_df = pd.DataFrame(feature_imp_data.T,columns = ['feature_name','delta_acc_mean','delta_acc_std'])
    final_feature_imp_df[['delta_acc_mean','delta_acc_std']] = final_feature_imp_df[['delta_acc_mean','delta_acc_std']].astype(float)
    sorted_final_feature_imp_df = final_feature_imp_df.sort_values('delta_acc_mean',ascending=False)
    sorted_final_feature_imp_df = sorted_final_feature_imp_df[sorted_final_feature_imp_df.feature_name != 'Kogia']
    sorted_final_feature_imp_df = sorted_final_feature_imp_df.reset_index(drop=True)
    sorted_final_feature_imp_df.to_csv(feature_importance_out_file,sep='\t',index=False,header=True)
else:
    sorted_final_feature_imp_df = pd.read_csv(feature_importance_out_file,sep='\t')
x = sorted_final_feature_imp_df.index.values
y = sorted_final_feature_imp_df.delta_acc_mean.values
e = sorted_final_feature_imp_df.delta_acc_std.values

#selected_indices = np.where((y-e)>0)[0]
selected_indices = np.arange(15)

type_id = np.zeros(len(x)).astype(int)
plant_bolean = ['eae' in i for i in sorted_final_feature_imp_df.feature_name.values]
type_id[plant_bolean] = 1
abiotic_bolean = [i in abiotic_feature_names for i in sorted_final_feature_imp_df.feature_name.values]
type_id[abiotic_bolean] = 2

formatted_selected_feature_names = sorted_final_feature_imp_df.feature_name.values[selected_indices]
formatted_selected_feature_names[type_id[selected_indices]<2] = ['$%s$'%i for i in formatted_selected_feature_names[type_id[selected_indices]<2]]



fig = plt.figure(figsize=[10, 5])
subplot = fig.add_subplot(111)
new_x = np.arange(len(selected_indices))
new_type_id = type_id[selected_indices]
if plot_error:
    subplot.errorbar(new_x[np.where(new_type_id==0)[0]], y[selected_indices][np.where(new_type_id==0)[0]], e[selected_indices][np.where(new_type_id==0)[0]], marker='.',linestyle='None', label='Mammal taxa',color='sienna',ms=15,elinewidth=5)
    subplot.errorbar(new_x[np.where(new_type_id==1)[0]], y[selected_indices][np.where(new_type_id==1)[0]], e[selected_indices][np.where(new_type_id==1)[0]], marker='.',linestyle='None', label='Plant taxa',color='green',ms=15,elinewidth=5)
    subplot.errorbar(new_x[np.where(new_type_id==2)[0]], y[selected_indices][np.where(new_type_id==2)[0]], e[selected_indices][np.where(new_type_id==2)[0]], marker='.',linestyle='None', label='Abiotic features',color='black',ms=15,elinewidth=5)
else:
    subplot.scatter(new_x[np.where(new_type_id==0)[0]], y[selected_indices][np.where(new_type_id==0)[0]], label='Mammal taxa',color='sienna',marker='x', s=100, linewidths = 3, zorder=3)
    subplot.scatter(new_x[np.where(new_type_id==1)[0]], y[selected_indices][np.where(new_type_id==1)[0]], label='Plant taxa',color='green',marker='x', s=100, linewidths = 3, zorder=3)
    subplot.scatter(new_x[np.where(new_type_id==2)[0]], y[selected_indices][np.where(new_type_id==2)[0]], label='Abiotic features',color='black',marker='x', s=100, linewidths = 3, zorder=3)

#subplot.errorbar(new_x, y[selected_indices], e[selected_indices], marker='.',linestyle='None', label='Significantly positive features',color='blue')
plt.xticks(new_x, formatted_selected_feature_names, rotation=45)
plt.ylim(-0.03,0.2)
plt.grid(which='major')
plt.gca().axhline(0,linestyle='--',color='black')
plt.tight_layout()
plt.show()
#plt.legend(loc='upper center')
fig.savefig('plots/feature_importance_major.pdf')


marker_size = 2.7
fig = plt.figure(figsize=[7, 2])
subplot = fig.add_subplot(111)
if plot_error:
    subplot.errorbar(x[np.where(type_id==0)[0]], y[np.where(type_id==0)[0]], e[np.where(type_id==0)[0]], marker='s',ms=marker_size-1,linestyle='None', label='Mammal taxa',color='sienna',elinewidth=marker_size)
    subplot.errorbar(x[np.where(type_id==1)[0]], y[np.where(type_id==1)[0]], e[np.where(type_id==1)[0]], marker='s',ms=marker_size-1,linestyle='None', label='Plant taxa',color='green',elinewidth=marker_size)
    subplot.errorbar(x[np.where(type_id==2)[0]], y[np.where(type_id==2)[0]], e[np.where(type_id==2)[0]], marker='s',ms=marker_size-1,linestyle='None', label='Abiotic features',color='black',elinewidth=marker_size)
else:
    subplot.scatter(x[np.where(type_id==0)[0]], y[np.where(type_id==0)[0]], label='Mammal taxa',color='sienna',marker='.', s=35, linewidths = 1, zorder=3)
    subplot.scatter(x[np.where(type_id==1)[0]], y[np.where(type_id==1)[0]], label='Plant taxa',color='green',marker='.', s=35, linewidths = 1, zorder=3)
    subplot.scatter(x[np.where(type_id==2)[0]], y[np.where(type_id==2)[0]], label='Abiotic features',color='black',marker='.', s=35, linewidths = 1, zorder=3)

plt.axvspan(-1,15,alpha=0.2)
#subplot.errorbar(x, y, e, marker='.',linestyle='None', label='All features',color='grey')
#subplot.errorbar(x[selected_indices], y[selected_indices], e[selected_indices], marker='.',linestyle='None', label='Significantly positive features',color='blue')
plt.gca().axhline(0,linestyle='--',color='black',zorder=0)
plt.ylim(-0.005,0.13)
#plt.grid(which='major')
#plt.xlabel('Feature importance rank')
#plt.ylabel('Delta prediction accuracy')
plt.tight_layout()
plt.show()
fig.savefig('plots/feature_importance_minor.pdf')




# Below is some code to compile feature importance for the test set across all CV folds_________________
# # load cv pkl files
# cv_pkl_files = glob.glob(
#     'results/results/n_current_331_n_paleo_331_nnodes_8_biotic_1_abiotic_1_sumpool_0_maxpool_0/continued_cv*.pkl')
#
# # get reference accuracy for each cv fold
# cv_acc_dict = {}
# for pkl_file in cv_pkl_files:
#     cv_i = int(os.path.basename(pkl_file).split('_')[1].replace('cv', ''))
#     bnn_obj, mcmc_obj, logger_obj, feature_gen_obj = bn.load_obj(pkl_file)
#     posterior_samples = logger_obj._post_weight_samples
#     posterior_samples = [posterior_samples[i] for i in selected_ids]
#     # get test indices
#     test_index_file = 'data/instance_selection_for_training/cv_instance_ids/n_paleo_331_n_current_331_cv_%i_of_5_test.txt' % cv_i
#     all_test_indices = np.loadtxt(test_index_file).astype(int)
#     if only_paleo_instances:
#         selected_test_indices = all_test_indices[abiotic_features[all_test_indices, 2] > 0]
#     else:
#         selected_test_indices = all_test_indices
#     # get test features and labels
#     test_labels = all_veg_labels[selected_test_indices]
#     pred_features_obj = PredictFeatures(spatial_distances_file,
#                                         temporal_distances_file,
#                                         scaled_additional_features,
#                                         feature_group_ids,
#                                         instance_index=selected_test_indices,
#                                         multiple_weights_per_species=multiple_weights_per_species,
#                                         sum_faunal_floral_features=sum_faunal_floral_features,
#                                         max_pooling=max_pooling,
#                                         actfun=feature_actfun
#                                         )
#     pred_acc_cv_fold = []
#     ref_acc_cv_fold = []
#     for i, weights_dict in enumerate(posterior_samples):
#         print(i)
#         feature_weights_rep = weights_dict['additional_prm']
#         pred_features_obj.update_weigths(feature_weights_rep[0], feature_weights_rep[1])
#         # extract final features for this rep
#         test_features = pred_features_obj.get_features_unseen_data()
#
#         post_softmax_probs, post_prob_predictions = bn.get_posterior_cat_prob(test_features,
#                                                                               posterior_samples,
#                                                                               post_summary_mode=1,
#                                                                               actFun=bnn_obj._act_fun,
#                                                                               output_act_fun=bnn_obj._output_act_fun)
#         ref_accuracy = bn.CalcAccuracy(post_prob_predictions, test_labels)
#         ref_acc_cv_fold.append(ref_accuracy)
#         # get all feature column ids
#         pred_acc_all_features_rep = []
#         for feature_column_id in np.arange(test_features.shape[1]):
#             print(feature_column_id)
#             post_softmax_probs, post_prob_predictions = bn.get_posterior_cat_prob(test_features,
#                                                                                   posterior_samples,
#                                                                                   feature_index_to_shuffle=[
#                                                                                       feature_column_id],
#                                                                                   post_summary_mode=1,
#                                                                                   unlink_features_within_block=True,
#                                                                                   actFun=bnn_obj._act_fun,
#                                                                                   output_act_fun=bnn_obj._output_act_fun)
#             accuracy = bn.CalcAccuracy(post_prob_predictions, test_labels)
#             # delta_acc = ref_accuracy-accuracy
#             pred_acc_all_features_rep.append(accuracy)
#         pred_acc_cv_fold.append(pred_acc_all_features_rep)
#     cv_acc_dict.setdefault(cv_i, [ref_acc_cv_fold, pred_acc_cv_fold])
#
# ref_accs = np.array([cv_acc_dict[i][0] for i in cv_acc_dict.keys()])
# feat_perm_accs = np.array([cv_acc_dict[i][1] for i in cv_acc_dict.keys()])

# np.save('results/feature_importance/ref_accs.npy',ref_accs)
# np.save('results/feature_importance/feat_perm_accs.npy',feat_perm_accs)
# ref_accs_mean = np.mean(ref_accs,axis=0)
# feat_perm_accs_mean = np.mean(feat_perm_accs,axis=0)

#_____________________________________________________________________________________________________




# fig = plt.figure(figsize=[10, 5])
# subplot = fig.add_subplot(121)
# subplot.errorbar(x[np.where(type_id==0)[0]], y[np.where(type_id==0)[0]], e[np.where(type_id==0)[0]], marker='.',linestyle='None', label='Mammal taxa',color='brown')
# subplot.errorbar(x[np.where(type_id==1)[0]], y[np.where(type_id==1)[0]], e[np.where(type_id==1)[0]], marker='.',linestyle='None', label='Plant taxa',color='green')
# subplot.errorbar(x[np.where(type_id==2)[0]], y[np.where(type_id==2)[0]], e[np.where(type_id==2)[0]], marker='.',linestyle='None', label='Abiotic features',color='black')
# #subplot.errorbar(x, y, e, marker='.',linestyle='None', label='All features',color='grey')
# #subplot.errorbar(x[selected_indices], y[selected_indices], e[selected_indices], marker='.',linestyle='None', label='Significantly positive features',color='blue')
# plt.gca().axhline(0,linestyle='--',color='black')
# plt.ylim(-0.02,0.2)
# #plt.grid(which='major')
# plt.xlabel('Feature importance rank')
# plt.ylabel('Delta prediction accuracy')
# plt.legend()
# plt.tight_layout()
#
# subplot = fig.add_subplot(122)
# new_x = np.arange(len(selected_indices))
# new_type_id = type_id[selected_indices]
# subplot.errorbar(new_x[np.where(new_type_id==0)[0]], y[selected_indices][np.where(new_type_id==0)[0]], e[selected_indices][np.where(new_type_id==0)[0]], marker='.',linestyle='None', label='Mammal taxa',color='brown')
# subplot.errorbar(new_x[np.where(new_type_id==1)[0]], y[selected_indices][np.where(new_type_id==1)[0]], e[selected_indices][np.where(new_type_id==1)[0]], marker='.',linestyle='None', label='Plant taxa',color='green')
# subplot.errorbar(new_x[np.where(new_type_id==2)[0]], y[selected_indices][np.where(new_type_id==2)[0]], e[selected_indices][np.where(new_type_id==2)[0]], marker='.',linestyle='None', label='Abiotic features',color='black')
# #subplot.errorbar(new_x, y[selected_indices], e[selected_indices], marker='.',linestyle='None', label='Significantly positive features',color='blue')
# plt.xticks(new_x, formatted_selected_feature_names, rotation=90)
# plt.ylim(-0.02,0.2)
# #plt.grid(which='major')
# plt.gca().axhline(0,linestyle='--',color='black')
# plt.tight_layout()
#
# fig.savefig('results/feature_importance/feature_importance_final_sorted.pdf')
# fig.savefig('plots/feature_importance_final_sorted.pdf')



#
#
#
#
#
#     # for each pkl, load weights
#     for pkl_file in cv_pkl_files:
#         #pkl_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_281_n_paleo_281_regular/cv1_p1_h0_l32_8_s1_binf_1234.pkl'
#         bnn_obj, mcmc_obj, logger_obj, feature_gen_obj = bn.load_obj(pkl_file)
#         posterior_samples = logger_obj._post_weight_samples
#
#         # get test indices for this cv fold
#
#
#         post_softmax_probs, post_prob_predictions = get_posterior_cat_prob(input_features, weights_posterior,
#                                                                            feature_index_to_shuffle=feature_block,
#                                                                            post_summary_mode=post_summary_mode,
#                                                                            unlink_features_within_block=unlink_features_within_block,
#                                                                            actFun=actFun,
#                                                                            output_act_fun=output_act_fun)
#         accuracy = CalcAccuracy(post_prob_predictions, true_labels)
#
#
#
# pred_features_obj = PredictFeatures(spatial_distances_file,
#                                     temporal_distances_file,
#                                     scaled_additional_features,
#                                     feature_group_ids,
#                                     instance_index=test_indices,
#                                     multiple_weights_per_species = multiple_weights_per_species,
#                                     sum_faunal_floral_features = sum_faunal_floral_features,
#                                     max_pooling = max_pooling,
#                                     actfun = feature_actfun
#                                     )
#
# abiotic_feature_names = ['Longitude','Latitude','Time','Precipitation','Temperature','Global_mean_temp']
# feature_names = list(taxon_names)+abiotic_feature_names
#
# # load pkl file
# pkl_file = 'cluster_content/feature_gen_mc3/new_runs_2021/testing_backup/n_current_281_n_paleo_281_regular/current_281_paleo_281_p1_h0_l32_8_s1_binf_1234_1.pkl'
# bnn_obj, mcmc_obj, logger_obj, feature_gen_obj = bn.load_obj(pkl_file)
# posterior_samples = logger_obj._post_weight_samples
# n_posterior_samples = 100
# selected_ids = np.random.choice(np.arange(len(posterior_samples)),n_posterior_samples,replace=False)
# posterior_samples = [posterior_samples[i] for i in selected_ids]
# feature_imp_dict = {}
# for i, weights_dict in enumerate(posterior_samples):
#     print(i)
#     # read fwature weights and apply to raw featrue values
#     feature_weights_rep = weights_dict['additional_prm']
#     pred_features_obj.update_weigths(feature_weights_rep[0],feature_weights_rep[1])
#     # extract final features for this rep
#     test_features = pred_features_obj.get_features_unseen_data()
#     # predict test set using feature importance
#     feature_importance_out = bn.feature_importance(test_features,
#                                                    weights_pkl=None,
#                                                    weights_posterior=[weights_dict],
#                                                    true_labels=paleo_test_labels,
#                                                    fname_stem='',
#                                                    feature_names=feature_names,
#                                                    verbose=False,
#                                                    post_summary_mode=0,
#                                                    n_permutations=100,
#                                                    feature_blocks=dict(),
#                                                    write_to_file=False,
#                                                    predictions_outdir='results/feature_importance',
#                                                    unlink_features_within_block=True,
#                                                    actFun=bnn_obj._act_fun,
#                                                    output_act_fun=bnn_obj._output_act_fun)
#     tmp = [feature_imp_dict.setdefault(i,[]) for i in feature_importance_out.feature_name]
#     tmp2 = [feature_imp_dict[i].append(feature_importance_out.iloc[np.where(feature_importance_out.feature_name.values==i)[0][0],2]) for i in feature_importance_out.feature_name]
#
#
# first_col = np.array(list(feature_imp_dict.keys()))
# remaining_cols = np.array([feature_imp_dict[i] for i in feature_imp_dict.keys()])
# data_array = np.vstack([first_col.T,remaining_cols.T]).T
# feature_imp_df = pd.DataFrame(data_array)
# feature_imp_df.to_csv('results/feature_importance/feature_importance_df_100_posterior_samples.txt',sep='\t',index=False,header=False)
#
# final_feat_imp_list = []
# for i in feature_imp_dict.keys():
#     feature_imp_res = [i,np.mean(feature_imp_dict[i]),np.std(feature_imp_dict[i])]
#     final_feat_imp_list.append(feature_imp_res)
# final_feature_imp_df = pd.DataFrame(final_feat_imp_list)
# final_feature_imp_df.columns = ['feature_name','delta_acc_mean','delta_acc_std']
# sorted_final_feature_imp_df = final_feature_imp_df.sort_values('delta_acc_mean',ascending=False)
# feature_importance_out_file = 'results/feature_importance/feature_importance_final_sorted.txt'
# sorted_final_feature_imp_df = sorted_final_feature_imp_df[sorted_final_feature_imp_df.feature_name != 'Kogia']
# sorted_final_feature_imp_df.to_csv(feature_importance_out_file,sep='\t',index=False,header=True)
#
#
# feature_imp_data = pd.read_csv(feature_importance_out_file,sep='\t')
# x = feature_imp_data.index.values
# y = feature_imp_data.delta_acc_mean.values
# e = feature_imp_data.delta_acc_std.values
# selected_indices = np.where((y-e)>0)[0]
#
# type_id = np.zeros(len(x)).astype(int)
# plant_bolean = ['eae' in i for i in feature_imp_data.feature_name.values]
# type_id[plant_bolean] = 1
# abiotic_bolean = [i in ['Time','Temperature','Global_mean_temp','Latitude','Longitude','Precipitation'] for i in feature_imp_data.feature_name.values]
# type_id[abiotic_bolean] = 2
#
#
# fig = plt.figure(figsize=[10, 5])
#
# subplot = fig.add_subplot(121)
# subplot.errorbar(x[np.where(type_id==0)[0]], y[np.where(type_id==0)[0]], e[np.where(type_id==0)[0]], marker='.',linestyle='None', label='Mammal taxa',color='brown')
# subplot.errorbar(x[np.where(type_id==1)[0]], y[np.where(type_id==1)[0]], e[np.where(type_id==1)[0]], marker='.',linestyle='None', label='Plant taxa',color='green')
# subplot.errorbar(x[np.where(type_id==2)[0]], y[np.where(type_id==2)[0]], e[np.where(type_id==2)[0]], marker='.',linestyle='None', label='Abiotic features',color='black')
# #subplot.errorbar(x, y, e, marker='.',linestyle='None', label='All features',color='grey')
# #subplot.errorbar(x[selected_indices], y[selected_indices], e[selected_indices], marker='.',linestyle='None', label='Significantly positive features',color='blue')
# plt.gca().axhline(0,linestyle='--',color='black')
# plt.ylim(-0.04,0.16)
# plt.xlabel('Feature importance rank')
# plt.ylabel('Delta prediction accuracy')
# plt.legend()
# plt.tight_layout()
#
# subplot = fig.add_subplot(122)
# new_x = np.arange(len(selected_indices))
# new_type_id = type_id[selected_indices]
# subplot.errorbar(new_x[np.where(new_type_id==0)[0]], y[selected_indices][np.where(new_type_id==0)[0]], e[selected_indices][np.where(new_type_id==0)[0]], marker='.',linestyle='None', label='Mammal taxa',color='brown')
# subplot.errorbar(new_x[np.where(new_type_id==1)[0]], y[selected_indices][np.where(new_type_id==1)[0]], e[selected_indices][np.where(new_type_id==1)[0]], marker='.',linestyle='None', label='Plant taxa',color='green')
# subplot.errorbar(new_x[np.where(new_type_id==2)[0]], y[selected_indices][np.where(new_type_id==2)[0]], e[selected_indices][np.where(new_type_id==2)[0]], marker='.',linestyle='None', label='Abiotic features',color='black')
# #subplot.errorbar(new_x, y[selected_indices], e[selected_indices], marker='.',linestyle='None', label='Significantly positive features',color='blue')
# plt.xticks(new_x, feature_imp_data.feature_name.values[selected_indices], rotation=45)
# plt.ylim(-0.04,0.16)
# plt.gca().axhline(0,linestyle='--',color='black')
# plt.tight_layout()
#
# fig.savefig('plots/feature_importance_final_sorted.pdf')

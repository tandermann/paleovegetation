#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:40:53 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import multiprocessing.pool
import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import pickle
# load BNN modules
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features, make_colormap
# sys.path.insert(0, r'/Users/tobiasandermann/Desktop/npBNN/')
import np_bnn as bn

np.set_printoptions(suppress=True)
np.random.seed(1234)


def get_accs(pred_with_posterior_weights_sample,post_prob,true_labels,summary_type,instance_ids_per_stage,all_test_indices,paleo_test_indices,current_test_indices,current_non_train_instances,post_threshold=0,sample_from_posterior = 0):
    indx = np.where(np.max(post_prob, axis=1) >= post_threshold)[0]
    bad_indx = np.where(np.max(post_prob, axis=1) < post_threshold)[0]
    #res_supported = pred_with_posterior_weights_sample[indx, :]
    #labels_supported = true_labels[indx]
    if sample_from_posterior == 1:
        predicted_labels = np.argmax(post_prob, axis=1)
    elif sample_from_posterior == 2:
        predicted_labels = np.array([np.random.choice([0, 1], p=i) for i in post_prob])
    else:
        predicted_labels = np.argmax(pred_with_posterior_weights_sample, axis=1)

    # subsample only those indices that pass the post-thres cutoff
    instance_ids_per_stage = [np.setdiff1d(j,bad_indx) for j in instance_ids_per_stage]
    all_test_indices = np.setdiff1d(all_test_indices, bad_indx)
    paleo_test_indices = np.setdiff1d(paleo_test_indices, bad_indx)
    current_test_indices = np.setdiff1d(current_test_indices, bad_indx)
    current_non_train_instances = np.array([i for i in current_non_train_instances if i in indx])

    if summary_type == 1:  # summarize by stage and average across stages
        all_accs = [sum(predicted_labels[i] == true_labels[i]) / len(i) for i in instance_ids_per_stage]
    elif summary_type == 4: # summarize across all input labels
        #print('Accuracy calculated across all input instances')
        acc_all = sum(predicted_labels[indx] == true_labels[indx]) / len(true_labels[indx])
        all_accs = [acc_all]
    else:  # get paleo and current acc
        accuracy_all_test = sum(predicted_labels[all_test_indices] == true_labels[all_test_indices]) / len(all_test_indices)
        accuracy_paleo_test = sum(predicted_labels[paleo_test_indices] == true_labels[paleo_test_indices]) / len(paleo_test_indices)
        accuracy_current_test = sum(predicted_labels[current_test_indices] == true_labels[current_test_indices]) / len(current_test_indices)
        accuracy_current_non_train_instances = sum(predicted_labels[current_non_train_instances] == true_labels[current_non_train_instances]) / len(current_non_train_instances)
        if summary_type == 0:
            all_accs = [accuracy_all_test, accuracy_paleo_test, accuracy_current_test, accuracy_current_non_train_instances]
        elif summary_type == 2:
            all_accs = [(accuracy_paleo_test * 10 + accuracy_current_non_train_instances * 1) / 11]
        elif summary_type == 3:
            all_accs = [accuracy_current_non_train_instances]
    return all_accs


def get_confidence_threshold(predicted_labels, true_labels, summary_type, instance_ids_per_stage,all_test_indices,paleo_test_indices,current_test_indices,current_non_train_instances, target_acc=None):
    # CALC TRADEOFFS
    tbl_results = []
    for i in np.linspace(0.01, 0.99, 99):
        try:
            scores = get_accuracy_threshold(predicted_labels, true_labels, summary_type, instance_ids_per_stage,all_test_indices,paleo_test_indices,current_test_indices,current_non_train_instances, threshold=i)
            tbl_results.append([i, scores['accuracy'], scores['retained_samples']])
        except:
            pass
    tbl_results = np.array(tbl_results)
    if target_acc is None:
        return tbl_results
    else:
        try:
            indx = np.min(np.where(np.round(tbl_results[:, 1], 2) >= target_acc))
        except ValueError:
            sys.exit('Target accuracy can not be reached. Set a lower threshold and try again.')
        selected_row = tbl_results[indx, :]
        return selected_row[0]


def get_accuracy_threshold(probs, labels, summary_type, instance_ids_per_stage,all_test_indices,paleo_test_indices,current_test_indices,current_non_train_instances,threshold=0.75):
    accuracies = get_accs(  probs,
                            probs,
                            labels,
                            summary_type,
                            instance_ids_per_stage,
                            all_test_indices,
                            paleo_test_indices,
                            current_test_indices,
                            current_non_train_instances,
                            post_threshold=threshold)
    accuracy = accuracies[0]
    indx = np.where(np.max(probs, axis=1) > threshold)[0]
    res_supported = probs[indx, :]
    pred = np.argmax(res_supported, axis=1)
    dropped_frequency = len(pred) / len(labels)
    return {'predictions': pred, 'accuracy': accuracy, 'retained_samples': dropped_frequency}


def get_geo_stage_index(age_array):
    geological_stages = np.array([0,0.0042,0.0082,0.0117,0.129,0.774,1.8,2.58,3.6,5.333,7.246,11.63,13.82,15.97,20.44,23.03,27.82,33.9])
    stage_id = [np.max(np.where(geological_stages<=i)) for i in age_array]
    return np.array(stage_id)



def get_test_accuracy(weight_pickle):
    data_folder = 'data'
    n_current = int(weight_pickle.split('/')[-2].split('_')[2])
    n_paleo = int(weight_pickle.split('/')[-2].split('_')[5])
    test_indices_file = os.path.join(data_folder, 'train_test_sets/test_instance_indices.txt')
    train_indices_file = os.path.join(data_folder, 'train_test_sets/train_instance_indices_ncurrent_%i_npaleo_%i.txt'%(n_current,n_paleo))
    outdir = os.path.join(os.path.dirname(weight_pickle), 'time_slice_predictions')

    summary_type = 2  # 0 is test,paleo,current,all; 1 is mean through geo-stages; 2 is weighted mean between paleo and current; 3 is only optimized for present
    sample_from_posterior = 1 # 0 is off (default), 1 is inferring predictions from posterior probability, and 2 is randomely sampling from posterior probs, introducing stochacisity

    multiple_weights_per_species = True
    sum_faunal_floral_features = False
    max_pooling = False
    if 'collapse' in weight_pickle:
        sum_faunal_floral_features = True
    if '_max_pooling' in weight_pickle:
        max_pooling = True
    load_predicted_labels = True
    burnin = 0
    post_summary_mode = 1
    post_thres = 0.95
    acc_thres = 0.9#.95

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # get pred feature raw data___________________________________________________
    # define data files (must be in same weight_folder folder as pkl files with weights)
    spatial_dists = os.path.join(data_folder, "spatial_distances_NN_input.pkl")
    temporal_dists = os.path.join(data_folder, "temporal_distances_NN_input.pkl")
    additional_features = np.load(os.path.join(data_folder, "abiotic_features.npy"))
    scaled_additional_features = rescale_abiotic_features(additional_features)
    true_labels_file = os.path.join(data_folder, 'veg_labels.txt')
    true_labels = np.loadtxt(true_labels_file).astype(int)
    taxon_names_file = os.path.join(data_folder,'selected_taxa.txt')
    taxon_names = np.loadtxt(taxon_names_file, dtype=str)
    feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])

    train_instances = np.loadtxt(train_indices_file).astype(int)
    all_non_train_instances = np.array([i for i in np.arange(len(additional_features)) if i not in train_instances])
    current_non_train_instances = all_non_train_instances[np.where(additional_features[all_non_train_instances][:, 2] == 0)[0]]
    all_current_instances = np.arange(len(additional_features))[additional_features[:,2]==0]

    all_test_indices = np.loadtxt(test_indices_file).astype(int)
    paleo_test_indices = all_test_indices[np.where(additional_features[all_test_indices][:, 2] > 0)[0]]
    current_test_indices = all_test_indices[np.where(additional_features[all_test_indices][:, 2] == 0)[0]]
    geo_stage_indices = get_geo_stage_index(additional_features[all_non_train_instances, 2])
    instance_ids_per_stage = [np.where(geo_stage_indices == i)[0] for i in np.unique(geo_stage_indices)]

    paleo_train_indices = train_instances[np.where(additional_features[train_instances,2]>0)[0]]
    all_paleo_indices = np.concatenate([paleo_train_indices,paleo_test_indices])
    geological_stages = np.array([0,0.0042,0.0082,0.0117,0.129,0.774,1.8,2.58,3.6,5.333,7.246,11.63,13.82,15.97,20.44,23.03,27.82,33.9])

    # fig = plt.figure(figsize=[8,8])
    # subplot = fig.add_subplot(221)
    # subplot.hist(additional_features[all_paleo_indices, 2],bins=geological_stages,edgecolor='black',color='r')
    # plt.title('Paleovegetation points - All')
    # plt.xlabel('Time (Mya)')
    # plt.ylabel('Count (per geological stage)')
    # subplot = fig.add_subplot(222)
    # subplot.hist(additional_features[paleo_train_indices, 2],bins=geological_stages,edgecolor='black',color='g')
    # plt.title('Paleovegetation points - Training')
    # plt.xlabel('Time (Mya)')
    # plt.ylabel('Count (per geological stage)')
    # subplot = fig.add_subplot(223)
    # subplot.hist(additional_features[paleo_test_indices, 2],bins=geological_stages,edgecolor='black',color='b')
    # plt.title('Paleovegetation points - Test')
    # plt.xlabel('Time (Mya)')
    # plt.ylabel('Count (per geological stage)')
    # plt.tight_layout()
    # fig.savefig('plots/paleoveg_points_through_time.pdf')


    predicted_labels_out_file = os.path.join(outdir,os.path.basename(weight_pickle).replace('.pkl','_predicted_labels.npy'))
    post_pred_out_file = predicted_labels_out_file.replace('predicted_labels.npy','post_pred.npy')
    if not load_predicted_labels:
        # prediction__________________________________________________________________
        # load the posterior samples
        if 'no_biotic' in weight_pickle:
            bnn_obj, mcmc_obj, logger_obj = bn.load_obj(weight_pickle)
            biotic_features = False
        else:
            bnn_obj, mcmc_obj, logger_obj, featgen_obj = bn.load_obj(weight_pickle)
            biotic_features = True
            # intialize the PredictFeatures object________________________________________
            pred_features_obj = PredictFeatures(spatial_dists,
                                                temporal_dists,
                                                scaled_additional_features,
                                                feature_group_ids,
                                                instance_index=[],
                                                multiple_weights_per_species=multiple_weights_per_species,
                                                sum_faunal_floral_features=sum_faunal_floral_features,
                                                max_pooling=max_pooling
                                                )
        posterior_samples = logger_obj._post_weight_samples
        post_pred = []
        #predicted_labels_all = []

        for i, weights_dict in enumerate(posterior_samples[burnin:]):
            if biotic_features:
                # read feature weights and apply to raw featrue values
                feature_weights_rep = weights_dict['additional_prm']
                pred_features_obj.update_weigths(feature_weights_rep[0],feature_weights_rep[1])
                # extract features for this rep
                predict_features = pred_features_obj.get_features_unseen_data()
                if 'no_abiotic' in weight_pickle:
                    predict_features = predict_features[:, :-6]
            else:
                predict_features = scaled_additional_features

            # apply features and bnn weights to predict labels
            actFun = bnn_obj._act_fun
            output_act_fun = bnn_obj._output_act_fun
            #bn.RunPredict(predict_features, weights_dict['weights'], actFun, output_act_fun)
            post_softmax_probs,post_prob_predictions = bn.get_posterior_cat_prob(predict_features,
                                                                                  [weights_dict],
                                                                                  post_summary_mode=post_summary_mode,
                                                                                  actFun=actFun,
                                                                                  output_act_fun=output_act_fun)
            #predicted_labels = np.argmax(post_prob_predictions, axis=1)
            post_pred.append(post_softmax_probs[0])
            #predicted_labels_all.append(predicted_labels)

            if i%10 == 0:
               print("Predicting labels for posterior samples: ",i)

        post_pred = np.array(post_pred)
        #predicted_labels_all = np.array(predicted_labels_all)
        np.save(post_pred_out_file,post_pred)
        #np.save(predicted_labels_out_file,predicted_labels_all)
    else:
        post_pred = np.load(post_pred_out_file)
        #predicted_labels_all = np.load(predicted_labels_out_file)

    # predicted_labels_all = np.array([np.argmax(i, axis=1) for i in post_pred])
    # accuracies = []
    # for predicted_labels in predicted_labels_all:
    #     accuracy_all_test = sum(predicted_labels[all_test_indices] == true_labels[all_test_indices]) / len(all_test_indices)
    #     accuracy_paleo_test = sum(predicted_labels[paleo_test_indices] == true_labels[paleo_test_indices]) / len(paleo_test_indices)
    #     accuracy_current_test = sum(predicted_labels[current_test_indices] == true_labels[current_test_indices]) / len(current_test_indices)
    #     accuracy_current_non_train_instances = sum(predicted_labels[current_non_train_instances] == true_labels[current_non_train_instances]) / len(current_non_train_instances)
    #     all_accs = [accuracy_all_test,accuracy_paleo_test,accuracy_current_test,accuracy_current_non_train_instances]
    #     accuracies.append(all_accs)
    # accuracies = np.array(accuracies)


    # calculate posterior probabilities
    if post_summary_mode == 0:
        prob_1 = np.mean(np.array([np.argmax(i, axis=1) for i in post_pred]), axis=0)
        prob_0 = 1 - prob_1
        post_prob = np.array([prob_0, prob_1]).T
    elif post_summary_mode == 1:
        post_prob = np.mean(post_pred, axis=0)


# fig = plt.figure(figsize=[10,5])
# subplot = fig.add_subplot(121)
# preds_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_1405_n_paleo_281/time_slice_predictions/continued_p1_h0_l32_8_s1_binf_1234_post_pred.npy'
# post_pred = np.load(preds_file)
# # calculate posterior probabilities
# if post_summary_mode == 0:
#     prob_1 = np.mean(np.array([np.argmax(i, axis=1) for i in post_pred]), axis=0)
#     prob_0 = 1 - prob_1
#     post_prob = np.array([prob_0, prob_1]).T
# elif post_summary_mode == 1:
#     post_prob = np.mean(post_pred, axis=0)
# plt.plot(post_prob[paleo_test_indices,1],true_labels[paleo_test_indices],'.')
# plt.xlabel('pred_pp')
#
# subplot = fig.add_subplot(122)
# preds_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_281_n_paleo_281_regular/time_slice_predictions/continued_p1_h0_l32_8_s1_binf_1234_post_pred.npy'
# post_pred = np.load(preds_file)
# # calculate posterior probabilities
# if post_summary_mode == 0:
#     prob_1 = np.mean(np.array([np.argmax(i, axis=1) for i in post_pred]), axis=0)
#     prob_0 = 1 - prob_1
#     post_prob = np.array([prob_0, prob_1]).T
# elif post_summary_mode == 1:
#     post_prob = np.mean(post_pred, axis=0)
# plt.plot(post_prob[paleo_test_indices,1],true_labels[paleo_test_indices],'.')
# plt.xlabel('pred_pp')



    # calculate accuracies_______________________
    accuracies = []
    for pred_with_posterior_weights_sample in post_pred:
        all_accs = get_accs(pred_with_posterior_weights_sample,
                            post_prob,
                            true_labels,
                            summary_type,
                            instance_ids_per_stage,
                            all_test_indices,
                            paleo_test_indices,
                            current_test_indices,
                            current_non_train_instances,
                            post_threshold=0,
                            sample_from_posterior = sample_from_posterior)
        accuracies.append(all_accs)
    accuracies = np.array(accuracies)


    # accthres table____________________________
    if not 'final_model_' in weight_pickle: # do not calculate the accthres tabel based on the final production model
        #post_pred_test = post_pred[:,all_test_indices,:]
        acc_thres_tbl = get_confidence_threshold(post_prob,
                                                 true_labels,
                                                 summary_type,
                                                 instance_ids_per_stage,
                                                 all_test_indices,
                                                 paleo_test_indices,
                                                 current_test_indices,
                                                 current_non_train_instances)
        np.savetxt(os.path.join(outdir,os.path.basename(weight_pickle).replace('.pkl','_acc_thres_tbl_post_mode_%i_sum_mode_%i_sample_from_post_%i.txt'%(post_summary_mode,summary_type,sample_from_posterior))),acc_thres_tbl,fmt='%.3f')

    if acc_thres:
        add_string = '_post_mode_%i_acc_thres_%.2f_sum_mode_%i_sample_from_post_%i'%(post_summary_mode,acc_thres,summary_type,sample_from_posterior)
        try:
            accthrestbl_rowid = min(np.where(acc_thres_tbl[:,1]>acc_thres)[0])
            post_thres = acc_thres_tbl[accthrestbl_rowid,0]
        except:
            post_thres = 1.1 #everything shoudl be uncertain in this case
    else:
        add_string = '_post_mode_%i_post_thres_%.2f_sum_mode_%i_sample_from_post_%i'%(post_summary_mode,post_thres,summary_type,sample_from_posterior)

    # select the paleo and current indices to summarize accuracy separately for these subsets
    current_map_coords = additional_features[all_current_instances,:2]
    current_map_labs = true_labels[all_current_instances]
    current_map_pred_labs = np.argmax(post_prob[all_current_instances,:],axis=1)
    delta_lab = np.abs(current_map_labs-current_map_pred_labs)*2
    confident_ids = np.where(post_prob[all_current_instances,:]>=post_thres)[0]
    unconfident_ids = [i for i in np.arange(post_prob[all_current_instances,:].shape[0]) if i not in confident_ids]
    delta_lab[unconfident_ids] = delta_lab[unconfident_ids]/2

    sample_acc_df = os.path.join(outdir, '%s_test_acc_%i_sample_from_post_%i.txt' %(os.path.basename(weight_pickle).replace('.pkl', ''),summary_type,sample_from_posterior))
    if summary_type == 0:
        test_accuracy_scores = np.matrix([np.mean(accuracies[:,0]),np.std(accuracies[:,0]),np.mean(accuracies[:,1]),np.std(accuracies[:,1]),np.mean(accuracies[:,2]),np.std(accuracies[:,2]),np.mean(accuracies[:,3]),np.std(accuracies[:,3])])
        test_accuracy_df = pd.DataFrame(test_accuracy_scores,columns=['all_mean','all_std','paleo_mean','paleo_std','current_mean','current_std','all_current_mean','all_current_std']).T
        test_accuracy_df = np.round(test_accuracy_df,3)
        test_accuracy_df.to_csv(sample_acc_df,sep=':',index=True,header=None)
    else:
        test_accuracy_scores = np.mean(accuracies,axis=1)
        test_accuracy_mean = np.mean(test_accuracy_scores)
        test_accuracy_std = np.std(test_accuracy_scores)
        test_accuracy_df = pd.DataFrame(np.matrix([test_accuracy_mean,test_accuracy_std]),columns=['all_mean','all_std']).T
        test_accuracy_df = np.round(test_accuracy_df,3)
        test_accuracy_df.to_csv(sample_acc_df,sep=':',index=True,header=None)

    # make map plot
    fig = plt.figure(figsize=[10,15])
    point_size = 3

    subplot = fig.add_subplot(311)
    subplot.scatter(current_map_coords[current_map_pred_labs==0][:, 0], current_map_coords[current_map_pred_labs==0][:, 1], c='darkgreen', marker=',', s=point_size, label='Closed habitat')
    subplot.scatter(current_map_coords[current_map_pred_labs==1][:, 0], current_map_coords[current_map_pred_labs==1][:, 1], c='goldenrod', marker=',', s=point_size, label='Open habitat')
    subplot.scatter(current_map_coords[unconfident_ids, 0], current_map_coords[unconfident_ids, 1], c='grey', marker=',', s=point_size, label='Low confidence')
    plt.title('Current predicted vegetation')
    plt.legend(loc='lower left')
    plt.tight_layout()

    subplot = fig.add_subplot(312)
    subplot.scatter(current_map_coords[current_map_labs==0][:, 0], current_map_coords[current_map_labs==0][:, 1], c='darkgreen', marker=',', s=point_size, label='Closed habitat')
    subplot.scatter(current_map_coords[current_map_labs==1][:, 0], current_map_coords[current_map_labs==1][:, 1], c='goldenrod', marker=',', s=point_size, label='Open habitat')
    plt.title('Current true vegetation')
    plt.legend(loc='lower left')
    plt.tight_layout()

    subplot = fig.add_subplot(313)
    subplot.scatter(current_map_coords[delta_lab==0][:, 0], current_map_coords[delta_lab==0][:, 1], c='grey', marker=',', s=point_size, label='Correctly classified')
    subplot.scatter(current_map_coords[delta_lab==1][:, 0], current_map_coords[delta_lab==1][:, 1], c='lightblue', marker=',', s=point_size, label='Low confidence misclassification')
    subplot.scatter(current_map_coords[delta_lab==2][:, 0], current_map_coords[delta_lab==2][:, 1], c='red', marker=',', s=point_size, label='Misclassification')
    plt.title('Prediction error')
    plt.legend(loc='lower left')
    plt.tight_layout()

    fig.savefig(os.path.join(outdir,os.path.basename(weight_pickle).replace('.pkl','_current_map_predictions%s.pdf'%add_string)))
    plt.close()

    # predict for all of current map
    # extract the non-train indices for prediction accuracy
    # subtract predictions from true labels
'''
    if make_accuracy_plot:
        fig = plt.figure(figsize=(15,5))
        bin_size = 0.02
        min_value = np.min(accuracies+accuracies_paleo+accuracies_current)
        max_value = np.max(accuracies+accuracies_paleo+accuracies_current)+bin_size

        fig.add_subplot(131)
        plt.title('All')
        plt.hist(accuracies,np.arange(min_value,max_value,bin_size),color='green')
        plt.axvline(np.mean(accuracies),color='green',linestyle='--')
        plt.axvline(np.mean(accuracies_paleo),color='blue',linestyle='--')
        plt.axvline(np.mean(accuracies_current),color='orange',linestyle='--')

        fig.add_subplot(132)
        plt.title('Only paleo')
        plt.hist(accuracies_paleo,np.arange(min_value,max_value,bin_size),color='blue')
        plt.axvline(np.mean(accuracies),color='green',linestyle='--')
        plt.axvline(np.mean(accuracies_paleo),color='blue',linestyle='--')
        plt.axvline(np.mean(accuracies_current),color='orange',linestyle='--')

        fig.add_subplot(133)
        plt.title('Only current')
        plt.hist(accuracies_current,np.arange(min_value,max_value,bin_size),color='orange')
        plt.axvline(np.mean(accuracies),color='green',linestyle='--')
        plt.axvline(np.mean(accuracies_paleo),color='blue',linestyle='--')
        plt.axvline(np.mean(accuracies_current),color='orange',linestyle='--')

        fig.savefig(sample_acc_df.replace('.txt','.pdf'))
'''

plot_figure_maps = False
if plot_figure_maps:
    plotoutdir = 'plots/map_predictions_high_res_model/'

    fig = plt.figure(figsize=[10,5])
    plt.scatter(current_map_coords[current_map_pred_labs == 0][:, 0],
                    current_map_coords[current_map_pred_labs == 0][:, 1], c='darkgreen', marker=',', s=point_size,
                    label='Closed habitat')
    plt.scatter(current_map_coords[current_map_pred_labs == 1][:, 0],
                    current_map_coords[current_map_pred_labs == 1][:, 1], c='goldenrod', marker=',', s=point_size,
                    label='Open habitat')
    plt.tight_layout()
    fig.savefig(os.path.join(plotoutdir,'predicted_map.pdf'))


    fig = plt.figure(figsize=[10,5])
    plt.scatter(current_map_coords[delta_lab == 0][:, 0], current_map_coords[delta_lab == 0][:, 1], c='grey',
                    marker=',', s=point_size, label='Correctly classified')
    plt.scatter(current_map_coords[delta_lab > 0][:, 0], current_map_coords[delta_lab > 0][:, 1], c='red', marker=',',
                    s=point_size, label='Misclassification')
    plt.tight_layout()
    fig.savefig(os.path.join(plotoutdir,'predicted_map_error.pdf'))


    fig = plt.figure(figsize=[10,5])
    plt.scatter(current_map_coords[current_map_pred_labs == 0][:, 0],
                    current_map_coords[current_map_pred_labs == 0][:, 1], c='darkgreen', marker=',', s=point_size,
                    label='Closed habitat')
    plt.scatter(current_map_coords[current_map_pred_labs == 1][:, 0],
                    current_map_coords[current_map_pred_labs == 1][:, 1], c='goldenrod', marker=',', s=point_size,
                    label='Open habitat')
    plt.scatter(current_map_coords[unconfident_ids, 0], current_map_coords[unconfident_ids, 1], c='grey', marker=',',
                    s=point_size, label='Low confidence')
    plt.tight_layout()
    fig.savefig(os.path.join(plotoutdir,'predicted_map_threshold.pdf'))


    fig = plt.figure(figsize=[10,5])
    plt.scatter(current_map_coords[delta_lab == 0][:, 0], current_map_coords[delta_lab == 0][:, 1], c='grey',
                    marker=',', s=point_size, label='Correctly classified')
    plt.scatter(current_map_coords[delta_lab == 1][:, 0], current_map_coords[delta_lab == 1][:, 1], c='lightblue',
                    marker=',', s=point_size, label='Low confidence misclassification')
    plt.scatter(current_map_coords[delta_lab == 2][:, 0], current_map_coords[delta_lab == 2][:, 1], c='red', marker=',',
                    s=point_size, label='Misclassification')
    plt.tight_layout()
    fig.savefig(os.path.join(plotoutdir,'predicted_map_error_threshold.pdf'))


    fig = plt.figure(figsize=[10,5])
    plt.scatter(current_map_coords[current_map_labs == 0][:, 0], current_map_coords[current_map_labs == 0][:, 1],
                    c='darkgreen', marker=',', s=point_size, label='Closed habitat')
    plt.scatter(current_map_coords[current_map_labs == 1][:, 0], current_map_coords[current_map_labs == 1][:, 1],
                    c='goldenrod', marker=',', s=point_size, label='Open habitat')
    plt.tight_layout()
    fig.savefig(os.path.join(plotoutdir,'current_veg_map.pdf'))


#weight_pkl_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/new_runs_2021/testing_backup/n_current_1405_n_paleo_281/current_1405_paleo_281_p1_h0_l32_8_s1_binf_1234.pkl'
#get_test_accuracy(weight_pkl_file)

basedir = 'new_runs_2021/testing'
weight_pickle_files = glob.glob(os.path.join(basedir,'*/*.pkl'))
selected_weight_pickle_files = []
for i in weight_pickle_files:
    if os.path.basename(i).startswith('continued_'):
        selected_weight_pickle_files.append(i)
#    if os.path.basename(i).startswith('final_model'):
#        pass
#    elif os.path.basename(i).startswith('continued_'):
#        pass
#    else:
#        selected_weight_pickle_files.append(i)
#        #get_test_accuracy(i)

#selected_weight_pickle_files = ['/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/new_runs_2021/testing_backup/n_current_1405_n_paleo_281/final_model_p1_h0_l32_8_s1_binf_1234.pkl']
cpus=14
pool = multiprocessing.Pool(cpus)
args = selected_weight_pickle_files
unkown_data_feature_array = np.array(pool.map(get_test_accuracy, args))
pool.close()

#fig = plt.figure(figsize=(10,5))
#mammuthus_occs_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/fossil_data/occurrences_by_taxon/current/Castor_current_occs.txt'
#mammuthus_occs_df = pd.read_csv(mammuthus_occs_file,sep='\t')
#plt.scatter(current_map_coords[:, 0], current_map_coords[:, 1], c='lightgrey', marker=',', s=point_size)
#plt.scatter(mammuthus_occs_df.lon,mammuthus_occs_df.lat,c='red', marker=',', s=point_size)
#plt.tight_layout()
#fig.savefig('/Users/tobiasandermann/GitHub/feature_gen_paleoveg/plots/castor_occs.pdf')


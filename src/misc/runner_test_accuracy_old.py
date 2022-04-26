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


def get_confidence_threshold(predicted_labels, true_labels, target_acc=None):
    # CALC TRADEOFFS
    tbl_results = []
    for i in np.linspace(0.01, 0.99, 99):
        try:
            scores = get_accuracy_threshold(predicted_labels, true_labels, threshold=i)
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


def get_accuracy_threshold(probs, labels, threshold=0.75):
    indx = np.where(np.max(probs, axis=1) > threshold)[0]
    res_supported = probs[indx, :]
    labels_supported = labels[indx]
    pred = np.argmax(res_supported, axis=1)
    accuracy = len(pred[pred == labels_supported]) / len(pred)
    dropped_frequency = len(pred) / len(labels)
    return {'predictions': pred, 'accuracy': accuracy, 'retained_samples': dropped_frequency}


def get_test_accuracy(weight_pickle):
    data_folder = 'data'
    n_current = int(weight_pickle.split('/')[-2].split('_')[2])
    n_paleo = int(weight_pickle.split('/')[-2].split('_')[5])
    test_indices_file = os.path.join(data_folder, 'train_test_sets/test_instance_indices.txt')
    train_indices_file = os.path.join(data_folder, 'train_test_sets/train_instance_indices_ncurrent_%i_npaleo_%i.txt'%(n_current,n_paleo))
    outdir = os.path.join(os.path.dirname(weight_pickle), 'time_slice_predictions')

    multiple_weights_per_species = True
    sum_faunal_floral_features = False
    max_pooling = False
    if 'collapse' in weight_pickle:
        sum_faunal_floral_features = True
    if '_max_pooling' in weight_pickle:
        max_pooling = True
    make_accuracy_plot = False
    load_predicted_labels = False
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

    all_test_indices = np.loadtxt(test_indices_file).astype(int)
    paleo_test_indices = all_test_indices[np.where(additional_features[all_test_indices][:, 2] > 0)[0]]
    current_test_indices = all_test_indices[np.where(additional_features[all_test_indices][:, 2] == 0)[0]]
    train_instances = np.loadtxt(train_indices_file).astype(int)
    all_non_train_instances = np.array([i for i in np.arange(len(additional_features)) if i not in train_instances])
    current_non_train_instances = all_non_train_instances[np.where(additional_features[all_non_train_instances][:, 2] == 0)[0]]
    all_current_instances = np.arange(len(additional_features))[additional_features[:,2]==0]


    predicted_labels_out_file = os.path.join(outdir,os.path.basename(weight_pickle).replace('.pkl','_predicted_labels.npy'))
    if not load_predicted_labels:
        # prediction__________________________________________________________________
        accuracies = []
        accuracies_paleo = []
        accuracies_current = []
        accuracies_current_non_train_instances = []
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
                                                max_pooling = max_pooling
                                                )
        posterior_samples = logger_obj._post_weight_samples
        post_pred = []
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
            predicted_labels = np.argmax(post_prob_predictions, axis=1)
            post_pred.append(post_softmax_probs[0])

            accuracy_all_test = sum(predicted_labels[all_test_indices] == true_labels[all_test_indices])/len(all_test_indices)
            accuracy_paleo_test = sum(predicted_labels[paleo_test_indices] == true_labels[paleo_test_indices])/len(paleo_test_indices)
            accuracy_current_test = sum(predicted_labels[current_test_indices] == true_labels[current_test_indices])/len(current_test_indices)
            accuracy_current_non_train_instances = sum(predicted_labels[current_non_train_instances] == true_labels[current_non_train_instances])/len(current_non_train_instances)

            accuracies.append(accuracy_all_test)
            accuracies_paleo.append(accuracy_paleo_test)
            accuracies_current.append(accuracy_current_test)
            accuracies_current_non_train_instances.append(accuracy_current_non_train_instances)
            if i%10 == 0:
               print("Predicting labels for posterior samples: ",i, ". Test acc = ", accuracy_all_test)

        # prepare accthrestbl
        post_pred = np.array(post_pred)
        np.save(predicted_labels_out_file,post_pred)
    else:
        post_pred = np.load(predicted_labels_out_file)

    #post_pred_test = post_pred[:,all_test_indices,:]
    if post_summary_mode == 0:
        prob_1 = np.mean(np.array([np.argmax(i, axis=1) for i in post_pred]),axis=0)
        prob_0 = 1-prob_1
        post_prob = np.array([prob_0,prob_1]).T
    elif post_summary_mode == 1:
        post_prob = np.mean(post_pred,axis=0)
    post_prob_test = post_prob[all_test_indices,:]
    acc_thres_tbl = get_confidence_threshold(post_prob_test,true_labels[all_test_indices])
    np.savetxt(os.path.join(outdir,os.path.basename(weight_pickle).replace('.pkl','_acc_thres_tbl_post_sum_mode_%i.txt'%post_summary_mode)),acc_thres_tbl,fmt='%.3f')

    if acc_thres:
        add_string = '_post_mode_%i_acc_thres_%.2f'%(post_summary_mode,acc_thres)
        accthrestbl_rowid = min(np.where(acc_thres_tbl[:,1]>acc_thres)[0])
        post_thres = acc_thres_tbl[accthrestbl_rowid,0]
    else:
        add_string = '_post_mode_%i_post_thres_%.2f'%(post_summary_mode,post_thres)


    # select the paleo and current indices to summarize accuracy separately for these subsets
    current_map_coords = additional_features[all_current_instances,:2]
    current_map_labs = true_labels[all_current_instances]
    current_map_pred_labs = np.argmax(post_prob[all_current_instances,:],axis=1)
    delta_lab = np.abs(current_map_labs-current_map_pred_labs)*2
    confident_ids = np.where(post_prob[all_current_instances,:]>=post_thres)[0]
    unconfident_ids = [i for i in np.arange(post_prob[all_current_instances,:].shape[0]) if i not in confident_ids]
    delta_lab[unconfident_ids] = delta_lab[unconfident_ids]/2




    #logfile_content = pd.read_csv(weight_pickle.replace('.pkl','.log'),sep='\t')
    #logfile_content.accuracy.values[-1000:]
    #np.array(accuracies)
    #sum(logfile_content.accuracy.values[-1000:]-np.array(accuracies))
    if not load_predicted_labels:
        test_accuracy_scores = np.matrix([np.mean(accuracies),np.std(accuracies),np.mean(accuracies_paleo),np.std(accuracies_paleo),np.mean(accuracies_current),np.std(accuracies_current),np.mean(accuracies_current_non_train_instances),np.std(accuracies_current_non_train_instances)])
        test_accuracy_df = pd.DataFrame(test_accuracy_scores,columns=['all_mean','all_std','paleo_mean','paleo_std','current_mean','current_std','all_current_mean','all_current_std']).T
        test_accuracy_df = np.round(test_accuracy_df,3)
        sample_acc_df = os.path.join(outdir,'%s_test_acc.txt'%os.path.basename(weight_pickle).replace('.pkl',''))
        test_accuracy_df.to_csv(sample_acc_df,sep=':',index=True,header=None)



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

    fig.savefig(os.path.join(outdir,os.path.basename(weight_pickle).replace('.pkl','_current_map_predictions%s.pdf'%add_string)),tightlayout=True)
    #plt.close()


    # predict for all of current map
    # extract the non-train indices for prediction accuracy
    # subtract predictions from true labels





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



#weight_pkl_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/new_runs_2021/testing_backup/n_current_1405_n_paleo_281/current_1405_paleo_281_p1_h0_l32_8_s1_binf_1234.pkl'
#get_test_accuracy(weight_pkl_file)




basedir = 'new_runs_2021/testing_backup'
weight_pickle_files = glob.glob(os.path.join(basedir,'*/*.pkl'))
#for i in weight_pickle_files:
#    get_test_accuracy(i)
cpus=3
pool = multiprocessing.Pool(cpus)
args = weight_pickle_files
unkown_data_feature_array = np.array(pool.map(get_test_accuracy, args))
pool.close()

#fig = plt.figure(figsize=(10,5))
#mammuthus_occs_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/fossil_data/occurrences_by_taxon/current/Castor_current_occs.txt'
#mammuthus_occs_df = pd.read_csv(mammuthus_occs_file,sep='\t')
#plt.scatter(current_map_coords[:, 0], current_map_coords[:, 1], c='lightgrey', marker=',', s=point_size)
#plt.scatter(mammuthus_occs_df.lon,mammuthus_occs_df.lat,c='red', marker=',', s=point_size)
#plt.tight_layout()
#fig.savefig('/Users/tobiasandermann/GitHub/feature_gen_paleoveg/plots/castor_occs.pdf')


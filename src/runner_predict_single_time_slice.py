#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:56:28 2020

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""


import sys,os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
# load BNN modules
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features, make_colormap
#sys.path.insert(0, r'/Users/tobiasandermann/Desktop/npBNN/')
import np_bnn as bn

np.set_printoptions(suppress=True)
np.random.seed(1234)


try:
    # this option is useful when parsing a single seed into the script
    # instead of running the loop for many seeds consecutively in this script
    data_folder = str(sys.argv[1])
    weight_pickle = str(sys.argv[2])
    timepoint = int(sys.argv[3])
    accthrestbl_file = str(sys.argv[4])
    precipiation_perturbation = float(sys.argv[5])
    temperature_perturbation = float(sys.argv[6])
    elevation_perturbation = float(sys.argv[7])
    n_burnin = int(sys.argv[8])
    try:
        outdir = str(sys.argv[9])
    except: # by default store the results in the folder with the pkl file
        outdir = os.path.join(os.path.dirname(weight_pickle),'time_slice_predictions')
except:
    # alternatively define a weight_folder and loop through all pkl files found in the dir
    data_folder = 'data/time_slice_features'
    weight_pickle = 'results/production_model_best_model/continued_final_model_p1_h0_l8_s1_binf_1234.pkl'
    timepoint = 0
    accthrestbl_file = 'results/results/n_current_331_n_paleo_331_nnodes_8_biotic_1_abiotic_1_sumpool_0_maxpool_0/testset_predictions/continued_current_331_paleo_331_p1_h0_l8_s1_binf_1234_acc_thres_tbl_post_mode_1_cv.txt'
    precipiation_perturbation = 0.0
    temperature_perturbation = 0.0
    elevation_perturbation = 0.0
    n_burnin = 500
    outdir = os.path.join(os.path.dirname(weight_pickle), 'time_slice_predictions')
    print('Falling back to default settings.')

#for timepoint in np.arange(31):
print(timepoint)
taxon_names_file = 'data/selected_taxa.txt'
#data_folder = 'data/time_slice_features'
#weight_pickle = 'results/production_model_best_model/final_model_current_331_paleo_331_p1_h0_l8_s1_binf_1234.pkl'
#timepoint = 0
#outdir = os.path.join(os.path.dirname(weight_pickle),'time_slice_predictions')

plot_feature_maps = True
only_plot_abiotic_featuremaps = False
small_fig_layout = True
accsum_type = 2
acc_thres = 0.#85#85#9#.9 #0.95
plotting_cutoffs = []#[0.05,0.95] # ignored when acc_thres is set to >0
colors = ['darkgreen','grey','goldenrod']
post_summary_mode=1
posterior_sample_mode=1
feature_actfun=2
load_predicted_labels = False
predicted_labels_out_dir = os.path.join(outdir,'predicted_labels')
if sum([precipiation_perturbation,temperature_perturbation,elevation_perturbation]) > 0:
    predicted_labels_out_dir = predicted_labels_out_dir+'_p%.1f_t%.1f_e%.1f'%(precipiation_perturbation,temperature_perturbation,elevation_perturbation)
    outdir = predicted_labels_out_dir
    modify_abiotic_features = True
else:
    modify_abiotic_features = False

# settings
multiple_weights_per_species = True
sum_faunal_floral_features = False
max_pooling = False
if '_sumpool_1' in weight_pickle:
    sum_faunal_floral_features = True
if '_maxpool_1' in weight_pickle:
    max_pooling = True
predicted_labels_out_file = os.path.join(predicted_labels_out_dir, "predicted_labels_%iMA.npy" % timepoint)


if not load_predicted_labels:
    # create output folders
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            print('Output folder already exists, continuing with processing...')
    if not os.path.exists(predicted_labels_out_dir):
        try:
            os.makedirs(predicted_labels_out_dir)
        except:
            pass

    spatial_dists = os.path.join(data_folder,"spatial_distances_%iMA.pkl"%timepoint)
    temporal_dists = os.path.join(data_folder,"temporal_distances_%iMA.pkl"%timepoint)
    additional_features = np.load(os.path.join(data_folder,"abiotic_features_%iMA.npy"%timepoint)).copy()
    # modify the abiotic features to see what effect it has on predicted maps
    if modify_abiotic_features:
        perturbated_prec_values = np.random.uniform(additional_features[:, 3]-precipiation_perturbation*additional_features[:, 3],
                                                    additional_features[:, 3]+precipiation_perturbation*additional_features[:, 3])
        additional_features[:, 3] = perturbated_prec_values
        perturbated_temp_values = np.random.uniform(additional_features[:, 4]-temperature_perturbation*additional_features[:, 4],
                                                    additional_features[:, 4]+temperature_perturbation*additional_features[:, 4])
        additional_features[:, 4] = perturbated_temp_values
        perturbated_elev_values = np.random.uniform(additional_features[:, 5]-elevation_perturbation*additional_features[:, 5],
                                                    additional_features[:, 5]+elevation_perturbation*additional_features[:, 5])
        additional_features[:, 5] = perturbated_elev_values
    scaled_additional_features = rescale_abiotic_features(additional_features)
    taxon_names_file = 'data/selected_taxa.txt'
    taxon_names = np.loadtxt(taxon_names_file, dtype=str)
    feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])

    # initiate output array and file
    # do predictions
    if 'no_biotic' in weight_pickle:
        bnn_obj, mcmc_obj, logger_obj = bn.load_obj(weight_pickle)
        predict_features = scaled_additional_features
    else:
        bnn_obj, mcmc_obj, logger_obj, featgen_obj = bn.load_obj(weight_pickle)
        # intialize the PredictFeatures object________________________________________

        pred_features_obj = PredictFeatures(spatial_dists,
                                            temporal_dists,
                                            scaled_additional_features,
                                            feature_group_ids,
                                            multiple_weights_per_species=multiple_weights_per_species,
                                            sum_faunal_floral_features=sum_faunal_floral_features,
                                            max_pooling = max_pooling,
                                            actfun = feature_actfun
                                            )



    posterior_samples = logger_obj._post_weight_samples
    labels = []
    post_pred = []
    feature_weigths_0 = []
    feature_weigths_1 = []
    for i, weights_dict in enumerate(posterior_samples[n_burnin:]):
        print(n_burnin+i)
        if not 'no_biotic' in weight_pickle:
            # read feature weights and apply to raw feature values
            feature_weights_rep = weights_dict['additional_prm']
            feature_weigths_0.append(feature_weights_rep[0])
            feature_weigths_1.append(feature_weights_rep[1])
            pred_features_obj.update_weigths(feature_weights_rep[0], feature_weights_rep[1])
            # extract final features for this rep
            predict_features = pred_features_obj.get_features_unseen_data()
            if 'no_abiotic' in weight_pickle:
                predict_features = predict_features[:, :-8]
        # apply features and bnn weights to predict labels
        actFun = bnn_obj._act_fun
        output_act_fun = bnn_obj._output_act_fun
        # bn.RunPredict(predict_features_feature_gen, weights_dict['weights'], actFun, output_act_fun)
        post_softmax_probs, post_prob_predictions = bn.get_posterior_cat_prob(predict_features,
                                                                              [weights_dict],
                                                                              post_summary_mode=post_summary_mode,
                                                                              actFun=actFun,
                                                                              output_act_fun=output_act_fun)
        post_pred.append(post_softmax_probs[0])
        predicted_labels = np.argmax(post_prob_predictions, axis=1)
        labels.append(predicted_labels)

    #labels = np.array(labels)
    post_pred = np.array(post_pred)
    np.save(predicted_labels_out_file,post_pred)

    if plot_feature_maps:
        # compile feature maps for all features, using mean weight estimates
        pred_features_obj.update_weigths(np.array(feature_weigths_0).mean(axis=0), np.array(feature_weigths_1).mean(axis=0))
        predict_features = pred_features_obj.get_features_unseen_data()
        # compile all feature names
        abiotic_feature_names = ['Longitude','Latitude','Time','Precipitation','Temperature','Elevation','Global_mean_temp','Global_mean_co2']
        taxon_names = np.loadtxt(taxon_names_file, dtype=str)
        feature_names = list(taxon_names)+abiotic_feature_names
        # create outdir for feature maps
        feature_maps_outpath = os.path.join('/Users/tobiasandermann/Desktop/tmp_bnn_pred','feature_maps_%i'%timepoint)
        if not os.path.exists(feature_maps_outpath):
            os.makedirs(feature_maps_outpath)
        # make the plot
        ylim_vals = [22, 81]
        xlim_vals = [-183, -40]
        for i,feature_values in enumerate(predict_features.T):
            save_plot=False
            if only_plot_abiotic_featuremaps: # if this is activated, only plot abiotic features
                if feature_names[i] in abiotic_feature_names:
                    save_plot = True
            else:
                save_plot = True
            if save_plot:
                fig = plt.figure(figsize=(10, 5))
                coords = additional_features[:, :2]
                plot = plt.scatter(coords[:, 0], coords[:, 1],c=feature_values,cmap='viridis', marker=',', s=2,zorder=3)
                # plt.xlim([-300,-140])
                plt.xlim(xlim_vals)
                plt.ylim(ylim_vals)
                outfile_name = os.path.join(feature_maps_outpath,'%s_feature_map.pdf'%feature_names[i])
                # plt.grid(axis='both', linestyle='dashed', which='major',zorder=0)
                # cbar = plt.colorbar(plot,orientation="vertical")
                # cbar.ax.tick_params(labelsize=20)
                # if small_fig_layout:
                #    plt.xticks(fontsize=20)
                #    plt.yticks(fontsize=20)
                # else:
                #    plt.title('Predicted vegetation %i MA'%timepoint)
                plt.gca().axes.xaxis.set_visible(False)
                plt.gca().axes.yaxis.set_visible(False)
                plt.gca().axis('off')
                plt.show()
                fig.savefig(outfile_name)
                plt.close()

else:
    post_pred = np.load(predicted_labels_out_file)
    # get coordinates
    additional_features = np.load(os.path.join(data_folder,"abiotic_features_%iMA.npy"%timepoint))

# calculate posterior probs
if post_summary_mode == 0:
    prob_1 = np.mean(np.array([np.argmax(i, axis=1) for i in post_pred]), axis=0)
    prob_0 = 1 - prob_1
    post_prob = np.array([prob_0, prob_1]).T
elif post_summary_mode == 1:
    post_prob = np.mean(post_pred, axis=0)

coords = additional_features[:,:2]

plt.scatter(coords[:, 0], coords[:, 1], c='grey', marker=',', s=2)
ylim_vals = [22,81]#plt.gca().get_ylim()
xlim_vals = [-183,-40]#plt.gca().get_xlim()
plt.show()
plt.close()



fig = plt.figure(figsize=(10,5))
c = mcolors.ColorConverter().to_rgb
if acc_thres > 0:
    accthrestbl = np.loadtxt(accthrestbl_file)
    try:
        row_index = np.min(np.where(accthrestbl[:,1]>=acc_thres))
        post_thres = accthrestbl[row_index,:][0]
    except:
        post_thres = 1.1

    pass_indices = np.where(post_prob>=post_thres)[0]
    coords_selected = coords[pass_indices]
    post_prob_selected = post_prob[pass_indices]

    # plt.scatter(coords[:,0],coords[:,1],c=colors[0],marker=',',s=2,zorder=3,label='Closed')
    # plt.scatter(coords[:,0],coords[:,1],c=colors[2],marker=',',s=2,zorder=3,label='Open')
    plt.scatter(coords[:,0],coords[:,1],c='grey',marker=',',s=2,zorder=3,label='Unknown')
    # lgnd = plt.legend(loc="lower left",fontsize=20,ncol=3,framealpha=1)
    # # change the marker size manually for both lines
    # lgnd.legendHandles[0]._sizes = [200]
    # lgnd.legendHandles[1]._sizes = [200]
    # lgnd.legendHandles[2]._sizes = [200]

    rvb = make_colormap([c(colors[0]), 0.5, c(colors[2])])
    plt.scatter(list(coords_selected[:, 0]) + [-200,-199], list(coords_selected[:, 1]) + [20,20],c=list(post_prob_selected[:,1]) + [0.0,1.0],cmap=rvb, marker=',', s=2,zorder=3)
    # plt.xlim([-300,-140])
    plt.xlim(xlim_vals)
    plt.ylim(ylim_vals)
    outfile_name = '%s_prediction_acc_thres_%.2f_mode_%i_sum_mode_%i_%02dMA.pdf' % (
                                os.path.basename(weight_pickle).replace('.pkl', ''),
                                acc_thres,
                                post_summary_mode,
                                accsum_type,
                                timepoint)
elif len(plotting_cutoffs)>0:
    rvb = make_colormap([c(colors[0]), plotting_cutoffs[0], c(colors[1]), plotting_cutoffs[1], c(colors[2])])
    plt.scatter(list(coords[:,0]) + [-200,-199],list(coords[:,1]) + [20,20],c=list(post_prob[:,1]) + [0.0,1.0],cmap=rvb,marker=',',s=2,zorder=3)
    plt.xlim(xlim_vals)
    plt.ylim(ylim_vals)
    outfile_name = '%s_prediction_%.2f_%.2f_mode_%i_sum_mode_%i_%02dMA.pdf' % (
                                os.path.basename(weight_pickle).replace('.pkl', ''),
                                plotting_cutoffs[0],
                                plotting_cutoffs[1],
                                post_summary_mode,
                                accsum_type,
                                timepoint)
else:
    rvb = make_colormap([c(colors[0]), c(colors[2])])
    plot = plt.scatter(list(coords[:, 0]) + [-200,-199], list(coords[:, 1]) + [20,20], c=list(post_prob[:, 1])+ [0.0,1.0], cmap=rvb, marker=',', s=2,zorder=3)
    plt.xlim(xlim_vals)
    plt.ylim(ylim_vals)
    # cbar = plt.colorbar(plot,orientation="horizontal")
    # cbar.ax.tick_params(labelsize=20)
    outfile_name = '%s_prediction_raw_mode_%i_sum_mode_%i_%02dMA.pdf' % (
                                os.path.basename(weight_pickle).replace('.pkl', ''),
                                post_summary_mode,
                                accsum_type,
                                timepoint)

plt.grid(axis='both', linestyle='dashed', which='major',zorder=0)
if small_fig_layout:
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
else:
    plt.title('Predicted vegetation %i MA'%timepoint)
fig.savefig(os.path.join(outdir,outfile_name))
plt.show()
plt.close()


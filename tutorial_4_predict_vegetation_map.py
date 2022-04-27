import os, sys
import numpy as np
import np_bnn as bn
import pandas as pd
sys.path.append('..')
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features, make_colormap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_posterior_prob_from_bnn_output(post_pred,
                                       post_summary_mode):
    # calculate posterior probabilities
    if post_summary_mode == 0:
        prob_1 = np.mean(np.array([np.argmax(i, axis=1) for i in post_pred]), axis=0)
        prob_0 = 1 - prob_1
        post_prob = np.array([prob_0, prob_1]).T
    elif post_summary_mode == 1:
        post_prob = np.mean(post_pred, axis=0)
    return post_prob


# define input and output folder
try:
    datadir_map = str(sys.argv[1])
except:
    datadir_map = 'tutorial/raw_data/compiled_data_current_map'

datadir = 'tutorial/training_data'
modeldir = 'tutorial/trained_model'
outdir = 'tutorial/model_predictions'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# load input data
weight_pickle = os.path.join(modeldir, "model_1_p1_h0_l32_8_s1_binf_1234.pkl")
spatial_dists = os.path.join(datadir_map, "spatial_distances_NN_input.pkl")
temporal_dists = os.path.join(datadir_map, "temporal_distances_NN_input.pkl")
additional_features = np.load(os.path.join(datadir_map, "abiotic_features.npy"))
scaled_additional_features = rescale_abiotic_features(additional_features,feature_set='public')
taxon_names_file = os.path.join(datadir,'selected_taxa.txt')
taxon_names = np.loadtxt(taxon_names_file, dtype=str)
feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])
predicted_labels_out_file = os.path.join(outdir, "%s_predicted_labels.npy"%(os.path.basename(datadir_map).replace('compiled_data_','')))


# prediction settings
n_burnin=150
post_threshold_for_plotting = 0.65
post_summary_mode=1
posterior_sample_mode=1
multiple_weights_per_species = True
predicted_labels_out_dir = os.path.join(outdir,'predicted_labels')

# load all objects from the saved model
bnn_obj, mcmc_obj, logger_obj, featgen_obj = bn.load_obj(weight_pickle)
featgen_obj.__dict__.keys()
# intialize the PredictFeatures object
pred_features_obj = PredictFeatures(spatial_dists,
                                    temporal_dists,
                                    scaled_additional_features,
                                    feature_group_ids,
                                    multiple_weights_per_species=multiple_weights_per_species,
                                    sum_faunal_floral_features=featgen_obj._sum_faunal_floral_features,
                                    max_pooling=featgen_obj._max_pooling,
                                    actfun=featgen_obj._actfun
                                    )


posterior_weight_samples = logger_obj._post_weight_samples
labels = []
post_pred = []

for i, weights_dict in enumerate(posterior_weight_samples[n_burnin:]):
    print("Predicting for posterior sample %i" %(n_burnin + i))
    # read feature weights and apply to raw feature values
    feature_weights_rep = weights_dict['additional_prm']
    pred_features_obj.update_weigths(feature_weights_rep[0], feature_weights_rep[1])
    # extract final features for this rep
    predict_features = pred_features_obj.get_features_unseen_data()
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

# labels = np.array(labels)
post_pred = np.array(post_pred)
np.save(predicted_labels_out_file, post_pred)

posterior_probs = get_posterior_prob_from_bnn_output(post_pred, post_summary_mode)
estimated_labels = np.argmax(posterior_probs, axis=1)
coords = additional_features[:,:2]
ylim_vals = [22,81]#plt.gca().get_ylim()
xlim_vals = [-183,-40]#plt.gca().get_xlim()


colors = ['darkgreen','grey','goldenrod']
fig = plt.figure(figsize=(10,15))
# posterior prob map____________________________
subplot1 = fig.add_subplot(311)
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c(colors[0]), c(colors[2])])
plot = plt.scatter(list(coords[:, 0]) + [-200,-199], list(coords[:, 1]) + [20,20], c=list(posterior_probs[:, 1])+ [0.0,1.0], cmap=rvb, marker=',', s=2,zorder=3)
plt.xlim(xlim_vals)
plt.ylim(ylim_vals)
# cbar = plt.colorbar(plot,orientation="horizontal")
# cbar.ax.tick_params(labelsize=20)
plt.grid(axis='both', linestyle='dashed', which='major',zorder=0)
plt.title('Posterior probability')
# all label call map____________________________

subplot2 = fig.add_subplot(312)
rvb = make_colormap([c(colors[0]), 0.5, c(colors[2])])
plt.scatter(list(coords[:, 0]) + [-200, -199], list(coords[:, 1]) + [20, 20],
            c=list(estimated_labels) + [0.0, 1.0], cmap=rvb, marker=',', s=2, zorder=3)
# plt.xlim([-300,-140])
plt.xlim(xlim_vals)
plt.ylim(ylim_vals)
plt.grid(axis='both', linestyle='dashed', which='major',zorder=0)
plt.title('Best estimate all cells')
# posterior cutoff map____________________________
subplot3 = fig.add_subplot(313)
pass_indices = np.where(np.sum(posterior_probs >= post_threshold_for_plotting,axis=1)==1)[0]
coords_selected = coords[pass_indices]
post_prob_selected = posterior_probs[pass_indices]
plt.scatter(coords[:, 0], coords[:, 1], c='grey', marker=',', s=2, zorder=3, label='Unknown')
rvb = make_colormap([c(colors[0]), 0.5, c(colors[2])])
plt.scatter(list(coords_selected[:, 0]) + [-200, -199], list(coords_selected[:, 1]) + [20, 20],
            c=list(post_prob_selected[:, 1]) + [0.0, 1.0], cmap=rvb, marker=',', s=2, zorder=3)
# plt.xlim([-300,-140])
plt.xlim(xlim_vals)
plt.ylim(ylim_vals)
plt.grid(axis='both', linestyle='dashed', which='major',zorder=0)
plt.title('Posterior cutoff %.2f'%post_threshold_for_plotting)


outfile_name = predicted_labels_out_file.replace('.npy','.pdf')
fig.savefig(outfile_name)
plt.show()
plt.close()

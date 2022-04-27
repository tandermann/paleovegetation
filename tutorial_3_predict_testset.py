import os, sys
import numpy as np
import np_bnn as bn
import pandas as pd
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features



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


try:
    modeldir = str(sys.argv[1])
except:
    modeldir = 'tutorial/trained_model'

# define input and output folder
datadir = 'tutorial/training_data'
outdir = 'tutorial/model_predictions'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# load input data
weight_pickle = os.path.join(modeldir, "model_1_p1_h0_l32_8_s1_binf_1234.pkl")
spatial_dists = os.path.join(datadir, "spatial_distances_NN_input.pkl")
temporal_dists = os.path.join(datadir, "temporal_distances_NN_input.pkl")
additional_features = np.load(os.path.join(datadir, "abiotic_features.npy"))
scaled_additional_features = rescale_abiotic_features(additional_features,feature_set='public')
taxon_names_file = os.path.join(datadir,'selected_taxa.txt')
taxon_names = np.loadtxt(taxon_names_file, dtype=str)
feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])
test_instances_path = os.path.join(datadir,"test_instances.txt")
test_indices = np.loadtxt(test_instances_path).astype(int)
predicted_labels_out_file = os.path.join(outdir, "predicted_labels.npy")
labels_file = os.path.join(datadir,"veg_labels.txt")
true_labels = np.loadtxt(labels_file).astype(int)
true_test_labels = true_labels[test_indices]

# prediction settings
n_burnin=150
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
                                    instance_index=test_indices,
                                    multiple_weights_per_species=multiple_weights_per_species,
                                    sum_faunal_floral_features=featgen_obj._sum_faunal_floral_features,
                                    max_pooling=featgen_obj._max_pooling,
                                    actfun=featgen_obj._actfun
                                    )


posterior_weight_samples = logger_obj._post_weight_samples
labels = []
post_pred = []

for i, weights_dict in enumerate(posterior_weight_samples[n_burnin:]):
    print(n_burnin + i)
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

test_pred_df = pd.DataFrame(np.hstack([np.array([estimated_labels,true_test_labels]).T,posterior_probs]),columns=['predicted_vegetation','true_vegetation','posterior_prob_closed','posterior_prob_open'])
test_pred_df[['predicted_vegetation', 'true_vegetation']] = test_pred_df[['predicted_vegetation', 'true_vegetation']].astype(int)
test_pred_df.to_csv(os.path.join(outdir,'test_set_predictions.txt'),sep='\t',index=False,float_format='%.3f')

n_misclassified =  sum(abs(true_test_labels-estimated_labels))
n_correctly_classified = len(true_test_labels)-n_misclassified
test_acc = 1 - n_misclassified/len(true_test_labels)
print('Predicted test set.\nCorrect predictions: %i of %i.\nPrediction accuracy: %.3f.\n' %(n_correctly_classified,len(true_test_labels),test_acc))



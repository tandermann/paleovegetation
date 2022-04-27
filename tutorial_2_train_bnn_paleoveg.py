import os, sys
import copy
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import np_bnn as bn
from feature_gen.feature_gen import FeatureGenerator
from feature_gen.utils import UpdateNormal, select_train_and_test_set, rescale_abiotic_features

# define input and output folder
datadir = 'tutorial/training_data'
outdir = 'tutorial/trained_model'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# load input data
spatial_dists = os.path.join(datadir,"spatial_distances_NN_input.pkl")
temporal_dists = os.path.join(datadir,"temporal_distances_NN_input.pkl")
additional_features_file = os.path.join(datadir,"abiotic_features.npy")
labels_file = os.path.join(datadir,"veg_labels.txt")
train_instances_path = os.path.join(datadir,"train_instances.txt")
additional_features = np.load(additional_features_file)
scaled_additional_features = rescale_abiotic_features(additional_features,feature_set='public')
train_indices = np.loadtxt(train_instances_path).astype(int)
taxon_names_file = os.path.join(datadir,'selected_taxa.txt')
taxon_names = np.loadtxt(taxon_names_file,dtype=str)
# get info if it's plant or mammal for each feature
feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])

# MCMC settings
n_iteration = 500000
n_post_samples = 1000 # defines how many posterior weight samples are being retained in the bnn-object (to limit file size)
sampling_f = 200
print_f = 10
prior = 1  # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
p_scale = 1  # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
update_freq_w1 = 0.25  # time and space weights
update_freq_w2 = 0.1  # taxon weights
window_size_w1 = 0.0075
window_size_w2 = 0.1
update_f = [0.05, 0.1, 0.2, 0.2]
update_ws = [0.075, 0.075, 0.075, 0.075]
mcmc_temp = 1.0
lik_temp = 1.0
init_std = 1.0
reload_features_weights = False

# BNN settings
n_nodes_list = [32,8]
n_current = 331
n_paleo = 331
feature_gen_actfun = 2 #0 is none, 1 is relu, 2 is swish
bias_node_setting = 3
use_class_weight = 0
sample_from_prior = 0
seed = 1234
act_fun = bn.ActFun(fun = 'swish')
multiple_weights_per_species = True
sum_pooling = 0
max_pooling = 0


# _______________________SET UP FEATURE-GENERATOR OBJECT________________________
# during training this part of the BNN architecture will reduce the dimensionality of the distance data to one proximity feature per taxon
feature_obj = FeatureGenerator(spatial_dists,
                               temporal_dists,
                               scaled_additional_features,
                               labels_file,
                               feature_group_ids,
                               instance_index=train_indices,
                               testsize=0.,
                               seed=seed,
                               transform_labels=False,
                               priorscale=p_scale,
                               prior_f=prior,
                               multiple_weights_per_species=multiple_weights_per_species,
                               sum_faunal_floral_features=sum_pooling,
                               max_pooling=max_pooling,
                               actfun=feature_gen_actfun)
dat, prior_prob_feature_weights = feature_obj.get_data()
add_prms = ['feat_w1_space', 'feat_w1_time', 'mean_feat_w2', 'std_feat_w2']
# _____________________________________________________________________


# _______________________SET UP BNN OBJECT________________________
init_weights = None
bnn_obj = bn.BNN_env.npBNN(dat,
                           use_bias_node=bias_node_setting,
                           n_nodes=n_nodes_list,
                           use_class_weights=use_class_weight,
                           prior_f=prior,
                           p_scale=p_scale,
                           seed=seed,
                           init_std=init_std,
                           actFun=act_fun,
                           init_weights=init_weights)
# _____________________________________________________________________


# _______________________SET UP MCMC OBJECT________________________
mcmc_obj = bn.BNN_env.MCMC(bnn_obj,
                           update_f=update_f,
                           update_ws=update_ws,
                           temperature=mcmc_temp,
                           likelihood_tempering=lik_temp,
                           n_iteration=n_iteration,
                           sampling_f=sampling_f,
                           print_f=print_f,
                           n_post_samples=n_post_samples,
                           adapt_f=0.3,
                           adapt_fM=0.6,
                           mcmc_id=1,
                           sample_from_prior=sample_from_prior,
                           init_additional_prob=prior_prob_feature_weights,
                           randomize_seed=False)
out_file_name = os.path.join(outdir, "model_1")
logger_obj = bn.BNN_env.postLogger(bnn_obj,
                                   out_file_name,
                                   add_prms=add_prms,
                                   log_all_weights=0)
# _____________________________________________________________________


# _______________________RUN TRAINING________________________
for i in range(n_iteration):
    rs = RandomState(MT19937(SeedSequence(mcmc_obj._current_iteration + mcmc_obj._mcmc_id)))
    # define temporary copies of feature_gen and bnn objects for which we try new weights
    feature_gen_prime = copy.deepcopy(feature_obj)
    bnn_prime = copy.deepcopy(bnn_obj)
    mcmc_update_n = mcmc_obj._update_n

    # 1. update weights
    rr = rs.random()
    if rr < update_freq_w1 + update_freq_w2:
        # no updates in fully connected NN
        mcmc_obj.reset_update_n([0, 0, 0, 0])
        if rr < update_freq_w1:
            # if mcmc._current_iteration % update_freq_w1 == 0:
            # note that this is the UpdateNormal function that I modified specifically for the feature weights
            w1_prime, index_w1 = UpdateNormal(feature_gen_prime._w1, d=window_size_w1, n=1, Mb=5, mb=-5, rs=rs)
            w2_prime = feature_gen_prime._w2
        else:
            w1_prime = feature_gen_prime._w1
            w2_prime, index_w2 = UpdateNormal(feature_gen_prime._w2, d=window_size_w2, n=1, Mb=5, mb=-5, rs=rs)

        feature_gen_prime.update_weigths(w1_prime, w2_prime)

    # 2.+ 3. update features and get new prior prob
    dat_prime, prior_prime = feature_gen_prime.get_data()

    # 4. update data in bnn
    bnn_prime.update_data(dat_prime)

    # 5. do MCMC step
    mcmc_obj.mh_step(bnn_prime, prior_prime)
    # bnn_obj_new, mcmc_obj_new = mcmc_obj.mh_step(bnn_obj, return_bnn=True)
    mcmc_obj.reset_update_n(mcmc_update_n)

    if mcmc_obj._last_accepted == 1:  # update objects with new weights if accepted
        # bnn = bnn_prime
        bnn_obj.reset_weights(bnn_prime._w_layers)
        bnn_obj._act_fun.reset_prm(bnn_prime._act_fun._prm)
        bnn_obj._act_fun.reset_accepted_prm()
        feature_obj = feature_gen_prime
    else:
        pass
    if mcmc_obj._current_iteration % mcmc_obj._sampling_f == 0:
        print('Current MCMC iteration:', mcmc_obj._current_iteration)
        logger_obj.log_sample(bnn_obj, mcmc_obj,
                              add_prms=list(np.array([feature_obj._w1[0],
                                                      feature_obj._w1[1],
                                                      np.mean(feature_obj._w2),
                                                      np.std(feature_obj._w2)])))
        logger_obj.log_weights(bnn_obj, mcmc_obj,
                               add_prms=[feature_obj._w1, feature_obj._w2],
                               add_obj=feature_obj)


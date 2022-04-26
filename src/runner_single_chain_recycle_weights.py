import os, sys
import copy
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import np_bnn as bn
from feature_gen.feature_gen import FeatureGenerator
from feature_gen.utils import UpdateNormal, select_train_and_test_set, rescale_abiotic_features

# set seed and working directory
try:
    pickle_file = str(sys.argv[1])
    nnodes_str = str(sys.argv[2])
    n_current = int(sys.argv[3])
    n_paleo = int(sys.argv[4])
    biotic_features = int(sys.argv[5])
    abiotic_features = int(sys.argv[6])
    continue_run = int(sys.argv[7])
    cv = int(sys.argv[8])
    final_model = int(sys.argv[9])
    outdir = str(sys.argv[10])
    sum_faunal_floral_features = int(sys.argv[11])
    max_pooling = int(sys.argv[12])
except:
    # define necessary paths
    pickle_file = 'results/production_model_best_model/final_model_current_331_paleo_331_p1_h0_l8_s1_binf_1234.pkl'
    nnodes_str = '8'
    n_current = 331
    n_paleo = 331
    biotic_features = 1
    abiotic_features = 1
    continue_run = 1
    cv = 0
    final_model = 1
    outdir = 'results/production_model_best_model'
    sum_faunal_floral_features = 0
    max_pooling = 0
    print('Using default settings')

datadir = 'data' #'cluster_content/feature_gen_mc3/data/'
if not os.path.exists(outdir):
    os.makedirs(outdir,exist_ok=True)
#outdir = 'runs_2021/modeltesting' #'cluster_content/feature_gen_mc3/runs_2021/modeltesting'
if cv: # don't need that many posterior samples
    n_iteration = 100100
    n_post_samples = 200
    sampling_f = 200
else:
    n_iteration = 201000
    n_post_samples = 1000
    sampling_f = 200
print_f = 10
update_freq_w1 = 0.25  # time and space weights
update_freq_w2 = 0.1  # taxon weights
window_size_w1 = 0.0075
window_size_w2 = 0.1
feature_gen_actfun = 2 #0 is none, 1 is relu, 2 is swish

if continue_run:
    if cv:
        # make sure to get the correct input cv file
        if os.path.basename(pickle_file).split('_')[0] == 'continued':
            pickle_file = os.path.join(os.path.dirname(pickle_file),
                                       'continued_cv' + str(cv) + '_' + '_'.join(os.path.basename(pickle_file).split('_')[2:]))
            continued_input_pkl = True
        else:
            pickle_file = os.path.join(os.path.dirname(pickle_file),
                                       'cv' + str(cv) + '_' + '_'.join(os.path.basename(pickle_file).split('_')[1:]))
            continued_input_pkl = False
    try: # automated reading of file names to make sure to get the settings right
        if 'sumpool_1' in pickle_file:
            sum_faunal_floral_features = 1
        else:
            sum_faunal_floral_features = 0
        if 'maxpool_1' in pickle_file:
            max_pooling = 1
        else:
            max_pooling = 0
        if max_pooling:
            sum_faunal_floral_features = 0  # don't activate both
        if 'nnodes_32_8' in pickle_file:
            nnodes_str = '32_8'
        elif 'nnodes_8' in pickle_file:
            nnodes_str = '8'
        if '_biotic_0' in pickle_file:
            biotic_features = False
        else:
            biotic_features = True

        if '_abiotic_0' in pickle_file:
            abiotic_features = False
        else:
            abiotic_features = True

        n_current = int(os.path.dirname(pickle_file).split('/')[-1].split('_')[2])
        n_paleo = int(os.path.dirname(pickle_file).split('/')[-1].split('_')[5])
        nnodes_str = os.path.basename(pickle_file).split('_l')[-1].split('_s')[0]
    except:
        pass

    # read the files from the pickle file
    if biotic_features:
        bnn_obj, mcmc_obj, logger_obj, feature_obj = bn.load_obj(pickle_file)
    else:
        bnn_obj, mcmc_obj, logger_obj = bn.load_obj(pickle_file)

    #mcmc_obj.__dict__.keys()
    mcmc_obj._current_iteration = 0
    mcmc_obj._n_iterations = n_iteration
    mcmc_obj._adapt_stop = int(mcmc_obj._n_iterations * 0.05)
    mcmc_obj._sampling_f = sampling_f
    if biotic_features:
        feature_obj._actfun = feature_gen_actfun
        if max_pooling:
            feature_obj._max_pooling = True
        else:
            feature_obj._max_pooling = False
    if final_model:
        out_file_name = os.path.join(os.path.dirname(pickle_file), 'continued_final_model')
    elif cv:
        if continued_input_pkl:
            out_file_name = os.path.join(os.path.dirname(pickle_file),
                                         'continued_continued_cv%i_current_%i_paleo_%i' % (cv, n_current, n_paleo))
        else:
            out_file_name = os.path.join(os.path.dirname(pickle_file),
                                         'continued_cv%i_current_%i_paleo_%i' % (cv, n_current, n_paleo))
    logger_obj = bn.postLogger(bnn_obj,
                               filename=out_file_name,
                               add_prms=['feat_w1_space', 'feat_w1_time', 'mean_feat_w2', 'std_feat_w2'],
                               log_all_weights=0)
    if biotic_features:
        loaded_bnn_obj, loaded_mcmc_obj, loaded_logger_obj, loaded_feature_obj = bn.load_obj(pickle_file)
    else:
        loaded_bnn_obj, loaded_mcmc_obj, loaded_logger_obj = bn.load_obj(pickle_file)
    reload_features_weights = True
else:
    reload_features_weights = False

# define input data
spatial_dists = os.path.join(datadir,"spatial_distances_NN_input.pkl")
temporal_dists = os.path.join(datadir,"temporal_distances_NN_input.pkl")
additional_features_file = os.path.join(datadir,"abiotic_features.npy")
labels_file = os.path.join(datadir,"veg_labels.txt")
additional_features = np.load(additional_features_file)
scaled_additional_features = rescale_abiotic_features(additional_features)
if final_model:
    train_instances_path = os.path.join(datadir,'instance_selection_for_training/selected_instances_paleo_%i_current_%i.txt'%(n_paleo,n_current))
elif cv:
    train_instances_path = os.path.join(datadir,'instance_selection_for_training/cv_instance_ids/n_paleo_%i_n_current_%i_cv_%i_of_5_train.txt'%(n_paleo,n_current,cv))
# else:
#     train_instances_path = os.path.join(datadir, 'train_test_sets/train_instance_indices_ncurrent_%i_npaleo_%i.txt' % (n_current, n_paleo))

train_indices = np.loadtxt(train_instances_path).astype(int)
print('Using train indices stored at %s' % train_instances_path)

taxon_names_file = 'data/selected_taxa.txt'
taxon_names = np.loadtxt(taxon_names_file,dtype=str)
feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])

# define settings
n_nodes_list = list(np.array(str.split(nnodes_str,'_')).astype(int))
bias_node_setting = 3
use_class_weight = 0
sample_from_prior = 0
seed = 1234
prior = 1  # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
p_scale = 1  # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
act_fun = bn.ActFun(fun = 'swish')
update_f = [0.05, 0.1, 0.2, 0.2]
update_ws = [0.075, 0.075, 0.075, 0.075]
mcmc_temp = 1.0
lik_temp = 1.0
init_std = 1.0
multiple_weights_per_species = True


if final_model or cv or not continue_run:
    # reinitialize feature object with previous weights
    if biotic_features:
        feature_obj = FeatureGenerator(  spatial_dists,
                                         temporal_dists,
                                         scaled_additional_features,
                                         labels_file,
                                         feature_group_ids,
                                         instance_index=train_indices,
                                         testsize=0.,
                                         seed=seed,
                                         transform_labels=False,
                                         priorscale = p_scale,
                                         prior_f = prior,
                                         multiple_weights_per_species=multiple_weights_per_species,
                                         sum_faunal_floral_features = sum_faunal_floral_features,
                                         max_pooling=max_pooling,
                                         actfun=feature_gen_actfun)

        if reload_features_weights:
            w1 = loaded_feature_obj._w1
            w2 = loaded_feature_obj._w2
            feature_obj.update_weigths(w1,w2)
        dat,prior_prob_feature_weights = feature_obj.get_data()
        add_prms = ['feat_w1_space','feat_w1_time','mean_feat_w2','std_feat_w2']
        if not abiotic_features:
            dat['data'] = dat['data'][:,:-8]
    else:
        features_train = scaled_additional_features[train_indices,:]
        labels = np.loadtxt(labels_file).astype(int)
        labels_train = labels[train_indices]
        prior_prob_feature_weights = 0
        add_prms = []
        dat = bn.get_data(features_train,
                          labels_train,
                          seed=seed,
                          from_file=False,
                          all_class_in_testset=False,
                          testsize=0.,  # 10% test set
                          header=0,  # input data has a header
                          instance_id=0)  # input data includes names of instances


    # set up the BNN object
    init_weights = None
    if continue_run:
        if final_model or cv:
            init_weights = bnn_obj._w_layers
            mcmc_obj._update_ws = update_ws



    bnn_obj = bn.BNN_env.npBNN(  dat,
                                 use_bias_node=bias_node_setting,
                                 n_nodes = n_nodes_list,
                                 use_class_weights=use_class_weight,
                                 prior_f = prior,
                                 p_scale = p_scale,
                                 seed=seed,
                                 init_std=init_std,
                                 actFun=act_fun,
                                 init_weights = init_weights)

    # set up mcmc object
    mcmc_obj = bn.BNN_env.MCMC(  bnn_obj,
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

if not continue_run:
    if final_model:
        out_file_name = os.path.join(outdir, "final_model_current_%i_paleo_%i" %(n_current, n_paleo))
    elif cv:
        out_file_name = os.path.join(outdir, "cv%i_current_%i_paleo_%i" %(cv, n_current, n_paleo))

    logger_obj = bn.BNN_env.postLogger(bnn_obj,
                                       out_file_name,
                                       add_prms=add_prms,
                                       log_all_weights=0)



# run MCMC
if not biotic_features:
    bn.run_mcmc(bnn_obj, mcmc_obj, logger_obj)
else:
    for i in range(n_iteration):
        rs = RandomState(MT19937(SeedSequence(mcmc_obj._current_iteration+mcmc_obj._mcmc_id)))
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
        if not abiotic_features:
            dat_prime['data'] = dat_prime['data'][:, :-8]

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

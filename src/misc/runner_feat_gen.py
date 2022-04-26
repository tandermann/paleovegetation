import numpy as np
import copy
import pandas as pd
import os,sys

# load BNN modules
from np_bnn import BNN_env, BNN_lib
from feature_gen.feature_gen import FeatureGenerator
from feature_gen.utils import UpdateNormal, select_train_and_test_set, rescale_abiotic_features

    
# set seed and working directory
try:
    datadir = str(sys.argv[1])
    outdir = str(sys.argv[2])
    seed = int(sys.argv[3])
except:
    datadir = '/Users/tobias/GitHub/feature_gen_paleoveg/data'
    outdir = '/Users/tobias/GitHub/feature_gen_paleoveg/test'
    seed = 1234

np.random.seed(seed)
try:
    os.makedirs(outdir)
except:
    pass



# settings____________________________________________________________________
reload_weights = False
continue_logfile = False
select_new_data = True
update_freq_w1 = 0.25 # time and space weights
update_freq_w2 = 0.25 # taxon weights
window_size_w1 = 0.0075
window_size_w2 = 0.1

# set up model architecture and priors
n_nodes_list = [5, 5] # 2 hidden layers with 5 nodes each
prior = 1 # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
p_scale = 1 # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
use_class_weight = 0 # set to 1 to use class weights for unbalanced classes
sample_from_prior = 0 # set to 1 to run an MCMC sampling the parameters from the prior only
#_____________________________________________________________________________




# define file paths of input data______________________________________________
spatial_dists = os.path.join(datadir,"spatial_distances_NN_input.pkl")
temporal_dists = os.path.join(datadir,"temporal_distances_NN_input.pkl")
additional_features_file = os.path.join(datadir,"abiotic_features.npy")
labels = os.path.join(datadir,"veg_labels.txt")

# load additional features
additional_features = np.load(additional_features_file)
scaled_additional_features = rescale_abiotic_features(additional_features)

# select train and test indices
if select_new_data:
    train_indices,test_indices = select_train_and_test_set(additional_features[:,2],seed,outdir,testsize=0.1,equal_paleo_and_current_labels=True,current_train_fraction=0.0)
else: # option to load test and train indices to use fixed data
    print('Using default train and test indices stored in %s'%datadir)
    train_indices = np.loadtxt(os.path.join(datadir,'train_instance_indices.txt')).astype(int)
    test_indices = np.loadtxt(os.path.join(datadir,'test_instance_indices.txt')).astype(int)
#______________________________________________________________________________




# initiate or load weights and output files____________________________________
if reload_weights:
    # define paths to pickle files with weights
    pkl_files = [os.path.join(outdir, 'BNN_p%i_h0_l%i_%i_s%i_binf_%i_feature_weights.pkl'%(prior,n_nodes_list[0],n_nodes_list[1],p_scale,seed)),
                 os.path.join(outdir, 'BNN_p%i_h0_l%i_%i_s%i_binf_%i.pkl'%(prior,n_nodes_list[0],n_nodes_list[1],p_scale,seed))]    
    # get data and prior prob of weights, load form pickle file
    feature_weights_file = pkl_files[0]
    feature_gen = FeatureGenerator(spatial_dists,temporal_dists,scaled_additional_features,labels,instance_index=train_indices,
                                   testsize=0.,pickle_file=feature_weights_file,seed=seed,transform_labels=False,priorscale=1,
                                   numba=False)
    dat,prior_prob_feature_weights = feature_gen.get_data()
    # create bnn object from pickle file    
    bnn_weights_file = pkl_files[1]
    bnn = BNN_env.npBNN(dat, n_nodes = n_nodes_list, use_class_weights=use_class_weight,
                     pickle_file=bnn_weights_file, use_bias_node = 0, prior_f = prior, p_scale = p_scale, seed=seed, init_std=1)

else:
    # get data and prior prob of weights
    feature_gen = FeatureGenerator(spatial_dists,temporal_dists,scaled_additional_features,labels,instance_index=train_indices,
                                   testsize=0.,seed=seed,transform_labels=False,priorscale=1,numba=False)
    dat,prior_prob_feature_weights = feature_gen.get_data()
    # set up the BNN model
    bnn = BNN_env.npBNN(dat, n_nodes = n_nodes_list, use_class_weights=use_class_weight,
                     use_bias_node = 0, prior_f = prior, p_scale = p_scale, seed=seed, init_std=1)

# set up the MCMC environment
mcmc = BNN_env.MCMC(bnn,update_f=[0.05, 0.1, 0.2], update_ws=[0.075, 0.075, 0.075],
                 temperature = 1, n_iteration=50000, sampling_f=10, print_f=10, n_post_samples=100,
                 sample_from_prior=sample_from_prior,init_additional_prob=prior_prob_feature_weights)

if continue_logfile:
    # reload previous logfile
    logger = BNN_env.postLogger(bnn, filename=os.path.join(outdir,"BNN"),add_prms=['feat_w1_space','feat_w1_time','mean_feat_w2','std_feat_w2'],
                                    continue_logfile=continue_logfile)
    mcmc._current_iteration = pd.read_csv(os.path.join(outdir,logger._w_file.replace('.pkl','.log')),sep='\t').iloc[-1,0]

else:
    # initialize new output files
    logger = BNN_env.postLogger(bnn, filename=os.path.join(outdir,"BNN"),add_prms=['feat_w1_space','feat_w1_time','mean_feat_w2','std_feat_w2'])
#______________________________________________________________________________




# start mcmc loop______________________________________________________________
feature_weights_log_list = []
# run MCMC
while True:
    # copy feature_gen and bnn objects
    feature_gen_prime = copy.deepcopy(feature_gen)
    bnn_prime = copy.deepcopy(bnn)
    mcmc_update_n = mcmc._update_n
    
    # 1. update weights
    rr = np.random.random()
    if rr < update_freq_w1 + update_freq_w2:
        # no updates in fully connected NN
        mcmc.reset_update_n([0,0,0])
        if rr < update_freq_w1:
            # if mcmc._current_iteration % update_freq_w1 == 0:
            w1_prime,index_w1 = UpdateNormal(feature_gen_prime._w1, d=window_size_w1, n=1, Mb=5, mb= -5)
            w2_prime = feature_gen_prime._w2
        else:
            w1_prime = feature_gen_prime._w1
            w2_prime,index_w2 = UpdateNormal(feature_gen_prime._w2, d=window_size_w2, n=1, Mb=5, mb= -5)
        
        feature_gen_prime.update_weigths(w1_prime,w2_prime)
    
    # 2.+ 3. update features and get new prior prob
    dat_prime,prior_prime = feature_gen_prime.get_data()
    
    # 4. update data in bnn
    bnn_prime.update_data(dat_prime)   
    
    # 5. do MCMC step
    mcmc.mh_step(bnn_prime, prior_prob_feature_weights)
    mcmc.reset_update_n(mcmc_update_n)
    if mcmc._last_accepted == 1:
        #bnn = bnn_prime
        bnn.reset_weights(bnn_prime._w_layers)
        feature_gen = feature_gen_prime      
        #bnn = copy.deepcopy(bnn_prime)
        #feature_gen = copy.deepcopy(feature_gen_prime)
    else:
        pass

    # print some stats (iteration number, likelihood, training accuracy, test accuracy
    if mcmc._current_iteration % mcmc._print_f == 0 or mcmc._current_iteration == 1:
        print(mcmc._current_iteration, np.round([mcmc._logLik, mcmc._accuracy, mcmc._test_accuracy,mcmc._logPrior],3))
        #print(mcmc._y)
        #print(feature_gen._w1,'\n',features_gen._w2)
        #print(feature_gen._features)
    # save to file
    if mcmc._current_iteration % mcmc._sampling_f == 0:
        "save additional weights"
        logger.log_sample(bnn,mcmc,add_prms=list(np.array([feature_gen._w1[0],feature_gen._w1[1],np.mean(feature_gen._w2),
                          np.std(feature_gen._w2)])))
        logger.log_weights(bnn,mcmc)    
        # log feature weights
        feature_weights_log_list.append([feature_gen._w1,feature_gen._w2])
    
    if len(mcmc._list_post_weights) == 0 and mcmc._current_iteration >= mcmc._sampling_f and len(feature_weights_log_list) > 0:
        BNN_lib.SaveObject(feature_weights_log_list,os.path.join(outdir,logger._w_file.replace('.pkl','_feature_weights.pkl')))
        feature_weights_log_list = []
        
    # stop MCMC after running desired number of iterations
    if mcmc._current_iteration == mcmc._n_iterations:
        break
#______________________________________________________________________________




"""
# testing speed of numba implementation
feature_gen = FeatureGenerator(spatial_dists,temporal_dists,scaled_additional_features,labels,testsize=0.1,seed=seed,transform_labels=False,priorscale=1,numba=False)
times=  []
for i in range(100):
    dat,prior_prob_feature_weights = feature_gen.get_data()
    times.append(feature_gen._time)

feature_gen = FeatureGenerator(spatial_dists,temporal_dists,scaled_additional_features,labels,testsize=0.1,seed=seed,transform_labels=False,priorscale=1,numba=True)
times_numba=  []
for i in range(100):
    dat,prior_prob_feature_weights = feature_gen.get_data()
    times_numba.append(feature_gen._time)


print('No numba:',np.mean(times))
print('Numba:',np.mean(times_numba))



import matplotlib.pyplot as plt
plt.hist(times,100)
plt.hist(times_numba,100)

"""



"""
from numba import jit
import time

# numba function to parallelize computation    
@jit(nopython=True, parallel=True)
def test_func_numba(n=1000000):
    value = 0
    for i in np.arange(n):
        value += np.sin(i)
    return value


def test_func(n=1000000):
    value = 0
    for i in np.arange(n):
        value += np.sin(i)
    return value


start = time.time()
test_func_numba()
end = time.time()
elapsed_time = end-start
print(elapsed_time)

start = time.time()
test_func()
end = time.time()
elapsed_time = end-start
print(elapsed_time)
"""


"""

# make predictions based on MCMC's estimated weights (test data)
post_pr = BNN_lib.predictBNN(dat['test_data'], pickle_file=logger._w_file, test_labels=dat['test_labels'])

# make predictions based on MCMC's estimated weights (train data)
post_pr = BNN_lib.predictBNN(dat['data'], pickle_file=logger._w_file, test_labels=dat['labels'])




# ADDITIONAL OPTIONS

# to restart a previous run you can provide the pickle file with the posterior parameters
# when initializing the BNN environment
pickle_file = logger._w_file


bnn = BNN_env.npBNN(dat, n_nodes = n_nodes_list,
                 use_bias_node = 0, prior_f = prior, p_scale = p_scale,
                 pickle_file=pickle_file, seed=seed, init_std=1)

mcmc = BNN_env.MCMC(bnn,update_f=[0.20, 0.20, 0.20, 0.20], update_ws=[0.075, 0.075, 0.075],
                 temperature = 1, n_iteration=50000, sampling_f=100, print_f=1, n_post_samples=100,
                 sample_from_prior=sample_from_prior,init_additional_prob=prior_prob_feature_weights)


"""

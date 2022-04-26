import numpy as np
import copy
import os,sys
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from concurrent.futures import ProcessPoolExecutor

# load BNN modules
import np_bnn as bn
from feature_gen.feature_gen import FeatureGenerator
from feature_gen.utils import UpdateNormal, select_train_and_test_set, rescale_abiotic_features

# set seed and working directory
try:
    datadir = str(sys.argv[1])
    outdir = str(sys.argv[2])
    seed = int(sys.argv[3])
    prior = int(sys.argv[4])
    p_scale = int(sys.argv[5])
    n_current = int(sys.argv[6])
    n_paleo = int(sys.argv[7])

except:
    print('Can\'t read settings input:',sys.argv)
    print('Using default settings')
    datadir = 'data'
    outdir = 'test'
    seed = 1234
    prior = 0 # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
    p_scale = 5 # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
    n_current = 0
    n_paleo = 281



trouble_shoot_mode = False
if trouble_shoot_mode:
    datadir = 'data/trouble_shooting_data/'
    outdir = 'data/trouble_shooting_data/results'


np.random.seed(seed)
try:
    os.makedirs(outdir)
except:
    pass




# settings____________________________________________________________________
reload_weights = False
continue_logfile = False
select_new_data = False
update_freq_w1 = 0.25 # time and space weights
update_freq_w2 = 0.1 # taxon weights
window_size_w1 = 0.0075
window_size_w2 = 0.1
lik_temp = 1.0
init_std = 1.0

# set up model architecture and priors
bias_node_setting = 3
n_nodes_list = [32, 8] # 2 hidden layers with n nodes each
use_class_weight = 0 # set to 1 to use class weights for unbalanced classes
sample_from_prior = 0 # set to 1 to run an MCMC sampling the parameters from the prior only
#_____________________________________________________________________________


# define the number of chains to run
n_chains = 10
# get a separate seed for each chain
rseeds = np.random.choice(range(1000,9999), n_chains, replace=False)


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
elif trouble_shoot_mode:
    train_instances_path = os.path.join(datadir,'train_instance_indices.txt')
    train_indices = np.loadtxt(train_instances_path).astype(int)
    print('Running small troubleshoot dataset.')
else: # option to load test and train indices to use fixed data
    train_instances_path = os.path.join(datadir,'train_test_sets/train_instance_indices_ncurrent_%i_npaleo_%i.txt'%(n_current,n_paleo))
    train_indices = np.loadtxt(train_instances_path).astype(int)
    print('Using train indices stored at %s'%train_instances_path)
    #test_indices = np.loadtxt(os.path.join(datadir,'train_test_sets/test_instance_indices.txt')).astype(int)
#______________________________________________________________________________

# define the activation functions

activation_function_list = [bn.ActFun(fun = 'swish') for i in range(n_chains)] # To use default ReLU: BNN_lib.genReLU()
#alphas = np.zeros(len(n_nodes_list))
#activation_function = BNN_lib.genReLU(prm=alphas, trainable=True) # To use default ReLU: BNN_lib.genReLU()

# get data and prior prob of weights
feature_gen_list = [FeatureGenerator(spatial_dists,
                                     temporal_dists,
                                     scaled_additional_features,
                                     labels,
                                     instance_index=train_indices,
                                     testsize=0.,
                                     seed=rseeds[i],
                                     transform_labels=False,
                                     priorscale=1,
                                     prior_f = prior,
                                     multiple_weights_per_species=True)
           for i in range(n_chains)]

dat_list,prior_prob_feature_weights_list = zip(*[feature_gen_list[i].get_data() for i in range(n_chains)])

# set up the BNN model
bnnList = [bn.BNN_env.npBNN( dat_list[i],
                             use_bias_node=bias_node_setting,
                             n_nodes = n_nodes_list,
                             use_class_weights=use_class_weight,
                             prior_f = prior,
                             p_scale = p_scale,
                             seed=rseeds[i],
                             init_std=init_std,
                             actFun=activation_function_list[i])
           for i in range(n_chains)]


# set temperatures for MCMCs
if n_chains == 1:
    temperatures = [1]
else:
    temperatures = np.linspace(0.8, 1, n_chains)

# set up the MCMC environment
mcmcList = [bn.BNN_env.MCMC( bnnList[i],
                             update_f=[0.05, 0.1, 0.2],
                             update_ws=[0.075, 0.075, 0.075],
                             temperature=temperatures[i],
                             likelihood_tempering=lik_temp,
                             n_iteration=3000000,
                             sampling_f=10,
                             print_f=10,
                             n_post_samples=1000,
                             adapt_f=0.3,
                             mcmc_id=i,
                             sample_from_prior=sample_from_prior,
                             init_additional_prob=prior_prob_feature_weights_list[i],
                             randomize_seed=True)
            for i in range(n_chains)]



singleChainArgs = [[bnnList[i],mcmcList[i],feature_gen_list[i]] for i in range(n_chains)]
n_iterations = 10
# initialize output files
logger = bn.BNN_env.postLogger(bnnList[0], os.path.join(outdir,"BNNMC3"),add_prms=['feat_w1_space','feat_w1_time','mean_feat_w2','std_feat_w2'], log_all_weights=0)
#______________________________________________________________________________



def run_single_mcmc(arg_list):
    [bnn_obj, mcmc_obj, feature_obj] = arg_list
    rs = RandomState(MT19937(SeedSequence(mcmc_obj._current_iteration+mcmc_obj._mcmc_id)))
    for i in range(n_iterations-1):
        # define temporary copies of feature_gen and bnn objects for which we try new weights
        feature_gen_prime = copy.deepcopy(feature_obj)
        bnn_prime = copy.deepcopy(bnn_obj)
        mcmc_update_n = mcmc_obj._update_n
        
        # 1. update weights
        rr = rs.random()
        if rr < update_freq_w1 + update_freq_w2:
            # no updates in fully connected NN
            mcmc_obj.reset_update_n([0,0,0,0])
            if rr < update_freq_w1:
                # if mcmc._current_iteration % update_freq_w1 == 0:
                    # note that this is the UpdateNormal function that I modified specifically for the feature weights
                w1_prime,index_w1 = UpdateNormal(feature_gen_prime._w1, d=window_size_w1, n=1, Mb=5, mb= -5, rs=rs)
                w2_prime = feature_gen_prime._w2
            else:
                w1_prime = feature_gen_prime._w1
                w2_prime,index_w2 = UpdateNormal(feature_gen_prime._w2, d=window_size_w2, n=1, Mb=5, mb= -5, rs=rs)
            
            feature_gen_prime.update_weigths(w1_prime,w2_prime)
        
        # 2.+ 3. update features and get new prior prob
        dat_prime,prior_prime = feature_gen_prime.get_data()
        
        # 4. update data in bnn
        bnn_prime.update_data(dat_prime)

        # 5. do MCMC step
        mcmc_obj.mh_step(bnn_prime, prior_prime)
        #bnn_obj_new, mcmc_obj_new = mcmc_obj.mh_step(bnn_obj, return_bnn=True)
        mcmc_obj.reset_update_n(mcmc_update_n)

        if mcmc_obj._last_accepted == 1: #update objects with new weights if accepted
            #bnn = bnn_prime
            bnn_obj.reset_weights(bnn_prime._w_layers)
            bnn_obj._act_fun.reset_prm(bnn_prime._act_fun._prm)
            bnn_obj._act_fun.reset_accepted_prm()
            
            
            feature_obj = feature_gen_prime
        else:
            pass
    return [bnn_obj, mcmc_obj, feature_obj]


feature_weights_log_list = []
for mc3_it in range(int(mcmcList[0]._n_iterations/n_iterations)):
    with ProcessPoolExecutor(max_workers=n_chains) as pool:
        singleChainArgs = list(pool.map(run_single_mcmc, singleChainArgs))
        
    # singleChainArgs = [i for i in tmp]
    if n_chains > 1:
        n1 = np.random.choice(range(n_chains),2,replace=False)
        [j, k] = n1
        temp_j = singleChainArgs[j][1]._temperature + 0
        temp_k = singleChainArgs[k][1]._temperature + 0
        r = (singleChainArgs[k][1]._logPost - singleChainArgs[j][1]._logPost) * temp_j + \
            (singleChainArgs[j][1]._logPost - singleChainArgs[k][1]._logPost) * temp_k
    
        # print(mc3_it, r, singleChainArgs[j][1]._logPost, singleChainArgs[k][1]._logPost, temp_j, temp_k)
        if mc3_it % 100 == 0:
            print(mc3_it, singleChainArgs[0][1]._logPost, singleChainArgs[1][1]._logPost, singleChainArgs[0][0]._w_layers[0][0][0:5])
        if r >= np.log(np.random.random()):
            singleChainArgs[j][1].reset_temperature(temp_k)
            singleChainArgs[k][1].reset_temperature(temp_j)
            print(mc3_it,"SWAPPED", singleChainArgs[j][1]._logPost, singleChainArgs[k][1]._logPost, temp_j, temp_k)

    for i in range(n_chains):
        if singleChainArgs[i][1]._temperature == 1 and singleChainArgs[i][1]._current_iteration % singleChainArgs[i][1]._sampling_f == 0:
            logger.log_sample(singleChainArgs[i][0],singleChainArgs[i][1],
                              add_prms=list(np.array([singleChainArgs[i][2]._w1[0],
                                                  singleChainArgs[i][2]._w1[1],
                                                  np.mean(singleChainArgs[i][2]._w2),
                                                  np.std(singleChainArgs[i][2]._w2)])))
            logger.log_weights(singleChainArgs[i][0],singleChainArgs[i][1],add_prms=[singleChainArgs[i][2]._w1,singleChainArgs[i][2]._w2],add_obj=singleChainArgs[i][2])
            

#pickle_file = '/Users/xhofmt/GitHub/feature_gen_bnn/data/trouble_shooting_data/results/BNNMC3_p0_h0_l32_8_s5_b5_5083.pkl'
#bnn_obj,mcmc_obj,logger_obj = bn.load_obj(pickle_file)
#mcmc_obj.__dict__.keys()
#logger_obj.__dict__.keys()

# import pickle
# feature_weights_pickle = '/Users/tobias/GitHub/feature_gen_paleoveg/results/test/BNNMC3_p1_h0_l5_5_s1_binf_7293_feature_weights.pkl'
# bnn_weight_pickle = '/Users/tobias/GitHub/feature_gen_paleoveg/results/test/BNNMC3_p1_h0_l5_5_s1_binf_7293.pkl'
# feature_weights = pickle.load(open(feature_weights_pickle,'rb'))
# bnn_weights = pickle.load(open(bnn_weight_pickle,'rb'))




"""


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
    
    
    if continue_logfile:
    # reload previous logfile
    logger = BNN_env.postLogger(bnn, filename=os.path.join(outdir,"BNN"),add_prms=['feat_w1_space','feat_w1_time','mean_feat_w2','std_feat_w2'],
                                    continue_logfile=continue_logfile)
    mcmc._current_iteration = pd.read_csv(os.path.join(outdir,logger._w_file.replace('.pkl','.log')),sep='\t').iloc[-1,0]

else:
    




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

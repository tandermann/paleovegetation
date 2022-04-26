#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:03:46 2021

@author: Tobias Andermann (tobiasandermann88@gmail.com)
"""

import numpy as np
import np_bnn as bn
import pandas as pd
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from concurrent.futures import ProcessPoolExecutor
import sys,os
import copy
from feature_gen.utils import UpdateNormal


pickle_file = str(sys.argv[1])
n_chains = int(sys.argv[2])
try:
    run_n_mcmc_it = int(sys.argv[3])
except:
    run_n_mcmc_it = 1000000

#pickle_file= '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/runs_2021/current_281_paleo_281/BNNMC3_p1_h0_l32_8_s1_binf_5083.pkl'
#n_chains=1
#run_n_mcmc_it=100000

def run_single_mcmc(arg_list):
    [bnn_obj, mcmc_obj, feature_obj] = arg_list
    rs = RandomState(MT19937(SeedSequence(mcmc_obj._current_iteration + mcmc_obj._mcmc_id)))
    for i in range(n_iterations):
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
    return [bnn_obj, mcmc_obj, feature_obj]


update_freq_w1 = 0.25 # time and space weights
update_freq_w2 = 0.1 # taxon weights
window_size_w1 = 0.0075
window_size_w2 = 0.1

# set temperatures for MCMCs

bnn_obj,mcmc_obj,logger_obj,feature_obj = bn.load_obj(pickle_file)
out_file_name = os.path.join(os.path.dirname(pickle_file),'continued')
logger_obj = bn.postLogger(bnn_obj,
                           filename=out_file_name,
                           add_prms=['feat_w1_space','feat_w1_time','mean_feat_w2','std_feat_w2'],
                           log_all_weights=0)

if n_chains == 1:
    n_iterations = 1
    mcmc_obj._sampling_f = 100

    mcmc_obj._adapt_f = 0.
    mcmc_obj._adapt_fM = 1
    mcmc_obj._mcmc_id = 1
    mcmc_obj._temperature = 1.0


    n_params = np.sum(np.array([np.size(i) for i in bnn_obj._w_layers]))
    if bnn_obj._act_fun._trainable:
        n_params += bnn_obj._n_layers
    bnn_obj._n_params = n_params

    args = [bnn_obj,mcmc_obj,feature_obj]

    mcmc_obj._n_iterations = run_n_mcmc_it
    mcmc_obj._acceptance_rate = 0.34
    mcmc_obj._last_accepted_mem = list(np.random.choice([0,1],100,p=[1-mcmc_obj._acceptance_rate,mcmc_obj._acceptance_rate]))
    mcmc_obj._current_iteration = 0

    print('Running MCMC for %i generations'%(run_n_mcmc_it))
    for i in np.arange(run_n_mcmc_it):
        bnn_obj,mcmc_obj,feature_obj = run_single_mcmc(args)
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
        args = [bnn_obj, mcmc_obj, feature_obj]

#import pandas as pd
#import matplotlib.pyplot as plt
#dat = pd.read_csv('/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/runs_2021/current_281_paleo_281/continued__p1_h0_l32_8_s1_binf_5715.log',sep='\t')
#plt.plot(dat.mean_feat_w2)

else:
    n_iterations = 10
    temperatures = np.linspace(0.8, 1, n_chains)
    feature_gen_list = [feature_obj for i in np.arange(n_chains)]
    bnnList = [bnn_obj for i in np.arange(n_chains)]
    mcmcList_tmp = [mcmc_obj for i in np.arange(n_chains)]
    mcmcList = []
    for i,mcmc_o in enumerate(mcmcList_tmp):
        mcmc_o = copy.deepcopy(mcmc_o)
        mcmc_o._temperature = temperatures[i]
        mcmc_o._mcmc_id = i
        mcmc_o._adapt_f = 0.3
        mcmc_o._adapt_fM = 0.6
        mcmcList.append(mcmc_o)
    logger = logger_obj

    singleChainArgs = [[bnnList[i],mcmcList[i],feature_gen_list[i]] for i in range(n_chains)]
    arg_list = singleChainArgs[0]

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

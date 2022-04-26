import os,glob,sys
import numpy as np
import np_bnn as bn
import pandas as pd
from feature_gen.feature_gen import PredictFeatures
from feature_gen.utils import rescale_abiotic_features
import multiprocessing.pool
#sys.path.append('src')
#from runner_test_accuracy import get_confidence_threshold


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


def get_accs(pred_with_posterior_weights_sample,
             post_prob,
             true_labels,
             summary_type,
             instance_ids_per_stage,
             all_test_indices,
             paleo_test_indices,
             current_test_indices,
             current_non_train_instances,
             post_threshold=0,
             sample_from_posterior = 0):
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


def get_confidence_threshold(predicted_labels,
                             true_labels,
                             summary_type,
                             instance_ids_per_stage,
                             all_test_indices,
                             paleo_test_indices,
                             current_test_indices,
                             current_non_train_instances,
                             target_acc=None):
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


def get_accuracy_threshold(probs,
                           labels,
                           summary_type,
                           instance_ids_per_stage,
                           all_test_indices,
                           paleo_test_indices,
                           current_test_indices,
                           current_non_train_instances,
                           threshold=0.75):
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


def predict_labels_from_weights(weight_pickle,
                                test_indices):
    post_pred_out_file = os.path.join(outdir,os.path.basename(weight_pickle).replace('.pkl','_test_pred.npy'))
    if continued:
        cv_i = int(os.path.basename(weight_pickle).split('_')[1].replace('cv',''))
    else:
        cv_i = int(os.path.basename(weight_pickle).split('_')[0].replace('cv',''))
    print('Processing cv fold ',cv_i)
    # load weights from pkl file
    if '_biotic_0' in weight_pickle:
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
                                            instance_index=test_indices,
                                            multiple_weights_per_species=multiple_weights_per_species,
                                            sum_faunal_floral_features=sum_faunal_floral_features,
                                            max_pooling=max_pooling,
                                            actfun=feature_gen_actfun
                                            )

    posterior_samples = logger_obj._post_weight_samples
    post_pred = []
    # predicted_labels_all = []

    for i, weights_dict in enumerate(posterior_samples[burnin:]):
        #print('Posterior sample %i of %i'%(i,len(posterior_samples[burnin:])))
        if biotic_features:
            # read feature weights and apply to raw featrue values
            feature_weights_rep = weights_dict['additional_prm']
            pred_features_obj.update_weigths(feature_weights_rep[0], feature_weights_rep[1])
            # extract features for this rep
            predict_features = pred_features_obj.get_features_unseen_data()
            if '_abiotic_0' in weight_pickle:
                predict_features = predict_features[:, :-8]
        else:
            predict_features = scaled_additional_features[test_indices]

        # apply features and bnn weights to predict labels
        actFun = bnn_obj._act_fun
        output_act_fun = bnn_obj._output_act_fun
        # bn.RunPredict(predict_features, weights_dict['weights'], actFun, output_act_fun)
        post_softmax_probs, post_prob_predictions = bn.get_posterior_cat_prob(predict_features,
                                                                              [weights_dict],
                                                                              post_summary_mode=post_summary_mode,
                                                                              actFun=actFun,
                                                                              output_act_fun=output_act_fun)
        # predicted_labels = np.argmax(post_prob_predictions, axis=1)
        post_pred.append(post_softmax_probs[0])
        # predicted_labels_all.append(predicted_labels)

        if i % 10 == 0:
            print("Predicting labels for posterior sample %i of %i"%(i,len(posterior_samples[burnin:])))

    post_pred = np.array(post_pred)
    # predicted_labels_all = np.array(predicted_labels_all)
    np.save(post_pred_out_file, post_pred)
    # np.save(predicted_labels_out_file,predicted_labels_all)


# manual settings
#indir = str(sys.argv[1])
indir = 'results/results/n_current_331_n_paleo_0_nnodes_32_8_biotic_1_abiotic_1_sumpool_0_maxpool_0'
burnin = 0
post_summary_mode = 1
continued=True
if 'old_results_relu' in indir:
    feature_gen_actfun=1
else:
    feature_gen_actfun=2
print('Using feature-gen actfun setting %i'%feature_gen_actfun,flush=True)

#______________________


# automated settings
outdir = os.path.join(indir,'testset_predictions')
if not os.path.exists(outdir):
    os.makedirs(outdir)
n_current = int(indir.split('n_current_')[-1].split('_')[0])
n_paleo = int(indir.split('n_paleo_')[-1].split('_')[0])
multiple_weights_per_species = True
sum_faunal_floral_features = False
if 'sumpool_1' in indir:
    sum_faunal_floral_features = True
max_pooling = False
if 'maxpool_1' in indir:
    max_pooling = True
#______________________


# load data for prediction
data_folder = 'data'
spatial_dists = os.path.join(data_folder, "spatial_distances_NN_input.pkl")
temporal_dists = os.path.join(data_folder, "temporal_distances_NN_input.pkl")
additional_features = np.load(os.path.join(data_folder, "abiotic_features.npy"))
scaled_additional_features = rescale_abiotic_features(additional_features)
taxon_names_file = os.path.join(data_folder, 'selected_taxa.txt')
taxon_names = np.loadtxt(taxon_names_file, dtype=str)
feature_group_ids = np.array([1 if i.endswith('aceae') or i.endswith('aceaee') else 0 for i in taxon_names])
# load test labels
veg_labels_file = os.path.join(data_folder, 'veg_labels.txt')
all_veg_labels = np.loadtxt(veg_labels_file).astype(int)


# load pkl files of cv-folds and predict test labels
if continued:
    cv_pkl_files = glob.glob(os.path.join(indir,'continued_cv*.pkl'))
else:
    cv_pkl_files = glob.glob(os.path.join(indir,'cv*.pkl'))

# cpus=3
# pool = multiprocessing.Pool(cpus)
# args = cv_pkl_files
# out_info = np.array(pool.map(predict_labels_from_weights, args))
# pool.close()
for i in cv_pkl_files:
    if continued:
        cv_i = int(os.path.basename(i).split('_')[1].replace('cv', ''))
    else:
        cv_i = int(os.path.basename(i).split('_')[0].replace('cv', ''))
    test_indices_file = os.path.join(data_folder,
                                     'instance_selection_for_training/cv_instance_ids/n_paleo_%i_n_current_%i_cv_%i_of_5_test.txt' %(n_paleo, n_current, cv_i))
    test_instances = np.loadtxt(test_indices_file).astype(int)
    time_array = additional_features[:,2]
    time_array_test_instances = time_array[test_instances]
    current_test_ids = np.where(time_array_test_instances==0)[0]
    paleo_test_ids = np.where(time_array_test_instances > 0)[0]
    if 'n_current_0_' in test_indices_file: # some models were trained without current indices. for these, select get current test set indices from alternative file
        other_test_indices_file =  test_indices_file.replace('n_current_0','n_current_331')
        other_test_instances = np.loadtxt(other_test_indices_file).astype(int)
        other_time_array_test_instances = time_array[other_test_instances]
        other_current_test_ids = np.where(other_time_array_test_instances==0)[0]
        additional_current_test_instances = other_test_instances[other_current_test_ids]
        test_instances = np.concatenate([test_instances,additional_current_test_instances])
        current_test_ids = paleo_test_ids[-1]+1+np.arange(len(additional_current_test_instances))
    elif 'n_paleo_0_' in test_indices_file: # some models were trained without paleo indices. for these, select get paleo test set indices from alternative file
        other_test_indices_file =  test_indices_file.replace('n_paleo_0','n_paleo_331')
        other_test_instances = np.loadtxt(other_test_indices_file).astype(int)
        other_time_array_test_instances = time_array[other_test_instances]
        other_paleo_test_ids = np.where(other_time_array_test_instances>0)[0]
        additional_paleo_test_instances = other_test_instances[other_paleo_test_ids]
        test_instances = np.concatenate([test_instances,additional_paleo_test_instances])
        paleo_test_ids = current_test_ids[-1]+1+np.arange(len(additional_paleo_test_instances))

    test_labels = all_veg_labels[test_instances]
    if continued:
        instance_text_file = os.path.join(outdir,'continued_cv%i_test_instances_paleo_and_current_and_labels.txt'%(cv_i))
    else:
        instance_text_file = os.path.join(outdir,'cv%i_test_instances_paleo_and_current_and_labels.txt'%(cv_i))
    with open(instance_text_file, 'w') as f:
        f.write(str(list(paleo_test_ids))+'\n')
        f.write(str(list(current_test_ids))+'\n')
        f.write(str(list(test_labels)) + '\n')
    predict_labels_from_weights(i,test_instances)


all_accs = []
cv_list = []
all_posterior_probs_paleo = []
all_posterior_probs_current = []
true_paleo_labels_sorted = []
true_current_labels_sorted = []
for weight_pickle in cv_pkl_files:
    post_pred_out_file = os.path.join(outdir,os.path.basename(weight_pickle).replace('.pkl','_test_pred.npy'))
    post_pred = np.load(post_pred_out_file)
    if continued:
        cv_i = int(os.path.basename(weight_pickle).split('_')[1].replace('cv',''))
        test_instance_text_file =  os.path.join(outdir,'continued_cv%i_test_instances_paleo_and_current_and_labels.txt'%(cv_i))
    else:
        cv_i = int(os.path.basename(weight_pickle).split('_')[0].replace('cv',''))
        test_instance_text_file =  os.path.join(outdir,'cv%i_test_instances_paleo_and_current_and_labels.txt'%(cv_i))

    # get test predictions for this cv fold
    posterior_probs = get_posterior_prob_from_bnn_output(post_pred, post_summary_mode)
    estimated_labels = np.argmax(posterior_probs,axis=1)
    # calculate accuracy, separately for current and paleo test set

    with open(test_instance_text_file) as f:
        lines = f.readlines()
    try:
        paleo_test_indices = np.array(lines[0].rstrip().replace('[','').replace(']','').split(', ')).astype(int)
    except:
        paleo_test_indices = []
    try:
        current_test_indices = np.array(lines[1].rstrip().replace('[','').replace(']','').split(', ')).astype(int)
    except:
        current_test_indices = []
    true_test_labels = np.array(lines[2].rstrip().replace('[','').replace(']','').split(', ')).astype(int)
    # calculate paleo-accuracy
    all_posterior_probs_paleo.append(posterior_probs[paleo_test_indices,:])
    paleo_test_labels = true_test_labels[paleo_test_indices]
    true_paleo_labels_sorted.append(paleo_test_labels)
    estimated_paleo_test_labels = estimated_labels[paleo_test_indices]
    try:
        paleo_acc = sum(estimated_paleo_test_labels == paleo_test_labels) / len(paleo_test_labels)
    except:
        paleo_acc = 0.0
    # calculate current-accuracy
    all_posterior_probs_current.append(posterior_probs[current_test_indices,:])
    current_test_labels = true_test_labels[current_test_indices]
    true_current_labels_sorted.append(current_test_labels)
    estimated_current_test_labels = estimated_labels[current_test_indices]
    try:
        current_acc = sum(estimated_current_test_labels == current_test_labels) / len(current_test_labels)
    except:
        current_acc = 0.0
    # calculate_total_acc
    total_acc = (10*paleo_acc + current_acc)/11
    all_accs.append([total_acc,paleo_acc,current_acc])
    cv_list.append('cv_rep_%i'%cv_i)
all_accs = np.array(all_accs)
acc_df = pd.DataFrame(all_accs,columns = ['test_acc','test_acc_paleo','test_acc_current'],index=cv_list)
acc_df = acc_df.sort_index()
acc_df = acc_df.append(pd.DataFrame(np.matrix(np.mean(all_accs,axis=0)),columns = ['test_acc','test_acc_paleo','test_acc_current'],index=['average_cv']))
if continued:
    acc_df.to_csv(os.path.join(outdir,'continued_cv_test_accs.txt'),sep='\t',float_format='%.3f')
else:
    acc_df.to_csv(os.path.join(outdir,'cv_test_accs.txt'),sep='\t',float_format='%.3f')

all_paleo_predicitons_pp = np.concatenate(all_posterior_probs_paleo)
true_paleo_labels_sorted = np.concatenate(true_paleo_labels_sorted)

acc_thres_tbl = get_confidence_threshold(all_paleo_predicitons_pp,
                                         true_paleo_labels_sorted,
                                         4,
                                         [],
                                         [],
                                         [],
                                         [],
                                         [])
if continued:
    acc_thres_tbl_filename = '_'.join(['continued']+os.path.basename(weight_pickle).replace('.pkl','_acc_thres_tbl_post_mode_%i_cv.txt' % (post_summary_mode)).split('_')[2:])
else:
    acc_thres_tbl_filename = '_'.join(os.path.basename(weight_pickle).replace('.pkl','_acc_thres_tbl_post_mode_%i_cv.txt' % (post_summary_mode)).split('_')[1:])
np.savetxt(os.path.join(outdir, acc_thres_tbl_filename), acc_thres_tbl,fmt='%.3f')



if False:
    predict_features.shape

    import matplotlib.pyplot as plt
    a = pd.DataFrame(predict_features)
    fig, ax = plt.subplots(figsize=(10,10))
    a.hist(figsize=[10,10], ax=ax)
    fig.savefig('plots/pred_data_hist.pdf',bbox_inches="tight")
    plt.show()
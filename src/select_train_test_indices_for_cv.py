import numpy as np
import os, glob, sys


def iter_test_indices(input_indices, n_splits=5, shuffle=True, seed=None):
    n_samples = input_indices.shape[0]
    indices = np.arange(n_samples)
    if seed:
        np.random.seed(seed)
    if shuffle:
        indices = np.random.choice(indices, len(indices), replace=False)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    index_output = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        index_output.append(input_indices[indices[start:stop]])
        current = stop
    return index_output

# set seed
seed = 1234
np.random.seed(seed)

# set how many instances for current and paleo
n_paleo = 331 #0,331
n_current = 0 #0,331,662,1655,3310,6620,11048
cv_splits = 5

# get indices for current and paleo instances
additional_features = np.load('data/abiotic_features.npy')
paleo_indices = np.where(additional_features[:,2]>0)[0]
current_indices = np.where(additional_features[:,2]==0)[0]
# randomly select these numbers of indices
paleo_indices_selected = np.random.choice(paleo_indices,n_paleo,replace=False)
current_indices_selected = np.random.choice(current_indices,n_current,replace=False)
selected_indices = np.concatenate([paleo_indices_selected,current_indices_selected])
np.random.shuffle(selected_indices)
np.savetxt('data/instance_selection_for_training/selected_instances_paleo_%i_current_%i.txt' %(n_paleo,n_current),selected_indices,fmt='%i')


# get n cv blocks for these indices, make sure to select same numbers of current and paleo
cv_index_blocks_paleo = iter_test_indices(paleo_indices_selected,n_splits = cv_splits,shuffle=True)
cv_index_blocks_current = iter_test_indices(current_indices_selected,n_splits = cv_splits,shuffle=True)
for it,_ in enumerate(cv_index_blocks_paleo):
    train_ids_paleo = np.concatenate(np.array([cv_index_blocks_paleo[i] for i in list(np.delete(np.arange(len(cv_index_blocks_paleo)), it))])).astype(int)
    train_ids_current = np.concatenate(np.array([cv_index_blocks_current[i] for i in list(np.delete(np.arange(len(cv_index_blocks_current)), it))])).astype(int)
    all_train_ids = np.concatenate([train_ids_paleo, train_ids_current])
    np.random.shuffle(all_train_ids)

    test_ids_paleo = cv_index_blocks_paleo[it]  # in case of cv, choose one of the k chunks as test set
    test_ids_current = cv_index_blocks_current[it]  # in case of cv, choose one of the k chunks as test set
    all_test_ids = np.concatenate([test_ids_paleo, test_ids_current])
    np.random.shuffle(all_test_ids)

    out_file_name_train = 'data/instance_selection_for_training/cv_instance_ids/n_paleo_%i_n_current_%i_cv_%i_of_5_train.txt' %(n_paleo,n_current,it+1)
    out_file_name_test = 'data/instance_selection_for_training/cv_instance_ids/n_paleo_%i_n_current_%i_cv_%i_of_5_test.txt' %(n_paleo,n_current,it+1)
    np.savetxt(out_file_name_train,all_train_ids,fmt='%i')
    np.savetxt(out_file_name_test,all_test_ids,fmt='%i')


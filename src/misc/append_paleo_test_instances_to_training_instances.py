import numpy as np
import os, glob, sys

input_folder = 'data/train_test_sets'#str(sys.argv[1]) #'/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/train_test_sets'
train_index_files = glob.glob(os.path.join(input_folder,'*.txt'))
test_index_file = [i  for i in train_index_files if os.path.basename(i).startswith('test')][0]
test_indices = np.loadtxt(test_index_file).astype(int)

abiotic_feature_file = 'data/abiotic_features.npy'
abiotic_features = np.load(abiotic_feature_file)

paleo_test_instances = test_indices[np.where(abiotic_features[test_indices,2]>0)]
index_files_containing_paleo = [i for i in train_index_files if os.path.basename(i).endswith('npaleo_281.txt')]
for paleo_file in index_files_containing_paleo:
    indices = np.loadtxt(paleo_file).astype(int)
    indices_new = np.concatenate([indices,paleo_test_instances])
    new_paleo_file = paleo_file.replace('npaleo_281.txt','npaleo_281+50.txt')
    np.savetxt(new_paleo_file,indices_new)



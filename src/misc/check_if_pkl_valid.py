import np_bnn as bn
import sys, shutil, os

pkl_file = str(sys.argv[1])
new_pkl_file = pkl_file.replace('testing','testing_backup')

print("Attempting to copy ",pkl_file, " to ", new_pkl_file)
n_attempts = 0
result = None
while result is None:
    shutil.copyfile(pkl_file, new_pkl_file)
    n_attempts += 1
    print('Trying for nth time: ', n_attempts)
    try:
        if 'no_biotic' in pkl_file:
            bnn_obj, mcmc_obj, logger_obj = bn.load_obj(new_pkl_file)
        else:
            bnn_obj, mcmc_obj, logger_obj, featgen_obj = bn.load_obj(new_pkl_file)
        result = 'Success.'
    except:
         pass



import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

basedir = 'cluster_content/feature_gen_mc3/new_runs_2021/testing'
summary_mode = 0 # 0 is test,paleo,current,all; 1 is mean trhough geo-stages; 2 is weighted mean between paleo and current
sample_from_posterior = 1
post_summary_mode = 1
acc_threshold = 0.9
acc_files = glob.glob(os.path.join(basedir,'*/time_slice_predictions/*_test_acc_%i_sample_from_post_%i.txt'%(summary_mode,sample_from_posterior)))
acc_files = [i for i in acc_files if os.path.basename(i).startswith('continued_')]
#acc_files = [i for i in acc_files if not 'n_current_0' in i]
#acc_files = [i for i in acc_files if not 'n_paleo_0' in i]
log_files = glob.glob(os.path.join(basedir,'*/*.log'))
log_files = [i for i in log_files if os.path.basename(i).startswith('continued_')]
#log_files = [i for i in log_files if not os.path.basename(i).startswith('final_model')]
#log_files = [i for i in log_files if not 'n_current_0' in i]
#log_files = [i for i in log_files if not 'n_paleo_0' in i]
acc_threshold_files = glob.glob(os.path.join(basedir,'*/time_slice_predictions/*_acc_thres_tbl_post_mode_%i_sum_mode_%i_sample_from_post_%i.txt'%(post_summary_mode,summary_mode,sample_from_posterior)))
#acc_threshold_files = [i for i in acc_threshold_files if not 'n_current_0' in i]
#acc_threshold_files = [i for i in acc_threshold_files if not 'n_paleo_0' in i]

# sort by folder name so manual naming later on is easier
subfolders = [i.split('/')[-3] for i in acc_files]
correct_order = np.argsort(subfolders)
acc_files=np.array(acc_files)[correct_order]
log_files = np.array(log_files)[correct_order]
acc_threshold_files = np.array(acc_threshold_files)[correct_order]

log_file_dfs = [pd.read_csv(i,sep='\t') for i in log_files]
liks = [i.likelihood.values for i in log_file_dfs]
accs = [i.accuracy.values for i in log_file_dfs]

acc_all_mean = []
acc_all_std = []
acc_paleo_mean = []
acc_paleo_std = []
acc_current_mean = []
acc_current_std = []
acc_all_current_mean = []
acc_all_current_std = []

all_accs = []
all_post_cutoffs = []
all_fraction_remaining = []
#scenario_names = []
for i,acc_file in enumerate(acc_files):
    #scenario = acc_file.split('/')[-3]
    #scenario_names.append(scenario)
    # read accuracy scores and interval
    acc_data = pd.read_csv(acc_file,sep=':',header=None).values[:,1]
    acc_thres_data = np.loadtxt(acc_threshold_files[i])
    try:
        cutoff_index = min(np.where(acc_thres_data[:,1]>acc_threshold)[0])
        post_cutoff, reached_acc, fraction_remaining_predictions = acc_thres_data[cutoff_index,:]
    except:
        print('Target accuracy could not be reaached. Setting posterior threshold to 1.00.')
        post_cutoff, reached_acc, fraction_remaining_predictions = [1.00, acc_threshold, 0.00]
    all_accs.append(acc_data)
    all_post_cutoffs.append(post_cutoff)
    all_fraction_remaining.append(fraction_remaining_predictions)
    #scenarios.append(scenario)
    acc_all_mean.append(acc_data[0])
    acc_all_std.append(acc_data[1])
    if summary_mode == 0:
        acc_paleo_mean.append(acc_data[2])
        acc_paleo_std.append(acc_data[3])
        acc_current_mean.append(acc_data[4])
        acc_current_std.append(acc_data[5])
        acc_all_current_mean.append(acc_data[6])
        acc_all_current_std.append(acc_data[7])

all_accs = np.array(all_accs).astype(float)
all_post_cutoffs = np.array(all_post_cutoffs).astype(float)
all_fraction_remaining = np.array(all_fraction_remaining).astype(float)

# define final order of scenarios for plot and table
order = [12,10,11,9,6,4,5,7,13,3,1,0,2] # this is the same order as the tbl in the ms
# name the scenarios in the basedir in alphabetical order of their original names
scenarios = ['All_32_8','Bio_32_8','Abio_32_8','All_32_8_SP','Bio_32_8_SP','All_8_SP','Bio_8_SP','Abio_8','All_32_8_MP','All_32_8_281_0','All_32_8_1405_0','All_32_8_0_281','All_32_8_1405_281']

all_accs_df = pd.DataFrame(np.concatenate([all_accs.T,np.matrix(all_post_cutoffs),np.matrix(all_fraction_remaining)]))
all_accs_df_sorted = all_accs_df.iloc[:,order]
all_accs_df_sorted.columns = scenarios
all_accs_df_sorted.T.to_csv('plots/acc_model_testing_sorted_for_ms_table_summary_mode_%i_sample_from_post_%i.txt'%(summary_mode,sample_from_posterior),sep='\t',index=True, header=False,float_format='%.3f')


liks = np.array(liks)[order]
fig = plt.figure(figsize=(10,15))
for i,lik in enumerate(liks):
    plt.subplot(5,3,i+1)
    plt.plot(lik)
    plt.xlim(-50,1050)
    plt.title('Lik. %s'%scenarios[i])
    plt.tight_layout()
fig.savefig('plots/lik_overview.pdf',bbox_inches="tight")

accs = np.array(accs)[order]
fig = plt.figure(figsize=(10,15))
for i,acc in enumerate(accs):
    plt.subplot(5,3,i+1)
    plt.plot(acc)
    plt.xlim(-50,1050)
    plt.ylim(0,1)
    plt.title('Train acc. %s'%scenarios[i])
    plt.tight_layout()
fig.savefig('plots/train_acc_overview.pdf',bbox_inches="tight")


fig = plt.figure(figsize=(10,4))
X = np.arange(len(order))
plt.grid(axis = 'y')
plt.gca().set_axisbelow(True)
if summary_mode == 0:
    acc_all_mean = (8*np.array(acc_paleo_mean) + np.array(acc_all_current_mean))/9
    plt.bar(X - 0.2, np.array(acc_all_mean)[order], yerr=np.array(acc_all_std)[order], edgecolor='black',color='blue', width = 0.2, label='Test set all',alpha=0.7)
    #plt.bar(X - 0.075, np.array(acc_current_mean)[order], yerr=np.array(acc_current_std)[order], color = 'blue', width = 0.15, label='Test set current')
    plt.bar(X , np.array(acc_paleo_mean)[order], yerr=np.array(acc_paleo_std)[order], width = 0.2, label='Test set paleo',edgecolor='black',color='grey',alpha=0.7)
    plt.bar(X + 0.2, np.array(acc_all_current_mean)[order], yerr=np.array(acc_all_current_std)[order], width = 0.2, label='Test set current',edgecolor='black',color='lightgrey',alpha=0.7)
    #plt.plot(np.array(acc_paleo_mean)[sorted_ids])
    plt.legend(loc='lower right')
    #plt.title('Prediction accuracy (test set)')
else:
    plt.bar(X,np.array(acc_all_mean)[order],yerr=np.array(acc_all_std)[order], color = 'blue' ,edgecolor='black', width = 0.5, label='Test set all',alpha=0.7)
    if summary_mode == 1:
        plt.title('Prediction accuracy (mean of geo-stages)')
    if summary_mode == 2:
        plt.title('Prediction accuracy')
#plt.xticks(X, scenarios, rotation=45)
plt.xticks(X, X+1, rotation=0)
plt.xlabel('Model ID')
plt.ylabel('Prediction accuracy')
plt.gca().set_yticks(np.linspace(0,1,11), minor=True)
plt.grid(axis='y',linestyle='dashed',which='minor')
plt.gca().axhline(max(acc_all_mean), color='blue', linestyle='--', linewidth=1)
plt.tight_layout()
fig.savefig('plots/acc_model_testing_summary_mode_%i_sample_from_post_%i.pdf'%(summary_mode,sample_from_posterior),bbox_inches="tight")

if summary_mode == 0:
    df_data = np.array([scenarios,
    np.array(acc_all_mean)[order],
    np.array(acc_current_mean)[order],
    np.array(acc_paleo_mean)[order],
    np.array(acc_all_current_mean)[order]
     ])

    acc_res_df = pd.DataFrame(df_data.T, columns = ['scenario','acc_test','acc_current','acc_paleo','acc_current_all'])
    acc_res_df.iloc[:,1] = pd.to_numeric(acc_res_df.iloc[:,1])
    acc_res_df.iloc[:,2] = pd.to_numeric(acc_res_df.iloc[:,2])
    acc_res_df.iloc[:,3] = pd.to_numeric(acc_res_df.iloc[:,3])
    acc_res_df.iloc[:,4] = pd.to_numeric(acc_res_df.iloc[:,4])
    acc_res_df = acc_res_df.sort_values('acc_paleo',ascending=False)
    acc_res_df.to_csv('plots/acc_model_testing_sorted_paleo.txt',sep='\t',index=False,float_format='%.3f')
    acc_res_df = acc_res_df.sort_values('acc_test',ascending=False)
    acc_res_df.to_csv('plots/acc_model_testing_sorted_test.txt',sep='\t',index=False,float_format='%.3f')

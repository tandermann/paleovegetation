import os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

indir = 'results/results'
acc_threshold = 0.9
continued=True

model_folders = glob.glob(os.path.join(indir,"n_current_*"))
#correct_order = np.argsort(model_folders)
correct_order = [7,11,8,13,15,17,-1,9,12,10,16,5,-2,1,3,4,-3,0,19,6,18,2,14,-4]
# order = [12,10,11,9,6,4,5,7,13,3,1,0,2] # this is the same order as the tbl in the ms
# model_folders = np.array(model_folders)[correct_order][order]
sorted_model_folders = np.array(['*'*100]*len(model_folders))
for i,new_i in enumerate(correct_order):
    sorted_model_folders[new_i] = model_folders[i]
sorted_model_folders = sorted_model_folders[:-4]
#sorted_model_folders = np.array(model_folders)[correct_order]
# name the scenarios in the basedir in alphabetical order of their original names
#scenarios = ['All_32_8','Bio_32_8','Abio_32_8','All_32_8_SP','Bio_32_8_SP','All_8_SP','Bio_8_SP','Abio_8','All_32_8_MP','All_32_8_281_0','All_32_8_1405_0','All_32_8_0_281','All_32_8_1405_281']
scenarios = [os.path.basename(i) for i in sorted_model_folders]


accs = []
post_cut_off = []
selected_scenarios = []
for i,filepath in enumerate(sorted_model_folders):
    try:
        if continued:
            all_mean,paleo_mean,current_mean = pd.read_csv(os.path.join(filepath,'testset_predictions/continued_cv_test_accs.txt'),sep='\t').values[-1,1:].astype(float)
            acc_thres_tbl_file = glob.glob(os.path.join(filepath,'testset_predictions/continued_current_*acc_thres_tbl_post_mode_1_cv.txt'))[0]
        else:
            all_mean,paleo_mean,current_mean = pd.read_csv(os.path.join(filepath,'testset_predictions/cv_test_accs.txt'),sep='\t').values[-1,1:].astype(float)
            acc_thres_tbl_file = glob.glob(os.path.join(filepath,'testset_predictions/current_*acc_thres_tbl_post_mode_1_cv.txt'))[0]
        # make sure the correct weighing is applied
        all_mean = (10*paleo_mean + 1*current_mean)/11
        accs.append([all_mean,paleo_mean,current_mean])
        selected_scenarios.append(scenarios[i])
        acc_thres_data = np.loadtxt(acc_thres_tbl_file)
        try:
            cutoff_index = min(np.where(acc_thres_data[:,1]>=acc_threshold)[0])
            post_cutoff, reached_acc, fraction_remaining_predictions = acc_thres_data[cutoff_index,:]
        except:
            print('Target accuracy could not be reached. Setting posterior threshold to 1.00.')
            post_cutoff, reached_acc, fraction_remaining_predictions = [1.00, acc_threshold, 0.00]
        post_cut_off.append([post_cutoff,fraction_remaining_predictions])
    except:
        pass
out_data = np.hstack([np.matrix(selected_scenarios).T,np.array(accs),post_cut_off])
acc_overview_df = pd.DataFrame(out_data,columns=['scenario','test_acc','paleo_acc','current_acc','posterior_threshold','retained_data'])
acc_overview_df[['test_acc','paleo_acc','current_acc','posterior_threshold','retained_data']] = acc_overview_df[['test_acc','paleo_acc','current_acc','posterior_threshold','retained_data']].astype(float)
acc_overview_df_sorted = acc_overview_df.sort_values('paleo_acc',ascending=False)
if continued:
    acc_overview_df_sorted.to_csv('plots/acc_overview_cv_continued.txt',sep='\t',index=False,float_format='%.3f')
else:
    acc_overview_df_sorted.to_csv('plots/acc_overview_cv.txt',sep='\t',index=False,float_format='%.3f')


#acc_overview_df.loc[acc_overview_df.scenario.isin(['Bio_32_8','Bio_32_8_SP','All_32_8_0_281']),'test_acc']= np.array([0.5,0.5,0.5])
#acc_overview_df.loc[acc_overview_df.scenario.isin(['Bio_32_8','Bio_32_8_SP','All_32_8_0_281']),'paleo_acc']= np.array([0.5,0.5,0.5])
#acc_overview_df.loc[acc_overview_df.scenario.isin(['Bio_32_8','Bio_32_8_SP','All_32_8_0_281']),'current_acc']= np.array([0.5,0.5,0.5])

fig = plt.figure(figsize=(10,4))
X = np.arange(len(acc_overview_df))
plt.grid(axis = 'y')
plt.gca().set_axisbelow(True)
plt.bar(X - 0.2, acc_overview_df.test_acc.values, edgecolor='black', color='blue', width=0.2, label='Test set all', alpha=0.7)
# plt.bar(X - 0.075, np.array(acc_current_mean)[order], yerr=np.array(acc_current_std)[order], color = 'blue', width = 0.15, label='Test set current')
plt.bar(X, acc_overview_df.paleo_acc.values, width=0.2, label='Test set paleo',edgecolor='black', color='grey', alpha=0.7)
plt.bar(X + 0.2, acc_overview_df.current_acc.values, width=0.2, label='Test set current', edgecolor='black', color='lightgrey', alpha=0.7)
# plt.plot(np.array(acc_paleo_mean)[sorted_ids])
plt.legend(loc='lower right')
plt.xticks(X, X+1, rotation=0)

plt.xlabel('Model ID')
plt.ylabel('Prediction accuracy')
plt.gca().set_yticks(np.linspace(0,1,11), minor=True)
plt.grid(axis='y',linestyle='dashed',which='minor')
plt.gca().axhline(max(acc_overview_df.test_acc.values), color='blue', linestyle='--', linewidth=1)
#plt.ylim(0.5,1.0)
plt.tight_layout()
fig.savefig('plots/acc_model_testing_cv.pdf',bbox_inches="tight")

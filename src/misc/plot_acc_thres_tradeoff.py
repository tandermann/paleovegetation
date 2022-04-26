import numpy as np
import matplotlib.pyplot as plt

acc_thres_tbl_1405_path = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_1405_n_paleo_0_regular/time_slice_predictions/continued_p1_h0_l32_8_s1_binf_1234_acc_thres_tbl_post_mode_1_sum_mode_3_sample_from_post_0.txt'
acc_thres_tbl_1405 = np.loadtxt(acc_thres_tbl_1405_path)
acc_thres_tbl_281_path = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/cluster_content/feature_gen_mc3/new_runs_2021/testing/n_current_281_n_paleo_0_regular/time_slice_predictions/continued_p1_h0_l32_8_s1_binf_1234_acc_thres_tbl_post_mode_1_sum_mode_3_sample_from_post_0.txt'
acc_thres_tbl_281 = np.loadtxt(acc_thres_tbl_281_path)



fig = plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.xlabel("Posterior threshold")
plt.ylabel("Prediction accuracy", color="b")
plt.tick_params(axis="y", labelcolor="b")
plt.plot(acc_thres_tbl_281[:,0],acc_thres_tbl_281[:,1], "b-", linewidth=2)
plt.ylim(0.87,1.01)
plt.grid(axis = 'both')

# Plot y2 vs x in red on the right vertical axis.
plt.twinx()
plt.ylabel("Fraction of valid labels", color="r")
plt.tick_params(axis="y", labelcolor="r")
plt.plot(acc_thres_tbl_281[:,0],acc_thres_tbl_281[:,2], "r-", linewidth=2)
plt.title('281 current training labels')
plt.ylim(0.2,1.1)
plt.tight_layout()

plt.subplot(1,2,2)
plt.xlabel("Posterior threshold")
plt.ylabel("Prediction accuracy", color="b")
plt.tick_params(axis="y", labelcolor="b")
plt.plot(acc_thres_tbl_1405[:,0],acc_thres_tbl_1405[:,1], "b-", linewidth=2)
plt.ylim(0.87,1.01)
plt.grid(axis = 'both')

# Plot y2 vs x in red on the right vertical axis.
plt.twinx()
plt.ylabel("Fraction of valid labels", color="r")
plt.tick_params(axis="y", labelcolor="r")
plt.plot(acc_thres_tbl_1405[:,0],acc_thres_tbl_1405[:,2], "r-", linewidth=2)
plt.title('1405 current training labels')
plt.ylim(0.2,1.1)
plt.tight_layout()

fig.savefig('plots/acc_thres_tradeoff.pdf')
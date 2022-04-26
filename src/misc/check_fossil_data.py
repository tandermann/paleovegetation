import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = 'data/raw/fossil_data/all_fossil_data_cleaned.txt'

data = pd.read_csv(data_file,sep='\t')
minage = data.min_age.values
maxage = data.max_age.values
meanage = data.mean_age.values

joined = np.vstack([minage,maxage])
calc_mean_age = np.mean(joined,axis=0)
plt.hist(calc_mean_age-meanage,bins=np.linspace(-5,5,500))
plt.ylim(0,1000)
plt.show()

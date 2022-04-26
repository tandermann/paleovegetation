import pandas as pd
import numpy as np
import os,glob
import matplotlib.pyplot as plt

global_temp_path = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/climatic_data/global_average_temperature.txt'
global_temp_data = pd.read_csv(global_temp_path,sep='\t').values.T
global_temp_data = global_temp_data[:,:301]
global_temp_data[1,:] = global_temp_data[1,:]-global_temp_data[1,0]

data_path = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/climatic_data/temperature'
file_list = glob.glob(os.path.join(data_path,'*.txt'))
age_temp_dict = {}
max_lat = 90
min_lat = -70 # cut off the bottom of the map since it has NA values and is anyways not of interest for this study of North America
for filepath in file_list:
    age = int(os.path.basename(filepath).split('_')[-1].split('.')[0].replace('ma',''))
    temp_data = pd.read_csv(filepath,sep='\t')
    target_ids = np.where(temp_data.y.values>=min_lat)[0]
    selected_cells = temp_data.iloc[target_ids]
    target_ids = np.where(selected_cells.y.values <= max_lat)[0]
    selected_cells = selected_cells.iloc[target_ids]
    mean_temp = selected_cells.temp.values.mean()
    age_temp_dict.setdefault(age,mean_temp)

age_temp_dict_sorted = dict(sorted(age_temp_dict.items()))
age_temp_array = np.array([list(age_temp_dict_sorted.keys()),list(age_temp_dict_sorted.values())])
age_temp_array[1,:] = age_temp_array[1,:]-age_temp_array[1,0]


plt.plot(age_temp_array[0,:],age_temp_array[1,:],'-',label='Mean of palaeoclimate models')
plt.plot(global_temp_data[0,:],global_temp_data[1,:],'-',label = 'Zachos curve')
plt.legend()
plt.xlabel('Time (Ma)')
plt.ylabel('$\Delta Temp$')
plt.show()



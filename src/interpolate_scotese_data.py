"""
Created on Tue Jul  2 17:42:29 2019

@author: Tobias Andermann (tobias.andermann@bioenv.gu.se)
"""

import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt

def interpolate_a_b_dT(a,b,dT,t,ta):
    slope = (b-a)/dT
    return a+(t-ta)*slope

max_age = 30


elevation_folder = 'data/raw/elevation/PaleoDEMS_long_lat_elev_csv_v2/PaleoDEMS_long_lat_elev_csv_v2.csv/*.csv'
files = sorted(glob.glob(elevation_folder))
# only continue with files up to the defined max_age
time_file_dict = dict(zip([int(os.path.basename(file).split('_')[-1].split('.')[0].replace('Ma','')) for file in files],files))
files = [time_file_dict[i] for i in np.arange(0,max_age+1,5)]

previous_data=[]
previous_timestamp=None
for file in files:
    filename = file.split('/')[-1]
    timestamp = int(os.path.basename(file).split('_')[-1].split('.')[0].replace('Ma',''))
    data = pd.read_csv(file,sep=',')
    data.columns = ['lon','lat','elev']
    data_array = data.values[:,-1]
    # after the first iteration when previous_data is empty, start this loop
    if len(previous_data) > 0:
        a = previous_data
        b = data_array
        ta = previous_timestamp
        tb = timestamp
        dT = tb-ta
        n_steps = dT+1
        for i in np.linspace(ta,tb,n_steps):
            interpolated_data = interpolate_a_b_dT(a,b,dT,i,ta)
            interpolated_df = data.copy()
            interpolated_df.elev = interpolated_data
            outfile = 'data/raw/elevation/global_elevation_%ima.txt'%int(i)
            interpolated_df.to_csv(outfile,sep='\t',index=False)
            plt.scatter(interpolated_df.lon,interpolated_df.lat,c=interpolated_df.elev)
            plt.show()
    previous_data = data_array
    previous_timestamp = timestamp


precipitation_folder = '/Users/tobias/GitHub/paleovegetation_mammals/data/raw/climate_data/precipitation_0_540Ma/*.csv'
files = sorted(glob.glob(precipitation_folder))
# only continue with files up to the defined max_age
time_file_dict = dict(zip([int(file.split('/')[-1].split('_')[0]) for file in files],files))
files = [time_file_dict[i] for i in np.arange(0,max_age+1,5)]

previous_data=[]
previous_timestamp=None
for file in files:
    filename = file.split('/')[-1]
    timestamp = int(file.split('/')[-1].split('_')[0])
    data = pd.read_csv(file,sep=',',header=-1)
    # remove first line and first column, since first cell contains outlier value and the total array should be 180x360
    # also remove last column, which only has NaN values
    final_data = data.iloc[1:,1:-1]
    data_array = final_data.values
    # after the first iteration when previous_data is empty, start this loop
    if len(previous_data) > 0:
        a = previous_data
        b = data_array
        ta = previous_timestamp
        tb = timestamp
        dT = tb-ta
        n_steps = dT+1
        for i in np.linspace(ta,tb,n_steps):
            interpolated_data = interpolate_a_b_dT(a,b,dT,i,ta)
            interpolated_df = final_data.copy()
            interpolated_df.loc[:,:] = interpolated_data
            outfile = '/Users/tobias/GitHub/paleovegetation_mammals/data/raw/climate_data/precipitation_0_540Ma/interpolated/%iMa_precipitation_interpolated.txt'%int(i)                
            interpolated_df.to_csv(outfile,sep=',',header=False,index=False)
    previous_data = data_array
    previous_timestamp = timestamp


temperature_folder = '/Users/tobias/GitHub/paleovegetation_mammals/data/raw/climate_data/temperature_0_540Ma/*.csv'
temp_files = glob.glob(temperature_folder)
# only continue with files up to the defined max_age
time_temp_file_dict = dict(zip([int(file.split('/')[-1].split('_')[0]) for file in temp_files],temp_files))
temp_files = [time_temp_file_dict[i] for i in np.arange(0,max_age+1,5)]

previous_data=[]
previous_timestamp=None
for file in temp_files:
    filename = file.split('/')[-1]
    timestamp = int(file.split('/')[-1].split('_')[0])
    data = pd.read_csv(file,sep=',',header=-1)
    # remove first line and first column, since first cell contains outlier value and the total array should be 180x360
    # also remove last column, which only has NaN values
    final_data = data.iloc[1:,1:-1]
    data_array = final_data.values
    # after the first iteration when previous_data is empty, start this loop
    if len(previous_data) > 0:
        a = previous_data
        b = data_array
        ta = previous_timestamp
        tb = timestamp
        dT = tb-ta
        n_steps = dT+1
        for i in np.linspace(ta,tb,n_steps):
            interpolated_data = interpolate_a_b_dT(a,b,dT,i,ta)
            interpolated_df = final_data.copy()
            interpolated_df.loc[:,:] = interpolated_data
            outfile = '/Users/tobias/GitHub/paleovegetation_mammals/data/raw/climate_data/temperature_0_540Ma/interpolated/%iMa_temperature_interpolated.txt'%int(i)                
            interpolated_df.to_csv(outfile,sep=',',header=False,index=False)
    previous_data = data_array
    previous_timestamp = timestamp



""" 
    This script is used to prepare the data for the ANN model.
    It get measurement spectrum from neuron sensors and environmental variables from the Zefyros dataset.
    It stores the data un the folder 'test_data' in the current directory.
"""
import xarray as xr
import os
# from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from Preprocessor_Zefyros_dataset_ANN import get_X_Y_train_test_from_xr_dataset, get_cos_sin_decomposition, Scale_Xscalar_data, Scale_Yspectrum_data

#### select X and Y variables
envir_variables = [ 'dp', 'tp', 'hs', 'mag10', 'theta10'] #, 'cur', 'cur_dir']
Neuron_variables = []
for i in ['x', 'y', 'z']:
    for j in [1,3]:
        Neuron_variables.append(f'simu_NeurNode{j}AL{i}t_psd')

####################################
#### OPTIONNAL : LOAD / PREPARE /SCALE DATA
url_base = r"C:\Users\romain.ribault\Documents\Dionysos\echanges_Morphosense\final_12-2022-accuracy_f.nc"
simu_path = os.path.join(url_base)
df = xr.open_dataset(simu_path)
#ds_envir = ds_final.drop_vars([key for key in list(ds_final.keys()) if key not in list_key])
ds_envir = df.drop_dims(['time_psd', 'stats', 'Frequency_psd', 'StatsDim', 'freq_ranges'])
ds_envir = ds_envir.rename({'time' : 'time_psd'})
df = df.drop_dims('time')
df = xr.merge([df, ds_envir], compat='override')


# remove nan values from df
df = df.dropna(dim='time_psd', how='any')
df = df.rename({'time_psd' : 'time'})
# Re-arrange direction columns because 0/360 discontinuity do not fit with neural networks.
for dict_key in [{'mag10':'theta10'}, {'hs':'dp'}] : #, {'cur':'cur_dir'}] :
    df = get_cos_sin_decomposition(dict_key, df)
    for magnitude, angle in dict_key.items() :
        envir_variables.remove(angle)
        envir_variables.append(f'{magnitude}_cos')
        envir_variables.append(f'{magnitude}_sin')

folder = r'C:\Users\romain.ribault\Documents\GitHub\subsee4d-mooring-monitoring\10_DATA\numpy databases\2022-12 Neuron Measurements'
# split data into train and test sets. The data are saved top disk to avoid re-computing them each time.
get_X_Y_train_test_from_xr_dataset(df=df, X_channel_list=envir_variables, Y_channel_list=Neuron_variables, folder = folder)


# LOAD final datasets : 
files = ['X_train', 'X_val', 'Y_train', 'Y_val'] 
train_test_sets = []
for file in files :
    file = os.path.join(folder, file)
    with open(file, 'rb') as f:
        train_test_sets.append(pickle.load(f) )
X_train, X_test, Y_train, Y_test  = train_test_sets[0], train_test_sets[1], train_test_sets[2], train_test_sets[3]  

y_train, y_test = Scale_Xscalar_data(X_train, X_test, folder= folder, file_names=['envir_train_scaled', 'envir_test_scaled'])
x_train, x_test = Scale_Yspectrum_data(Y_train, Y_test, folder= folder, file_names=['spectrum_train_scaled', 'spectrum_test_scaled'])
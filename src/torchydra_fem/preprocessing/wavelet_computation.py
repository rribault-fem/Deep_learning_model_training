# -*- coding: utf-8 -*-

"""
This Script takes a date, wavelet scales and wavelet name as input.

Go to the end of the script for user parameters in if __name__ == '__main__' at the en

For each hour of the day, it takes time series of Neuron nodes and calculates associated wavelets.
It performs the calculation of One dimensional Continuous Wavelet Transform for both : 
- measurement data from Neuron sensors
- simulation data from Openfast model.


@author: rribault
"""

import glob
import os
import xarray as xr
import pywt
import numpy as np



def merge_dataset(file):
    # Open the netCDF file using xarray
    dataset = xr.open_dataset(file)
    return dataset

def merge_all_files_for_day(pattern) :
    file_list = glob.glob(pattern)
    datasets = map(merge_dataset, file_list)
    return xr.merge(datasets, compat='no_conflicts')

def compute_wavelet_coeff_for_CNN(input_set, wavelet_scales,  waveletname) :
    print('####')
    print(f'start computing wavelet for {np.shape(input_set)[0]} samples on {np.shape(input_set)[2]} channels')
    print('####')
    input_data_cwt = np.ndarray(shape=(np.shape(input_set)[0], len(wavelet_scales), len(wavelet_scales),np.shape(input_set)[2] ))

    # loop on each samples
    for ii in range(0,np.shape(input_set)[0]):
        if ii % 10 == 0:
            print('computing for sample nb {}'.format(ii))
        # loop on each channel
        for jj in range(0,np.shape(input_set)[2]):
            signal = input_set[ii, :, jj]
            coeff, freq = pywt.cwt(signal, wavelet_scales, waveletname, 1)
            coeff_ = coeff[:,:len(wavelet_scales)]
            input_data_cwt[ii, :, :, jj] = coeff_

    return input_data_cwt

def get_numpy_input_channel_set(df, channels) :
    # loading channels data in numpy for CNN 

    input_channel_set = np.empty_like(np.expand_dims(df[channels[0]].values, axis = 2))
    for channel in channels :
        input_channel = np.expand_dims(df[channel].values, axis = 2)
        input_channel_set = np.append(input_channel_set, input_channel, axis = 2)
    input_channel_set = np.delete(input_channel_set, 0, axis=2)
    return input_channel_set

def compute_pywt_cwt(da : xr.DataArray, wavelet_scales : list,  waveletname : str):
    func = lambda x, y, z: pywt.cwt(x, y, z, 1)
    return xr.apply_ufunc(func, da, wavelet_scales, waveletname)

if __name__ == '__main__':
    # User parameters :
    dataset_path = r"data\netcdf_databases\20231019_for_torchydra_ins_to_tension.nc"
    df = xr.open_dataset(dataset_path)
    df = df.sel(time=slice('2023-02-19 00:00:00', '2023-02-28 02:00:00'))
    # Define chanels on which to apply wavelet transformation
    channels = ['x', 'y', 'y', 'roll', 'pitch', 'yaw']
    variable_list = channels
    coordinate_list = list(df.coords)
    not_drop_list = coordinate_list + variable_list
        
    # drop all variables not in variable_list
    df = df.drop_vars([ var for var in df.variables if var not in not_drop_list] )
    df = df.dropna(dim='time', how='any')

    # Define wavelet scales and name :
    #l1 = [np.around(1/p,6) for p in np.arange(16.5, 200, 0.5)][::-1] # Low frequencies (example : slow drift)
    l2 = [np.around(1/p,6) for p in np.arange(2, 16, 0.1)][::-1] # Wave frequencies including rotor frequencies
    l3 = [np.around(i,6) for i in np.arange(0.52,4.01,0.1)] # High Frequencies  (0.52 not to have duplicate 0.5 and 4.01 to include 4.0)
    wavelet_scales = l2 + l3
    waveletname = 'morl'
    
    #Perform wavelet transformation on measurements data
    data_ts = get_numpy_input_channel_set(df, channels)
    data_cwt = compute_wavelet_coeff_for_CNN(data_ts, wavelet_scales,  waveletname)
    hour = 1
    channel = 0
    
    print(f'cwt coefficients for simulation on at {hour} for {channel}')
    print(data_cwt[hour, : , : , channel])
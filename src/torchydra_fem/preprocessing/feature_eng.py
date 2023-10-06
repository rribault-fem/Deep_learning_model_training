
from dataclasses import dataclass
from typing import List, Any, Optional, Dict
import xarray as xr
import logging
import numpy as np
import os

@dataclass
class FeatureEng:
    envir_direction_dict : Optional[Dict]
    sin_cos_method: str

    def get_cos_sin_decomposition(self, dict_key:dict, df:xr.Dataset):
        """
        This function decomposes 0 to 365 values into cosine and sine values. It takes in a dictionary and an xarray Dataset as input.

        The function first logs information about the decomposition. 
        It then calculates the cosine and sine values for each magnitude and angle specified in the input dictionary.

        The calculated cosine and sine values are then merged into the input Dataset.

        Args:
            dict_key (dict): A dictionary containing magnitude-angle pairs to be decomposed.
            df (xr.Dataset): An xarray Dataset containing the data to be decomposed.

        Returns:
            xr.Dataset: An xarray Dataset containing the original data as well as the calculated cosine and sine values.
        """
         
        log = logging.getLogger(os.environ['logger_name'])
        log.info('###')
        log.info(f'get cos and sin decomposition of the data {dict_key}')
        for magnitude, angle in dict_key.items() :
            log.info(f'get cos and sin decomposition of the data {magnitude} and {angle}')
            df_cos = self.get_xr_dataset_time(magnitude+'_cos', df[magnitude]*np.cos(2*np.pi*df[angle].values/365), f'{magnitude} * cos of {angle} values', df.time.values)
            df_sin = self.get_xr_dataset_time(magnitude+'_sin', df[magnitude]*np.sin(2*np.pi*df[angle].values/365), f'{magnitude} * sin of {angle} values', df.time.values)
            df = xr.merge([df, df_cos, df_sin], compat = 'no_conflicts')

        return df

    def get_xr_dataset_time(self, array_name : str, array_values : np.array, description : str , time_values : np.array) :
        
        data_array_dict = {
                "attrs":{
                    "description" : description},
                "dims" : "time",
                "data" : array_values,
                "name" : array_name
            }

        DataArray  =  xr.DataArray.from_dict(data_array_dict)
        Dataset = xr.Dataset(
            { array_name : DataArray},
            coords={"time" : time_values}
        )

        return Dataset
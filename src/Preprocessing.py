from dataclasses import dataclass
from typing import Optional, Dict
from preprocessing.envir_scaler import Envir_scaler
from preprocessing.decomp_y_spectrum import Decomp_y_spectrum
from preprocessing.y_spectrum_scaler import Y_spectrum_scaler	
from preprocessing.split_transform import Split_transform
from preprocessing.feature_eng import FeatureEng
from preprocessing.inputs_outputs import Inputs_ouputs_v1
import xarray as xr

@dataclass
class Preprocessing :    
    inputs_outputs : Inputs_ouputs_v1
    feature_eng : FeatureEng
    split_transform : Split_transform
    envir_scaler : Envir_scaler
    perform_decomp : bool
    decomp_y_spectrum: Decomp_y_spectrum
    y_spectrum_scaler : Y_spectrum_scaler
    unit_dictionnary:  Optional[Dict] = None
    Frequency_psd : xr.DataArray = None


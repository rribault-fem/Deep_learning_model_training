from dataclasses import dataclass
from typing import Optional, Dict
import xarray as xr
from torchydra_fem.preprocessing.decomp_y_spectrum import Decomp_y_spectrum
from torchydra_fem.preprocessing.split_transform import Split_transform
from torchydra_fem.preprocessing.feature_eng import FeatureEng
from torchydra_fem.preprocessing.inputs_outputs_v1 import Inputs_ouputs_v1
from torchydra_fem.preprocessing.scaler import Scaler

@dataclass
class Preprocessing :    
    inputs_outputs : Inputs_ouputs_v1
    feature_eng : FeatureEng
    split_transform : Split_transform
    input_scaler : Scaler
    perform_decomp : bool
    decomp_y_spectrum: Decomp_y_spectrum
    output_scaler : Scaler
    unit_dictionnary:  Optional[Dict] = None
    Frequency_psd : xr.DataArray = None


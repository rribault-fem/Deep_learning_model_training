import xarray as xr
import os
import pickle
import numpy as np
import torch
from typing import List
import yaml
import hydra
from omegaconf import DictConfig
import logging
from lightning import LightningDataModule
from torch.utils.data import TensorDataset

from torchydra_fem.utils.load_env_file import load_env_file
import torchydra_fem.utils as utils
from torchydra_fem.Preprocessing import Preprocessing
from torchydra_fem.model.surrogate_module import SurrogateModule

@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="test_time_series_to_stats.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    inference(cfg)


@utils.task_wrapper
def inference(cfg : DictConfig):
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    
    os.environ['logger_name'] = cfg.task_name
    SAVE_PATH = cfg.save_path
    EXPERIMENT_PATH = os.path.join(cfg.paths.root_dir, cfg.experiment_folder)

    inference_dataset_path = os.path.join(cfg.paths.dataset)
    infer_dataset = xr.open_dataset(inference_dataset_path)
    infer_dataset = infer_dataset.sel(time = slice('2023-06-20 13', '2023-06-30-14'))

    # load preprocessing pipeline
    # Use of pickle object to load the scaler already fit to the data
    preprocess_path = os.path.join(EXPERIMENT_PATH, "preprocessing.pkl")
    with open(preprocess_path, 'rb') as f:
        preprocess : Preprocessing = pickle.load(f)

    # Load inputs list
    X_channel_list : List[str] = preprocess.inputs_outputs.input_variables
    Y_channel_list : List[str] = preprocess.inputs_outputs.output_variables

    # Use preprocessing pipeline to prepare test data for inference
    df= infer_dataset
    X_infer = preprocess.split_transform.get_numpy_input_2D_set(df, X_channel_list)
    Y = preprocess.split_transform.get_numpy_input_2D_set(df, Y_channel_list)
    
    x_infer = preprocess.input_scaler.scale_data_infer(X_infer)
    y_test = preprocess.output_scaler.scale_data_infer(Y)

    # load model
    hydra_config_path = os.path.join(EXPERIMENT_PATH, r'.hydra/config.yaml' )

    with open(hydra_config_path, 'r') as f:
        hydra_config = yaml.safe_load(f)

    datamodule : LightningDataModule = hydra.utils.instantiate(hydra_config['data'])

    # setup x_infer to datamodule for model evaluation
    datamodule.setup(stage='test', x_test=x_infer, y_test=y_test)
    
    if hydra_config['preprocessing']['perform_decomp'] == True:
        decomp_length = 24
    else : decomp_length = 512

    # load a dummy model with dummy kwargs to load the checkpoint
    try :
        kwargs = hydra_config['model_net']['kwargs']
    
    except KeyError as e :
        kwargs = {
            "nb_obs" : 10,
            "two_dims_decomp_length" : 600,
            "two_dims_channel_nb" : 3}
        
    model_path = os.path.join(EXPERIMENT_PATH, r"checkpoints\last.ckpt")
    
    net: torch.nn.Module = hydra.utils.instantiate(hydra_config['model_net'], **kwargs)

    # model kwargs parameters are infered from checkpoint
    model : SurrogateModule = SurrogateModule.load_from_checkpoint(model_path, net=net)
    

    # predict nominal spectrum thanks to the surrogate model
    def model_predict(x_infer: torch.Tensor , model :SurrogateModule ) -> np.ndarray :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        x_infer = x_infer.to(device)
        model.eval()
        y_hat = model(x_infer)
        
        
        return y_hat
    
    log = logging.getLogger(os.environ['logger_name'])
    log.info('Perform model predictions')

    y_hat = model_predict(datamodule.x_val, model)
    y_hat = y_hat.detach().cpu().numpy()
    # y_hat = datamodule._undo_shift_reshape_data(y_hat)
    

    # unscale y_hat
    if preprocess.output_scaler.donot_scale :
        Y_hat : np.array = y_hat
        Y_test = Y
    else: 
        Y_hat: np.array = preprocess.output_scaler.inverse_transform_data_infer(y_hat)
        Y_test = preprocess.output_scaler.inverse_transform_data_infer(y_test)
    
    # reshape Y_hat and Y to be able to save them in a netcdf file
    y_hat = y_hat.reshape(-1, 60, 4)

    y_test= datamodule.y_test.detach().cpu().numpy()
    y_test = y_test.reshape(-1, 60, 4)

    variables = ['max', 'min', 'mean', 'std']
    minutes = np.arange(0,60,1)


    for var_name in variables :
        array_NN = prepare_new_array_NN(Y_hat[:, :, variables.index(var_name)], var_name)
        array_val = prepare_new_array_val(Y_test[:, :, variables.index(var_name)], var_name)
        infer_dataset['NNet_tension_'+var_name] = array_NN
        infer_dataset['tension_sensor'+var_name] = array_val

    log.info('Save netcdf file')
    infer_dataset.to_netcdf(r"C:\Users\romain.ribault\Documents\git_folders\torchydra\data\netcdf_databases\inference_stats.nc")

    metric_dict = {}
    object_dict = {}

    return metric_dict, object_dict



def prepare_new_array_NN(var : np.array, var_name : str) -> xr.DataArray:

    array =  xr.DataArray.from_dict(
        {
            "attrs" : {
                "description" : f" Inference of the {var_name} using a neural network",
                "source" : "Neural network"},
            "dims" : ["time", "minutes"],
            "data"  : var,
            "name" : f"NNet_tension_{var_name}",
        })
     
    return array

def prepare_new_array_val(var : np.array, var_name : str) -> xr.DataArray:

    array =  xr.DataArray.from_dict(
        {
            "attrs" : {
                "description" : f" Statistics of the {var_name} from sensor measurements",
                "source" : "Sensor measurements"},
            "dims" : ["time", "minutes"],
            "data"  : var,
            "name" : f"tension_sensor{var_name}",
        })
     
    return array


if __name__ == "__main__":
    load_env_file(f"env.yaml")
    main()

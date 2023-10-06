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

from torchydra_fem.utils.load_env_file import load_env_file
import torchydra_fem.utils as utils
from torchydra_fem.Preprocessing import Preprocessing
from torchydra_fem.model.surrogate_module import SurrogateModule

@hydra.main(version_base="1.3", config_path="..../configs", config_name="inference_tseries.yaml")
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
    infer_dataset = infer_dataset.sel(time = slice('2023-06-20 13', '2023-06-20-14'))

    # load preprocessing pipeline
    # Use of pickle object to load the scaler already fit to the data
    preprocess_path = os.path.join(EXPERIMENT_PATH, "preprocessing.pkl")
    with open(preprocess_path, 'rb') as f:
        preprocess : Preprocessing = pickle.load(f)

    # load model
    model_path = os.path.join(EXPERIMENT_PATH, r"checkpoints\last.ckpt")
    hydra_config_path = os.path.join(EXPERIMENT_PATH, r'.hydra/config.yaml' )

    with open(hydra_config_path, 'r') as f:
        hydra_config = yaml.safe_load(f)

    if hydra_config['preprocessing']['perform_decomp'] == True:
        decomp_length = 24
    else : decomp_length = 512

    # load a dummy model with dummy kwargs to load the checkpoint
    try :
        kwargs = hydra_config['model_net']['kwargs']
    
    except KeyError as e :
        kwargs = {
            "nb_obs" : 35985,
            "two_dims_decomp_length" : 1,
            "two_dims_channel_nb" : 3}
    
    net: torch.nn.Module = hydra.utils.instantiate(hydra_config['model_net'], **kwargs)

    # model kwargs parameters are infered from checkpoint
    model : SurrogateModule = SurrogateModule.load_from_checkpoint(model_path, net=net)

    # Load inputs list
    X_channel_list : List[str] = preprocess.inputs_outputs.input_variables

    # Use preprocessing pipeline to prepare test data for inference
    df= infer_dataset
    X_infer = preprocess.split_transform.get_numpy_input_2D_set(df, X_channel_list)
    x_infer = preprocess.input_scaler.scale_data_infer(X_infer)

    # predict nominal spectrum thanks to the surrogate model
    def model_predict(x_infer: np.array, model :SurrogateModule ) -> np.ndarray :
        
        x_infer = torch.from_numpy(x_infer).float()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_infer = x_infer.to(device)
        model = model.to(device)
        model.eval()
        y_hat = model(x_infer)
        y_hat = y_hat.detach().cpu().numpy()
        
        return y_hat
    
    log = logging.getLogger(os.environ['logger_name'])
    log.info('Perform model predictions')
    y_hat = model_predict(x_infer, model)

    # unscale y_hat
    Y_hat = preprocess.output_scaler.inverse_transform_data_infer(y_hat)

    variables = preprocess.inputs_outputs.output_variables

    for var in variables :
        array = prepare_new_array(Y_hat[:, :, variables.index(var)], 'NNet_'+var)
        infer_dataset['NNet_'+var] = array

    log.info('Save netcdf file')
    infer_dataset.to_netcdf(r"C:\Users\romain.ribault\Documents\git_folders\torchydra\data\netcdf_databases\inference.nc")

    metric_dict = {}
    object_dict = {}

    return metric_dict, object_dict



def prepare_new_array(NN_var : np.array, vars : str) -> xr.DataArray:

    array =  xr.DataArray.from_dict(
        {
            "attrs" : {
                "description" : f" Inference of the {vars} using a neural network",
                "source" : "Neural network"},
            "dims" : ["time", "time_sensor"],
            "data"  : NN_var,
            "name" : f"NN_{vars}",
        })
     
    return array

if __name__ == "__main__":
    load_env_file(f"env.yaml")
    main()

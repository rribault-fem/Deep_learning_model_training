import xarray as xr
import os
import pickle
import numpy as np
import virtual_sensor.export_data.Neuron as Neuron
from Preprocessing import Preprocessing
from scipy.stats import qmc
from model.surrogate_module import SurrogateModule
import torch
from typing import List
import yaml
import utils
from utils.load_env_file import load_env_file
import hydra
from omegaconf import DictConfig
import logging

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

    infer_dataset.to_netcdf(r"C:\Users\romain.ribault\Documents\git_folders\torchydra\data\netcdf_databases\inference.nc")

    # if preprocess.perform_decomp :
    #     PCAs = preprocess.decomp_y_spectrum.decomps

    #     # Prepare np arrays to receive inference results
    #     Y_hat_inter = np.zeros_like(y_hat)
    #     batch_size = np.shape(PCAs[0].inverse_transform(y_hat[:,:,0]))[0]
    #     spectrum_length = np.shape(PCAs[0].inverse_transform(y_hat[:,:,0]))[1]
    #     channel_nb = len(Yscalers)
    #     Y_hat = np.zeros((batch_size, spectrum_length, channel_nb ))


    #     i=0
    #     for pca, scaler in zip(PCAs, Yscalers) :
    #         Y_hat_inter[:,:,i] =  scaler.inverse_transform(y_hat[:,:,i])
    #         Y_hat[:,:,i] = pca.inverse_transform(Y_hat_inter[:,:,i])
    #         i+=1


    # perform uncertainty propagation and calculate 95% confidence interval
    # define input variables uncertainty :
    # define variables uncertainty
    log.info('uncertainty propagation and calculate 95% confidence interval based on input uncertainties')
    env_uncertainty = {}
    for key in preprocess.inputs_outputs.envir_variables:
        env_uncertainty[key] = 0.2 # +/- 20% uncertainty

    sampler = qmc.LatinHypercube(d=len(env_uncertainty), seed=12345)
    sample = sampler.random(n=100)
    sample = np.repeat(sample.reshape((sample.shape[0], 1, sample.shape[1])), 24, axis=1)
    from scipy.stats import norm


    y_output_sample = []
    y_hat_max_env = []
    y_hat_min_env = []
    hour = 0
    for env_sample in X_env :
        # stretch the sample to the uncertainty range and center it around the nominal value
        for i, key in enumerate(env_uncertainty.keys()):
            sample[:, hour, i] = norm(loc=(env_sample[preprocess.inputs_outputs.envir_variables.index(key)]),
                            scale=(abs(env_sample[preprocess.inputs_outputs.envir_variables.index(key)]*env_uncertainty[key]))/1.97).ppf(sample[:,hour, i])
        hour+=1
    for i in range(sample.shape[0]):
        # scale the input sample
        x_env_sample = preprocess.input_scaler.scaler.transform(sample[i, :, :])
        y_sample = model_predict(x_env_sample, model)
        y_output_sample.append(y_sample)
        
    y_output_sample = np.array(y_output_sample)
        
    # take the 99% percentice of the sample (none, 512, 6) on axis 1  
        
    y_hat_max_env = np.percentile(y_output_sample, 99.9, axis=0)
    y_hat_min_env = np.percentile(y_output_sample, 0.1, axis=0)
    y_hat_mean = np.mean(y_output_sample, axis=0)

    # unscale y_hat
    Y_hat_max_env_int= np.zeros_like(y_hat_max_env)
    Y_hat_min_env_int= np.zeros_like(y_hat_min_env)
    Y_hat_mean_env_int = np.zeros_like(y_hat_mean)
    Y_hat_max_env = np.zeros((batch_size, spectrum_length, channel_nb ))
    Y_hat_min_env = np.zeros((batch_size, spectrum_length, channel_nb ))
    Y_hat_mean_env = np.zeros((batch_size, spectrum_length, channel_nb ))

    if preprocess.perform_decomp :
        i=0
        for pca, scaler in zip(PCAs, Yscalers):
            Y_hat_max_env_int[:,:,i] =  scaler.inverse_transform(y_hat_max_env[:,:,i])
            Y_hat_max_env[:,:,i] =  pca.inverse_transform(Y_hat_max_env_int[:,:,i])
            Y_hat_min_env_int[:,:,i] =  scaler.inverse_transform(y_hat_min_env[:,:,i])
            Y_hat_min_env[:,:,i] = pca.inverse_transform(Y_hat_min_env_int[:,:,i])
            Y_hat_mean_env_int[:,:,i] =  scaler.inverse_transform(y_hat_mean[:,:,i])     
            Y_hat_mean_env[:, :, i] = pca.inverse_transform(Y_hat_mean_env_int[:,:,i])
            i+=1
    else :
        i=0
        for scaler in Yscalers:
            Y_hat_max_env[:,:,i] =  scaler.inverse_transform(y_hat_max_env[:,:,i])
            Y_hat_min_env[:,:,i] =  scaler.inverse_transform(y_hat_min_env[:,:,i])
            Y_hat_mean_env[:,:,i] =  scaler.inverse_transform(y_hat_mean[:,:,i])        

            i+=1

    # Allocate to Neuron object
    Y_channel_list = preprocess.inputs_outputs.output_variables
    neuron = Neuron.Neuron()
    neuron.institution = 'FEM'
    model_id = os.path.basename(os.path.normpath(EXPERIMENT_PATH))
    neuron.source = f'surrogate_{model_id}'

    i=0
    for channel in Y_channel_list:
        units_dict = preprocess.unit_dictionnary
        units_dict[f'{channel}_max_env'] = units_dict[channel]
        units_dict[f'{channel}_min_env'] = units_dict[channel]
        
        neuron.allocate_ann_psd_inference(channel, units_dict, Y_hat_mean_env[:,:,i])
        neuron.allocate_ann_psd_inference(f'{channel}_max_env', units_dict, Y_hat_max_env[:,:,i])
        neuron.allocate_ann_psd_inference(f'{channel}_min_env', units_dict, Y_hat_min_env[:,:,i])
        i+=1

    #neuron.Frequency_psd = preprocess.Frequency_psd
    # Temporary fix for the frequency vector :
    neuron.Frequency_psd = preprocess.Frequency_psd.where(preprocess.Frequency_psd>(preprocess.split_transform.cut_low_frequency), drop=True).values
    neuron.time_psd = infer_dataset.time.values

    # Save netcdf file
    log.info('Save netcdf file')
    saved_path = neuron.save_nc(DATE, SAVE_PATH, ann_name=neuron.source)
    log.info(f'MODEL INFERENCE OUTPUTS saved in {saved_path} ')

    metric_dict = {}
    object_dict = {}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference_tseries.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    inference(cfg)


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

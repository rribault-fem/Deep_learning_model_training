import xarray as xr
import numpy as np
import pickle
import hydra
from omegaconf import DictConfig
import logging
from torchydra_fem.Preprocessing import Preprocessing
from torchydra_fem.model.surrogate_module import SurrogateModule
from sklearn.utils import shuffle
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger
import torch
from typing import List
from torchydra_fem.utils import *
from torchydra_fem.utils.load_env_file import load_env_file
import os
import math

# version_base=1.1 is used to make hydra change the current working directory to the hydra output path
@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg :  DictConfig):
        """
        This function serves as the main entry point for the script.
        It takes in a configuration object from hydra config files and uses it to train a surrogate model.
        The function first creates an instance of the `Preprocessing` class using the provided configuration. 
        It then pre-processes the data using the `Pre_process_data` function.

        After pre-processing, the Preprocessing object is saved for future use. 
        The specified model type is then imported and trained on the pre-processed data using the `Train_model` function.

        Finally, after training, both the pipeline and trained model are saved for future use.

        Args:
        cfg (DictConfig): The configuration object used to specify training parameters.

        Returns:
        None
        """

        # load environment variables foir logger name
        load_env_file(f"{hydra.utils.get_original_cwd()}/env.yaml")
        os.environ['logger_name'] = cfg.task_name
        log = logging.getLogger(os.environ['logger_name'])

        # Instantiate preprocessing pipeline
        log.info(f"Instantiating Preprocessing <{cfg.preprocessing._target_}>")
        preprocess: Preprocessing = hydra.utils.instantiate(cfg.preprocessing)
        
        # Pre-process data
        x_train, y_train, x_test, y_test = Pre_process_data(cfg, preprocess)
        
        # save the pipeline for future use and inverse transform
        log.info("Saving preprocessing")
        file_path = os.path.join(cfg.paths.output_dir, 'preprocessing.pkl')
        with open(file_path, 'wb') as f:
                pickle.dump(preprocess, f)

        x_input_size, spectrum_decomp_length, spectrum_channel_nb = np.shape(x_train)[1], np.shape(y_train)[1], np.shape(y_train)[2]

        # instanciate DataModule. Parameters depending on dataset are passed as args. 
        kwargs = {
                "x_train" : x_train,
                "y_train" : y_train,
                "x_test" : x_test,
                "y_test" : y_test}
        
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, **kwargs)

        # instanciate model. Parameters depending on dataset are passed as kwargs.
        kwargs = {
                "x_input_size" : x_input_size,
                "spectrum_decomp_length" : spectrum_decomp_length,
                "spectrum_channel_nb" : spectrum_channel_nb}
        
        log.info(f"Importing model net {cfg.model_net._target_}")
        # can be passed as *args because all arguments are defined above, no argument defined in .yaml config file.
        model_net : torch.nn.Module = hydra.utils.instantiate(cfg.model_net, **kwargs)
        
        log.info(f"Importing model  {cfg.model._target_}")
        model : SurrogateModule = hydra.utils.instantiate(cfg.model)
        # model.net cannot be instanciated in the config file because it depends on the dataset:
        model.net = model_net
        
        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
        
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        torch.set_float32_matmul_precision('medium')
        
        object_dict = {
                "cfg": cfg,
                "datamodule": datamodule,
                "model": model,
                "callbacks": callbacks,
                "logger": logger,
                "trainer": trainer,
        }

        if logger:
                log.info("Logging hyperparameters!")
                log_hyperparameters(object_dict)

        if cfg.get("compile"):
                log.info("Compiling model!")
                model = torch.compile(model)

        if cfg.get("train"):
                log.info("Starting training!")
                trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        train_metrics = trainer.callback_metrics

        # return the metric to optimise for hyper parameter search
        return train_metrics['train/loss']



def Pre_process_data(cfg: DictConfig, preprocess : Preprocessing):
        """
        This function pre-processes the data before training. It takes in an instance of the `PipeConfig` class and uses it to perform various operations on the data.

        The function first loads the dataset and drops any missing values. It then creates a dictionary to store the units of each variable.

        Next, direction columns are rearranged to fit with neural networks. This is done using the `get_cos_sin_decomposition` function.

        After rearranging direction columns, the data is split into training and testing sets using the specified split transform. The input data is then scaled using the specified scaler.

        Args:
        pipe (PipeConfig): An instance of the `PipeConfig` class containing configuration information.

        Returns:
        tuple: A tuple containing four elements: X_train, X_test, Y_train, Y_test.
        """
        ####
        #Start pipeline
        ####
        df = xr.open_dataset(cfg.paths.dataset)
        df = df.dropna(dim='time', how='any')

        preprocess.unit_dictionnary = {}
        for var in preprocess.inputs_outputs.envir_variables :
                preprocess.unit_dictionnary[var] = df[var].attrs['unit']
        for var in preprocess.inputs_outputs.neuron_variables :
                preprocess.unit_dictionnary[var] = df[var].attrs['unit']
        
        if not preprocess.perform_decomp :
                cut_low_freq_arg = np.argwhere(df.Frequency_psd.values>(preprocess.split_transform.cut_low_frequency))[0][0]
                if math.log(cut_low_freq_arg,2) - int(math.log(cut_low_freq_arg,2)) != 0 :
                        cut_low_freq_arg = 345
                        preprocess.split_transform.cut_low_frequency =float(df["Frequency_psd"].isel(Frequency_psd= cut_low_freq_arg-1))
     
        preprocess.Frequency_psd =df['Frequency_psd'].where(df['Frequency_psd']>(preprocess.split_transform.cut_low_frequency), drop=True)

        ####
        # Re-arrange direction columns because 0/360 discontinuity do not fit with neural networks.
        ####
        if preprocess.feature_eng.envir_direction_dict is not None :
                # Select the sin_cos_method 
                # The user can define a custom  method by adding a method to this class and then replace the sin_cos_method name to the hydra config file
                if hasattr(preprocess.feature_eng, preprocess.feature_eng.sin_cos_method):
                        log = logging.getLogger(os.environ['logger_name'])
                        log.info(f" run <{preprocess.feature_eng.sin_cos_method}> method")
                        df = getattr(preprocess.feature_eng, preprocess.feature_eng.sin_cos_method ) (preprocess.feature_eng.envir_direction_dict, df)
                        for magnitude, angle in preprocess.feature_eng.envir_direction_dict.items() :
                                preprocess.inputs_outputs.envir_variables.remove(angle)
                                preprocess.inputs_outputs.envir_variables.append(f'{magnitude}_cos')
                                preprocess.inputs_outputs.envir_variables.append(f'{magnitude}_sin')
        
        ####
        # Split data into train and test sets. 
        ####
        X_train, X_test, Y_train, Y_test = preprocess.split_transform.process_data(df=df, 
                                                                        X_channel_list=preprocess.inputs_outputs.envir_variables,
                                                                        Y_channel_list=preprocess.inputs_outputs.neuron_variables,
                                                                        df_train_set_envir_filename=cfg.paths.training_env_dataset)
        ####
        # Scale input data with scaler defined in hydra config file
        ####
        x_train, x_test  = preprocess.input_scaler.scale_data(X_train, X_test)

        ####
        # Decompose y data with decomposition methode defined in hydra config file
        #### 
        if preprocess.perform_decomp :
                Y_train, Y_test = preprocess.decomp_y_spectrum.decomp_data(Y_train, Y_test)

        ####
        # Scale Y spectrum data with scaler defined in hydra config file
        ####
        y_train, y_test = preprocess.output_scaler.scale_data(Y_train, Y_test)

        ####
        # Shuffle training data
        ####
        x_train, y_train = shuffle(x_train, y_train)


        log.info(f'x_train shape: {np.shape(x_train)}')
        log.info(f'y_train shape: {np.shape(y_train)}')

        return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    main()
import xarray as xr
import numpy as np
import pickle
import hydra
from omegaconf import DictConfig
import logging
import yaml
import sys

from sklearn.utils import shuffle
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger
from torch.nn import Module
from torch import set_float32_matmul_precision, compile, device, cuda
from typing import List
from torchydra_fem.utils.load_env_file import load_env_file
import os

from torchydra_fem.Preprocessing import Preprocessing
from torchydra_fem.model.surrogate_module import SurrogateModule
import torchydra_fem.utils as utils



# version_base=1.1 is used to make hydra change the current working directory to the hydra output path
@hydra.main(config_path="../../../../configs", config_name="monabiop_conv1d.yaml", version_base="1.3")
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
        utils.load_env_file(f"{hydra.utils.get_original_cwd()}/env.yaml")
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

        # y_output_size, two_dims_decomp_length, two_dims_channel_nb = np.shape(y_train)[1], np.shape(x_train)[1], np.shape(x_train)[2]
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
        datamodule.setup(stage='train', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        # instanciate model. Parameters depending on dataset are passed as kwargs.
        kwargs = {
                        "nb_obs" : datamodule.shape_x[0],
                        "two_dims_decomp_length" : datamodule.shape_x[1],
                        "two_dims_channel_nb" : datamodule.shape_x[2],}
        
        # save kwargs to hydra config
        # add kwargs to hydra config
        # cfg.model_net = DictConfig(cfg.model_net).update(kwargs)
        # # save hydra config to disk
        # hydra_config_path = os.path.join(cfg.paths.output_dir, '.hydra/config.yaml' )
        # with open(hydra_config_path, 'w') as f:
        #         yaml.dump(cfg, f)
        
        log.info(f"Importing model net {cfg.model_net._target_}")
        # can be passed as *args because all arguments are defined above, no argument defined in .yaml config file.
        model_net : Module = hydra.utils.instantiate(cfg.model_net, **kwargs)
        
        log.info(f"Importing model  {cfg.model._target_}")
        model : SurrogateModule = hydra.utils.instantiate(cfg.model)
        # model.net cannot be instanciated in the config file because it depends on the dataset:
        d= device('cuda' if cuda.is_available() else 'cpu')
        model.net = model_net
        model = model.to(d)

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
        
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        set_float32_matmul_precision('medium')
        
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
                utils.log_hyperparameters(object_dict)

        if cfg.get("compile"):
                log.info("Compiling model!")
                model = compile(model)

        if cfg.get("train"):
                log.info("Starting training!")
                trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        train_metrics = trainer.callback_metrics

        log.info("Training finished!")

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
        variable_list = preprocess.inputs_outputs.input_variables + preprocess.inputs_outputs.output_variables
        coordinate_list = list(df.coords)

        not_drop_list = coordinate_list + variable_list
        
        # drop all variables not in variable_list
        df = df.drop_vars([ var for var in df.variables if var not in not_drop_list] )
        df = df.dropna(dim='time', how='any')

        preprocess.unit_dictionnary = {}
        for var in preprocess.inputs_outputs.input_variables :
                preprocess.unit_dictionnary[var] = df[var].attrs['unit']
        for var in preprocess.inputs_outputs.output_variables :
                preprocess.unit_dictionnary[var] = 'kN'
        
        ####
        # Split data into train and test sets. 
        ####
        X_train, X_test, Y_train, Y_test = preprocess.split_transform.process_data(df=df, 
                                                                        X_channel_list=preprocess.inputs_outputs.input_variables,
                                                                        Y_channel_list=preprocess.inputs_outputs.output_variables,
                                                                        df_train_set_envir_filename=cfg.paths.training_env_dataset)
        
        # drop nan on training & test sets


        ####
        # Scale 1D output data with scaler defined in hydra config file
        ####
        if preprocess.output_scaler.donot_scale : 
                y_train = Y_train
                y_test = Y_test

        else: 
                y_train, y_test  = preprocess.output_scaler.scale_data(Y_train, Y_test)

        ####
        # Decompose x data with decomposition methode defined in hydra config file
        #### 
        if preprocess.perform_decomp :
                x_train, x_test = preprocess.decomp_y_spectrum.decomp_data(X_train, X_test)

        ####
        # Scale 2D data with scaler defined in hydra config file
        ####
        x_train, x_test = preprocess.input_scaler.scale_data(X_train, X_test)

        ####
        # Shuffle training data
        ####
        x_train, y_train = shuffle(x_train, y_train)

        os.environ['logger_name'] = cfg.task_name
        log = logging.getLogger(os.environ['logger_name'])

        log.info(f'x_train shape: {np.shape(x_train)}')
        log.info(f'y_train shape: {np.shape(y_train)}')

        return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    # get args from command line
    main()
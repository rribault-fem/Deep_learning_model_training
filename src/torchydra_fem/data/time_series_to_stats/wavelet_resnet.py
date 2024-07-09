from typing import Any, Dict, Optional, Tuple
import os
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset 
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import numpy as np
import logging
from torchydra_fem.preprocessing.wavelet_computation import compute_wavelet_coeff_for_CNN

from torchydra_fem.data.surrogate_time_series import SurrogateDataModule


class WaveletResnet(SurrogateDataModule):
    """A subclass of LightningDataModule and Surrogate module .

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    
    def setup(self,
            stage: str = None,
            x_train: np.array = None,
            y_train: np.array = None,
            x_test: np.array = None,
            y_test: np.array = None) :
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
    
        if stage == "evaluate" :
            self.x_eval = self._shift_reshape_data(x_test)

        if stage == "test" :
            self.x_test, self.y_test = self._shift_reshape_data(x_test, y_test)

        if stage == "train" or stage == "fit" and stage is None and x_train and y_train and x_test and y_test:

            # load and split datasets only if not loaded already
            if not self.data_train and not self.data_val and not self.data_test:
                
                # x_train, y_train = self._shift_reshape_data(x_train, y_train)
                x_train = self._calculate_wavelet(x_train)
                self.data_train = TensorDataset(x_train, y_train)

                # x_test, y_test = self._shift_reshape_data(x_test, y_test)
                x_test = self._calculate_wavelet(x_test)
                self.data_val = TensorDataset(x_test, y_test)

                self.data_test = ConcatDataset(datasets=[self.data_train, self.data_val])

    def _calculate_wavelet(self, x : np.array) -> np.array:
        l2 = [np.around(1/p,6) for p in np.arange(2, 16, 0.1)][::-1] # Wave frequencies including rotor frequencies
        l3 = [np.around(i,6) for i in np.arange(0.52,4.01,0.1)] # High Frequencies  (0.52 not to have duplicate 0.5 and 4.01 to include 4.0)
        wavelet_scales = l2 + l3
        waveletname = 'morl'

        x = compute_wavelet_coeff_for_CNN(x, wavelet_scales,  waveletname)
    
        return x
    
    def _undo_shift_reshape_data(self, x: torch.Tensor = None ) -> torch.Tensor:
        x = x.reshape(-1, 36000-abs(self.step_diff_y_x), np.shape(x)[-1])
        if self.step_diff_y_x !=0:
            x_add = np.zeros((np.shape(x)[0], abs(self.step_diff_y_x), np.shape(x)[-1]))
            x = np.concatenate((x, x_add), axis=1)
        return x


    def _shift_data(self, x: np.array, y: np.array = None )-> TensorDataset:
        "specific for Monamoor data to shift time discrepancy and reshape tensors"
        log =  logging.getLogger(os.environ['logger_name'])
        log.info(f"Adjust x and y for time discrepancy of config : step_diff_y_x: {self.step_diff_y_x}")

        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.step_diff_y_x > 0:
            x = x[:, :-self.step_diff_y_x, :]
            if y is not None: y = y[:, self.step_diff_y_x:, :]
        
        if self.step_diff_y_x <0:
            x = x[:, -self.step_diff_y_x:, :]

            if y is not None: y = y[:, :self.step_diff_y_x, :]

        return x, y

    def reshape_sample_length(self, x:np.array, y:np.array=None):
        
        log =  logging.getLogger(os.environ['logger_name'])
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.from_numpy(x).float()
        x.to(device)
        x = x.reshape(-1, self.reshape_length, np.shape(x)[-1])
        log.info(f"x shape: {np.shape(x)}")
        self.shape_x = np.shape(x)
    
        if y is not None:
            y = y.reshape(-1, self.reshape_length, np.shape(y)[-1])
            return x, y

        else : return x
        
    def compute_y_stats(self, y:np.array):
            
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        new_y = np.zeros((np.shape(y)[0], 4 , np.shape(y)[-1]))
        new_y[:, 0, :] = np.max(y, axis=1)
        new_y[:, 1, :] = np.min(y, axis=1)
        new_y[:, 2, :] = np.mean(y, axis=1)
        new_y[:, 3, :] = np.std(y, axis=1)
        y = torch.from_numpy(new_y).float()
        y.to(device)
        log =  logging.getLogger(os.environ['logger_name'])
        log.info(f"y shape: {np.shape(y)}")
        return y


if __name__ == "__main__":
    _ = WaveletResnet(reshape_length = 600,
        step_diff_y_x = 0)

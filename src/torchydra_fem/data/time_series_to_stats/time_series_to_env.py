from typing import Any, Dict, Optional, Tuple
import os
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset 
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import logging

from torchydra_fem.data.surrogate_time_series import SurrogateDataModule


class TimeToEnv(SurrogateDataModule):
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
     
    def _undo_shift_reshape_data(self, x: torch.Tensor = None ) -> torch.Tensor:
        "function to apply after model prediction to undo the shift and reshape tensors"
        
        x = x.reshape(-1, 36000-abs(self.step_diff_y_x), np.shape(x)[-1])
        if self.step_diff_y_x !=0:
            x_add = np.zeros((np.shape(x)[0], abs(self.step_diff_y_x), np.shape(x)[-1]))
            x = np.concatenate((x, x_add), axis=1)

        return x

    def _shift_reshape_data(self, x: np.array, y: np.array = None )-> TensorDataset:
        "specific for Environment data reshape tensors"
        log =  logging.getLogger(os.environ['logger_name'])
        log.info(f"Adjust x and y for time discrepancy of config : step_diff_y_x: {self.step_diff_y_x}")
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = torch.from_numpy(x).float()
        x.to(device)
        x = x.reshape(-1, self.reshape_length, np.shape(x)[-1])
        log.info(f"x shape: {np.shape(x)}")
        self.shape_x = np.shape(x)

        if y is not None:
            y = np.repeat(y, int(18000/self.reshape_length), axis=0)
            y = torch.from_numpy(y).float()
            y = y.unsqueeze(1)  # Add a new dimension at index 1
            y.to(device)
            # y = y.reshape(-1, self.reshape_length, np.shape(y)[-1])
            log.info(f"x shape: {np.shape(y)}")
            return (x, y)

        else : return x


if __name__ == "__main__":
    _ = TimeToEnv(reshape_length = 6000,
        step_diff_y_x = 0)

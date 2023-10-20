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


class SurrogateDataModule(LightningDataModule):
    """LightningDataModule for Surrogate model dataset.

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

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        reshape_length: int = 600,
        step_diff_y_x: int = 0, # steps difference beween y and x. positive if y is shifted to the right. Negative if y is shifted to the left
        **kwargs
    ):
        super().__init__()
        self.reshape_length = reshape_length
        self.step_diff_y_x = step_diff_y_x

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.data_test = None
        self.data_train = None
        self.data_val = None

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
            self.x_val, self.y_val = self._shift_reshape_data(x_test, y_test)

        if stage == "train" or stage == "fit" and stage is None and x_train and y_train and x_test and y_test:

            # load and split datasets only if not loaded already
            if not self.data_train and not self.data_val and not self.data_test:
                
                x_train, y_train = self._shift_reshape_data(x_train, y_train)
                self.data_train = TensorDataset(x_train, y_train)

                x_test, y_test = self._shift_reshape_data(x_test, y_test)
                self.data_val = TensorDataset(x_test, y_test)

                self.data_test = ConcatDataset(datasets=[self.data_train, self.data_val])
     
    def _undo_shift_reshape_data(self, x: torch.Tensor = None ) -> torch.Tensor:
        "function to apply after model prediction to undo the shift and reshape tensors"
        
        x = x.reshape(-1, 36000-abs(self.step_diff_y_x), np.shape(x)[-1])
        if self.step_diff_y_x !=0:
            x_add = np.zeros((np.shape(x)[0], abs(self.step_diff_y_x), np.shape(x)[-1]))
            x = np.concatenate((x, x_add), axis=1)

        return x


    def _shift_reshape_data(self, x: np.array, y: np.array = None )-> TensorDataset:
        "specific for Monamoor data to shift time discrepancy and reshape tensors"
        log =  logging.getLogger(os.environ['logger_name'])
        log.info(f"Adjust x and y for time discrepancy of config : step_diff_y_x: {self.step_diff_y_x}")

        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.step_diff_y_x > 0:
            x = x[:, :-self.step_diff_y_x, :]
            if y is not None: y = y[:, self.step_diff_y_x:, :]
        
        elif self.step_diff_y_x <0:
            x = x[:, -self.step_diff_y_x:, :]

            if y is not None: y = y[:, :self.step_diff_y_x, :]
        
        elif self.step_diff_y_x == 0:
            pass

        x = torch.from_numpy(x).float()
        x.to(device)
        x = x.reshape(-1, self.reshape_length, np.shape(x)[-1])
        log.info(f"x shape: {np.shape(x)}")
        self.shape_x = np.shape(x)
    
        if y is not None:
            y = torch.from_numpy(y).float()
            y.to(device)
            y = y.reshape(-1, self.reshape_length, np.shape(y)[-1])
            log.info(f"x shape: {np.shape(y)}")
            return (x, y)
    
        else : return x

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = SurrogateDataModule()

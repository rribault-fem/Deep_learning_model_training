from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset 
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np


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
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

        required_kwargs_list = ['x_train', 'y_train', 'x_test', 'y_test']

        for kwarg in required_kwargs_list:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required kwarg: {kwarg}")       

        x_train_tensor = torch.from_numpy(kwargs['x_train']).float()
        y_train_tensor = torch.from_numpy(kwargs['y_train']).float()
        x_test_tensor = torch.from_numpy(kwargs['x_test']).float()
        y_test_tensor = torch.from_numpy(kwargs['y_test']).float()
        self.data_train = TensorDataset(x_train_tensor, y_train_tensor)
        self.data_val = TensorDataset(x_test_tensor, y_test_tensor)
        self.data_test = ConcatDataset(datasets=[self.data_train, self.data_val])


    def setup(self,
            stage: Optional[str] = None,
            x_train: Optional[np.array] = None,
            y_train: Optional[np.array] = None,
            x_test: Optional[np.array] = None,
            y_test: Optional[np.array] = None) :
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # load data
            x_train_tensor = torch.from_numpy(x_train).float()
            y_train_tensor = torch.from_numpy(y_train).float()
            x_test_tensor = torch.from_numpy(x_test).float()
            y_test_tensor = torch.from_numpy(y_test).float()
            self.data_train = TensorDataset(x_train_tensor, y_train_tensor)
            self.data_val = TensorDataset(x_test_tensor, y_test_tensor)
            self.data_test = ConcatDataset(datasets=[self.data_train, self.data_val])

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

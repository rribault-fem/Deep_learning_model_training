from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MeanSquaredError, MaxMetric, MeanMetric, MinMetric, MeanAbsolutePercentageError
import torchmetrics


class SurrogateModule(LightningModule):
    """Example of LightningModule for generation of Neuron Acceleration from environment.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        net: torch.nn.Module = None,
        criterion = torch.nn.MSELoss(),
        train_loss = torchmetrics.MeanSquaredError(),
        val_loss = torchmetrics.MeanSquaredError(),
        test_loss = torchmetrics.MeanSquaredError(),
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net', 'val_loss', 'train_loss', 'criterion', 'test_loss'])

        self.net = net
        
        self.criterion = criterion

        # # metric objects for calculating and averaging accuracy across batches
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # # for tracking best so far validation accuracy
        self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net.forward(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_mse_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.net.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss.update(loss)
        # self.train_mse(preds, targets)

        # log metrics for the logger (default with tensorboard)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/mse", self.train_mse, on_step=True, on_epoch=True, prog_bar=True)
        

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss.update(loss)
        # self.val_mse(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('hp_metric', self.val_loss)

    def on_validation_epoch_end(self):
        pass
        # acc = self.val_mse.compute()  # get current val acc
        # self.val_mse_best(acc)  # update best so far val acc
        # # log `val_mse_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/mse_best", self.val_mse_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SurrogateModule(None, None, None)

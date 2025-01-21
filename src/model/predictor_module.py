from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import logging
import os
import torch
from torchmetrics import MeanMetric, SpearmanCorrCoef

from src.model.predictors import BaseCNN


class my_predictor_module(LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        self._cfg = model_cfg
        self.predictor = BaseCNN()
        self.optimizer = torch.optim.Adam(
            params=self.predictor.parameters(),
            **self._cfg.optimizer,
        )

        self.criterion = torch.nn.MSELoss()
        self.train_loss = MeanMetric()
        self.train_sr = SpearmanCorrCoef()

    def forward(self, x):
        return self.predictor(x)
    
    def training_step(self, batch, batch_idx):
        features, targets = batch
        # targets = targets.float()
        pred = self.forward(features)
        # pred = pred.float()

        loss = self.criterion(targets, pred)
        
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_sr(pred, targets)
        self.log("train_sr", self.train_sr, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    
    # def on_train_epoch_end(self):
    #     self.log("train_loss_epoch", self.train_loss.compute(), prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer
from __future__ import annotations

import copy
import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset


class DotDict(dict):
    def __init__(self, d=None):
        super().__init__()
        if d:
            for k, v in d.items():
                self[k] = self._wrap(v)

    def _wrap(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, DotDict) else x for x in v]
            else:
                out[k] = v
        return out

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(self.to_dict(), memo))


class Anscombe(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2.0 * torch.sqrt(x + 3.0 / 8.0)

    def inv(self, x):
        return (x / 2.0) ** 2 - 3.0 / 8.0


class PositionalEncoding(nn.Module):
    def __init__(self, n_bins, n_positional):
        super().__init__()
        self.encoding = nn.Parameter(
            torch.empty(1, n_bins, n_positional)
        )
        nn.init.normal_(self.encoding, mean=0.0, std=0.02)

    def forward(self, batch_size):
        return self.encoding.expand(batch_size, -1, -1)


class Model(nn.Module):
    def __init__(self, Conf=None, **kwargs):
        super().__init__()

        self.Conf = copy.deepcopy(Conf)

        model_conf = Conf.model_type[Conf.model_type.name]

        self.n_channels = Conf.data.n_channels
        self.n_pixels = Conf.data.n_pixels
        self.n_bins = Conf.data.n_bins
        self.n_days = Conf.data.n_days
        self.n_electrodes_list = Conf.data.n_electrodes_list

        self.cnn_n_hidden = model_conf.cnn_n_hidden
        self.cnn_n_layers = model_conf.cnn_n_layers
        self.n_latent = model_conf.n_latent
        self.dropout = model_conf.dropout

        self.cnn_n_out = self.n_channels if self.cnn_n_layers == 0 else self.cnn_n_hidden * 2 ** (self.cnn_n_layers - 1)
        self.cnn_height = self.n_pixels if self.cnn_n_layers == 0 else self.n_pixels // 2 ** self.cnn_n_layers
        self.cnn_width = self.n_pixels if self.cnn_n_layers == 0 else self.n_pixels // 2 ** self.cnn_n_layers

        cnn_layers = []

        for layer_idx in range(self.cnn_n_layers):
            in_channels = self.n_channels if layer_idx == 0 else self.cnn_n_hidden * 2 ** (layer_idx - 1)
            out_channels = self.cnn_n_hidden * 2 ** layer_idx

            cnn_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]

        self.cnn = nn.Sequential(*cnn_layers)

        self.latent_channel = nn.Parameter(torch.empty(self.n_latent, self.cnn_n_out))
        self.latent_height = nn.Parameter(torch.empty(self.n_latent, self.cnn_height))
        self.latent_width = nn.Parameter(torch.empty(self.n_latent, self.cnn_width))
        self.latent_bias = nn.Parameter(torch.zeros(self.n_latent))

        self.latent_activation = nn.GELU()
        self.latent_dropout = nn.Dropout(self.dropout)

        self.latent_time = nn.Parameter(torch.empty(self.n_latent, self.n_bins))


        self.latent_electrode_list = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_latent, n_electrodes))
            for n_electrodes in self.n_electrodes_list
        ])
        
        self.output_bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_bins, n_electrodes))
            for n_electrodes in self.n_electrodes_list
        ])

        nn.init.normal_(self.latent_channel, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_height, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_width, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_time, mean=0.0, std=0.02)
        for latent_electrode in self.latent_electrode_list:
            nn.init.normal_(latent_electrode, mean=0.0, std=0.02)

    def forward(self, x, day_idx):
        cnn_features = self.cnn(x)

        latent_filter = (
            self.latent_channel[:, :, None, None]
            * self.latent_height[:, None, :, None]
            * self.latent_width[:, None, None, :]
        )

        latent_features = torch.einsum(
            "bchw,kchw->bk",
            cnn_features,
            latent_filter,
        )

        latent_features = latent_features + self.latent_bias
        latent_features = self.latent_activation(latent_features)
        latent_features = self.latent_dropout(latent_features)

        latent_time = (
            latent_features[:, :, None]
            * self.latent_time[None, :, :]
        )

        y_hat = torch.einsum(
            "bkt,ke->bte",
            latent_time,
            self.latent_electrode_list[day_idx],
        )
        
        y_hat = y_hat + self.output_bias_list[day_idx]

        return y_hat


class History(L.Callback):
    def __init__(self):
        self.train_loss = []
        self.test_loss = []
        self.lr = []
        self.current_train_loss = None
        self.current_lr = None

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.current_train_loss = metrics["train_loss"].item()
        self.current_lr = trainer.optimizers[0].param_groups[0]["lr"]

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        self.train_loss.append(self.current_train_loss)
        self.test_loss.append(metrics["test_loss"].item())
        self.lr.append(self.current_lr)


class LitModel(L.LightningModule):
    def __init__(self, Conf, model):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer_type", "scheduler_type", "data", "model_params"]) 
        self.Conf = Conf      
        self.optimizer_params = Conf.optimization.Adam
        self.scheduler_params = Conf.optimization.Reduce
        self.model = Model(Conf)
        self.mse_loss = nn.MSELoss()
        self.anscombe = Anscombe()

    def forward(self, x):
        return self.full_model(x)

    def training_step(self, batch):
        losses = []
    
        for day_idx, day_batch in batch.items():
            x, y = day_batch
            y_hat = self.model(x, int(day_idx))
            losses.append(self.mse_loss(y_hat, y))
    
        loss = torch.stack(losses).mean()
    
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
        return loss

    def validation_step(self, batch):
        losses = []
    
        for day_idx, day_batch in batch.items():
            x, y = day_batch
            y_hat = self.model(x, int(day_idx))
            losses.append(self.mse_loss(y_hat, y))
    
        loss = torch.stack(losses).mean()
    
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_params.factor,
            patience=self.scheduler_params.patience,
            min_lr=self.scheduler_params.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "test_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


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
        self.n_bins = Conf.data.n_bins
        self.n_electrodes = Conf.data.n_electrodes

        self.cnn_n_hidden = model_conf.cnn_n_hidden
        self.cnn_n_layers = model_conf.cnn_n_layers
        self.positional_n_hidden = model_conf.positional_n_hidden
        self.transformer_n_hidden = model_conf.transformer_n_hidden
        self.transformer_n_heads = model_conf.transformer_n_heads
        self.transformer_n_layers = model_conf.transformer_n_layers
        self.dropout = model_conf.dropout
        self.transformer_nonlinearity = model_conf.transformer_nonlinearity

        self.cnn_n_out = self.cnn_n_hidden * 2 ** (self.cnn_n_layers - 1)
        self.image_n_hidden = self.transformer_n_hidden - self.positional_n_hidden

        assert self.transformer_n_hidden % self.transformer_n_heads == 0

        cnn_layers = []

        for layer_idx in range(self.cnn_n_layers):
            in_channels = self.n_channels if layer_idx == 0 else self.cnn_n_hidden * 2 ** (layer_idx - 1)
            out_channels = self.cnn_n_hidden * 2 ** layer_idx

            cnn_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]

        cnn_layers += [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]

        self.cnn = nn.Sequential(*cnn_layers)

        self.image_proj = nn.Sequential(
            nn.Linear(self.cnn_n_out, self.image_n_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.positional_encoding = PositionalEncoding(
            n_bins=self.n_bins,
            n_positional=self.positional_n_hidden,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_n_hidden,
            nhead=self.transformer_n_heads,
            dim_feedforward=self.transformer_n_hidden * 4,
            dropout=self.dropout,
            activation=self.transformer_nonlinearity,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_n_layers)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.output_proj = nn.Linear(self.transformer_n_hidden, self.n_electrodes)

    def forward(self, x):
        image_features = self.cnn(x)
        image_features = self.image_proj(image_features)

        image_features = image_features.unsqueeze(1).expand(-1, self.n_bins, -1)

        positional_features = self.positional_encoding(batch_size=x.shape[0])

        transformer_input = torch.cat([image_features, positional_features], dim=-1)

        transformed = self.transformer_encoder(transformer_input)

        transformed = self.dropout_layer(transformed)

        y_hat = self.output_proj(transformed)

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
        x, y = batch
        y_hat = self.model(x)
        loss = self.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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


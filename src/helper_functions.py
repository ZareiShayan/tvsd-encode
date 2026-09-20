from __future__ import annotations

import copy
import os
import random
import pickle
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    DeepLift,
    GuidedBackprop,
    Occlusion,
    GradientShap,
    NoiseTunnel,
)

from src.classes import *


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)


def prepare_dataset(Conf, images, spikes, train_idx, test_idx):

    seed_everything(Conf.seed)

    train_mask = train_idx > 0
    test_mask = test_idx > 0

    images_train = images[train_mask]
    images_test = images[test_mask]
    
    spikes_train = spikes[train_mask]
    spikes_test = spikes[test_mask]

    images_train = torch.as_tensor(images_train, dtype=torch.uint8).permute(0, 3, 1, 2)
    images_test = torch.as_tensor(images_test, dtype=torch.uint8).permute(0, 3, 1, 2)

    spikes_train = torch.as_tensor(spikes_train, dtype=torch.float32).permute(0, 2, 1)
    spikes_test = torch.as_tensor(spikes_test, dtype=torch.float32).permute(0, 2, 1)

    Y_mean = spikes_train.mean(dim=0, keepdim=True)
    Y_std = spikes_train.std(dim=0, keepdim=True)


    Y_train = (spikes_train - Y_mean) / (Y_std + 1e-4)
    Y_test = (spikes_test - Y_mean) / (Y_std + 1e-4)

    train_dataset = TensorDataset(images_train, Y_train)
    test_dataset = TensorDataset(images_test, Y_test)

    return train_dataset, test_dataset, Y_mean, Y_std


def prepare_loader(Conf, train_dataset, test_dataset):

    seed_everything(Conf.seed)

    train_loader = DataLoader(train_dataset, batch_size=Conf.training.batch_size, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=Conf.training.batch_size, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, test_loader


def build_lit_model(Conf, model, enable_progress_bar_epoch):

    seed_everything(Conf.seed)

    early_stop = EarlyStopping(
        monitor="test_loss",
        mode="min",
        min_delta=Conf.training.min_delta,
        patience=Conf.training.patience,
    )

    history = History()
    
    trainer = L.Trainer(
        max_epochs=Conf.training.max_epoch,
        accelerator="gpu" if Conf.device.type == "cuda" else "cpu",
        devices=1,
        precision="16-mixed",
        deterministic=True,
        num_sanity_val_steps=0,
        logger=False,
        callbacks=[early_stop, history],
        enable_progress_bar=enable_progress_bar_epoch,
        enable_checkpointing=False,
        enable_model_summary=False,
        check_val_every_n_epoch=1,
    )
    
    lit_model = LitModel(Conf, model).to(Conf.device)
    trainer.history = history
    
    return trainer, lit_model


def predict_loader(
    Conf,
    lit_model,
    loader_list,
    Y_mean_list,
    Y_std_list,
):
    device = Conf.device

    lit_model = lit_model.to(device)
    lit_model.eval()

    Y_list = []
    Y_hat_list = []

    with torch.inference_mode():
        for day_idx, loader in enumerate(loader_list):
            Y_mean = Y_mean_list[day_idx].to(device, non_blocking=True)
            Y_std = Y_std_list[day_idx].to(device, non_blocking=True)

            Y = []
            Y_hat = []

            for x, y in loader:
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_hat = lit_model.model(x, day_idx)

                Y.append(y * Y_std + Y_mean)
                Y_hat.append(y_hat * Y_std + Y_mean)

            Y_list.append(torch.cat(Y, dim=0).cpu().numpy())
            Y_hat_list.append(torch.cat(Y_hat, dim=0).cpu().numpy())

    return Y_list, Y_hat_list


def compute_correlation(y, y_hat):

    if np.std(y) == 0 or np.std(y_hat) == 0:
        return 0.0

    return float(np.corrcoef(y, y_hat)[0, 1])

def compute_r2(y, y_hat):

    denominator = np.sum((y - np.mean(y)) ** 2)

    if denominator <= 1e-3:
        return np.nan

    return float(
        1.0 - np.sum((y - y_hat) ** 2) / denominator
    )

def compute_mse(y, y_hat):

    return float(np.mean((y - y_hat) ** 2))

def compute_metrics(
    Y_list,
    Y_hat_list,
):
    correlation_list = []
    r2_list = []
    mse_list = []

    for Y, Y_hat in zip(Y_list, Y_hat_list):
        n_electrodes = Y.shape[2]
        n_bins = Y.shape[1]

        correlation = np.full(
            (n_electrodes, n_bins),
            np.nan,
            dtype=float,
        )

        r2 = np.full(
            (n_electrodes, n_bins),
            np.nan,
            dtype=float,
        )

        mse = np.full(
            (n_electrodes, n_bins),
            np.nan,
            dtype=float,
        )

        for electrode_idx in range(n_electrodes):
            for bin_idx in range(n_bins):
                y = Y[:, bin_idx, electrode_idx]
                y_hat = Y_hat[:, bin_idx, electrode_idx]

                correlation[electrode_idx, bin_idx] = compute_correlation(
                    y,
                    y_hat,
                )

                r2[electrode_idx, bin_idx] = compute_r2(
                    y,
                    y_hat,
                )

                mse[electrode_idx, bin_idx] = compute_mse(
                    y,
                    y_hat,
                )

        correlation_list.append(correlation)
        r2_list.append(r2)
        mse_list.append(mse)

    correlation = np.concatenate(
        correlation_list,
        axis=0,
    )

    r2 = np.concatenate(
        r2_list,
        axis=0,
    )

    mse = np.concatenate(
        mse_list,
        axis=0,
    )

    return correlation, r2, mse


def compute_attribution_map(
    Conf,
    lit_model,
    explain_sample,
    day_idx,
    electrode_idx,
    bin_idx,
    method="integrated_gradients",
    baseline=None,
    smooth=False,
    nt_samples=10,
    nt_stdevs=0.1,
    **kwargs,
):
    seed_everything(Conf.seed)

    device = Conf.device

    model = lit_model.model.to(device).eval()

    explain_x = explain_sample[0].to(device).unsqueeze(0)
    
    if baseline is None:
        baseline = torch.zeros_like(explain_x)
    else:
        baseline = baseline.to(
            device=device,
            dtype=explain_x.dtype,
        )

    def forward_fn(x):
        x = x.to(torch.float32)
        return model(x, day_idx)[:, bin_idx, electrode_idx]

    methods = {
        "integrated_gradients": IntegratedGradients,
        "saliency": Saliency,
        "input_x_gradient": InputXGradient,
        "deeplift": DeepLift,
        "guided_backprop": GuidedBackprop,
        "occlusion": Occlusion,
        "gradient_shap": GradientShap,
    }

    attributor = methods[method](forward_fn)

    if smooth:
        attributor = NoiseTunnel(attributor)

    attribute_kwargs = dict(kwargs)

    if method in ["integrated_gradients", "deeplift", "occlusion", "gradient_shap"]:
        attribute_kwargs["baselines"] = baseline

    if smooth:
        attribute_kwargs["nt_samples"] = nt_samples
        attribute_kwargs["stdevs"] = nt_stdevs
        

    attribution = attributor.attribute(
        explain_x,
        **attribute_kwargs,
    )

    attribution_map = attribution.squeeze(0).detach().cpu().numpy()

    return attribution_map






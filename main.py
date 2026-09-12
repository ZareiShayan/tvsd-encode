# %% [markdown]
# # Neural Encoder Interpretation

# %% [markdown]
# ## Imports and Setup

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

%load_ext autoreload
%autoreload 2

import h5py
import numpy as np
import pandas as pd
import torch

from src.classes import *
from src.helper_functions import *
from src.plot_functions import *

SEED = 1
seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Configuration

# %%
Conf = DotDict({
    "run_id": 1,
    "seed": SEED,
    "paths": {
        "data": r"D:\Research\Project\THINGS\data\data.mat",
    },
    "device": device,
    "data": {
        "bin_size": 0.01,
    },
    "training": {
        "batch_size": 16,
        "max_epoch": 100,
        "min_delta": 1e-5,
        "patience": 5,
    },
    "model_type": {
        "name": "cnn_transformer",
        "cnn_transformer": {
            "cnn_hidden": 4,
            "n_hidden": 128,
            "n_positional": 16,
            "n_heads": 8,
            "n_layers": 2,
            "dropout": 0,
            "nonlinearity": "gelu",
        },
    },
    "optimization": {
        "Adam": {
            "lr": 0.0001,
            "weight_decay": 0,
        },
        "Reduce": {
            "factor": 0.5,
            "patience":2,
            "min_lr": 1e-10,
        },
        "entropy_lambda": 0.0,
        "coverage_lambda": 0.0,
    }
})

# %% [markdown]
# ## Data Preparation

# %%
with h5py.File(Conf.paths.data, "r") as f:
    g = f["data"]

    allmat = np.array(g["ALLMAT"]).T
    spikes = np.array(g["ALLMUA"]).transpose(1, 2, 0)
    images = np.array(g["IMAGES"])
    bin_times = np.array(g["tb"]).ravel()
    electrode_names = np.array(g["selectedElectrodes"]).ravel().astype(int)
    mapping = np.array(g["mapping"]).ravel().astype(int)

train_idx = allmat[:, 1].astype(int)
test_idx = allmat[:, 2].astype(int)
rep = allmat[:, 3].astype(int)
count = allmat[:, 4].astype(int)
day = allmat[:, 5].astype(int)


electrode_roi = np.empty(len(electrode_names), dtype="<U2")
electrode_roi[electrode_names <= 512] = "V1"
electrode_roi[(electrode_names >= 513) & (electrode_names <= 832)] = "IT"
electrode_roi[electrode_names >= 833] = "V4"

# %%
trials_to_remove, electrodes_to_remove = plot_spike_summary(
    spikes=spikes,
    file_name="spike-summary-before",
    file_path="./plot/spikes",
    metric="outlier_rate",
    k=5.0,
    electrode_threshold=0.02,
    trial_threshold=0.02,
    electrode_names=electrode_names,
)

for electrode_idx in electrodes_to_remove:
    plot_electrode_spikes(
        electrode_spikes=spikes[:, electrode_idx, :],
        title=f"Electrode {electrode_names[electrode_idx]}",
        file_name=f"spike-activity-electrode-{electrode_idx}",
        file_path="./plot/spikes/electrodes_to_remove/",
        bin_times=bin_times,
    )

# %%
electrodes_to_remove = [172, 355, 462, 723, 731, 755, 767, 852, 863]

clean_trials_mask = np.ones(spikes.shape[0], dtype=bool)
clean_trials_mask[trials_to_remove] = False

spike_trials_mask_finite = np.isfinite(spikes).any(axis=(1, 2))
image_trials_mask_finite = np.isfinite(images).any(axis=(1, 2, 3))
finite_trials_mask = spike_trials_mask_finite & image_trials_mask_finite

keep_trials_mask = clean_trials_mask & finite_trials_mask


clean_electrodes_mask = np.ones(spikes.shape[1], dtype=bool)
clean_electrodes_mask[electrodes_to_remove] = False

nonnegative_electrodes_mask = (spikes >= 0).all(axis=(0, 2))

keep_electrodes_mask = clean_electrodes_mask & nonnegative_electrodes_mask


spikes = spikes[keep_trials_mask][:, keep_electrodes_mask, :]
images = images[keep_trials_mask]
allmat = allmat[keep_trials_mask]

train_idx = train_idx[keep_trials_mask]
test_idx = test_idx[keep_trials_mask]
rep = rep[keep_trials_mask]
count = count[keep_trials_mask]
day = day[keep_trials_mask]

electrode_names = electrode_names[keep_electrodes_mask]
electrode_roi = electrode_roi[keep_electrodes_mask]

n_trials, n_electrodes, n_bins = spikes.shape
n_trials, n_pixels, n_pixels, n_channels = images.shape

# %%
Conf.data.n_trials = n_trials
Conf.data.n_electrodes = n_electrodes
Conf.data.n_bins = len(bin_times)
Conf.data.n_pixels = n_pixels
Conf.data.n_channels = n_channels

# %%
train_dataset, test_dataset, image_mean, image_std, Y_mean, Y_std = prepare_dataset(Conf, images, spikes, train_idx, test_idx)
train_loader, test_loader, loader_generators = prepare_loader(Conf, train_dataset, test_dataset)

# %%
electrode_idx = 0

plot_electrode_spikes(
    electrode_spikes=spikes[train_idx > 0, electrode_idx, :],
    title=f"Electrode {electrode_names[electrode_idx]}",
    file_name=f"spike-activity-electrode-{electrode_idx}-raw",
    file_path=f"./plot/spikes/electrodes/",
    bin_times=bin_times,
)

plot_electrode_spikes(
    electrode_spikes=train_dataset[:][1][:, :, electrode_idx].numpy(),
    title=f"Electrode {electrode_names[electrode_idx]}",
    file_name=f"spike-activity-electrode-{electrode_idx}-prep",
    file_path=f"./plot/spikes/electrodes/",
    bin_times=bin_times,
)

# %%
trial_idx = 56

plot_image(
    image=images[train_idx > 0][trial_idx],
    title=f"Image {trial_idx}",
    file_name=f"image-{trial_idx}-raw",
    file_path="./plot/images/",
    vmin=[0, 0, 0],
    vmax=[255, 255, 255],
)

plot_image(
    image=train_dataset[trial_idx][0].permute(1, 2, 0).numpy(),
    title=f"Image {trial_idx}",
    file_name=f"image-{trial_idx}-prep",
    file_path="./plot/images/",
    vmin=(0 - image_mean.squeeze().numpy()) / image_std.squeeze().numpy(),
    vmax=(255 - image_mean.squeeze().numpy()) / image_std.squeeze().numpy(),
)

# %% [markdown]
# ## Model Training

# %%
trainer, lit_model = build_lit_model(Conf, loader_generators, "cnn_transformer", True)
trainer.fit(lit_model, train_loader, test_loader)

# %%
plot_training_history(
    trainer=trainer,
    title=None,
    file_name="training-history",
    file_path="./plot/model/",
)

# %% [markdown]
# ## Model Evaluation

# %%
Y_train, Y_hat_train = predict_loader(Conf, lit_model, train_loader, Y_mean, Y_std)
Y_test, Y_hat_test = predict_loader(Conf, lit_model, test_loader, Y_mean, Y_std)

# %%
correlation_test, r2_test, mse_test = compute_metrics(Conf, Y_test, Y_hat_test)

# %%
for electrode_idx in range(Conf.data.n_electrodes):
    plot_electrode_metrics(
        correlation_electrode=correlation_test[electrode_idx],
        r2_electrode=r2_test[electrode_idx],
        mse_electrode=mse_test[electrode_idx],
        bin_times=bin_times,
        title=f"Electrode {electrode_idx}",
        file_name=f"electrode_{electrode_idx}_metrics",
        file_path="./plot/model/electrode_metrics",
        show=False,
    )

# %% [markdown]
# ## Model Interpretation

# %%
trial_idx = 56

shap_values = compute_shap_values(
    Conf,
    lit_model,
    test_dataset[trial_idx],
    n_samples=1000,
)

# %%
import torch

trial_idx = 56

device_old = Conf.device
Conf.device = torch.device("cpu")

lit_model = lit_model.cpu()

shap_values = compute_patch_shap_values(
    Conf,
    lit_model,
    test_dataset[trial_idx],
    patch_size=10,
    n_permutations=2000,
    batch_size=16,
)

Conf.device = device_old
lit_model = lit_model.to(device_old).eval()

# %%
trial_idx = 56

shap_values = compute_patch_shap_values(
    Conf,
    lit_model,
    test_dataset[trial_idx],
    patch_size=10,
    n_permutations=5000,
    batch_size=16,
)

# %%
trial_idx = 56

shap_values = compute_patch_perturbation_values(
    Conf,
    lit_model,
    test_dataset[trial_idx],
    patch_size=10,
    n_permutations=10000,
    batch_size=16,
)

# %%
import torch

trial_idx = 56

device_old = Conf.device
Conf.device = torch.device("cpu")

lit_model = lit_model.cpu()

shap_values = compute_patch_perturbation_values(
    Conf,
    lit_model,
    test_dataset[trial_idx],
    patch_size=10,
    n_permutations=3000,
    batch_size=16,
)

Conf.device = device_old
lit_model = lit_model.to(device_old).eval()

# %%
electrode_idx = 7
for bin_idx in range(Conf.data.n_bins):
    plot_image_shap(
        image=test_dataset[trial_idx][0].permute(1, 2, 0).numpy(),
        shap=shap_values[:, :, :, bin_idx, electrode_idx].transpose(1, 2, 0),
        vmax_shap = np.abs(shap_values[:, :, :, :, electrode_idx]).max(),
        vmin_shap = -np.abs(shap_values[:, :, :, :, electrode_idx]).max(),
        title=f"Electrode {electrode_idx}, Bin {bin_times[bin_idx]}",
        file_name=f"electrode_{electrode_idx}_bin_{bin_idx}_shap_values", 
        file_path="./plot/model/shap",
        show=False
    )

# %%
electrode_idx = 8

for bin_idx in range(Conf.data.n_bins):
    plot_image_shap(
        image=test_dataset[trial_idx][0].permute(1, 2, 0).numpy(),
        shap=(shap_values[:, :, :, bin_idx, electrode_idx].transpose(1, 2, 0) - np.mean(shap_values[:, :, :, 0:10, electrode_idx], axis=3).transpose(1, 2, 0))/np.std(shap_values[:, :, :, 0:10, electrode_idx], axis=3).transpose(1, 2, 0),
        vmax_shap = np.abs((shap_values[:, :, :, bin_idx, electrode_idx].transpose(1, 2, 0) - np.mean(shap_values[:, :, :, 0:10, electrode_idx], axis=3).transpose(1, 2, 0))/np.std(shap_values[:, :, :, 0:10, electrode_idx], axis=3).transpose(1, 2, 0)).max(),
        vmin_shap = -np.abs((shap_values[:, :, :, bin_idx, electrode_idx].transpose(1, 2, 0) - np.mean(shap_values[:, :, :, 0:10, electrode_idx], axis=3).transpose(1, 2, 0))/np.std(shap_values[:, :, :, 0:10, electrode_idx], axis=3).transpose(1, 2, 0)).max(),
        title=f"Electrode {electrode_idx}, Bin {bin_times[bin_idx]}",
        file_name=f"electrode_{electrode_idx}_bin_{bin_idx}_shap_values", 
        file_path="./plot/model/shap",
        show=False
    )



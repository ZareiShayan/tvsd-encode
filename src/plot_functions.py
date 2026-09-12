from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch

import sys
from pathlib import Path

def save_figure(fig, file_name, file_path, ext=".png", transparent=False, **savefig_kwargs):
    file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)

    out = file_path / f"{file_name}{ext}"
    if out.exists():
        out.unlink()

    fig.savefig(out, dpi=100, bbox_inches="tight", pad_inches=0.25, transparent=transparent, **savefig_kwargs)
    plt.close(fig)
    return file_path


def plot_electrode_spikes(
    electrode_spikes,
    title,
    file_name,
    file_path,
    bin_times,
    figsize=(4, 6),
    show=False,
):
    
    mean_by_bin = np.nanmean(electrode_spikes, axis=0)
    sem_by_bin = np.nanstd(electrode_spikes, axis=0) / np.sqrt(electrode_spikes.shape[0])
    ci_low = mean_by_bin - 1.96 * sem_by_bin
    ci_high = mean_by_bin + 1.96 * sem_by_bin

    if bin_times[0] >= 0:
        xticks = [
            0,
            bin_times[-1],
        ]
        xticklabels = [
            "0",
            f"{bin_times[-1]:g}",
        ]
    else:
        xticks = [
            bin_times[0],
            0,
            bin_times[-1],
        ]
        xticklabels = [
            f"{bin_times[0]:g}",
            "0",
            f"{bin_times[-1]:g}",
        ]

    fig = plt.figure(figsize=figsize, layout="constrained")

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 1],
        hspace=0.05,
    )

    ax_heatmap = fig.add_subplot(gs[0])

    ax_psth = fig.add_subplot(
        gs[1],
        sharex=ax_heatmap,
    )

    im = ax_heatmap.imshow(
        electrode_spikes,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap='Greys',
        extent=[
            bin_times[0],
            bin_times[-1],
            0,
            electrode_spikes.shape[0],
        ],
    )

    ax_heatmap.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_heatmap.set_ylabel(
        "Trial",
    )

    ax_heatmap.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_heatmap.spines[["top", "right"]].set_visible(False)

    ax_psth.fill_between(
        bin_times,
        ci_low,
        ci_high,
        color="black",
        alpha=0.25,
    )

    ax_psth.plot(
        bin_times,
        mean_by_bin,
        color="black",
        linewidth=2,
    )

    ax_psth.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_psth.set(
        xlabel="Time from press onset (s)",
        ylabel="Mean\nspikes per s",
        xticks=xticks,
        xticklabels=xticklabels,
    )

    ax_psth.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax_heatmap,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Spikes per s",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_spike_summary(
    spikes,
    file_name,
    file_path,
    title=None,
    metric="mean",
    k=4.0,
    electrode_threshold=None,
    trial_threshold=None,
    electrode_names=None,
    trial_names=None,
    cmap="viridis",
    figsize=(8, 5),
    show=False,
):
    to_numpy = lambda x: (
        x.detach().cpu().numpy()
        if torch.is_tensor(x)
        else np.asarray(x)
    )

    metric_functions = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
        "std": np.nanstd,
        "var": np.nanvar,
        "sum": np.nansum,
    }

    if metric not in [*metric_functions.keys(), "outlier_rate"]:
        raise ValueError(
            "metric must be one of: "
            f"{[*metric_functions.keys(), 'outlier_rate']}."
        )

    def parse_threshold(threshold):
        if threshold is None:
            return None, None

        if np.isscalar(threshold):
            return None, float(threshold)

        threshold = tuple(threshold)

        if len(threshold) != 2:
            raise ValueError(
                "threshold must be None, a scalar, or (low, high)."
            )

        low, high = threshold

        low = None if low is None else float(low)
        high = None if high is None else float(high)

        return low, high

    spikes = to_numpy(spikes).astype(float)

    if spikes.ndim != 3:
        raise ValueError(
            "spikes must have shape (n_trials, n_electrodes, n_bins). "
            f"Received shape {spikes.shape}."
        )

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    n_trials, n_electrodes, n_bins = spikes.shape

    if electrode_names is None:
        electrode_names = np.arange(n_electrodes)

    if trial_names is None:
        trial_names = np.arange(n_trials)

    electrode_names = np.asarray(electrode_names)
    trial_names = np.asarray(trial_names)

    if electrode_names.size != n_electrodes:
        raise ValueError("electrode_names must contain one name per electrode.")

    if trial_names.size != n_trials:
        raise ValueError("trial_names must contain one name per trial.")

    if metric == "outlier_rate":
        median = np.nanmedian(
            spikes,
            axis=0,
            keepdims=True,
        )

        mad = np.nanmedian(
            np.abs(spikes - median),
            axis=0,
            keepdims=True,
        )

        robust_scale = np.maximum(
            1.4826 * mad,
            1.0,
        )

        flagged = spikes > median + k * robust_scale

        summary = np.nanmean(
            flagged,
            axis=2,
        )

        electrode_score = np.nanmean(
            flagged,
            axis=(0, 2),
        )

        trial_score = np.nanmean(
            flagged,
            axis=(1, 2),
        )

        metric_name = f"Outlier rate"
        colorbar_label = "Proportion of flagged bins"

    else:
        metric_function = metric_functions[metric]

        summary = metric_function(
            spikes,
            axis=2,
        )

        electrode_score = metric_function(
            spikes,
            axis=(0, 2),
        )

        trial_score = metric_function(
            spikes,
            axis=(1, 2),
        )

        metric_name = metric.title()
        colorbar_label = f"{metric.title()} spike value"

    electrode_low, electrode_high = parse_threshold(
        electrode_threshold,
    )

    trial_low, trial_high = parse_threshold(
        trial_threshold,
    )

    bad_electrodes = np.zeros(
        n_electrodes,
        dtype=bool,
    )

    bad_trials = np.zeros(
        n_trials,
        dtype=bool,
    )

    if electrode_low is not None:
        bad_electrodes |= electrode_score < electrode_low

    if electrode_high is not None:
        bad_electrodes |= electrode_score > electrode_high

    if trial_low is not None:
        bad_trials |= trial_score < trial_low

    if trial_high is not None:
        bad_trials |= trial_score > trial_high

    electrodes_to_remove = np.where(
        bad_electrodes,
    )[0]

    trials_to_remove = np.where(
        bad_trials,
    )[0]

    electrode_tick_idx = np.linspace(
        0,
        n_electrodes - 1,
        min(10, n_electrodes),
        dtype=int,
    )

    trial_tick_idx = np.linspace(
        0,
        n_trials - 1,
        min(10, n_trials),
        dtype=int,
    )

    fig = plt.figure(
        figsize=figsize,
        layout="constrained",
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.25, 1],
        height_ratios=[1, 0.25],
        wspace=0.03,
        hspace=0.03,
    )

    ax_main = fig.add_subplot(gs[0, 1])

    ax_left = fig.add_subplot(
        gs[0, 0],
        sharey=ax_main,
    )

    ax_bottom = fig.add_subplot(
        gs[1, 1],
        sharex=ax_main,
    )

    ax_corner = fig.add_subplot(gs[1, 0])
    ax_corner.axis("off")

    im = ax_main.imshow(
        summary.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    ax_main.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_main.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    ax_main.spines[["top", "right"]].set_visible(False)

    electrode_idx = np.arange(n_electrodes)
    trial_idx = np.arange(n_trials)

    ax_left.scatter(
        electrode_score,
        electrode_idx,
        color="C0",
        s=8,
        alpha=0.7,
    )

    if electrode_low is not None:
        ax_left.axvline(
            electrode_low,
            color="tab:red",
            linewidth=1,
        )

    if electrode_high is not None:
        ax_left.axvline(
            electrode_high,
            color="tab:red",
            linewidth=1,
        )

    ax_left.scatter(
        electrode_score[bad_electrodes],
        electrode_idx[bad_electrodes],
        color="tab:red",
        s=18,
        zorder=3,
    )

    ax_left.set(
        xlabel=metric_name,
        ylabel="Electrode",
        yticks=electrode_tick_idx,
        yticklabels=electrode_names[electrode_tick_idx],
    )

    ax_left.tick_params(
        axis="x",
        labelsize=7,
    )

    ax_left.tick_params(
        axis="y",
        labelsize=7,
    )

    ax_left.spines[["top", "right"]].set_visible(False)

    ax_bottom.scatter(
        trial_idx,
        trial_score,
        color="C0",
        s=8,
        alpha=0.7,
    )

    if trial_low is not None:
        ax_bottom.axhline(
            trial_low,
            color="tab:red",
            linewidth=1,
        )

    if trial_high is not None:
        ax_bottom.axhline(
            trial_high,
            color="tab:red",
            linewidth=1,
        )

    ax_bottom.scatter(
        trial_idx[bad_trials],
        trial_score[bad_trials],
        color="tab:red",
        s=18,
        zorder=3,
    )

    ax_bottom.set(
        xlabel="Trial",
        ylabel=metric_name,
        xticks=trial_tick_idx,
        xticklabels=trial_names[trial_tick_idx],
    )

    ax_bottom.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=7,
    )

    ax_bottom.tick_params(
        axis="y",
        labelsize=7,
    )

    ax_bottom.spines[["top", "right"]].set_visible(False)

    for electrode_idx in electrodes_to_remove:
        ax_main.axhline(
            electrode_idx - 0.5,
            color="tab:red",
            linewidth=0.6,
        )

        ax_main.axhline(
            electrode_idx + 0.5,
            color="tab:red",
            linewidth=0.6,
        )

    for trial_idx in trials_to_remove:
        ax_main.axvline(
            trial_idx - 0.5,
            color="tab:red",
            linewidth=0.6,
        )

        ax_main.axvline(
            trial_idx + 0.5,
            color="tab:red",
            linewidth=0.6,
        )

    cbar = fig.colorbar(
        im,
        ax=ax_main,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        colorbar_label,
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    save_figure(
        fig,
        file_name,
        file_path,
    )

    return trials_to_remove, electrodes_to_remove


    image, 
    title, 
    file_name, 
    file_path, 
    vmin=None, 
    vmax=None, 
    figsize=(16, 4), 
    show=False


def plot_image(
    image, 
    title, 
    file_name, 
    file_path, 
    vmin=None, 
    vmax=None, 
    show=False
):

    if vmin is None:
        vmin = image.min(axis=(0, 1))
    if vmax is None:
        vmax = image.max(axis=(0, 1))

    vmin = np.asarray(vmin).reshape(-1)
    vmax = np.asarray(vmax).reshape(-1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    channel_names = ["Red", "Green", "Blue"]
    channel_cmaps = ["Reds", "Greens", "Blues"]

    for i in range(3):
        axes[i].imshow(image[:, :, i], cmap=channel_cmaps[i], vmin=vmin[i], vmax=vmax[i])
        axes[i].set_title(channel_names[i])
        axes[i].axis("off")

    rgb_image = image.copy()

    if np.issubdtype(rgb_image.dtype, np.floating):
        rgb_min = float(np.min(vmin))
        rgb_max = float(np.max(vmax))
        rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min + 1e-8)
        rgb_image = np.clip(rgb_image, 0, 1)

    axes[3].imshow(rgb_image)
    axes[3].set_title("RGB")
    axes[3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_training_history(
    trainer,
    title,
    file_name,
    file_path,
    show=False,
):
    history = trainer.history

    train = np.asarray(history.train_loss, dtype=float)
    test = np.asarray(history.test_loss, dtype=float)
    lr = np.asarray(history.lr, dtype=float)

    n = min(len(train), len(test), len(lr))

    if n == 0:
        raise ValueError("Training history is empty.")

    train = train[:n]
    test = test[:n]
    lr = lr[:n]
    epochs = np.arange(1, n + 1)

    def relative_trend(values):
        return np.diff(values) / np.maximum(np.abs(values[:-1]), 1e-8)

    train_trend = relative_trend(train)
    test_trend = relative_trend(test)
    trend_epochs = epochs[1:]

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(12, 3.5),
        layout="constrained",
    )

    ax[0].plot(epochs, train, color="tab:blue", label="Training")
    ax[0].plot(epochs, test, color="tab:orange", label="Test")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("MSE loss")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.25)

    ax[1].axhline(0, color="black", linestyle="--", linewidth=1)
    ax[1].plot(
        trend_epochs,
        train_trend,
        color="tab:blue",
        label="Training",
    )
    ax[1].plot(
        trend_epochs,
        test_trend,
        color="tab:orange",
        label="Test",
    )
    ax[1].set_title("Relative loss trend")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Relative change")
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.25, which="both")

    ax[2].plot(epochs, lr, color="tab:green")
    ax[2].set_title("Learning rate")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Learning rate")
    ax[2].set_yscale("log")
    ax[2].grid(alpha=0.25, which="both")

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_electrode_metrics(
    correlation_electrode,
    r2_electrode,
    mse_electrode,
    bin_times,
    title,
    file_name,
    file_path,
    figsize=(4, 7),
    show=False,
):

    if bin_times[0] >= 0:
        xticks = [0, bin_times[-1]]
        xticklabels = ["0", f"{bin_times[-1]:g}"]
    else:
        xticks = [bin_times[0], 0, bin_times[-1]]
        xticklabels = [
            f"{bin_times[0]:g}",
            "0",
            f"{bin_times[-1]:g}",
        ]

    fig, ax = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        layout="constrained",
    )

    metrics = [
        (correlation_electrode, "Correlation", "tab:blue"),
        (r2_electrode, "R\u00b2", "tab:orange"),
        (mse_electrode, "MSE", "tab:green"),
    ]

    for axis, (values, ylabel, color) in zip(ax, metrics):
        mean_val = np.nanmean(values, axis=0)
        sem_val = np.nanstd(values, axis=0) / np.sqrt(np.sum(~np.isnan(values), axis=0))
        ci_low = mean_val - 1.96 * sem_val
        ci_high = mean_val + 1.96 * sem_val

        axis.fill_between(
            bin_times,
            ci_low,
            ci_high,
            color=color,
            alpha=0.25,
        )

        axis.plot(
            bin_times,
            mean_val,
            color=color,
            linewidth=2,
        )

        axis.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
        )

        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)

    ax[-1].set(
        xlabel="Time from stimulus onset",
        xticks=xticks,
        xticklabels=xticklabels,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_image_metrics(
    correlation_image,
    r2_image,
    mse_image,
    image_names,
    title,
    file_name,
    file_path,
    figsize=(20, 12),
    show=False,
):

    fig, ax = plt.subplots(
        3,
        1,
        figsize=figsize,
        layout="constrained",
    )

    metrics = [
        (
            correlation_image,
            "Correlation",
            "tab:blue",
        ),
        (
            r2_image,
            "R²",
            "tab:orange",
        ),
        (
            mse_image,
            "MSE",
            "tab:green",
        ),
    ]

    for axis, (values, ylabel, color) in zip(
        ax,
        metrics,
    ):
        mean_values = np.nanmean(
            values,
            axis=1,
        )

        sem_values = np.nanstd(
            values,
            axis=1,
        ) / np.sqrt(
            np.sum(
                ~np.isnan(values),
                axis=1,
            )
        )

        ci_values = 1.96 * sem_values

        sort_idx = np.argsort(
            mean_values,
        )[::-1]

        sorted_mean_values = mean_values[
            sort_idx
        ]

        sorted_ci_values = ci_values[
            sort_idx
        ]

        sorted_names = np.asarray(
            image_names,
        )[sort_idx]

        axis.bar(
            np.arange(
                len(sorted_mean_values),
            ),
            sorted_mean_values,
            yerr=sorted_ci_values,
            color=color,
            alpha=0.8,
            capsize=2,
            error_kw={
                "elinewidth": 0.8,
                "capthick": 0.8,
            },
        )

        axis.set(
            ylabel=ylabel,
            xticks=np.arange(
                len(sorted_mean_values),
            ),
            xticklabels=sorted_names,
        )

        axis.tick_params(
            axis="x",
            labelrotation=90,
            labelsize=7,
        )

        axis.spines[
            ["top", "right"]
        ].set_visible(False)

        axis.grid(
            axis="y",
            alpha=0.25,
        )

    ax[-1].set_xlabel(
        "Image",
    )

    fig.suptitle(
        title,
    )

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_electrode_metrics_hist(
    correlation_electrode,
    r2_electrode,
    mse_electrode,
    title,
    file_name,
    file_path,
    bins=20,
    figsize=(4, 7),
    show=False,
):

    fig, ax = plt.subplots(
        3,
        1,
        figsize=figsize,
        layout="constrained",
    )

    metrics = [
        (correlation_electrode, "Correlation", "tab:blue"),
        (r2_electrode, "R²", "tab:orange"),
        (mse_electrode, "MSE", "tab:green"),
    ]

    for axis, (values, xlabel, color) in zip(ax, metrics):
        axis.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.8,
        )

        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_attribution(
    image,
    attribution,
    vmax_attribution,
    vmin_attribution,
    title,
    file_name,
    file_path,
    show=False,
):
    if image.shape[2] != 3:
        raise ValueError(
            "image must have shape (H, W, 3) for RGB."
        )

    if attribution.shape[2] != 3:
        raise ValueError(
            "attribution must have shape (H, W, 3) for RGB."
        )

    attribution_map = attribution.sum(axis=2)

    vmin_image = image.min(axis=(0, 1))
    vmax_image = image.max(axis=(0, 1))

    rgb_image = image.copy()

    if np.issubdtype(rgb_image.dtype, np.floating):
        rgb_min = float(np.min(vmin_image))
        rgb_max = float(np.max(vmax_image))
        rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min + 1e-8)
        rgb_image = np.clip(rgb_image, 0, 1)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        layout="constrained",
    )

    axes[0].imshow(rgb_image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    im = axes[1].imshow(
        attribution_map,
        cmap="bwr",
        vmin=vmin_attribution,
        vmax=vmax_attribution,
    )
    axes[1].set_title("Attribution")
    axes[1].axis("off")

    cbar = fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.02,
    )

    cbar.ax.tick_params(labelsize=7)

    axes[2].imshow(rgb_image)
    axes[2].imshow(
        attribution_map,
        cmap="bwr",
        vmin=vmin_attribution,
        vmax=vmax_attribution,
        alpha=0.5,
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


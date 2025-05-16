"""Plotting utils."""
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch as t

from einops import asnumpy
from jaxtyping import Float
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

from toy_cis.models import Cis


def plot_weight_bars(
    W: Float[Tensor, "dim1 dim2"],
    xax: str = "neuron",
    palette: str = "inferno",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plots weights for each input_feature-hidden_neuron pair as stacked bars.
    
    Plots the first dimension as bars per item in the second dimension.
    """
    bar_label = "feature" if xax == "neuron" else "neuron"
    W = asnumpy(W)
    assert len(W.shape) == 2
    dim1, dim2 = W.shape  # (neurons, features) or (features, neurons)
    x = np.arange(dim2)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    colors = sns.color_palette(palette, dim1)
    bottom_pos = np.zeros(dim2)
    bottom_neg = np.zeros(dim2)
    
    # Plot bars ensuring that none cross the x-axis (each should be clearly pos or neg).
    for i in range(dim1):
        mask_pos = W[i] >= 0
        mask_neg = W[i] < 0
        
        if np.any(mask_pos):
            ax.bar(x[mask_pos], W[i][mask_pos], bottom=bottom_pos[mask_pos], 
                  label=f"{bar_label} {i}", color=colors[i])
            bottom_pos[mask_pos] += W[i][mask_pos]
            
        if np.any(mask_neg):
            ax.bar(x[mask_neg], W[i][mask_neg], bottom=bottom_neg[mask_neg], 
                  label=f"{bar_label} {i}", color=colors[i])
            bottom_neg[mask_neg] += W[i][mask_neg]
    
    # Add colorbar.
    norm = mpl.colors.Normalize(vmin=0, vmax=(dim1 - 1))
    cmap = mpl.cm.get_cmap(palette, dim1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(bar_label)

    ax.set_xlabel(xax)
    ax.set_ylabel("Weight Value")
    
    return ax

def plot_input_output_response(
    Y: Float[Tensor, "feat val"],
    x: Float[Tensor, ""],
    sorted_idxs: list[int],
    ax: plt.Axes | None = None
) -> plt.Axes:
    """Plots the input-output response for each feature, in the sorted order of residual error."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=(Y.shape[0] -1))

    x = asnumpy(x)

    # Plot a line for each feature.
    for feat_idx in sorted_idxs:
        y = asnumpy(Y[feat_idx, :])
        color = cmap(norm(feat_idx))
        ax.plot(x, y, color=color)

    # Add colorbar for feature index.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Feature")

    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title("Input-Output Response per Feature")

    return ax

@t.no_grad()
def plot_input_output_heatmap(
    x: float,
    model: Cis,
    model_idx: int = 0,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plots heatmap of out response of all features given single-feature in, for all features."""
    device, dtype = model.device, model.dtype
    n_feat, n_inst = model.cfg.n_feat, model.cfg.n_instances
    title = f"Onehot input-output heatmap ({x=})"

    # Generate onehot input 
    x = (t.eye(n_feat, device=device, dtype=dtype) * x).reshape(n_feat, n_inst, n_feat)
    
    # Generate output response matrix
    Y = model.forward(x)
    Y = asnumpy(Y[:, model_idx, :].squeeze())
    
    # Plot it
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(
        Y,
        cmap = "viridis" if kwargs.get("cmap") is None else kwargs["cmap"],
        annot = True if kwargs.get("annot") is None else kwargs["annot"],
        fmt = ".2f" if kwargs.get("fmt") is None else kwargs["fmt"],
        annot_kws = (
            {"fontsize": 12, "color": "white"} 
            if kwargs.get("annot_kws") is None else kwargs["annot_kws"]
        ),
        vmax = 1 if kwargs.get("vmax") is None else kwargs["vmax"],
        vmin = -1 if kwargs.get("vmin") is None else kwargs["vmin"],
        ax=ax,
    )
    ax.set_title(title, fontsize=18)

    return ax

@t.no_grad()
def plot_loss_across_sparsities(
    sparsities: list[float],
    model: Cis,
    eval_model: callable,
    target: str,
    train_sparsity: float,
    batch_sz: int = 100000,
    n_steps: int = 100,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plots loss across feature probabilities."""
    loss_data = []
    # Get loss across feature probabilities.
    pbar = tqdm(sparsities, desc="Testing across feature sparsities")
    for s in pbar:
        eval_loss = eval_model(
            model=model,
            batch_sz=batch_sz,
            n_batches=n_steps,
            feat_sparsity=s,
            device=model.device,
            target=target,
        ).mean().item()

        loss_dict = {"sparsity": s, "loss": eval_loss}
        loss_data.append(loss_dict)

    # Format it for plotting.
    df_loss = pd.DataFrame(loss_data)
    df_loss["loss"] = df_loss["loss"] / (1 - df_loss["sparsity"])
    df_loss = df_loss.sort_values(by="sparsity")
    df_loss["p"] = 1 - df_loss["sparsity"]

    n_feat = model.cfg.n_feat
    d_mlp = model.cfg.n_hidden
    naive_loss = (n_feat - d_mlp) / n_feat * 1/6
    
    # Plot it.
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    # loss across sparsities
    sns.lineplot(
        data=df_loss,
        x="p",
        y="loss",
        errorbar=None,
        ax=ax,
    )
    ax.axhline(naive_loss, color="black", linestyle="--", label="Naive Loss", alpha=0.5)
    
    # Prettify.
    train_p = np.round(1 - train_sparsity, 3).item()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Loss across feature probabilities \n({train_p=})")
    ax.set_xlabel("feature probability (p)")
    ax.set_ylabel("adjusted loss (L / p)")
    ax.grid(True)

    ax.legend()

    return ax, df_loss

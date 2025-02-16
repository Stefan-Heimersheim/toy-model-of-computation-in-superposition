"""Plotting utils."""
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch as t

from einops import asnumpy
from jaxtyping import Float
from matplotlib import pyplot as plt
from torch import Tensor


def plot_weight_bars(
    W: Float[Tensor, "dim1 dim2"], xax: str = "neuron", palette: str = "inferno"
) -> plt.Figure:
    """Plots weights for each input_feature-hidden_neuron pair as stacked bars.
    
    Plots the first dimension as bars per item in the second dimension.
    """
    bar_label = "feature" if xax == "neuron" else "neuron"
    W = asnumpy(W)
    assert len(W.shape) == 2
    dim1, dim2 = W.shape  # (neurons, features) or (features, neurons)
    x = np.arange(dim2)
    
    sns.set_style("whitegrid")
    colors = sns.color_palette(palette, dim1)
    fig, ax = plt.subplots(figsize=(10, 6))
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
    
    return fig

def plot_input_output_response(
        Y: Float[Tensor, "feat val"], vals: Float[Tensor, ""], sorted_idxs: list[int]
) -> plt.Figure:
    """Plots the input-output response for each feature, in the sorted order of residual error."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=(Y.shape[0] -1))

    x_vals = asnumpy(vals)

    # Plot a line for each feature.
    for feat_idx in sorted_idxs:
        y_vals = asnumpy(Y[feat_idx, :])
        color = cmap(norm(feat_idx))
        ax.plot(x_vals, y_vals, color=color)

    # Add colorbar for feature index.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Feature")

    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    plt.tight_layout()

    return fig

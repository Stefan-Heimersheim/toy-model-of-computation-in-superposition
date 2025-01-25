"""Plotting utils."""
import numpy as np
import seaborn as sns
import torch as t

from einops import asnumpy
from jaxtyping import Float
from matplotlib import pyplot as plt
from torch import Tensor


def plot_weight_bars(W: Float[Tensor, "dim1 dim2"], xax="neuron") -> plt.Figure:
    """Plots weights for each input_feature-hidden_neuron pair as stacked bars.
    
    Plots the first dimension as bars per item in the second dimension.
    """
    bar_label = "feature" if xax == "neuron" else "neuron"
    W = asnumpy(W)
    assert len(W.shape) == 2
    dim1, dim2 = W.shape  # (neurons, features) or (features, neurons)
    x = np.arange(dim2)
    
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", dim1)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom_pos = np.zeros(dim2)
    bottom_neg = np.zeros(dim2)
    
    # Plot bars ensuring that none cross the x-axis (each should be clearly pos or neg).
    for h in range(dim1):
        mask_pos = W[h] >= 0
        mask_neg = W[h] < 0
        
        if np.any(mask_pos):
            ax.bar(x[mask_pos], W[h][mask_pos], bottom=bottom_pos[mask_pos], 
                  label=f"{bar_label} {h}", color=colors[h])
            bottom_pos[mask_pos] += W[h][mask_pos]
            
        if np.any(mask_neg):
            ax.bar(x[mask_neg], W[h][mask_neg], bottom=bottom_neg[mask_neg], 
                  label=f"{bar_label} {h}", color=colors[h])
            bottom_neg[mask_neg] += W[h][mask_neg]
    
    ax.set_xlabel(xax)
    ax.set_ylabel("Weight Value")
    
    return fig

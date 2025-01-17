"""Plotting utils."""
import numpy as np
import seaborn as sns
import torch as t

from einops import asnumpy
from matplotlib import pyplot as plt


def plot_weight_bars(W: t.Tensor) -> plt.Figure:
    """Plot weights from each hidden neuron as stacked bars across input features."""
    W = asnumpy(W)
    assert len(W.shape) == 2
    n_hidden, n_feat = W.shape
    x = np.arange(n_feat)
    
    # Set seaborn style but use plt.bar
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom_pos = np.zeros(n_feat)
    bottom_neg = np.zeros(n_feat)
    
    # Use seaborn color palette
    colors = sns.color_palette("husl", n_hidden)
    
    for h in range(n_hidden):
        mask_pos = W[h] >= 0
        mask_neg = W[h] < 0
        
        if np.any(mask_pos):
            ax.bar(x[mask_pos], W[h][mask_pos], bottom=bottom_pos[mask_pos], 
                  label=f"Feature {h}", color=colors[h])
            bottom_pos[mask_pos] += W[h][mask_pos]
            
        if np.any(mask_neg):
            ax.bar(x[mask_neg], W[h][mask_neg], bottom=bottom_neg[mask_neg], 
                  label=f"Feature {h}", color=colors[h])
            bottom_neg[mask_neg] += W[h][mask_neg]
    
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Weight Value")
    
    return fig

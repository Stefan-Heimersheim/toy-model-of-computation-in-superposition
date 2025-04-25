"""Plotting utils."""
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch as t
import pandas as pd

from einops import asnumpy, rearrange,reduce
from jaxtyping import Float
from matplotlib import pyplot as plt
from torch import Tensor

def plot_weight_bars(
    W: Float[Tensor, "dim1 dim2"],  model_name, feat_prob, xax: str = "neuron", palette: str = "inferno",
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
    if xax == 'feature':
        fig.get_axes()[0].set_title(f"{model_name} at FeatProb ={feat_prob:.2f} \n neuron weights per feature")
    else: 
        fig.get_axes()[0].set_title(f"{model_name} at FeatProb = {feat_prob:.2f} \n feature weights per neuron")
    
    return fig

def plot_input_output_response(
    Y: Float[Tensor, "feat val"], vals: Float[Tensor, ""], model_name: str, feat_prob: Float,
    losses: list[Float]
) -> plt.Figure:
    """Plots the input-output response for each feature, in the sorted order of residual error."""
    # Rank features in ascending order of residual error to ReLU(vals).
    target = rearrange(t.relu(vals), "val -> 1 val")  # shape for broadcasting during err calculation
    err = reduce((Y - target) ** 2, "feat val -> feat", "sum")
    sorted_idxs = t.argsort(err).tolist()
    
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

    plt.grid()
    ax = fig.get_axes()[0]
    ax.plot(
        asnumpy(vals), asnumpy(vals + t.relu(vals)), color="red", linestyle="--", label="target: x + relu(x)"
    )
    ax.set_title(
        f"{model_name} at featProb = {feat_prob:.2f}" 
        + f"\n input-output response per feature  (loss = {losses[-1]:.4f})"
    )
    ax.legend()

    return fig

def plot_loss_across_sparsities(
    loss_data, sparsities, model_name, trained_sparsity
)-> plt.Figure:
    """
    Plots normalized loss per feature vs. input sparsity.

    Args:
        loss_data (dict): Dictionary with keys 'sparsity' and 'loss_per_feature'.
        sparsities (np.ndarray): 1D array of sparsity values used to compute naive loss.
        model_name (str): Name of the model (for plot title and legend).
    """
    
    # convert loss data to DataFrame, convert sparsity to probability and adjust loss
    df_loss = pd.DataFrame(loss_data) 
    df_loss["FeatProb"] = 1 - df_loss["sparsity"]
    df_loss["loss/FeatProb"] = df_loss["loss_per_feature"] / df_loss["FeatProb"]
    
    naive_loss = 0.5 * (1 - sparsities) / 6 # compute naive loss (monosemantic solution)
    norm_loss = naive_loss.ravel() / (1 - sparsities)
    
    # plot performance vs input sparsity
    fig = plt.figure(figsize=(15, 8))
    sns.lineplot(data = df_loss, x = "FeatProb", y = "loss/FeatProb")
    
    # add naive loss line
    plt.plot(1- sparsities, norm_loss, linestyle="dashed", color="black", label=r"Naive loss") 
    
    # add labels and scale
    plt.xlabel('Feature probability')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend(loc = "best", fontsize = 16)
    plt.ylabel('Loss/Feature probability ')
    trained_probability = 1 - trained_sparsity
    plt.title(f"{model_name} trained at featProb = {trained_probability:.2f}:\n loss vs input probability")
    plt.tight_layout()
    return fig

def plot_phase_diagram (loss_data, model_name): 
    
    """Plot phase diagram"""
    loss = pd.DataFrame(loss_data)
    
    # Aggregate loss_per_feature over feature_idx
    loss_avg = loss.groupby(["train_sparsity", "input_sparsity"])["loss_per_feature"].mean().reset_index()
    loss_avg["input_probs"] = np.round(1 - loss_avg["input_sparsity"],2)
    loss_avg["training_probs"] = np.round(1 - loss_avg["train_sparsity"],2)
    loss_avg["loss/1-S"] = loss_avg["loss_per_feature"] / (1 - loss_avg["input_sparsity"])
    
    # Create pivot table for heatmap
    pivot_table = loss_avg.pivot(index="training_probs", columns="input_probs", values="loss/1-S")
    
    # Plot phase diagram
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_table, cmap="viridis", annot=True, fmt=".2f", cbar_kws={'label': 'Adjusted loss'})
    
    plt.xlabel("Input FeatProb")
    plt.ylabel("Training FeatProb")
    plt.title(f"{model_name} model: Phase Diagram of \n Training FeatProb vs Input FeatProb (Loss)")
    return fig

def plot_phase_diagram_training(loss_data, model_name): 

    """Plot phase diagram"""
    loss = pd.DataFrame(loss_data)
    
    # Aggregate loss_per_feature over feature_idx
    loss["featProbs"] = np.round(1 - loss["sparsity"],2)
    loss["loss/1-S"] = loss["loss"] / (1 - loss["sparsity"])
    
    # Create pivot table for heatmap
    pivot_table = loss.pivot(index="featProbs", columns="n_steps", values="loss/1-S")
    
    # Plot phase diagram
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_table, cmap="viridis", annot=True, fmt=".2f", cbar_kws={'label': 'Adjusted loss'})
    
    plt.ylabel("Training feature Probability")
    plt.xlabel("Training Steps")
    plt.title(f"{model_name} model: Phase Diagram of \n Feature probability vs Training steps (Loss)")
    return fig

def plot_phase_diagram_polysem(poly_data, model_name): 
    """Plot phase diagram"""
    poly = pd.DataFrame(poly_data)
    
    # Aggregate loss_per_feature over feature_idx
    poly["featProbs"] = np.round(1 - poly["sparsity"],2)
    poly["mean_featPerNeuron"] = poly["features_per_neuron"].apply(np.mean)
    
    # Create pivot table for heatmap
    pivot_table = poly.pivot(index="featProbs", columns="n_steps", values="mean_featPerNeuron")
    
    # Plot phase diagram
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_table, cmap="viridis", annot=True, fmt=".2f", cbar_kws={'label': 'Features per neuron'})
    
    plt.ylabel("Training feature Probability")
    plt.xlabel("Training Steps")
    plt.title(f"{model_name} model: Phase Diagram of \n Feature probability vs Training steps (Features per neuron)")
    return fig
    
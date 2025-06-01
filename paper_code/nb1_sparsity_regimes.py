import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from jaxtyping import Float
from mlpinsoup import MLP, ResidTransposeDataset, plot_loss_of_input_sparsity, train
from torch import Tensor

sns.set_style("whitegrid")

# Input feature probabilities to test
plot_ps = np.geomspace(0.001, 1, 100)
# Train feature probabilities
str_train_ps = ["0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"]
# Dataset following Braun et al. 2025
apd_dataset = ResidTransposeDataset(n_features=100, d_embed=1000, p=0)

models, labels = [], []
for str_p in str_train_ps:
    p = float(str_p)
    apd_dataset.set_p(p)
    model = MLP(n_features=100, d_mlp=50)
    frac_zero_samples = (1 - p) ** model.n_features
    n_steps = int(10_000 / (1 - frac_zero_samples))
    train(model, apd_dataset, batch_size=1024, steps=n_steps)
    models.append(model)
    labels.append("p=" + str_p)

train_ps = [float(p) for p in str_train_ps]
fig = plot_loss_of_input_sparsity(models, apd_dataset, ps=plot_ps, labels=labels, highlight_ps=train_ps)
fig.savefig("nb1_sparsity_regimes.png")

# Plot the input-output behaviour of one sparse and dense model, for illustration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5), sharey=True)
fig.suptitle("Input-output behaviour for individual features")
models[0].plot_input_output_behaviour(ax1)
ax1.set_title("p=" + str_train_ps[0])
x = torch.linspace(-1, 1, 100)
ax1.plot(x, F.relu(x), color="tab:red", ls="--")
ax2.plot(x, F.relu(x), color="tab:red", ls="--")
norm = plt.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04, label="Feature index")
models[-1].plot_input_output_behaviour(ax2)
ax2.set_title("p=" + str_train_ps[-1])
fig.savefig("nb1a_input_output_behaviour.png")
plt.show()


# Plot a stacked bar chart of model weights
def plot_weight_bars(W: Float[Tensor, "dim1 dim2"], bar_label: str = "MLP neuron index", cmap: bool = True, ax: plt.Axes | None = None) -> plt.Axes:
    """Plots weights for each input_feature-hidden_neuron pair as stacked bars. Closely follows Jai's toy-model-of-computation-in-superposition/toy_cis/plot.py"""
    palette = "inferno"
    W = einops.asnumpy(W)
    d_mlp, n_features = W.shape
    x = np.arange(n_features)
    colors = sns.color_palette(palette, d_mlp)
    fig, ax = plt.subplots(figsize=(10, 5)) if ax is None else (ax.figure, ax)
    bottom_pos = np.zeros(n_features)
    bottom_neg = np.zeros(n_features)
    for i in range(d_mlp):
        mask_pos = W[i] >= 0
        mask_neg = W[i] < 0
        if np.any(mask_pos):
            ax.bar(x[mask_pos], W[i][mask_pos], bottom=bottom_pos[mask_pos], label=f"{bar_label} {i}", color=colors[i])
            bottom_pos[mask_pos] += W[i][mask_pos]
        if np.any(mask_neg):
            ax.bar(x[mask_neg], W[i][mask_neg], bottom=bottom_neg[mask_neg], label=f"{bar_label} {i}", color=colors[i])
            bottom_neg[mask_neg] += W[i][mask_neg]
    if cmap:
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=1, vmax=d_mlp), cmap=mpl.colormaps[palette])
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(bar_label)
    ax.set_xlim(-0.5, n_features - 0.5)
    return ax


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 5), sharey=True)
fig.suptitle(r"Model weights $W_{\rm in} \odot W_{\rm out}$ stacked by neuron")
for model, ax in zip([models[0], models[-1]], [ax1, ax2]):
    ax.set_title("p=" + str_train_ps[0])
    w_in: Float[Tensor, "n_features d_mlp"] = model.w_in.cpu().detach()
    w_out: Float[Tensor, "d_mlp n_features"] = model.w_out.cpu().detach()
    W: Float[Tensor, "d_mlp n_features"] = einops.einsum(w_in, w_out, "n_features d_mlp, d_mlp n_features -> d_mlp n_features")
    plot_weight_bars(W, ax=ax, cmap=ax is ax2)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Weight Value")
fig.savefig("nb1b_weight_bars.png")
plt.show()

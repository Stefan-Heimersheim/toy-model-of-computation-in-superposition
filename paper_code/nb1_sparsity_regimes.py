import einops
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mlpinsoup import MLP, NoisyDataset, ResidTransposeDataset, plot_loss_of_input_sparsity, train
from torch import Tensor

sns.set_style("whitegrid")

# Input feature probabilities to test
plot_ps = np.geomspace(0.001, 1, 100)
# Train feature probabilities
str_train_ps = ["0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"]
# Dataset following Braun et al. 2025
# apd_dataset = ResidTransposeDataset(n_features=100, d_embed=1000, p=0)
apd_dataset = NoisyDataset(n_features=100, p=0, scale=0.02, zero_diagonal=False)

models, labels = [], []
for str_p in str_train_ps:
    p = float(str_p)
    apd_dataset.set_p(p)
    model = MLP(n_features=100, d_mlp=50)
    frac_zero_samples = (1 - p) ** model.n_features
    n_steps = int(10_000 / (1 - frac_zero_samples))
    train(model, apd_dataset, steps=n_steps)
    models.append(model)
    labels.append("p=" + str_p)

train_ps = [float(p) for p in str_train_ps]


# Generate colours
def get_gradient_colors(start_hex: str, end_hex: str, n_lines: int) -> list[str]:
    # Convert hex to RGB (0-1 range)
    start_rgb = np.array(mcolors.to_rgb(start_hex))
    end_rgb = np.array(mcolors.to_rgb(end_hex))

    # Create linear gradient between start and end
    return [mcolors.to_hex(start_rgb + (end_rgb - start_rgb) * i / (n_lines - 1)) for i in range(n_lines)]  # type: ignore


colors = get_gradient_colors("#1abc9c", "#9b59b6", len(str_train_ps))  # purple to teal gradient
fig = plot_loss_of_input_sparsity(models, apd_dataset, ps=plot_ps, labels=labels, highlight_ps=train_ps, colors=colors)
fig.suptitle("Loss over input sparsity for different input feature probabilities $p$")
fig.savefig("./plots/nb1_sparsity_regimes.png")

# Plot the input-output behaviour of one sparse and dense model, for illustration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5), sharey=True)  # type: ignore
fig.suptitle("Input-output behaviour for individual features")
models[0].plot_input_output_behaviour(ax1)
ax1.set_title("p=" + str_train_ps[0])
ax1.grid(True, alpha=0.3)
x = torch.linspace(-1, 1, 100)
ax1.plot(x, F.relu(x), color="tab:red", ls="--")
norm = Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04, label="Feature index")
models[-1].plot_input_output_behaviour(ax2)
ax2.plot(x, F.relu(x), color="tab:red", ls="--")
ax2.set_title("p=" + str_train_ps[-1])
ax2.grid(True, alpha=0.3)
fig.savefig("plots/nb1a_input_output_behaviour.png")
plt.show()


# Plot a stacked bar chart of model weights


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5), sharey=True)
fig.suptitle(r"Model weights $W_{\rm in} \odot W_{\rm out}$ stacked by neuron")
for model, ax, p in zip([models[0], models[-1]], [ax1, ax2], [str_train_ps[0], str_train_ps[-1]]):
    ax.set_title("p=" + p)
    model.plot_weight_bars(ax=ax)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Weight Value")
    ax.grid(True, alpha=0.3)
fig.savefig("plots/nb1b_weight_bars.png")


@torch.no_grad()
def plot_input_output_heatmap(
    x: float,
    model: MLP,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Plots heatmap of output response of all features given single-feature input, for all features.

    Args:
        x: Value to set for each input feature (e.g., 1 or -1)
        model: MLP model from mlpinsoup.py
        ax: Matplotlib axis to plot on
        **kwargs: Additional arguments for seaborn heatmap
    """
    device = model.device
    n_features = model.n_features
    title = f"Input-output heatmap (x={x})"

    # Generate one-hot input: each row has exactly one feature set to x
    # Shape: [n_features, n_features] where row i has feature i set to x
    inputs = torch.eye(n_features, device=device, dtype=torch.float32) * x

    # Generate output response matrix
    outputs = model.forward(inputs)  # Shape: [n_features, n_features]
    outputs_np = outputs.cpu().numpy()

    # Plot it
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Set default parameters
    heatmap_kwargs = {"cmap": "viridis", "annot": False, "fmt": ".2f", "vmax": 1, "vmin": 0, "center": 0, "cbar_kws": {"label": "Output value"}}
    heatmap_kwargs.update(kwargs)

    ax = sns.heatmap(outputs_np, ax=ax, **heatmap_kwargs)
    ax.set_title(title)
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Output feature")
    ax.set_xticks(np.arange(0, n_features, step=5))
    ax.set_yticks(np.arange(0, n_features, step=5))
    ax.set_xticklabels(np.arange(0, n_features, step=5))
    ax.set_yticklabels(np.arange(0, n_features, step=5))
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    return ax


# Create 4-panel interference heatmap figure

model_low_p = models[0]  # p=0.001
model_high_p = models[-1]  # p=1.0
p_low = str_train_ps[0]
p_high = str_train_ps[-1]

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle("Input-Output Heatmap Response for all features given a particular input value")

# Top row: x = 1
plot_input_output_heatmap(1.0, model_low_p, ax=axes[0, 0], annot=False, fmt=".1f", cbar=False)
axes[0, 0].set_title(f"x=+1, p={p_low}")

plot_input_output_heatmap(1.0, model_high_p, ax=axes[0, 1], annot=False, fmt=".1f", cbar=True)
axes[0, 1].set_title(f"x=+1, p={p_high}")

# Bottom row: x = -1
plot_input_output_heatmap(-1.0, model_low_p, ax=axes[1, 0], annot=False, fmt=".1f", cbar=False)
axes[1, 0].set_title(f"x=-1, p={p_low}")

plot_input_output_heatmap(-1.0, model_high_p, ax=axes[1, 1], annot=False, fmt=".1f", cbar=True)
axes[1, 1].set_title(f"x=-1, p={p_high}")

fig.savefig("plots/nb1c_interference_heatmaps.png")

import time
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from jaxtyping import Float
from plotly.subplots import make_subplots
from torch import Tensor, nn
from tqdm import tqdm

from mlpinsoup import MLP, CleanDataset, ResidTransposeDataset, evaluate, plot_loss_of_input_sparsity, train

# %%
naive_model = MLP(n_features=100, d_mlp=50)
dense_dataset = CleanDataset(n_features=100, p=1)
sparse_dataset = CleanDataset(n_features=100, p=0.01)

bias_strengths = np.geomspace(0.002, 0.05, 100)
dense_losses = []
sparse_losses = []
for bias_strength in bias_strengths:
    naive_model.handcode_naive_mlp(bias_strength=bias_strength)
    dense_loss = evaluate(naive_model, dense_dataset, batch_size=100_000)
    sparse_loss = evaluate(naive_model, sparse_dataset, batch_size=5_000_000)
    dense_losses.append(dense_loss / 1)
    sparse_losses.append(sparse_loss / 0.01)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(bias_strengths, dense_losses, label="Dense (p=1)")
ax.plot(bias_strengths, sparse_losses, label="Sparse (p=0.01)")
minimum_dense_loss = np.argmin(np.array(dense_losses))
ax.scatter(bias_strengths[minimum_dense_loss], dense_losses[minimum_dense_loss], color="C0", marker="o")
ax.text(bias_strengths[minimum_dense_loss], dense_losses[minimum_dense_loss], f"Optimal bias strength (dense): {bias_strengths[minimum_dense_loss]:.3f}", ha="right", va="top")
ax.legend()
ax.set_xlabel("Bias Strength")
ax.set_ylabel("Adjusted loss L / p")
ax.set_xscale("log")

# %%

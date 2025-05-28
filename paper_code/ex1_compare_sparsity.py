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

plot_ps = np.geomspace(0.001, 1, 100)

# %% APD version (noisy dataset)

train_ps = ["0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"]
apd_dataset = ResidTransposeDataset(n_features=100, d_embed=1000, p=0)

models, labels = [], []
for str_p in train_ps:
    p = float(str_p)
    apd_dataset.set_p(p)
    model = MLP(n_features=100, d_mlp=50)
    train(model, apd_dataset, batch_size=1024, steps=int(10_000 * 0.01 / min(0.01, p)))
    models.append(model)
    labels.append("p=" + str_p)

fig = plot_loss_of_input_sparsity(models, apd_dataset, ps=plot_ps, labels=labels)
fig.savefig("apd_loss_of_input_sparsity.png")

# %% Clean dataset

train_ps = ["0.001", "0.01", "0.1", "1"]
colors = ["C0", "C3", "C6", "C9"]
clean_dataset = CleanDataset(n_features=100, p=0)

models, labels = [], []
for str_p in train_ps:
    p = float(str_p)
    clean_dataset.set_p(p)
    model = MLP(n_features=100, d_mlp=50)
    # TODO: Would be nice to scale steps exactly with the binomal number of non-zero features
    train(model, clean_dataset, batch_size=1024, steps=int(10_000 * 0.01 / min(0.01, p)))
    models.append(model)
    labels.append("p=" + str_p)

fig = plot_loss_of_input_sparsity(models, clean_dataset, ps=plot_ps, labels=labels, colors=colors)

ax = fig.axes[0]
naive_model = MLP(n_features=100, d_mlp=50)
naive_model.handcode_naive_mlp(bias_strength=0.019)
plot_loss_of_input_sparsity([naive_model], clean_dataset, ps=plot_ps, colors=["black"], ax=ax, labels=["Monosemantic + emulate bias"])

fig.suptitle("Clean dataset")
fig.savefig("clean_loss_of_input_sparsity.png")

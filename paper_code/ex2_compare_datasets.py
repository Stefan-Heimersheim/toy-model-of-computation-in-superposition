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

from mlpinsoup import MLP, CleanDataset, NoisyDataset, ResidTransposeDataset, evaluate, plot_loss_of_input_sparsity, train

p = 0.01
n_features = 100
d_mlp = 50
n_steps = 10_000
batch_size_train = 1024

clean_dataset = CleanDataset(n_features=n_features, p=p)
clean_model = MLP(n_features=n_features, d_mlp=d_mlp)
train(clean_model, clean_dataset, batch_size=batch_size_train, steps=n_steps)
resid_transpose_dataset = ResidTransposeDataset(n_features=n_features, d_embed=1000, p=p)
resid_transpose_model = MLP(n_features=n_features, d_mlp=d_mlp)
train(resid_transpose_model, resid_transpose_dataset, batch_size=batch_size_train, steps=n_steps)

noisy_dataset = NoisyDataset(n_features=n_features, p=p, scale=0.0225)
noisy_model = MLP(n_features=n_features, d_mlp=d_mlp)
train(noisy_model, noisy_dataset, batch_size=batch_size_train, steps=n_steps)

naive_model = MLP(n_features=n_features, d_mlp=d_mlp)
naive_model.handcode_naive_mlp(bias_strength=0.019)


fig, ax = plt.subplots()
plot_loss_of_input_sparsity(
    models=[clean_model, naive_model, resid_transpose_model, noisy_model, naive_model, naive_model],
    ps=np.geomspace(0.001, 1, 100),
    ax=ax,
    labels=["Clean", "Naive solution on clean dataset", "ResidTranspose", "Noisy (0.0225)", "Naive solution on noisy dataset", "Naive solution on resid transpose dataset"],
    colors=["C0", "C1", "C2", "C3", "C4", "C5"],
    datasets=[clean_dataset, clean_dataset, resid_transpose_dataset, noisy_dataset, noisy_dataset, resid_transpose_dataset],
)
ax.legend().remove()
ax.legend(loc="upper left", ncols=2)
fig.savefig("dataset_comparison.png")

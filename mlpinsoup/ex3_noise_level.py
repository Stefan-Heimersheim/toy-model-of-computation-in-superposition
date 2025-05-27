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

noise_levels = np.linspace(0.00, 0.05, 51)
models, datasets, losses = [], [], []

for i, noise_level in enumerate(noise_levels):
    noisy_dataset = NoisyDataset(n_features=100, p=0.01, scale=noise_level, exactly_one_active_feature=True)
    noisy_model = MLP(n_features=100, d_mlp=50)
    train(noisy_model, noisy_dataset, batch_size=1024, steps=10_000)
    models.append(noisy_model)
    datasets.append(noisy_dataset)

losses = []
for i, (noisy_model, noisy_dataset) in tqdm(enumerate(zip(models, datasets, strict=True)), total=len(models), desc="Evaluating"):
    loss = np.mean([evaluate(noisy_model, noisy_dataset, batch_size=1_000_000) for _ in range(100)])
    losses.append(loss)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(noise_levels, losses, marker="o")
ax.set_xlabel("Dataset noise scale")
ax.set_ylabel("Loss of trained model")
fig.savefig("dataset_noise_level.png")
plt.show()

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
d_embed = 1000


def zero_function(x):
    return torch.zeros_like(x)


def relu_function(x):
    return F.relu(x)


def identity_function(x):
    return x


# In the linear case (no ReLU), the performance is fully explained by the SVD.
# We're training a model on y = W_E W_E^T x where W_E is a random matrix of shape
# (100, 1000), so M = W_E W_E^T has ones on the diagonal, and small numbers off
# the diagonal. The inputs x are random vectors of shape (100).

linear_dataset = ResidTransposeDataset(n_features=n_features, d_embed=d_embed, p=p)
linear_dataset.relu = nn.Identity()
embedding_matrix = linear_dataset.W_E
M = einops.einsum(embedding_matrix, embedding_matrix.T, "n_1 d_embed, d_embed n_2 -> n_1 n_2")
U, S, V = torch.linalg.svd(M)

zero_loss = evaluate(zero_function, linear_dataset)
print(f"Linear: zero function: {zero_loss / p:.3f}")
identity_loss = evaluate(identity_function, linear_dataset)
print(f"Linear: identity function: {identity_loss / p:.3f}")
relu_loss = evaluate(relu_function, linear_dataset)
print(f"Linear: relu function: {relu_loss / p:.3f}")

half_model = MLP(n_features=n_features, d_mlp=d_mlp)
half_model.relu = nn.Identity()
half_model.handcode_naive_mlp(bias_strength=0)
half_loss = evaluate(half_model, linear_dataset)
print(f"Linear: half (no bias): {half_loss / p:.3f}")

svd_model = MLP(n_features=n_features, d_mlp=d_mlp)
svd_model.relu = nn.Identity()
svd_model.w_in.data = U[:, :d_mlp].clone() * S[:d_mlp].clone()
svd_model.w_out.data = V[:d_mlp, :].clone()
svd_loss = evaluate(svd_model, linear_dataset)
print(f"Linear: SVD: {svd_loss / p:.3f}")

trained_model = MLP(n_features=n_features, d_mlp=d_mlp)
trained_model.relu = nn.Identity()
training_losses = train(trained_model, linear_dataset, batch_size=batch_size_train, steps=n_steps)
final_loss = evaluate(trained_model, linear_dataset)
print(f"Linear: trained: {final_loss / p:.3f}")

# %% Confirm (a) that the losses for inputs proportional to the top 50 SVD directions
# are near-perfect, and (b) that the matrices are similar to the SVD directions.?

fig, (ax_loss, ax_cosinesim) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))


def get_loss_for_svd_direction(model, dataset, svd_direction):
    batch_size = 1024
    test_inputs = torch.zeros(batch_size, n_features, device=dataset.device)
    random_amplitudes = torch.rand((batch_size, 1), device=dataset.device) * 2 - 1
    test_inputs = test_inputs + random_amplitudes * svd_direction
    test_targets = einops.einsum(test_inputs, M, "batch n_in, n_in n_out -> batch n_out")
    predictions = model(test_inputs)
    loss = ((predictions - test_targets) ** 2).mean().item()
    return loss


svd_model_direction_losses = [get_loss_for_svd_direction(svd_model, linear_dataset, U[:, i]) for i in range(n_features)]
trained_model_losses = [get_loss_for_svd_direction(trained_model, linear_dataset, U[:, i]) for i in range(n_features)]


def get_cosine_sim_for_svd_direction(model, singular_vector):
    model_transformation = einops.einsum(model.w_in, model.w_out, "d_in d_mlp, d_mlp d_out -> d_in d_out")
    projected_sv = einops.einsum(singular_vector, model_transformation, "d_in, d_in d_out -> d_out")
    cosine_sim = F.cosine_similarity(singular_vector, projected_sv, dim=0)
    return cosine_sim.item()


svd_model_cosine_sims = [get_cosine_sim_for_svd_direction(svd_model, U[:, i]) for i in range(n_features)]
trained_model_cosine_sims = [get_cosine_sim_for_svd_direction(trained_model, U[:, i]) for i in range(n_features)]

ax_sv = ax_loss.twinx()

tab20 = plt.get_cmap("tab20")
ax_loss.plot(svd_model_direction_losses, label="SVD", color=tab20(0))
ax_loss.plot(trained_model_losses, label="Trained", color=tab20(1), linestyle="--")
ax_sv.plot(S.cpu(), label="singular values", color=tab20(2), linestyle="--")

ax_loss.set_xlabel("Singular vector index")
ax_loss.set_ylabel("Loss", color=tab20(0))
ax_loss.legend(loc="upper left")
ax_sv.set_ylabel("Singular values", color=tab20(2))
ax_sv.legend(loc="upper right")

ax_cosinesim.plot(svd_model_cosine_sims, label="SVD", color=tab20(0))
ax_cosinesim.plot(trained_model_cosine_sims, label="Trained", color=tab20(1), linestyle="--")
ax_cosinesim.set_xlabel("Singular vector index")
ax_cosinesim.set_ylabel("Cosine similarity", color=tab20(0))
ax_cosinesim.legend(loc="upper left")

plt.show()
# %%
# Linear case:
# Loss_zero = 0.365: 0.033 (noise) + 2x0.166 (function)
# Loss_idty = 0.032: 0.033 (noise) + 0.000 (function)
# Loss_relu = 0.200: 0.033 (noise) + 0.166 (function)
# Loss_half = 0.199: 0.033 (noise) + 2x0.083 (function)
# Loss_svd = 0.094: 0.033 (noise) + ?
# trained 0.093

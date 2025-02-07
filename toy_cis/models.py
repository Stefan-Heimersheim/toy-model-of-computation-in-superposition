"""Contains a toy model class that can (possibly) illustrate computation in superposition.

tms-cis, simple-relu, and res-mlp toy models can all be created with the same class, `Cis`.
"""

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import torch as t

from einops import einsum, rearrange
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F


@dataclass
class CisConfig:
    """Config class for single hidden layer CiS model."""
    n_instances: int  # number of model instances
    n_feat: int  # number of features (elements) in input vector
    n_hidden: int  # number of hidden units in the model
    act_fn: List[Callable] = field(default_factory=lambda: [F.relu, F.relu])  # layer act funcs
    # Bias terms for hidden and output layers. For a given layer, if None, biases are not learned;
    # if scalar, all biases have the same value; if tensor, each bias has the corresponding 
    # tensor element value.
    b1: float | Float[t.Tensor, "inst hid"] | None = None
    b2: float | Float[t.Tensor, "inst hid"] | None = 0.0
    W1_as_W2T: bool = False  # W2 is learned if False, else W2 = W1.T
    We_and_Wu: bool = False  # if True, use fixed, random orthogonal embed and unembed matrices
    We_dim: int = 1000  # if We_and_Wu, this is the dim of the embedding space
    skip_cnx: bool = False  # if True, skip connection from in to out is added


    def __post_init__(self):
        """Ensure attribute values are valid."""
        # Handle `b1` tensor validation
        if isinstance(self.b1, t.Tensor):
            expected_shape = (self.n_instances, self.n_hidden)
            if self.b1.shape != expected_shape:
                raise ValueError(f"{self.b1.shape=} does not match {expected_shape=}")
        
        # Handle `b2` tensor validation
        if isinstance(self.b2, t.Tensor):
            expected_shape = (self.n_instances, self.n_hidden)
            if self.b2.shape != expected_shape:
                raise ValueError(f"{self.b2.shape=} does not match {expected_shape=}")



class Cis(nn.Module):
    """A generic computation-in-superposition toy model."""
    # Some attribute type hints
    W1: Float[t.Tensor, "inst hid feat"]
    W2: Float[t.Tensor, "inst feat hid"]
    b1: Float[t.Tensor, "inst hid"]
    b2: Float[t.Tensor, "inst feat"]


    def __init__(self, cfg: CisConfig, device: t.device):
        """Initializes model params."""
        super().__init__()
        self.cfg = cfg
        n_feat = cfg.n_feat

        # Embed and Unembed Matrices
        if cfg.We_and_Wu:
            rand_unit_mats = [
                F.normalize(t.randn(cfg.We_dim, cfg.n_feat), dim=0) for _ in range(cfg.n_instances)
            ]
            self.We = t.stack(rand_unit_mats).to(device)
            self.Wu = rearrange(self.We, "inst emb feat -> inst feat emb")
            n_feat = cfg.We_dim

        # Model Weights
        self.W1 = t.empty(cfg.n_instances, cfg.n_hidden, n_feat)
        self.W1 = nn.Parameter(nn.init.xavier_normal_(self.W1))
        if cfg.W1_as_W2T:
            self.W2 = self.W1.T
        else:
            self.W2 = t.empty(cfg.n_instances, n_feat, cfg.n_hidden)
            self.W2 = nn.Parameter(nn.init.xavier_normal_(self.W2))

        # Model Biases
        if cfg.b1 is None:
            self.b1 = t.zeros(cfg.n_instances, cfg.n_hidden, device=device)
        elif np.isscalar(cfg.b1):
            self.b1 = nn.Parameter(t.full((cfg.n_instances, cfg.n_hidden), cfg.b1))
        else:
            self.b1 = nn.Parameter(cfg.b1)

        if cfg.b2 is None:
            self.b2 = t.zeros(cfg.n_instances, n_feat, device=device)
        elif np.isscalar(cfg.b2):
            self.b2 = nn.Parameter(t.full((cfg.n_instances, n_feat), cfg.b2))
        else:
            self.b2 = nn.Parameter(cfg.b2)


    def forward(
        self, 
        x: Float[t.Tensor, "batch inst feat"],
    ) -> Float[t.Tensor, "batch inst feat"]:
        """Runs a forward pass through the model."""

        # Embedding layer
        if self.cfg.We_and_Wu:
            x = einsum(x, self.We, "batch inst feat, inst emb feat -> batch inst emb")

        # Hidden layer
        h = einsum(x, self.W1, "batch inst feat, inst hid feat -> batch inst hid")
        h = self.cfg.act_fn[0](h + self.b1)

        # Output layer
        y = einsum(h, self.W2, "batch inst hid, inst feat hid -> batch inst feat")
        y = self.cfg.act_fn[1](y + self.b2)
        
        # Skip connection
        if self.cfg.skip_cnx:
            y += x
        
        # Unembedding layer
        if self.cfg.We_and_Wu:
            y = einsum(y, self.Wu, "batch inst emb, inst feat emb -> batch inst feat")

        return y

# Note:
# Feature sparsity should be used in a function that generates batches.
# Feature importance should be used in a loss function.
# Both of these should be defined outside of this class, and called in a training loop.

"""Contains a toy model class that can (possibly) illustrate computation in superposition.

tms-cis, simple-relu, and res-mlp toy models can all be created with the same class, `Cis`.
"""

from dataclasses import dataclass, field
from typing import Callable, List

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
    skip_cnx: bool = False  # if True, skip connection from in to out is added
    feat_sparsity: float| Float[t.Tensor, "inst n_feat"] = 0.0  # sparsity of all or each feat
    feat_importance: float | Float[t.Tensor, "inst n_feat"] = 1.0  # importance of all or each feat

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
        
        # Handle `feat_sparsity` tensor validation
        if isinstance(self.feat_sparsity, t.Tensor):
            if self.feat_sparsity.shape != (self.n_feat,):
                raise ValueError(f"{self.feat_sparsity.shape=} must be ({self.n_feat},)")
        
        # Handle `feat_importance` tensor validation
        if isinstance(self.feat_importance, t.Tensor):
            if self.feat_importance.shape != (self.n_feat,):
                raise ValueError(f"{self.feat_importance.shape=} must be ({self.n_feat},)")


class Cis(nn.Module):
    """Anthropic Toy Models of Superposition Computation in Superposition toy model."""

    # Some attribute type hints
    W1: Float[t.Tensor, "inst hid feat"]
    W2: Float[t.Tensor, "inst feat hid"]
    b1: Float[t.Tensor, "inst hid"]
    b2: Float[t.Tensor, "inst feat"]
    s: Float[t.Tensor, "inst feat"]  # feature sparsity
    i: Float[t.Tensor, "inst feat"]  # feature importance


    def __init__(self, cfg: CisConfig):
        """Initializes model params."""
        super().__init__()
        self.cfg = cfg

        # Model Weights
        self.W1 = t.empty(cfg.n_instances, cfg.n_hidden, cfg.n_feat)
        self.W1 = nn.Parameter(nn.init.xavier_normal_(self.W1))
        if self.W1_as_W2T:
            self.W2 = self.W1.T
        else:
            self.W2 = t.empty(cfg.n_instances, cfg.n_feat, cfg.n_hidden)
            self.W2 = nn.Parameter(nn.init.xavier_normal_(self.W2))

        # Model Biases
        if cfg.b1 is None:
            self.b1 = t.zeros(cfg.n_instances, cfg.n_hidden)
        elif isinstance(cfg.b1, float):
            self.b1 = nn.Parameter(t.full((cfg.n_instances, cfg.n_hidden), cfg.b1))
        else:
            self.b1 = cfg.b1

        if cfg.b2 is None:
            self.b2 = t.zeros(cfg.n_instances, cfg.n_feat)
        elif isinstance(cfg.b2, float):
            self.b2 = nn.Parameter(t.full((cfg.n_instances, cfg.n_feat), cfg.b2))
        else:
            self.b2 = cfg.b2

        # Embed and Unembed Matrices
        if cfg.We_and_Wu:
            rand_ortho_mats = [
                t.linalg.qr(t.randn(cfg.n_feat, cfg.n_feat))[0] for _ in range(cfg.n_instances)
            ]
            self.We = t.stack(rand_ortho_mats)
            self.Wu = rearrange(self.We, "inst row col -> inst col row")

        # Sparsities
        if isinstance(cfg.feat_sparsity, float):
            self.s = t.full((cfg.n_instances, cfg.n_feat), cfg.feat_sparsity)
        else:
            self.s = cfg.feat_sparsity

        # Importances
        if isinstance(cfg.feat_importance, float):
            self.i = t.full((cfg.n_instances, cfg.n_feat), cfg.feat_importance)
        else:
            self.i = cfg.feat_importance

    def forward(
        self, 
        x: Float[t.Tensor, "batch inst feat"],
    ) -> Float[t.Tensor, ""]:
        """Runs a forward pass through the model."""

        # Embedding layer
        if self.cfg.We_and_Wu:
            x = einsum(x, self.We, "batch inst feat, inst feat col -> batch inst feat")

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
            y = einsum(y, self.Wu, "batch inst feat, inst feat col -> batch inst feat")

        return y

"""Contains a toy model class that can (possibly) illustrate computation in superposition.

tms-cis, simple-relu, and res-mlp toy models can all be created with the same class, `Cis`.
"""

from dataclasses import dataclass, field
from typing import Callable, List
from tqdm.notebook import tqdm

import numpy as np
import torch as t

from einops import einsum, rearrange, reduce
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
    b1: float | Float[t.Tensor, "inst hid"] | None = None
    b2: float | Float[t.Tensor, "inst hid"] | None = 0.0
    W1_as_W2T: bool = False  # W2 is learned if False, else W2 = W1.T
    We_and_Wu: bool = False  # if True, use fixed, random orthogonal embed and unembed matrices
    We_dim: int = 1000  # if We_and_Wu, this is the dim of the embedding space
    skip_cnx: bool = False  # if True, skip connection from in to out is added
    noise_params: dict | None = None  # params for the noise matrix
    dtype: t.dtype = t.float32  # dtype for all tensors in the model

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
        self.device = device
        self.dtype = cfg.dtype
        n_feat = cfg.n_feat

        # Embed and Unembed Matrices
        if cfg.We_and_Wu:
            rand_unit_mats = [
                F.normalize(t.randn(cfg.We_dim, cfg.n_feat, dtype=self.dtype), dim=0, p=2)
                for _ in range(cfg.n_instances)
            ]
            self.We = t.stack(rand_unit_mats).to(device)
            self.Wu = rearrange(self.We, "inst emb feat -> inst feat emb")
            n_feat = cfg.We_dim
        
        # Noise matrix
        if cfg.noise_params is not None:
            self.noise_coeff = cfg.noise_params["noise_coeff"]
            if cfg.noise_params.get("learned", False):
                self.noise_coeff = nn.Parameter(
                    t.tensor(self.noise_coeff, dtype=self.dtype, device=device)
                )
            
            matrix_type = cfg.noise_params["matrix_type"]
            self.noise_base = t.zeros(
                cfg.n_instances, cfg.n_feat, cfg.n_feat, dtype=self.dtype, device=device
            )
            
            if matrix_type == "asymmetric":
                # Fill with random noise except on diagonal
                self.noise_base = t.randn_like(self.noise_base)
                idx = t.arange(cfg.n_feat, device=device)
                self.noise_base[:, idx, idx] = 0.0
                
            elif matrix_type == "symmetric":
                # Create a symmetric noise matrix with zeros on diagonal
                tril_idxs = t.tril_indices(cfg.n_feat, cfg.n_feat, offset=-1)
                base_noise = t.randn(cfg.n_instances, tril_idxs.shape[1], device=device)
                self.noise_base[:, tril_idxs[0], tril_idxs[1]] = base_noise
                self.noise_base[:, tril_idxs[1], tril_idxs[0]] = base_noise
                
            elif matrix_type == "rank-r":
                # Precompute rank-r base with zeros on diagonal
                r = cfg.noise_params["r"]
                Q, _ = t.linalg.qr(t.randn(cfg.n_instances, cfg.n_feat, r, device=device), mode="reduced")
                self.noise_base = einsum(Q, Q, "inst feat r, inst feat2 r -> inst feat feat2")
                idx = t.arange(cfg.n_feat, device=device)
                self.noise_base[:, idx, idx] = 0.0
            
            elif matrix_type != "identity":
                raise ValueError(f"Unknown noise matrix type: {matrix_type}")

        # Model Weights
        self.W1 = t.empty(cfg.n_instances, cfg.n_hidden, n_feat, dtype=self.dtype, device=device)
        self.W1 = nn.Parameter(nn.init.xavier_normal_(self.W1)).type(self.dtype)
        if cfg.W1_as_W2T:
            self.W2 = self.W1.transpose(-1, -2)
        else:
            self.W2 = t.empty(cfg.n_instances, n_feat, cfg.n_hidden, dtype=self.dtype, device=device)
            self.W2 = nn.Parameter(nn.init.xavier_normal_(self.W2)).type(self.dtype)

        # Model Biases
        if cfg.b1 is None:
            self.b1 = t.zeros(cfg.n_instances, cfg.n_hidden, dtype=self.dtype, device=device)
        elif np.isscalar(cfg.b1):
            self.b1 = nn.Parameter(t.full((cfg.n_instances, cfg.n_hidden), cfg.b1, dtype=self.dtype))
        else:
            self.b1 = nn.Parameter(cfg.b1.to(dtype=self.dtype, device=device))

        if cfg.b2 is None:
            self.b2 = t.zeros(cfg.n_instances, n_feat, dtype=self.dtype, device=device)
        elif np.isscalar(cfg.b2):
            self.b2 = nn.Parameter(t.full((cfg.n_instances, n_feat), cfg.b2, dtype=self.dtype))
        else:
            self.b2 = nn.Parameter(cfg.b2.to(dtype=self.dtype, device=device))
        
        self.to(device)

    def forward(
        self, 
        x: Float[t.Tensor, "batch inst feat"],
        res_factor: float = 1.0,  # factor for skip connection
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
        if self.cfg.skip_cnx and not self.cfg.noise_params:  # avoid 2x skip cnx if noise
            y += (x * res_factor)
        
        # Unembedding layer
        if self.cfg.We_and_Wu:
            y = einsum(y, self.Wu, "batch inst emb, inst feat emb -> batch inst feat")

        # Noise residual
        if self.cfg.noise_params is not None:
            # Update the noise matrix by adding identity and scaled noise_base
            Wn = t.eye(self.cfg.n_feat, dtype=self.dtype, device=self.device).expand(
                self.cfg.n_instances, -1, -1
            )
            Wn = Wn + self.noise_coeff * self.noise_base
                
            y += einsum(
                x * res_factor,
                Wn, 
                "batch inst feat, inst feat feat_out -> batch inst feat_out"
            )

        return y


# Note:
# Feature sparsity should be used in a function that generates batches.
# Feature importance should be used in a loss function.
# Both of these should be defined outside of this class, and called in a training loop.

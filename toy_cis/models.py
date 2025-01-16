"""Contains toy models that (possibly) illustrate computation in superposition."""

from dataclasses import dataclass, field
from typing import Callable, List

import torch as t

from einops import einsum
from jaxtyping import Float
from torch import nn
from t.nn import functional as F
from tqdm.notebook import tqdm

@dataclass
class CisConfig:
    """Config class for single hidden layer CiS model."""
    n_instances: int  # number of model instances
    n_feat: int  # number of features (elements) in input vector
    n_hidden: int  # number of hidden units in the model
    act_fn: List[Callable] = field(default_factory=lambda: [F.relu, F.relu])  # layer act funcs
    # Bias terms for hidden and output layers. For a given layer, if "0", biases are not trained 
    # on; if scalar, all biases have the same value; if tensor, each bias has the corresponding 
    # tensor element value.
    b1: str | float | Float[t.Tensor, "inst hid"] = field(default_factory=lambda: "0")
    b2: str | float | Float[t.Tensor, "inst hid"] = 0.0
    W1_as_W2T: bool = False  # W2 is learned if False, else W2 = W1.T
    We_and_Wu: bool = False  # if False, no embedding and unembedding layers
    Wu_as_WeT: bool = False  # if `We_and_Wu`, Wu is learned if False, else Wu = We.T
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
    W1: Float[t.Tensor, "inst feat hid"]
    W2: Float[t.Tensor, "inst hid feat"]
    b1: Float[t.Tensor, "inst hid"]
    b2: Float[t.Tensor, "inst feat"]
    s: Float[t.Tensor, "inst feat"]  # feature sparsity
    i: Float[t.Tensor, "inst feat"]  # feature importance


    def __init__(self, cfg: CisConfig):
        """Initializes model params."""
        super().__init__()
        self.cfg = cfg

        # Model Weights
        self.W1 = nn.Parameter(nn.init.xavier_normal_(t.empty(cfg.n_instances, cfg.n_feat, cfg.n_hidden)))
        self.W2 = nn.Parameter(nn.init.xavier_normal_(t.empty(cfg.n_instances, cfg.n_hidden, cfg.n_feat)))

        # Model Biases
        if cfg.b1 == "0":
            self.b1 = t.zeros(cfg.n_instances, cfg.n_hidden)
        elif isinstance(cfg.b1, float):
            self.b1 = nn.Parameter(t.full((cfg.n_instances, cfg.n_hidden), cfg.b1))
        else:
            self.b1 = cfg.b1

        if cfg.b2 == "0":
            self.b2 = t.zeros(cfg.n_instances, cfg.n_feat)
        elif isinstance(cfg.b2, float):
            self.b2 = nn.Parameter(t.full((cfg.n_instances, cfg.n_feat), cfg.b2))
        else:
            self.b2 = cfg.b2

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
        y_true: Float[t.Tensor, "batch inst feat"],
        loss_fn: Callable
    ) -> Float[t.Tensor, ""]:
        """Runs a forward pass through the model."""

        # Hidden layer
        h = einsum(x, self.W1, "batch inst feat, inst feat hid -> batch inst hid")
        h = self.cfg.act_fn[0](h + self.b1)

        # Output layer
        y = einsum(h, self.W2, "batch inst hid, inst hid feat -> batch inst feat")
        y = self.cfg.act_fn[1](y + self.b2)
        return y

    def optimize(
        self,
        x: Float[t.Tensor, "batch inst feat"],
        y_true: Float[t.Tensor, "batch inst feat"],
        loss_fn: Callable,
        optimizer: t.optim.Optimizer,
        steps: int,
        logging_freq: int
    ):
        """Optimizes the model."""
        losses = []
        pbar = tqdm(range(steps), desc="Training")

        for step in pbar:
            y = self.forward(x, y_true, loss_fn)
            loss = loss_fn(y, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Log progress
            if step % logging_freq == 0 or (step + 1 == steps):
                losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return losses

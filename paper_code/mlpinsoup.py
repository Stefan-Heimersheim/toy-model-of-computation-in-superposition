import math
import time
from abc import abstractmethod
from collections.abc import Iterable
from typing import Callable, Literal

import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torch import Tensor, nn
from tqdm import tqdm


class MLP(nn.Module):
    """A simple MLP module without biases."""

    def __init__(self, n_features: int, d_mlp: int, device: str = None):
        super().__init__()
        self.n_features = n_features
        self.d_mlp = d_mlp
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

        self.w_in: Float[Tensor, "n_features d_mlp"] = nn.Parameter(torch.empty(n_features, d_mlp, device=self.device))
        self.w_out: Float[Tensor, "d_mlp n_features"] = nn.Parameter(torch.empty(d_mlp, n_features, device=self.device))
        self.relu = nn.ReLU()

        nn.init.kaiming_uniform_(self.w_in)
        nn.init.kaiming_uniform_(self.w_out)

    def forward(self, x: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        mid_pre_act: Float[Tensor, "batch d_mlp"] = einops.einsum(x, self.w_in, "batch n_features, n_features d_mlp -> batch d_mlp")
        mid: Float[Tensor, "batch d_mlp"] = self.relu(mid_pre_act)
        out: Float[Tensor, "batch n_features"] = einops.einsum(mid, self.w_out, "batch d_mlp, d_mlp n_features -> batch n_features")
        return out

    def handcode_naive_mlp(self, bias_strength: float = 0.0, implement_features: list[int] = None):
        """Set weights to implement the first half of the features, optionally with bias for the second half."""
        n_features = self.n_features
        d_mlp = self.d_mlp

        # Set weights to identity for first half of features
        self.w_in.data = torch.zeros(n_features, d_mlp, device=self.device)
        self.w_out.data = torch.zeros(d_mlp, n_features, device=self.device)

        for i, f in enumerate(range(d_mlp)):
            self.w_in.data[f, i] = 1.0
            self.w_out.data[i, f] = 1.0

        # Add bias for second half of output features
        if n_features > d_mlp:
            self.w_out.data[:, d_mlp:n_features] = bias_strength

    def plot_weights(self) -> plt.Figure:
        """Plot the weights of the MLP."""
        fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
        cmap = plt.cm.get_cmap("RdBu")

        axes[0].set_title("W_in")
        axes[1].set_title("W_out.T")

        absmax = self.w_in.data.abs().max()
        im0 = axes[0].imshow(self.w_in.data.detach().cpu().numpy(), cmap=cmap, vmin=-absmax, vmax=absmax)
        absmax = self.w_out.data.abs().max()
        im1 = axes[1].imshow(self.w_out.data.T.detach().cpu().numpy(), cmap=cmap, vmin=-absmax, vmax=absmax)

        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        return fig

    def plot_input_output_behaviour(self, ax: plt.Axes | None = None) -> plt.Figure:
        """Plot the input-output behaviour of the MLP."""
        fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.get_figure(), ax)
        cmap = plt.cm.get_cmap("viridis")
        for i in range(self.n_features):
            test_input = torch.zeros(1024, self.n_features, device=self.device)
            test_input[:, i] = torch.linspace(-1, 1, 1024, device=self.device)
            test_output = self(test_input).detach()
            c = cmap(i / self.n_features)
            ax.plot(test_input[:, i].cpu(), test_output[:, i].cpu(), color=c)
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.grid(True, alpha=0.3)
        fig.suptitle("Input-output behaviour for individual features")
        return fig

    def plot_weight_bars(self, bar_label: str = "MLP neuron index", cmap: bool = True, ax: plt.Axes | None = None) -> plt.Axes:
        """Plots weights for each input_feature-hidden_neuron pair as stacked bars. Closely follows Jai's toy-model-of-computation-in-superposition/toy_cis/plot.py"""
        palette = "inferno"
        W = einops.einsum(self.w_in, self.w_out, "n_features d_mlp, d_mlp n_features -> d_mlp n_features")
        W = W.cpu().detach().numpy()
        d_mlp, n_features = W.shape
        x = np.arange(n_features)
        colors = sns.color_palette(palette, d_mlp)
        fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.get_figure(), ax)
        bottom_pos = np.zeros(n_features)
        bottom_neg = np.zeros(n_features)
        for i in range(d_mlp):
            mask_pos = W[i] >= 0
            mask_neg = W[i] < 0
            if np.any(mask_pos):
                ax.bar(x[mask_pos], W[i][mask_pos], bottom=bottom_pos[mask_pos], label=f"{bar_label} {i}", color=colors[i])
                bottom_pos[mask_pos] += W[i][mask_pos]
            if np.any(mask_neg):
                ax.bar(x[mask_neg], W[i][mask_neg], bottom=bottom_neg[mask_neg], label=f"{bar_label} {i}", color=colors[i])
                bottom_neg[mask_neg] += W[i][mask_neg]
        if cmap:
            sm = ScalarMappable(norm=Normalize(vmin=1, vmax=d_mlp), cmap=mpl.colormaps[palette])
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)  # type: ignore
            cbar.set_label(bar_label)
        ax.set_xlim(-0.5, n_features - 0.5)
        return ax


class SparseDataset:
    """Dataset that generates inputs and labels for a sparse MLP."""

    def __init__(self, n_features: int, p: float, device: str = None, seed: int | None = None, exactly_one_active_feature: bool = False):
        self.n_features = n_features
        self.set_p(p, exactly_one_active_feature)
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed if seed is not None else int(time.time())
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.relu = nn.ReLU()

    def set_p(self, p: float, exactly_one_active_feature: bool = False):
        self.p = p
        self.exactly_one_active_feature = exactly_one_active_feature

    def _generate_inputs(self, batch_size: int, p: float | None = None) -> Float[Tensor, "batch n_features"]:
        p = p or self.p
        batch = torch.zeros((batch_size, self.n_features), device=self.device)
        mask = torch.rand(batch.shape, device=self.device, generator=self.generator) < p
        if self.exactly_one_active_feature:
            active_features = torch.randint(0, self.n_features, (batch_size,), device=self.device, generator=self.generator).unsqueeze(1)
            values = torch.rand(batch_size, 1, device=self.device, generator=self.generator) * 2 - 1
            batch.scatter_(1, active_features, values)
        else:
            values = torch.rand(batch.shape, device=self.device, generator=self.generator) * 2 - 1
            batch = values * mask
        return batch

    @abstractmethod
    def _generate_labels(self, inputs: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        pass

    def generate_batch(self, batch_size: int, p: float | None = None) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch n_features"]]:
        inputs = self._generate_inputs(batch_size, p)
        labels = self._generate_labels(inputs)
        return inputs, labels


class CleanDataset(SparseDataset):
    """Dataset that generates inputs and labels = ReLU(inputs)."""

    def __init__(self, n_features: int, p: float, device: str = None, seed: int | None = None, exactly_one_active_feature: bool = False):
        super().__init__(n_features, p, device=device, seed=seed, exactly_one_active_feature=exactly_one_active_feature)

    def _generate_labels(self, inputs: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        return self.relu(inputs)


class ResidTransposeDataset(SparseDataset):
    """Dataset that generates inputs and labels = ReLU(inputs) + (W_E.T W_E) @ inputs."""

    def __init__(self, n_features: int, d_embed: int, p: float, device: str = None, seed: int | None = None, exactly_one_active_feature: bool = False, scale: float | nn.Parameter = 1.0):
        super().__init__(n_features, p, device=device, seed=seed, exactly_one_active_feature=exactly_one_active_feature)
        self.d_embed = d_embed
        self.W_E: Float[Tensor, "n_features d_embed"] = torch.randn(self.n_features, d_embed, device=self.device, generator=self.generator)
        self.W_E = F.normalize(self.W_E, dim=1)
        self.scale = scale if isinstance(scale, nn.Parameter) else nn.Parameter(torch.tensor(scale, device=self.device, dtype=torch.float32))

    @property
    def M(self) -> Float[Tensor, "n_features n_features"]:
        return einops.einsum(self.W_E, self.W_E.T, "n_in d_embed, d_embed n_out -> n_out n_in") - torch.eye(self.n_features, self.n_features, device=self.device)

    @property
    def Mscaled(self) -> Float[Tensor, "n_features n_features"]:
        return self.M * torch.abs(self.scale)

    def _generate_labels(self, inputs: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        return self.relu(inputs) + einops.einsum(self.Mscaled, inputs, "n_out n_in, batch n_in -> batch n_out")


class NoisyDataset(SparseDataset):
    """Dataset that generates inputs and labels = ReLU(inputs) + M @ inputs."""

    def __init__(
        self,
        n_features: int,
        p: float,
        device: str | None = None,
        seed: int | None = None,
        scale: float | nn.Parameter = 0.0225,
        exactly_one_active_feature: bool = False,
        **kwargs,
    ):
        super().__init__(n_features, p, device=device, seed=seed, exactly_one_active_feature=exactly_one_active_feature)
        self.M = self._generate_M(**kwargs)
        self.scale = torch.tensor(scale, device=self.device) if isinstance(scale, float) else scale

    @property
    def Mscaled(self) -> Float[Tensor, "n_features n_features"]:
        return self.M * torch.abs(self.scale)

    def _generate_M(
        self,
        rank: int | None = None,
        symmetric: bool = False,
        zero_diagonal: bool = True,
        U_equals_V: bool = False,  # only used if rank is not None
        normalize_rows: bool = False,  # only used if rank is not None
        flat_spectrum: bool = False,  # only used if rank is not None
        align_to_neurons: bool = False,  # only used if rank is not None
    ) -> Float[Tensor, "n_features n_features"]:
        if U_equals_V and not zero_diagonal:
            print("Warning: U_equals_V is True but zero_diagonal is False, this will result in a priviledged diagonal")
        if rank is None:
            M = torch.randn(self.n_features, self.n_features, device=self.device, generator=self.generator)
        elif align_to_neurons:
            d_mlp = self.n_features // 2
            U_half = torch.randn(d_mlp, rank, device=self.device, generator=self.generator)
            V_half = torch.randn(d_mlp, rank, device=self.device, generator=self.generator) if not U_equals_V else U_half
            zeros = torch.zeros(self.n_features - d_mlp, rank, device=self.device)
            U = torch.cat([U_half, zeros], dim=0)
            V = torch.cat([V_half, zeros], dim=0)
            M = U @ V.T
        else:
            U = torch.randn(self.n_features, rank, device=self.device, generator=self.generator)
            V = torch.randn(self.n_features, rank, device=self.device, generator=self.generator) if not U_equals_V else U
            if flat_spectrum:
                U, _ = torch.linalg.qr(U)
                V, _ = torch.linalg.qr(V)
                assert U.shape == (self.n_features, rank) and V.shape == (self.n_features, rank)
            if normalize_rows:
                U = F.normalize(U, dim=1)
                V = F.normalize(V, dim=1)
            M = U @ V.T
        if zero_diagonal:
            M.fill_diagonal_(0)
        if symmetric:
            M = torch.triu(M) + torch.triu(M, diagonal=1).T
        return M

    def _generate_labels(self, inputs: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        return self.relu(inputs) + einops.einsum(self.Mscaled, inputs, "n_out n_in, batch n_in -> batch n_out")


def train(model: MLP, dataset: SparseDataset, batch_size: int = 2048, steps: int = 10_000, cosine_scheduler: bool = True) -> list[float]:
    parameters = list(model.parameters())
    if isinstance(dataset, NoisyDataset) and isinstance(dataset.scale, nn.Parameter):
        parameters.append(dataset.scale)
    optimizer = torch.optim.AdamW(parameters, lr=3e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps) if cosine_scheduler else None
    losses = []
    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        inputs, labels = dataset.generate_batch(batch_size)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ((outputs - labels) ** 2).mean()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        if step % 100 == 0:
            pbar.set_postfix({"loss": loss.item()})
    return losses


def evaluate(model: Callable[[Float[Tensor, "batch n_features"]], Float[Tensor, "batch n_features"]], dataset: SparseDataset, n_samples: int | None = None, batch_size: int = 100_000) -> float:
    frac_zero_samples = (1 - dataset.p) ** dataset.n_features
    n_samples = n_samples or int(1_000_000 / (1 - frac_zero_samples))
    n_batches = math.ceil(n_samples / batch_size)
    with torch.no_grad():
        losses = []
        for _ in range(n_batches):
            inputs, labels = dataset.generate_batch(batch_size)
            outputs = model(inputs)
            loss = ((outputs - labels) ** 2).mean()
            losses.append(loss.item())
    return np.mean(losses)


def plot_loss_of_input_sparsity(
    models: MLP | list[MLP],
    datasets: SparseDataset | list[SparseDataset],
    ps: Iterable[float] = np.geomspace(1e-3, 1, 100),
    n_batches: int = 10,
    batch_size: int | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    ax: plt.Axes | None = None,
    highlight_ps: list[float] | float | None = None,
    show_naive: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.get_figure(), ax)
    models = [models] if isinstance(models, MLP) else models
    highlight_ps = [highlight_ps] if isinstance(highlight_ps, float) else highlight_ps
    datasets = [datasets] * len(models) if isinstance(datasets, SparseDataset) else datasets

    n_features = models[0].n_features
    d_mlp = models[0].d_mlp
    ps = np.array(ps)

    def naive_loss(n_features: int, d_mlp: int, p: float) -> float:
        return (n_features - d_mlp) / n_features * p / 6

    naive_adj_losses = naive_loss(n_features, d_mlp, ps) / ps
    highlight_adj_losses = []
    for i, (model, dataset) in tqdm(enumerate(zip(models, datasets, strict=True)), total=len(models), desc="Plotting"):
        with torch.no_grad():
            losses = []
            for p in ps:
                dataset.set_p(p)
                loss = evaluate(model, dataset)
                losses.append(loss)
        adj_losses = np.array(losses) / ps
        color = colors[i] if colors else None
        ls = linestyles[i] if linestyles else None
        label = labels[i] if labels else None
        ax.plot(ps, adj_losses, label=label, color=color, ls=ls)
        if highlight_ps is not None:
            with torch.no_grad():
                p = highlight_ps[i]
                dataset.set_p(p)
                loss_at_p = evaluate(model, dataset)
                highlight_adj_losses.append(loss_at_p / p)
    if show_naive:
        ax.plot(ps, naive_adj_losses, color="k", ls="--", label="Naive solution")
    if highlight_ps is not None:
        ax.plot(highlight_ps, highlight_adj_losses, color="k", marker="o", ls=":", label="p_eval = p_train")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Feature probability $p$")
    ax.set_ylabel("Loss per feature $L / p$")
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(ncols=3, loc="upper left")
    ax.grid(True, alpha=0.3)
    return fig


sns_colorblind = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"]


def compare_WoutWin_Mscaled(model: MLP, dataset: NoisyDataset, ax: plt.Axes | None = None):
    WoutWin = einops.einsum(model.w_in, model.w_out, "d_in d_mlp, d_mlp d_out -> d_out d_in").cpu().detach()
    Mscaled = dataset.Mscaled.cpu().detach()
    ax = ax or plt.subplots(constrained_layout=True)[1]
    ax.set_title("Entries of $M$ and $W_{\\rm out} W_{\\rm in}$")
    ax.set_xlabel("$(W_{\\rm out} W_{\\rm in})_{i,j}$")
    ax.set_ylabel("$M_{i,j}$")
    ax.scatter(WoutWin.flatten(), Mscaled.flatten())

    diagonal_M = torch.diag(Mscaled)
    diagonal_WoutWin = torch.diag(WoutWin)
    offdiag_M = Mscaled - torch.diag(torch.diag(Mscaled))
    offdiag_WoutWin = WoutWin - torch.diag(torch.diag(WoutWin))
    ax.scatter(offdiag_WoutWin.flatten(), offdiag_M.flatten(), label="Off-diagonal", alpha=0.5)
    ax.scatter(diagonal_WoutWin.flatten(), diagonal_M.flatten(), label="Diagonal", alpha=0.7)

    offdiag_x = offdiag_WoutWin.flatten().numpy()
    offdiag_y = offdiag_M.flatten().numpy()
    slope = np.sum(offdiag_x * offdiag_y) / np.sum(offdiag_x**2)
    offdiag_y_pred = slope * offdiag_x
    ax.plot(offdiag_x, offdiag_y_pred, "r-", label=f"Off-diag fit: y={slope:.3f}x")
    diag_x = diagonal_WoutWin.flatten().numpy()
    diag_y = diagonal_M.flatten().numpy()
    diag_x_mean = np.mean(diag_x)
    diag_y_mean = np.mean(diag_y)
    slope = np.sum((diag_x - diag_x_mean) * (diag_y - diag_y_mean)) / np.sum((diag_x - diag_x_mean) ** 2)
    intercept = diag_y_mean - slope * diag_x_mean
    x_intercept = -intercept / slope
    diag_x_range = np.linspace(np.min(diag_x), np.max(diag_x), 100)
    diag_y_pred = slope * diag_x_range + intercept
    ax.plot(diag_x_range, diag_y_pred, "g-", label=f"Diag fit: y={slope:.3f}(x-{x_intercept:.3f})")
    ax.legend()

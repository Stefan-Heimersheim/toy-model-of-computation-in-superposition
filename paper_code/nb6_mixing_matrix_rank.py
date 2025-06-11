import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from mlpinsoup import MLP, NoisyDataset, evaluate, train
from torch import Tensor
from tqdm import tqdm


def create_mixing_matrix(n_features: int, rank: int | None = None, device: str = None) -> Float[Tensor, "n_features n_features"]:
    """Create a mixing matrix with specified rank."""
    device = device or "cuda" if torch.cuda.is_available() else "cpu"
    if rank is None or rank >= n_features:
        M = torch.randn(n_features, n_features, device=device)
    else:
        U = torch.randn(n_features, rank, device=device)
        V = torch.randn(n_features, rank, device=device)
        M = torch.mm(U, V.T)
    M.fill_diagonal_(0)
    return M


def train_with_learnable_scale(model: MLP, dataset: NoisyDataset, batch_size: int = 1024, steps: int = 10_000) -> tuple[list[float], list[float]]:
    """Train model with learnable noise scale added to the optimization parameters."""
    all_params = list(model.parameters()) + [dataset.scale]
    optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-2)
    losses = []
    scales = []
    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        inputs, labels = dataset.generate_batch(batch_size)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ((outputs - labels) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        scales.append(torch.abs(dataset.scale).item())
        if step % 100 == 0:
            pbar.set_postfix({"loss": loss.item(), "scale": dataset.scale.item()})
    return losses, scales


# Experiment parameters
p = 0.01
n_features = 100
d_mlp = 50
n_steps = 30_000
batch_size_train = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"
baseline_scale = 0.035

ranks = [None, 100, 50, 20, 15, 10, 7, 5]  # None means full rank
rank_labels = [f"Rank {r}" if r is not None else "Full rank" for r in ranks]

models = []
datasets = []
final_losses = []
final_scales = []
effective_ranks = []
svd_spectra = []

print("Training models with different mixing matrix ranks...")

for rank, label in zip(ranks, rank_labels):
    print(f"\nTraining {label} model...")

    model = MLP(n_features=n_features, d_mlp=d_mlp, device=device)
    # Create learnable scale parameter and dataset
    learnable_scale = nn.Parameter(torch.tensor(0.01, device=device))
    dataset = NoisyDataset(n_features=n_features, p=p, device=device, scale=learnable_scale, exactly_one_active_feature=True, symmetric=False, zero_diagonal=True)

    # Create mixing matrix with specified rank
    M = create_mixing_matrix(n_features, rank, device)
    dataset.M = M

    # Calculate effective rank (sum of normalized singular values)
    U, S, V = torch.linalg.svd(dataset.M)
    S_normalized = S / S.sum()
    eff_rank = torch.exp(-torch.sum(S_normalized * torch.log(S_normalized + 1e-12))).item()
    effective_ranks.append(eff_rank)

    losses, scales = train_with_learnable_scale(model, dataset, batch_size=batch_size_train, steps=n_steps)

    _, S_trained, _ = torch.linalg.svd(dataset.M * dataset.scale.detach())
    svd_spectra.append(S_trained.cpu())

    final_loss = evaluate(model, dataset, n_samples=1_000_000)
    final_losses.append(final_loss / p)
    final_scales.append(torch.abs(dataset.scale).item())

    models.append(model)
    datasets.append(dataset)

    print(f"{label}: Final loss/p = {final_loss / p:.4f}, Final scale = {torch.abs(dataset.scale).item():.4f}, Effective rank = {eff_rank:.2f}")

# Add baseline comparison with original NoisyDataset (optimal scale from nb3)
print(f"\nTraining baseline with original NoisyDataset (optimal scale ~{baseline_scale})...")
baseline_dataset = NoisyDataset(
    n_features=n_features,
    p=p,
    scale=baseline_scale,  # Approximate optimal scale from nb3
    exactly_one_active_feature=True,
    symmetric=False,
    zero_diagonal=True,
    device=device,
)
baseline_model = MLP(n_features=n_features, d_mlp=d_mlp, device=device)
train(baseline_model, baseline_dataset, batch_size=batch_size_train, steps=n_steps)
baseline_loss = evaluate(baseline_model, baseline_dataset, n_samples=1_000_000)

# Calculate effective rank of baseline dataset
U_baseline, S_baseline, V_baseline = torch.linalg.svd(baseline_dataset.M)
S_baseline_normalized = S_baseline / S_baseline.sum()
baseline_eff_rank = torch.exp(-torch.sum(S_baseline_normalized * torch.log(S_baseline_normalized + 1e-12))).item()

print(f"Baseline (original): Final loss/p = {baseline_loss / p:.4f}, Fixed scale = {baseline_scale}, Effective rank = {baseline_eff_rank:.2f}")

# Add baseline to results for plotting
models.append(baseline_model)
datasets.append(baseline_dataset)
final_losses.append(baseline_loss / p)
final_scales.append(baseline_scale)
effective_ranks.append(baseline_eff_rank)
rank_labels.append("Baseline (fixed scale)")
ranks.append("Baseline")
svd_spectra.append(S_baseline.cpu())


####################################################################
#                          Plotting                                #
####################################################################


# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)

# Plot masks for separating learnable vs baseline
learnable_mask = [i < len(effective_ranks) - 1 for i in range(len(effective_ranks))]
baseline_mask = [not m for m in learnable_mask]

# Plot 1: Performance - both effective rank and theoretical rank on same plot
learnable_ranks = [effective_ranks[i] for i in range(len(effective_ranks)) if learnable_mask[i]]
learnable_losses = [final_losses[i] for i in range(len(final_losses)) if learnable_mask[i]]
baseline_ranks = [effective_ranks[i] for i in range(len(effective_ranks)) if baseline_mask[i]]
baseline_losses = [final_losses[i] for i in range(len(final_losses)) if baseline_mask[i]]

# Theoretical ranks
theoretical_ranks = [n_features if r is None else r for r in ranks[:-1]]  # Exclude baseline
theoretical_ranks.append(baseline_eff_rank)  # Use effective rank for baseline
learnable_theoretical_ranks = theoretical_ranks[:-1]
learnable_theoretical_losses = final_losses[:-1]

# Plot both effective and theoretical rank curves
# Handle full rank separately with different marker
full_rank_idx = next(i for i, r in enumerate(ranks[:-1]) if r is None)
other_indices = [i for i, r in enumerate(ranks[:-1]) if r is not None]

# Plot non-full-rank points
if other_indices:
    other_theoretical_ranks = [learnable_theoretical_ranks[i] for i in other_indices]
    other_theoretical_losses = [learnable_theoretical_losses[i] for i in other_indices]
    other_effective_ranks = [learnable_ranks[i] for i in other_indices]
    other_effective_losses = [learnable_losses[i] for i in other_indices]
    ax1.plot(other_effective_ranks, other_effective_losses, "o-", color="tab:blue", label="Learned scale")

# Plot full rank with different marker (effective only)
ax1.plot(learnable_ranks[full_rank_idx], learnable_losses[full_rank_idx], "s", color="tab:orange", label="Learned scale (full rank)")

# Plot baseline (effective only)
ax1.plot(baseline_ranks, baseline_losses, "v", color="tab:red", label=f"Fixed scale ({baseline_scale}; full rank)")

# Add labels on plot points
for i, label in enumerate(rank_labels[1:-1]):  # Exclude baseline from annotations
    ax1.annotate(label.replace("Rank ", ""), (learnable_ranks[i], learnable_losses[i]), xytext=(5, 5), textcoords="offset points", color="black")
for i, label in enumerate(rank_labels[-1:]):  # Just baseline
    ax1.annotate("Baseline (full rank)", (baseline_ranks[0], baseline_losses[0]), xytext=(5, 5), textcoords="offset points", color="red")

ax1.set_xlabel("Mixing Matrix Rank")
ax1.set_ylabel("Final Loss / p")
ax1.set_title("Model Performance vs Matrix Rank")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Learned scale vs specified rank
learnable_scales = [final_scales[i] for i in range(len(final_scales)) if learnable_mask[i]]
baseline_scales = [final_scales[i] for i in range(len(final_scales)) if baseline_mask[i]]

# Handle full rank separately with different marker
if other_indices:
    other_theoretical_ranks = [learnable_theoretical_ranks[i] for i in other_indices]
    other_scales = [learnable_scales[i] for i in other_indices]
    ax2.plot(other_theoretical_ranks, other_scales, "o-", color="tab:blue", label="Learned scale")

# Plot full rank with different marker
ax2.plot([learnable_ranks[full_rank_idx]], [learnable_scales[full_rank_idx]], "s", color="tab:orange", label="Learned scale (full rank)")

# Plot baseline
ax2.plot([baseline_eff_rank], baseline_scales, "o", color="tab:red", label=f"Fixed scale ({baseline_scale}; full rank)")

ax2.annotate("Baseline (full rank)", (baseline_eff_rank, baseline_scales[0]), xytext=(5, 5), textcoords="offset points", color="red")

ax2.set_xlabel("Specified Rank")
ax2.set_ylabel("Noise Scale")
ax2.set_title("Optimal Noise Scale vs Matrix Rank")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Singular value spectrum for different ranks
colors = plt.cm.viridis(np.linspace(0, 1, len(svd_spectra) - 1))
for i, (svd_spectrum, label, color) in enumerate(zip(svd_spectra[:-1], rank_labels[:-1], colors)):
    if ranks[i] is None:  # Full rank case
        ax3.plot(svd_spectrum, color="tab:orange", label=label, zorder=10, ls="--")
    else:
        ax3.plot(svd_spectrum, color=color, label=label)

# Add baseline dataset singular values
U_baseline, S_baseline, V_baseline = torch.linalg.svd(baseline_dataset.M * baseline_dataset.scale)
ax3.plot(S_baseline.cpu(), color="tab:red", label="Baseline")

ax3.set_xlabel("Singular Value Index")
ax3.set_ylabel("Singular Value Magnitude")
ax3.set_title("Mixing Matrix Singular Value Spectra")
ax3.legend(ncol=2)
ax3.set_yscale("log")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"plots/nb6_mixing_matrix_rank_{n_steps=}.png", dpi=150)
plt.show()

# Print summary
print("\n" + "=" * 60)
print("SUMMARY: Mixing Matrix Rank Analysis")
print("=" * 60)
print(f"{'Rank':<12} {'Eff. Rank':<10} {'Loss/p':<10} {'Scale':<10}")
print("-" * 60)
for rank, eff_rank, loss, scale, label in zip(ranks, effective_ranks, final_losses, final_scales, rank_labels):
    rank_str = str(rank) if rank is not None else "Full"
    print(f"{rank_str:<12} {eff_rank:<10.2f} {loss:<10.4f} {scale:<10.4f}")

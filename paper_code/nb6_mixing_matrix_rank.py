import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mlpinsoup import MLP, NoisyDataset, evaluate, train


def effective_rank(M):
    """Calculate effective rank using entropy of normalized singular values."""
    S = torch.linalg.svdvals(M)
    S_norm = S / S.sum()
    return torch.exp(-torch.sum(S_norm * torch.log(S_norm + 1e-12))).item()


# Parameters
p, n_features, d_mlp = 0.01, 100, 50
n_steps = 25_000
baseline_scale = 0.035
device = "cuda" if torch.cuda.is_available() else "cpu"
ranks = [None, 100, 50, 20, 15, 10, 7, 5]

results = []

# Train models with learnable scale
for rank in ranks:
    model = MLP(n_features, d_mlp, device)
    scale = nn.Parameter(torch.tensor(0.01, device=device))
    dataset = NoisyDataset(n_features, p, device=device, rank=rank, scale=scale, exactly_one_active_feature=True)
    train(model, dataset, steps=n_steps)
    loss = evaluate(model, dataset, n_samples=1_000_000) / p
    results.append({"rank": rank, "model": model, "dataset": dataset, "loss": loss, "scale": abs(scale.item()), "effective_rank": effective_rank(dataset.M), "svd_vals": torch.linalg.svdvals(dataset.M * scale.detach()).cpu()})
    print(f"Rank {rank}: loss/p={loss:.4f}, scale={abs(scale.item()):.4f}")

# Baseline with fixed scale
baseline_model = MLP(n_features, d_mlp, device)
baseline_dataset = NoisyDataset(n_features, p, device=device, scale=baseline_scale, exactly_one_active_feature=True)
train(baseline_model, baseline_dataset, steps=n_steps)
baseline_loss = evaluate(baseline_model, baseline_dataset, n_samples=1_000_000) / p
baseline_eff_rank = effective_rank(baseline_dataset.M)
results.append({"rank": "Baseline", "model": baseline_model, "dataset": baseline_dataset, "loss": baseline_loss, "scale": baseline_scale, "effective_rank": baseline_eff_rank, "svd_vals": torch.linalg.svdvals(baseline_dataset.M * baseline_dataset.scale).cpu()})
print(f"Baseline: loss/p={baseline_loss:.4f}, scale={baseline_scale}")

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Separate learnable vs baseline
learnable = [r for r in results[:-1]]
baseline = results[-1]

# Plot 1: Performance vs effective rank
eff_ranks = [r["effective_rank"] for r in learnable]
losses = [r["loss"] for r in learnable]
full_idx = next(i for i, r in enumerate(learnable) if r["rank"] is None)
# Non-full rank points
other_idx = [i for i, r in enumerate(learnable) if r["rank"] is not None]
# Plot with annotations for specified rank
for i in other_idx:
    ax1.plot(eff_ranks[i], losses[i], "o", color="tab:green")
    ax1.annotate(f"{learnable[i]['rank']}", (eff_ranks[i], losses[i]), xytext=(5, 0), textcoords="offset points")
ax1.plot([eff_ranks[i] for i in other_idx], [losses[i] for i in other_idx], "-", color="tab:green", label="Effective rank")
# Full rank and baseline
ax1.plot(eff_ranks[full_idx], losses[full_idx], "o", color="tab:orange", label="Full rank")
ax1.plot(baseline["effective_rank"], baseline["loss"], "s", color="tab:red", label="Baseline")
ax1.set_xlabel("Effective rank")
ax1.set_ylabel("Final loss / p")
ax1.set_title("Performance vs matrix rank")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Scale vs specified rank
scales = [r["scale"] for r in learnable]
ax2.plot([eff_ranks[i] for i in other_idx], [scales[i] for i in other_idx], "o-", color="tab:blue", label="Learned scale")
ax2.plot(eff_ranks[full_idx], scales[full_idx], "o", color="tab:orange", label="Full rank")
ax2.plot(baseline["effective_rank"], baseline["scale"], "s", color="tab:red", label="Baseline")
ax2.set_xlabel("Effective rank")
ax2.set_ylabel("Noise scale")
ax2.set_title("Noise scale vs matrix rank")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: SVD spectra
colors = plt.cm.viridis(np.linspace(0, 1, len(learnable)))
for i, (result, color) in enumerate(zip(learnable, colors)):
    style = ("tab:orange", "--") if result["rank"] is None else (color, "-")
    ax3.plot(result["svd_vals"], color=style[0], ls=style[1], label=f"Rank {result['rank']}" if result["rank"] else "Full rank")
ax3.plot(baseline["svd_vals"], color="tab:red", label="Baseline")
ax3.set_xlabel("Singular value index")
ax3.set_ylabel("Singular value magnitude")
ax3.set_title("Singular value spectra")
ax3.set_yscale("log")
ax3.legend(ncol=2)
ax3.grid(True, alpha=0.3)

fig.savefig(f"plots/nb6_mixing_matrix_rank_n_steps={n_steps}.png", dpi=150)
plt.show()

# Summary
print(f"\n{'Rank':<12}{'Eff. Rank':<10}{'Loss/p':<10}{'Scale':<10}")
print("-" * 50)
for r in results:
    rank_str = str(r["rank"]) if r["rank"] is not None else "Full"
    print(f"{rank_str:<12}{r['effective_rank']:<10.2f}{r['loss']:<10.4f}{r['scale']:<10.4f}")

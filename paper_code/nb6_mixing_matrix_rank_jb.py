import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mlpinsoup import MLP, NoisyDataset, evaluate, train


def effective_rank(M):
    """Calculate effective rank using entropy of normalized singular values."""
    S = torch.linalg.svdvals(M)
    S_norm = S / S.sum()
    return torch.exp(-torch.sum(S_norm * torch.log(S_norm + 1e-12))).item()


def main():
    """Run mixing matrix rank analysis and generate plots."""
    # Set environment and plotting parameters
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    
    # Parameters
    p, n_features, d_mlp = 0.01, 100, 50
    n_steps = 30_000
    n_runs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ranks = [1000, 70, 30, 20, 9]  # equivalent to 80, 60, 40, 30, 20
    xax_ranks = [80, 60, 40, 30, 20]
    
    print(f"Using device: {device}")
    print(f"Running analysis with {n_runs} runs per rank, {n_steps} training steps each")
    print(f"Ranks to test: {ranks}")
      # Structure to collect all data for error bars
    results_by_rank = {rank: {'losses': [], 'scales': [], 'eff_ranks': []} for rank in ranks}
    
    # Main training loop
    print("Starting training...")
    for rank in tqdm(ranks, desc="Ranks"):
        for run in tqdm(range(n_runs), desc="Runs", leave=False):
            model = MLP(n_features, d_mlp, device)
            scale = nn.Parameter(torch.tensor(0.05, device=device))
            dataset = NoisyDataset(
                n_features, 
                p, 
                device=device, 
                rank=rank, 
                scale=scale,
                exactly_one_active_feature=True
            )
            train(model, dataset, steps=n_steps)
            loss = evaluate(model, dataset, n_samples=1_000_000) / p
            eff_rank = effective_rank(dataset.M)
              # Store data for this run
            results_by_rank[rank]['losses'].append(loss)
            results_by_rank[rank]['scales'].append(abs(scale.item()))
            results_by_rank[rank]['eff_ranks'].append(eff_rank)
            
            print(f"Run {run+1}, Rank {rank}: loss/p={loss:.4f}, scale={abs(scale.item()):.4f}, eff_rank={eff_rank:.2f}")
    
    # Generate plots and summary
    print("Generating plots and summary...")
    create_plots_and_summary(results_by_rank, ranks, xax_ranks, n_runs)
    
    print("Analysis complete! Plot saved to plots/nb6_mixing_matrix_rank_jb.png")


def create_plots_and_summary(results_by_rank, ranks, xax_ranks, n_runs):
    """Create plots and print summary statistics."""
    # Extract statistics for each rank
    rank_stats = {}
    for rank in ranks:
        rank_stats[rank] = {
            'mean_loss': np.mean(results_by_rank[rank]['losses']),
            'sem_loss': np.std(results_by_rank[rank]['losses']) / np.sqrt(n_runs),
            'mean_scale': np.mean(results_by_rank[rank]['scales']),
            'sem_scale': np.std(results_by_rank[rank]['scales']) / np.sqrt(n_runs),
            'mean_eff_rank': np.mean(results_by_rank[rank]['eff_ranks']),
            'sem_eff_rank': np.std(results_by_rank[rank]['eff_ranks']) / np.sqrt(n_runs),
        }

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Extract data for plotting using xax_ranks for x-axis
    losses_mean = [rank_stats[rank]['mean_loss'] for rank in ranks]
    losses_sem = [rank_stats[rank]['sem_loss'] for rank in ranks]
    scales_mean = [rank_stats[rank]['mean_scale'] for rank in ranks]
    scales_sem = [rank_stats[rank]['sem_scale'] for rank in ranks]

    # Plot 1: Performance vs xax_ranks with shaded bounds
    ax1.plot(xax_ranks, losses_mean, 'o-', color='tab:green', linewidth=2, markersize=8)
    ax1.fill_between(xax_ranks, 
                     np.array(losses_mean) - np.array(losses_sem),
                     np.array(losses_mean) + np.array(losses_sem),
                     color='tab:green', alpha=0.3)
    ax1.axhline(y=0.083, color='black', linestyle='--', linewidth=1, label='Naive solution')

    ax1.set_xlabel('Effective rank')
    ax1.set_ylabel('L / p')
    ax1.set_title('Loss over $M$ effective rank')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scale vs xax_ranks with shaded bounds
    ax2.plot(xax_ranks, scales_mean, 'o-', color='tab:blue', linewidth=2, markersize=8)
    ax2.fill_between(xax_ranks,
                     np.array(scales_mean) - np.array(scales_sem),
                     np.array(scales_mean) + np.array(scales_sem),
                     color='tab:blue', alpha=0.3)

    ax2.set_xlabel('Effective rank')
    ax2.set_ylabel('Noise scale')
    ax2.set_title('Optimal noise scale per $M$ effective rank')
    ax2.grid(True, alpha=0.3)

    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Save the figure
    fig.savefig("plots/nb6_mixing_matrix_rank_jb.png", dpi=150)
    plt.close(fig)  # Close to free memory

    # Print summary statistics
    print(f"\n{'Rank':<6}{'Eff. Rank':<12}{'Loss/p':<12}{'Scale':<12}")
    print("-" * 50)
    for rank in ranks:
        stats = rank_stats[rank]
        print(f"{rank:<6}{stats['mean_eff_rank']:<6.2f}±{stats['sem_eff_rank']:<4.2f} "
              f"{stats['mean_loss']:<6.4f}±{stats['sem_loss']:<4.4f} "
              f"{stats['mean_scale']:<6.4f}±{stats['sem_scale']:<4.4f}")


if __name__ == "__main__":
    main()

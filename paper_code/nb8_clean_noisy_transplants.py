import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mlpinsoup import MLP, CleanDataset, NoisyDataset, evaluate, train

# Set KMP_DUPLICATE_LIB_OK=TRUE to avoid MKL errors when plotting with mpl
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rcParams.update({"font.size": 14})


def main():
    # Parameters
    p, n_features, d_mlp = 0.01, 100, 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize datasets (shared between experiments)
    clean_dataset = CleanDataset(n_features, p, device=device, exactly_one_active_feature=True)
    noisy_dataset = NoisyDataset(n_features, p, device=device, scale=0.035, exactly_one_active_feature=True)

    # Experiment 1: Clean → Noisy Training
    print("=== Experiment 1: Clean → Noisy Training ===")
    model_clean_to_noisy = MLP(n_features, d_mlp, device)

    print("Phase 1: Training on clean data for 10k steps...")
    train(model_clean_to_noisy, clean_dataset, steps=10000)

    # Get evaluation losses every 1k steps during clean training
    losses_clean_phase = []
    for step in range(0, 10000, 1000):
        eval_loss = evaluate(model_clean_to_noisy, clean_dataset, n_samples=100000) / p
        losses_clean_phase.append(eval_loss)
        print(f"Step {step}: Clean loss/p = {eval_loss:.4f}")

    print("\nPhase 2: Continuing training on noisy data for 10k steps...")
    train(model_clean_to_noisy, noisy_dataset, steps=10000)

    # Get evaluation losses every 1k steps during noisy training
    losses_noisy_phase = []
    for step in range(0, 10000, 1000):
        eval_loss = evaluate(model_clean_to_noisy, noisy_dataset, n_samples=100000) / p
        losses_noisy_phase.append(eval_loss)
        print(f"Step {step}: Noisy loss/p = {eval_loss:.4f}")

    print(f"\nFinal clean-to-noisy model loss/p: {losses_noisy_phase[-1]:.4f}")

    # Experiment 2: Noisy → Clean Training
    print("\n=== Experiment 2: Noisy → Clean Training ===")
    model_noisy_to_clean = MLP(n_features, d_mlp, device)

    print("Phase 1: Training on noisy data for 10k steps...")
    train(model_noisy_to_clean, noisy_dataset, steps=10000)

    # Get evaluation losses every 1k steps during noisy training
    losses_noisy_phase_2 = []
    for step in range(0, 10000, 1000):
        eval_loss = evaluate(model_noisy_to_clean, noisy_dataset, n_samples=100000) / p
        losses_noisy_phase_2.append(eval_loss)
        print(f"Step {step}: Noisy loss/p = {eval_loss:.4f}")

    print("\nPhase 2: Continuing training on clean data for 10k steps...")
    train(model_noisy_to_clean, clean_dataset, steps=10000)

    # Get evaluation losses every 1k steps during clean training
    losses_clean_phase_2 = []
    for step in range(0, 10000, 1000):
        eval_loss = evaluate(model_noisy_to_clean, clean_dataset, n_samples=100000) / p
        losses_clean_phase_2.append(eval_loss)
        print(f"Step {step}: Clean loss/p = {eval_loss:.4f}")

    print(f"\nFinal noisy-to-clean model loss/p: {losses_clean_phase_2[-1]:.4f}")

    # Plot the training curves for both experiments
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Plot 1: Clean-to-Noisy training
    steps_clean = np.arange(0, 10000, 1000)
    steps_noisy = np.arange(10000, 20000, 1000)
    ax1.plot(steps_clean, losses_clean_phase, 'b-', label='Clean phase', marker='o')
    ax1.plot(steps_noisy, losses_noisy_phase, 'r-', label='Noisy phase', marker='s')
    ax1.axvline(x=10000, color='gray', linestyle='--', alpha=0.7, label='Phase transition')
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Loss / p')
    ax1.set_title('Clean → Noisy Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(6.5e-2, 8.5e-2)
    ax1.set_xticks([0, 5000, 10000, 15000, 20000])

    # Plot 2: Noisy-to-Clean training
    ax2.plot(steps_clean, losses_noisy_phase_2, 'r-', label='Noisy phase', marker='o')
    ax2.plot(steps_noisy, losses_clean_phase_2, 'b-', label='Clean phase', marker='s')
    ax2.axvline(x=10000, color='gray', linestyle='--', alpha=0.7, label='Phase transition')
    ax2.set_xlabel('Training step')
    ax2.set_ylabel('Loss / p')
    ax2.set_title('Noisy → Clean Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(6.5e-2, 8.5e-2)
    ax2.set_xticks([0, 5000, 10000, 15000, 20000])

    # Save the figure
    plt.savefig("./plots/nb8_clean_noisy_transplants.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n=== Training Transplant Results ===")
    print(f"Clean→Noisy: Clean phase final loss/p = {losses_clean_phase[-1]:.4f}")
    print(f"Clean→Noisy: Noisy phase final loss/p = {losses_noisy_phase[-1]:.4f}")
    print(f"Noisy→Clean: Noisy phase final loss/p = {losses_noisy_phase_2[-1]:.4f}")
    print(f"Noisy→Clean: Clean phase final loss/p = {losses_clean_phase_2[-1]:.4f}")
    print(f"\nLoss change when switching Clean→Noisy: {losses_noisy_phase[0]/losses_clean_phase[-1]:.2f}x")
    print(f"Loss change when switching Noisy→Clean: {losses_clean_phase_2[0]/losses_noisy_phase_2[-1]:.2f}x")


if __name__ == "__main__":
    main()

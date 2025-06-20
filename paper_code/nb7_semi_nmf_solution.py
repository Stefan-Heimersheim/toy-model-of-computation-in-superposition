import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from mlpinsoup import MLP, CleanDataset, NoisyDataset, compare_WoutWin_Mscaled, evaluate
from sklearn.decomposition import NMF

# Parameters
p, n_features, d_mlp = 0.01, 100, 50
device = "cuda" if torch.cuda.is_available() else "cpu"

# Naive Solution on clean dataset
clean_dataset = CleanDataset(n_features, p, device=device, exactly_one_active_feature=True)
naive_model = MLP(n_features, d_mlp, device)
naive_model.handcode_naive_mlp(bias_strength=0.0)
naive_clean_loss = evaluate(naive_model, clean_dataset, n_samples=500_000) / p
print(f"Naive on clean: {naive_clean_loss:.4f}")

# Semi-NMF Solution on noisy dataset
scales = []
losses = []
models = []
for scale in [0.0, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
    noisy_dataset = NoisyDataset(n_features, p, device=device, exactly_one_active_feature=True, scale=scale)
    Mscaled = noisy_dataset.Mscaled.cpu().detach().numpy()
    target_matrix = np.eye(n_features) + Mscaled
    ## NMF
    nmf = NMF(n_components=d_mlp, init="random", random_state=42, max_iter=5000)
    U_init = nmf.fit_transform(np.abs(target_matrix))
    V_init = nmf.components_
    # Semi NMF
    U_pinv = np.linalg.pinv(U_init)
    V_optimal = U_pinv @ target_matrix
    U = U_init.copy()
    V = V_optimal.copy()
    initial_lr = 0.1
    final_lr = 0.001
    max_iterations = 500
    for iteration in range(max_iterations):
        progress = iteration / max_iterations
        step_size = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * progress))
        # Fix U, solve for V (unconstrained least squares)
        if np.linalg.cond(U) < 1e12:
            U_pinv = np.linalg.pinv(U)
            V = U_pinv @ target_matrix
        else:
            print("U is ill-conditioned, skipping iteration")
        # Fix V, solve for U (non-negative least squares via projected gradient)
        for _ in range(5):
            grad_U = (U @ V - target_matrix) @ V.T
            U = U - step_size * grad_U
            U = np.maximum(U, 1e-8)
    semi_nmf_model = MLP(n_features, d_mlp, device)
    semi_nmf_model.w_in.data = torch.from_numpy(U).float().to(device)
    semi_nmf_model.w_out.data = torch.from_numpy(V).float().to(device)
    semi_nmf_noisy_loss = evaluate(semi_nmf_model, noisy_dataset, n_samples=500_000) / p
    print(f"Semi-NMF on noisy scale {scale:.3f}: {semi_nmf_noisy_loss:.4f}")
    scales.append(scale)
    losses.append(semi_nmf_noisy_loss)
    models.append(semi_nmf_model)

# Plot
plt.axhline(y=naive_clean_loss, color="red", linestyle="--", label="Naive on clean")
plt.plot(scales, losses, label="Semi-NMF on noisy")
plt.xlabel("Scale")
plt.ylabel("Loss")
plt.legend()
plt.show()

for model in models:
    model.plot_weights()
    compare_WoutWin_Mscaled(model, noisy_dataset)

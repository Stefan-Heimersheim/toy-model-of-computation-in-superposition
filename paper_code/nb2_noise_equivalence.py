import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from mlpinsoup import MLP, CleanDataset, NoisyDataset, ResidTransposeDataset, evaluate, get_sns_colorblind, plot_loss_of_input_sparsity, set_seed, train
from torch import nn
from tqdm import tqdm

p = 0.01
n_features = 100
d_mlp = 50
n_steps = 10_000
batch_size_train = 2048

set_seed(42)

sns_colorblind = get_sns_colorblind()

# %% Compare clean, embed, asymmetric and symmetric noise

clean_dataset = CleanDataset(n_features=n_features, p=p)
clean_model = MLP(n_features=n_features, d_mlp=d_mlp)
losses_clean = train(clean_model, clean_dataset, batch_size=batch_size_train, steps=n_steps)
resid_transpose_dataset = ResidTransposeDataset(n_features=n_features, d_embed=1000, p=p)
resid_transpose_model = MLP(n_features=n_features, d_mlp=d_mlp)
losses_resid_transpose = train(resid_transpose_model, resid_transpose_dataset, batch_size=batch_size_train, steps=n_steps)
sigma_embed = resid_transpose_dataset.Mscaled.std().item()

scale_asym = nn.Parameter(torch.tensor(0.01))
noise_dataset = NoisyDataset(n_features=n_features, p=p, scale=scale_asym, symmetric=False, zero_diagonal=False)
noisy_model = MLP(n_features=n_features, d_mlp=d_mlp)
losses_noisy = train(noisy_model, noise_dataset, batch_size=batch_size_train, steps=n_steps)

scale_sym = nn.Parameter(torch.tensor(0.01))
symmetric_noise_dataset = NoisyDataset(n_features=n_features, p=p, scale=scale_sym, symmetric=True, zero_diagonal=False)
symmetric_noise_model = MLP(n_features=n_features, d_mlp=d_mlp)
losses_noise_sym = train(symmetric_noise_model, symmetric_noise_dataset, batch_size=batch_size_train, steps=n_steps)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
plot_loss_of_input_sparsity(
    models=[resid_transpose_model, clean_model, noisy_model, symmetric_noise_model],
    ps=np.geomspace(0.001, 1, 100),
    ax=ax1,
    labels=[f"Embedding-like $M = 1 - W_E W_E^T$ ($\\mathrm{{std}}(M) = {sigma_embed:.3f}$)", "No mixing $M = 0$", f"Asymmetric noisy-mixing $M \\sim \\mathcal{{N}}(0, {scale_asym.abs().item():.3f}^2)$", f"Symmetric noisy-mixing $M \\sim \\mathcal{{N}}(0, {scale_sym.abs().item():.3f}^2)$"],
    colors=sns_colorblind[:4],
    datasets=[resid_transpose_dataset, clean_dataset, noise_dataset, symmetric_noise_dataset],
)
ax1.set_title("Loss over input sparsity for different mixing matrices $M$")
ax1.set_xlabel("Feature probability $p$")
ax1.set_ylabel("Loss per feature $L / p$")
ax1.legend().remove()
ax1.legend(loc="upper left", ncols=1)
ax1.grid(True, alpha=0.3)


# %% Plot loss vs noise scale for symmetric and asymmetric noise

noise_levels = np.linspace(0.00, 0.08, 41)  # 10 minutes
sym_models, sym_datasets, sym_losses = [], [], []
asym_models, asym_datasets, asym_losses = [], [], []

for i, noise_level in enumerate(noise_levels):
    sym_noisy_dataset = NoisyDataset(n_features=n_features, p=p, scale=noise_level, exactly_one_active_feature=True, symmetric=True, zero_diagonal=True)
    asym_noisy_dataset = NoisyDataset(n_features=n_features, p=p, scale=noise_level, exactly_one_active_feature=True, symmetric=False, zero_diagonal=True)
    sym_noisy_model = MLP(n_features=n_features, d_mlp=d_mlp)
    asym_noisy_model = MLP(n_features=n_features, d_mlp=d_mlp)
    train(sym_noisy_model, sym_noisy_dataset, batch_size=batch_size_train, steps=n_steps)
    train(asym_noisy_model, asym_noisy_dataset, batch_size=batch_size_train, steps=n_steps)
    sym_models.append(sym_noisy_model)
    asym_models.append(asym_noisy_model)
    sym_datasets.append(sym_noisy_dataset)
    asym_datasets.append(asym_noisy_dataset)

sym_losses = []
asym_losses = []
for i, (sym_noisy_model, sym_noisy_dataset, asym_noisy_model, asym_noisy_dataset) in tqdm(enumerate(zip(sym_models, sym_datasets, asym_models, asym_datasets, strict=True)), total=len(sym_models), desc="Evaluating"):
    sym_loss = evaluate(sym_noisy_model, sym_noisy_dataset, n_samples=100_000_000)
    asym_loss = evaluate(asym_noisy_model, asym_noisy_dataset, n_samples=10_000_000)
    sym_losses.append(sym_loss)
    asym_losses.append(asym_loss)

ax2.plot(noise_levels, np.array(sym_losses) / p, marker="o", label="Symmetric noise", color=sns_colorblind[2])
ax2.plot(noise_levels, np.array(asym_losses) / p, marker="o", label="Asymmetric noise", color=sns_colorblind[3])
ax2.axhline(0.083, color="k", ls="--", label="Naive solution")
ax2.set_xlabel("Label noise scale $\\sigma$")
ax2.set_ylabel("Loss per feature $L / p$")
ax2.legend()
ax2.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax2.grid(True, alpha=0.3)
ax2.set_title("Loss over a range of mixing matrix magnitudes")
fig.savefig("plots/nb2_noise_comparison_and_optimum.png")
plt.show()

# %% Try transplant noisy to clean and fine-tuning

losses_clean_ft = train(resid_transpose_model, clean_dataset, batch_size=batch_size_train, steps=n_steps)
fig_ft, ax_ft = plt.subplots(constrained_layout=True)
losses_noisy_avg = np.convolve(losses_noisy, np.ones(100) / 100, mode="valid")
losses_clean_ft_avg = np.convolve(losses_clean_ft, np.ones(100) / 100, mode="valid")
scale = noise_dataset.scale.abs().item()
ax_ft.plot(np.arange(len(losses_noisy_avg)), losses_noisy_avg / p, label=f"Train with $M \\sim \\mathcal{{N}}(0, {scale:.3f}^2)$")
ax_ft.plot(
    np.arange(len(losses_noisy_avg) - 1, len(losses_noisy_avg) + len(losses_clean_ft_avg)),
    np.concatenate([losses_noisy_avg[-1:], losses_clean_ft_avg]) / p,
    label="Fine-tune with $M = 0$",
)
ax_ft.set_ylabel("Loss per feature $L / p$ (running average)")
ax_ft.set_xlabel("Training step")
ax_ft.semilogy()
ax_ft.legend()
ax_ft.grid(True, alpha=0.3)
fig_ft.suptitle("Transplanting weights from noisy case to clean case")
fig_ft.savefig("plots/nb2a_transplant_finetune.png")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from mlpinsoup import MLP, NoisyDataset, evaluate, train
from tqdm import tqdm

noise_levels = np.linspace(0.00, 0.06, 31)
sym_models, sym_datasets, sym_losses = [], [], []
asym_models, asym_datasets, asym_losses = [], [], []

for i, noise_level in enumerate(noise_levels):
    sym_noisy_dataset = NoisyDataset(n_features=100, p=0.01, scale=noise_level, exactly_one_active_feature=True, symmetric=True, zero_diagonal=True)
    asym_noisy_dataset = NoisyDataset(n_features=100, p=0.01, scale=noise_level, exactly_one_active_feature=True, symmetric=False, zero_diagonal=True)
    sym_noisy_model = MLP(n_features=100, d_mlp=50)
    asym_noisy_model = MLP(n_features=100, d_mlp=50)
    train(sym_noisy_model, sym_noisy_dataset, batch_size=1024, steps=10_000)
    train(asym_noisy_model, asym_noisy_dataset, batch_size=1024, steps=10_000)
    sym_models.append(sym_noisy_model)
    asym_models.append(asym_noisy_model)
    sym_datasets.append(sym_noisy_dataset)
    asym_datasets.append(asym_noisy_dataset)

sym_losses = []
asym_losses = []
for i, (sym_noisy_model, sym_noisy_dataset, asym_noisy_model, asym_noisy_dataset) in tqdm(enumerate(zip(sym_models, sym_datasets, asym_models, asym_datasets, strict=True)), total=len(sym_models), desc="Evaluating"):
    print("Iteration", i + 1, "of", len(sym_models))
    sym_loss = evaluate(sym_noisy_model, sym_noisy_dataset, n_samples=100_000_000)
    asym_loss = evaluate(asym_noisy_model, asym_noisy_dataset, n_samples=10_000_000)
    sym_losses.append(sym_loss)
    asym_losses.append(asym_loss)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(noise_levels, np.array(sym_losses) / 0.01, marker="o", label="Symmetric noise", color="C2")
ax.plot(noise_levels, np.array(asym_losses) / 0.01, marker="o", label="Asymmetric noise", color="C3")
ax.axhline(0.083, color="k", ls="--")
ax.set_xlabel("Dataset noise scale")
ax.set_ylabel("Loss per feature L / p")
ax.legend()
fig.savefig("nb3_noise_optimum.png")
plt.show()

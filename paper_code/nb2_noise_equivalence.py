import matplotlib.pyplot as plt
import numpy as np
from mlpinsoup import MLP, CleanDataset, NoisyDataset, ResidTransposeDataset, plot_loss_of_input_sparsity, train

p = 0.01
n_features = 100
d_mlp = 50
n_steps = 10_000
batch_size_train = 1024

clean_dataset = CleanDataset(n_features=n_features, p=p)
clean_model = MLP(n_features=n_features, d_mlp=d_mlp)
train(clean_model, clean_dataset, batch_size=batch_size_train, steps=n_steps)
resid_transpose_dataset = ResidTransposeDataset(n_features=n_features, d_embed=1000, p=p)
resid_transpose_model = MLP(n_features=n_features, d_mlp=d_mlp)
train(resid_transpose_model, resid_transpose_dataset, batch_size=batch_size_train, steps=n_steps)

noise_dataset = NoisyDataset(n_features=n_features, p=p, scale=0.03, symmetric=False)
noisy_model = MLP(n_features=n_features, d_mlp=d_mlp)
train(noisy_model, noise_dataset, batch_size=batch_size_train, steps=n_steps)

symmetric_noise_dataset = NoisyDataset(n_features=n_features, p=p, scale=0.04, symmetric=True)
symmetric_noise_model = MLP(n_features=n_features, d_mlp=d_mlp)
train(symmetric_noise_model, symmetric_noise_dataset, batch_size=batch_size_train, steps=n_steps)


fig, ax = plt.subplots(constrained_layout=True)
plot_loss_of_input_sparsity(
    models=[clean_model, resid_transpose_model, symmetric_noise_model, noisy_model],
    ps=np.geomspace(0.001, 1, 100),
    ax=ax,
    labels=["Clean", "Residual", "Symmetric mixing", "Asymmetric mixing"],
    colors=["C0", "C1", "C2", "C3"],
    datasets=[clean_dataset, resid_transpose_dataset, symmetric_noise_dataset, noise_dataset],
)
ax.legend().remove()
ax.legend(loc="upper left", ncols=1, title="Dataset")
ax.grid(True, alpha=0.3)
fig.savefig("plots/nb2_noise_equivalence.png")

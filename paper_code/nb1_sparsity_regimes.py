import numpy as np
from mlpinsoup import MLP, ResidTransposeDataset, plot_loss_of_input_sparsity, train

# Input feature probabilities to test
plot_ps = np.geomspace(0.001, 1, 100)
# Train feature probabilities
str_train_ps = ["0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"]
# Dataset following Braun et al. 2025
apd_dataset = ResidTransposeDataset(n_features=100, d_embed=1000, p=0)

models, labels = [], []
for str_p in str_train_ps:
    p = float(str_p)
    apd_dataset.set_p(p)
    model = MLP(n_features=100, d_mlp=50)
    frac_zero_samples = (1 - p) ** model.n_features
    n_steps = int(10_000 / (1 - frac_zero_samples))
    train(model, apd_dataset, batch_size=1024, steps=n_steps)
    models.append(model)
    labels.append("p=" + str_p)

train_ps = [float(p) for p in str_train_ps]
fig = plot_loss_of_input_sparsity(models, apd_dataset, ps=plot_ps, labels=labels, highlight_ps=train_ps)
fig.savefig("nb1_sparsity_regimes.png")

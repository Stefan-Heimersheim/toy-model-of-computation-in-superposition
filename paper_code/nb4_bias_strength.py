import colorsys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mlpinsoup import MLP, CleanDataset, evaluate, plot_loss_of_input_sparsity, train


def make_dark_color(color):
    """https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib"""
    hls = colorsys.rgb_to_hls(*mc.to_rgb(color))
    darker_hls = (hls[0], 1 - (1 - hls[1]) * 1.5, hls[2])
    return colorsys.hls_to_rgb(*darker_hls)


bias_ps = ["0.001", "0.01", "0.1", "1"]
bias_colors = ["C0", "C1", "C2", "C3"]
bias_strengths = np.geomspace(0.002, 0.05, 100)
dataset = CleanDataset(n_features=100, p=0)
optimal_bias_strengths = {}

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

for bias_p, bias_color in zip(bias_ps, bias_colors):
    losses = []
    dataset.set_p(float(bias_p))
    print(f"Evaluating p={bias_p}")
    for bias_strength in bias_strengths:
        naive_model = MLP(n_features=100, d_mlp=50)
        naive_model.handcode_naive_mlp(bias_strength=bias_strength)
        loss = evaluate(naive_model, dataset)
        losses.append(loss / float(bias_p))
    ax1.plot(bias_strengths, losses, label=f"p={bias_p}", color=bias_color)
    opt_idx = np.argmin(losses)
    ax1.scatter(bias_strengths[opt_idx], losses[opt_idx], color=bias_color, marker="o")
    if float(bias_p) >= 0.1:
        ax1.text(
            bias_strengths[opt_idx],
            losses[opt_idx],
            f"Optimal bias: {bias_strengths[opt_idx]:.3f}  ",
            ha="right",
            va="top",
        )
    optimal_bias_strengths[bias_p] = bias_strengths[opt_idx]

ax1.legend()
ax1.set_xlabel("Offset strength")
ax1.set_ylabel("Adjusted loss L / p")
ax1.set_xscale("log")
ax1.grid(True, alpha=0.3)


plot_ps = np.geomspace(0.001, 1, 100)
train_ps = ["0.001", "0.01", "0.1", "1"]
tab20 = plt.get_cmap("tab20")
colors = [tab20(2 * i + 1) for i in [0, 1, 2, 3]]
linestyles = ["-", "-", "-", "-"]
clean_dataset = CleanDataset(n_features=100, p=0)

trained_models, trained_labels = [], []
for str_p in train_ps:
    p = float(str_p)
    clean_dataset.set_p(p)
    model = MLP(n_features=100, d_mlp=50)
    frac_zero_samples = (1 - p) ** model.n_features
    n_steps = int(10_000 / (1 - frac_zero_samples))
    train(model, clean_dataset, batch_size=1024, steps=n_steps)
    trained_models.append(model)
    trained_labels.append("p=" + str_p)

handcoded_models, handcoded_labels = [], []
handcoded_colors = [tab20(2 * i) for i in [0, 1, 2, 3]]
handcoded_linestyles = ["--", "--", "--", "--"]
for bias_p in bias_ps:
    naive_model = MLP(n_features=100, d_mlp=50)
    naive_model.handcode_naive_mlp(bias_strength=optimal_bias_strengths[bias_p])
    handcoded_models.append(naive_model)
    handcoded_labels.append(None)  # f"Handcoded w/ bias={optimal_bias_strengths[bias_p]:.3f}")

plot_loss_of_input_sparsity(
    trained_models + handcoded_models,
    clean_dataset,
    ps=plot_ps,
    labels=trained_labels + handcoded_labels,
    colors=colors + handcoded_colors,
    linestyles=linestyles + handcoded_linestyles,
    show_naive=False,
    ax=ax2,
)
ax2.legend().remove()
ax2.legend(loc="upper left", ncols=2, title="Feature probability")
ax2.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax2.grid(True, alpha=0.3)
fig.suptitle("Trained vs handcoded models on the clean label")
fig.savefig("plots/nb4_opt_offset_and_comparison.png")

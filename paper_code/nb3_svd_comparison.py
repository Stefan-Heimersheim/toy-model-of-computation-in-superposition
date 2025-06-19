import einops
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from mlpinsoup import MLP, NoisyDataset, ResidTransposeDataset, compare_WoutWin_Mscaled, evaluate, train
from torch import Tensor

p = 0.01
n_features = 100
d_mlp = 50
n_steps = 20_000
batch_size_train = 1024
d_embed = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

noisy_dataset = NoisyDataset(n_features=n_features, p=p, zero_diagonal=False, symmetric=True, scale=0.05)
embed_dataset = ResidTransposeDataset(n_features=n_features, d_embed=d_embed, p=p)
trained_model_noisy = MLP(n_features=n_features, d_mlp=d_mlp)
trained_model_embed = MLP(n_features=n_features, d_mlp=d_mlp)
train(trained_model_noisy, noisy_dataset, batch_size=batch_size_train, steps=n_steps)
train(trained_model_embed, embed_dataset, batch_size=batch_size_train, steps=n_steps)
trained_final_loss_noisy = evaluate(trained_model_noisy, noisy_dataset)
trained_final_loss_embed = evaluate(trained_model_embed, embed_dataset)
print(f"Trained model (noisy): {trained_final_loss_noisy / p:.3f}")
print(f"Trained model (embed): {trained_final_loss_embed / p:.3f}")


def get_cosine_sim_for_direction(model: MLP, v_in: Float[Tensor, "d_in"], v_out: Float[Tensor, "d_out"]) -> float:
    """Get cosine similarity between direction d and W_out @ W_in @ d"""
    model_transformation = einops.einsum(model.w_in, model.w_out, "d_in d_mlp, d_mlp d_out -> d_out d_in")
    projected_sv = einops.einsum(v_in, model_transformation, "d_in, d_out d_in -> d_out")
    cosine_sim = F.cosine_similarity(v_out, projected_sv, dim=0)
    return cosine_sim.item()


MplusID_noisy = noisy_dataset.Mscaled + torch.eye(n_features, device=device)
MplusID_embed = embed_dataset.Mscaled + torch.eye(n_features, device=device)
U_noisy, S_noisy, V_noisy = torch.linalg.svd(MplusID_noisy)
U_embed, S_embed, V_embed = torch.linalg.svd(MplusID_embed)
eigenvalues_noisy, eigenvectors_noisy = torch.linalg.eigh((noisy_dataset.Mscaled + noisy_dataset.Mscaled.T) / 2)
ord = torch.argsort(eigenvalues_noisy, descending=True)
eigenvalues_noisy, eigenvectors_noisy = eigenvalues_noisy[ord], eigenvectors_noisy[:, ord]
eigenvalues_embed, eigenvectors_embed = torch.linalg.eigh((embed_dataset.Mscaled + embed_dataset.Mscaled.T) / 2)
ord = torch.argsort(eigenvalues_embed, descending=True)
eigenvalues_embed, eigenvectors_embed = eigenvalues_embed[ord], eigenvectors_embed[:, ord]

trained_model_noisy_cosine_sims_cross = []
trained_model_noisy_cosine_sims_crossinv = []
trained_model_noisy_cosine_sims_u = []
trained_model_noisy_cosine_sims_v = []
trained_model_embed_cosine_sims_cross = []
trained_model_embed_cosine_sims_crossinv = []
trained_model_embed_cosine_sims_u = []
trained_model_embed_cosine_sims_v = []
for i in range(n_features):
    u_noisy = U_noisy[:, i]
    v_noisy = V_noisy[i, :]
    u_embed = U_embed[:, i]
    v_embed = V_embed[i, :]
    e_noisy = eigenvectors_noisy[:, i]
    e_embed = eigenvectors_embed[:, i]
    # trained_model_noisy_cosine_sims_cross.append(get_cosine_sim_for_direction(trained_model_noisy, u_noisy, v_noisy))
    trained_model_noisy_cosine_sims_crossinv.append(get_cosine_sim_for_direction(trained_model_noisy, v_noisy, v_noisy))
    # trained_model_noisy_cosine_sims_u.append(get_cosine_sim_for_direction(trained_model_noisy, u_noisy, u_noisy))
    # trained_model_noisy_cosine_sims_v.append(get_cosine_sim_for_direction(trained_model_noisy, v_noisy, v_noisy))
    # trained_model_embed_cosine_sims_cross.append(get_cosine_sim_for_direction(trained_model_embed, v_embed, u_embed))
    trained_model_embed_cosine_sims_crossinv.append(get_cosine_sim_for_direction(trained_model_embed, v_embed, v_embed))
    # trained_model_embed_cosine_sims_u.append(get_cosine_sim_for_direction(trained_model_embed, u_embed, u_embed))
    # trained_model_embed_cosine_sims_v.append(get_cosine_sim_for_direction(trained_model_embed, v_embed, v_embed))

fig = plt.figure(figsize=(15, 8), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], figure=fig)
right_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 1])
ax = fig.add_subplot(gs[:, 0])
ax_top_left = fig.add_subplot(right_gs[0, 0])
ax_top_right = fig.add_subplot(right_gs[0, 1])
ax_bottom_left = fig.add_subplot(right_gs[1, 0])
ax_bottom_right = fig.add_subplot(right_gs[1, 1])
ax.plot(trained_model_noisy_cosine_sims_crossinv, label=f"Noisy dataset (loss={trained_final_loss_noisy / p:.3f})")
ax.plot(trained_model_embed_cosine_sims_crossinv, label=f"Resid dataset (loss={trained_final_loss_embed / p:.3f})")
ax.set_title("Testing how well eigenvectors are captured by $W_{\\rm out} W_{\\rm in}$")
ax.set_xlabel("Eigenvector index $i$ (descending eigenvalues)")
ax.set_ylabel("Cosine similarity of $V_i$ with $W_{\\rm out} W_{\\rm in} V_i$")
ax.legend(loc="lower left")
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax.grid(True, alpha=0.3)
projs1 = torch.abs(V_noisy @ trained_model_noisy.w_in)
projs2 = torch.abs(V_noisy @ trained_model_noisy.w_out.T)
projs3 = torch.abs(U_noisy.T @ trained_model_noisy.w_in)
projs4 = torch.abs(U_noisy.T @ trained_model_noisy.w_out.T)

ax_top_left.imshow(projs1.detach().cpu().numpy(), aspect="auto", cmap="magma")
ax_top_left.set_title("Noisy dataset, $V_i^T W_{\\rm in}$")
ax_top_right.imshow(projs2.detach().cpu().numpy(), aspect="auto", cmap="magma")
ax_top_right.set_title("Noisy dataset, $V_i^T W_{\\rm out}^T$")
ax_bottom_left.imshow(projs3.detach().cpu().numpy(), aspect="auto", cmap="magma")
ax_bottom_left.set_title("Noisy dataset, $U_i^T W_{\\rm in}$")
ax_bottom_right.imshow(projs4.detach().cpu().numpy(), aspect="auto", cmap="magma")
ax_bottom_right.set_title("Noisy dataset, $U_i^T W_{\\rm out}^T$")

fig.savefig("plots/nb3a_svd_comparison.png")
plt.show()

#  %% Compare W_out @ W_in and M_scaled entry correlation

print(trained_model_noisy.w_in.shape)
print(trained_model_noisy.w_out.shape)
compare_WoutWin_Mscaled(trained_model_noisy, noisy_dataset)

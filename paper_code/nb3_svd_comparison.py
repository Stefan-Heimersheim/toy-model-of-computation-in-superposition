import einops
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from mlpinsoup import MLP, NoisyDataset, ResidTransposeDataset, compare_WoutWin_Mscaled, evaluate, train
from sklearn.decomposition import NMF
from torch import Tensor

p = 0.01
n_features = 100
d_mlp = 50
n_steps = 20_000
batch_size_train = 2048
d_embed = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

noisy_dataset = NoisyDataset(n_features=n_features, p=p, zero_diagonal=False, symmetric=False, scale=0.035)
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
trained_model_noisy_cosine_sims_u = []
trained_model_noisy_cosine_sims_v = []
trained_model_noisy_cosine_sims_e = []
trained_model_embed_cosine_sims_cross = []
trained_model_embed_cosine_sims_u = []
trained_model_embed_cosine_sims_v = []
trained_model_embed_cosine_sims_e = []

for i in range(n_features):
    u_noisy = U_noisy[:, i]
    v_noisy = V_noisy[i, :]
    u_embed = U_embed[:, i]
    v_embed = V_embed[i, :]
    e_noisy = eigenvectors_noisy[:, i]
    e_embed = eigenvectors_embed[:, i]
    trained_model_noisy_cosine_sims_cross.append(get_cosine_sim_for_direction(trained_model_noisy, v_noisy, u_noisy))
    trained_model_noisy_cosine_sims_e.append(get_cosine_sim_for_direction(trained_model_noisy, e_noisy, e_noisy))
    trained_model_noisy_cosine_sims_u.append(get_cosine_sim_for_direction(trained_model_noisy, u_noisy, u_noisy))
    trained_model_noisy_cosine_sims_v.append(get_cosine_sim_for_direction(trained_model_noisy, v_noisy, v_noisy))
    # trained_model_embed_cosine_sims_cross.append(get_cosine_sim_for_direction(trained_model_embed, v_embed, u_embed))
    # trained_model_embed_cosine_sims_u.append(get_cosine_sim_for_direction(trained_model_embed, u_embed, u_embed))
    # trained_model_embed_cosine_sims_v.append(get_cosine_sim_for_direction(trained_model_embed, v_embed, v_embed))
    # trained_model_embed_cosine_sims_e.append(get_cosine_sim_for_direction(trained_model_embed, e_embed, e_embed))

fig = plt.figure(figsize=(10, 5), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], figure=fig)
left_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 0])
ax = fig.add_subplot(gs[:, 1])
ax_top_left = fig.add_subplot(left_gs[0, 0])
ax_top_right = fig.add_subplot(left_gs[0, 1])
ax_bottom_left = fig.add_subplot(left_gs[1, 0])
ax_bottom_right = fig.add_subplot(left_gs[1, 1])
ax.plot(trained_model_noisy_cosine_sims_cross, label="$\\cos(U_i, W_{\\rm out} W_{\\rm in} V_i)$")
ax.plot(trained_model_noisy_cosine_sims_e, label="$\\cos(E_i, W_{\\rm out} W_{\\rm in} E_i)$")
# ax.plot(trained_model_noisy_cosine_sims_u, label="$\\cos(U_i, W_{\\rm out} W_{\\rm in} U_i)$")
# ax.plot(trained_model_noisy_cosine_sims_v, label="$\\cos(V_i, W_{\\rm out} W_{\\rm in} V_i)$")
ax.set_title("Testing how well eigenvectors are captured by $W_{\\rm out} W_{\\rm in}$")
ax.set_xlabel("Eigen- / singular vector index $i$ (descending eigenvalues)")
ax.set_ylabel("Cosine similarity of $V_i$ with $W_{\\rm out} W_{\\rm in} V_i$")
ax.legend(loc="lower left")
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax.grid(True, alpha=0.3)
projs1 = torch.abs(einops.einsum(V_noisy, trained_model_noisy.w_in, "r d_in, d_in d_mlp -> r d_mlp"))
projs2 = torch.abs(einops.einsum(U_noisy, trained_model_noisy.w_out, "d_out r, d_mlp d_out -> r d_mlp"))
projs3 = torch.abs(einops.einsum(eigenvectors_noisy, trained_model_noisy.w_in, "d_in r, d_in d_mlp -> r d_mlp"))
projs4 = torch.abs(einops.einsum(eigenvectors_noisy, trained_model_noisy.w_out, "d_out r, d_mlp d_out -> r d_mlp"))
# projs3 = torch.abs(einops.einsum(V_noisy, trained_model_noisy.w_out, "r d_out, d_mlp d_out -> r d_out"))
# projs4 = torch.abs(einops.einsum(U_noisy, trained_model_noisy.w_in, "d_in r, d_in d_mlp -> r d_mlp"))

ax_bottom_left.imshow(projs1.T.detach().cpu().numpy(), aspect="auto", cmap="magma")
ax_bottom_right.imshow(projs2.T.detach().cpu().numpy(), aspect="auto", cmap="magma")
ax_bottom_left.set_title("Noisy dataset, $V \\cdot W_{\\rm in}$")
ax_bottom_right.set_title("Noisy dataset, $U \\cdot W_{\\rm out}$")
ax_top_left.set_title("Noisy dataset, $E \\cdot W_{\\rm in}$")
ax_top_right.set_title("Noisy dataset, $E \\cdot W_{\\rm out}$")
ax_top_left.imshow(projs3.T.detach().cpu().numpy(), aspect="auto", cmap="magma")
ax_top_right.imshow(projs4.T.detach().cpu().numpy(), aspect="auto", cmap="magma")
ax_bottom_left.set_ylabel("MLP neuron index", ha="left")
ax_bottom_left.set_xlabel("Eigen- / singular vector index", ha="left")

fig.savefig("plots/nb3_eigenvecrtors_svd_comparison.png")
plt.show()

#  %% Compare W_out @ W_in and M_scaled entry correlation

fig = compare_WoutWin_Mscaled(trained_model_noisy, noisy_dataset)
fig.savefig("plots/nb3a_WoutWin_Mscaled_noisy.png")
# %%

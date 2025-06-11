import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from baselines import get_half_identity_model, get_semi_nmf_model, get_svd_model
from mlpinsoup import MLP, ResidTransposeDataset, evaluate, train

p = 0.01
n_features = 100
d_mlp = 50
n_steps = 10_000
batch_size_train = 1024
d_embed = 1000


relu_dataset = ResidTransposeDataset(n_features=n_features, d_embed=d_embed, p=p)
M = einops.einsum(relu_dataset.W_E, relu_dataset.W_E.T, "n_1 d_embed, d_embed n_2 -> n_1 n_2")
U, S, V = torch.linalg.svd(M)

half_identity_model = get_half_identity_model(n_features, d_mlp)
half_identity_final_loss = evaluate(half_identity_model, relu_dataset)
print(f"ReLU: half identity: {half_identity_final_loss / p:.3f}")

svd_model = get_svd_model(n_features, d_mlp, M)
svd_final_loss = evaluate(svd_model, relu_dataset)
print(f"ReLU: SVD: {svd_final_loss / p:.3f}")

nmf_model = get_semi_nmf_model(n_features, d_mlp, M)
nmf_final_loss = evaluate(nmf_model, relu_dataset)
print(f"ReLU: NMF: {nmf_final_loss / p:.3f}")

trained_model = MLP(n_features=n_features, d_mlp=d_mlp)
training_losses = train(trained_model, relu_dataset, batch_size=batch_size_train, steps=n_steps)
trained_final_loss = evaluate(trained_model, relu_dataset)
print(f"ReLU: trained: {trained_final_loss / p:.3f}")


def get_cosine_sim_for_direction(model, d):
    model_transformation = einops.einsum(model.w_in, model.w_out, "d_in d_mlp, d_mlp d_out -> d_in d_out")
    projected_sv = einops.einsum(d, model_transformation, "d_in, d_in d_out -> d_out")
    cosine_sim = F.cosine_similarity(d, projected_sv, dim=0)
    return cosine_sim.item()


svd_model_cosine_sims = [get_cosine_sim_for_direction(svd_model, U[:, i]) for i in range(n_features)]
nmf_model_cosine_sims = [get_cosine_sim_for_direction(nmf_model, U[:, i]) for i in range(n_features)]
trained_model_cosine_sims = [get_cosine_sim_for_direction(trained_model, U[:, i]) for i in range(n_features)]
half_identity_model_cosine_sims = [get_cosine_sim_for_direction(half_identity_model, U[:, i]) for i in range(n_features)]

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(trained_model_cosine_sims, label=f"Trained, loss={trained_final_loss / p:.3f}")
# ax.plot(nmf_model_cosine_sims, label=f"NMF, loss={nmf_final_loss / p:.3f}")
# ax.plot(half_identity_model_cosine_sims, label=f"Half identity, loss={half_identity_final_loss / p:.3f}")
ax.plot(svd_model_cosine_sims, label=f"SVD, loss={svd_final_loss / p:.3f}")
ax.set_title("Testing how well SVD directions are captured by $W_{\\rm out} W_{\\rm in}$")
ax.set_xlabel("Singular vector index $i$")
ax.set_ylabel("Cosine similarity of $v_i$ with $W_{\\rm out} W_{\\rm in} v_i$")
ax.legend(loc="lower left")
fig.savefig("plots/nb5_svd_comparison.png")
plt.show()

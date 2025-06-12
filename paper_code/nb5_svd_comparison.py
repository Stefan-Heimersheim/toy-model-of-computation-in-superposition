import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from baselines import get_half_identity_model, get_semi_nmf_model, get_svd_model
from mlpinsoup import MLP, ResidTransposeDataset, evaluate, get_cosine_sim_for_direction, train

p = 0.01
n_features = 100
d_mlp = 50
n_steps = 10_000
batch_size_train = 1024
d_embed = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

relu_dataset = ResidTransposeDataset(n_features=n_features, d_embed=d_embed, p=p)
# Note: The optimal factor to minimize the SVD loss seems to be 0.5, but using
# a factor of 1 seems to give nicer SVD directions to visualize.
MplusID = relu_dataset.Mscaled + 0.5 * torch.eye(n_features, device=device)
U, S, V = torch.linalg.svd(MplusID)

half_identity_model = get_half_identity_model(n_features, d_mlp)
half_identity_final_loss = evaluate(half_identity_model, relu_dataset)
print(f"ReLU: half identity: {half_identity_final_loss / p:.3f}")

svd_model = get_svd_model(n_features, d_mlp, MplusID)
svd_final_loss = evaluate(svd_model, relu_dataset)
print(f"ReLU: SVD: {svd_final_loss / p:.3f}")

nmf_model = get_semi_nmf_model(n_features, d_mlp, MplusID)
nmf_final_loss = evaluate(nmf_model, relu_dataset)
print(f"ReLU: NMF: {nmf_final_loss / p:.3f}")

trained_model = MLP(n_features=n_features, d_mlp=d_mlp)
training_losses = train(trained_model, relu_dataset, batch_size=batch_size_train, steps=n_steps)
trained_final_loss = evaluate(trained_model, relu_dataset)
print(f"ReLU: trained: {trained_final_loss / p:.3f}")

MplusID = relu_dataset.Mscaled.detach() + 1 * torch.eye(n_features, device=relu_dataset.device)
U, _, _ = torch.linalg.svd(MplusID)

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

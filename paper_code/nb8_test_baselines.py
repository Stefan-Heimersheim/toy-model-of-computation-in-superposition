import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from mlpinsoup import MLP, NoisyDataset, evaluate, plot_loss_of_input_sparsity, train
from sklearn.decomposition import NMF
from torch import Tensor


def get_zero_model(n_features: int, d_mlp: int):
    model = MLP(n_features=n_features, d_mlp=d_mlp)
    model.w_in.data = torch.zeros_like(model.w_in.data)
    model.w_out.data = torch.zeros_like(model.w_out.data)
    return model


def get_half_identity_model(n_features: int, d_mlp: int, bias_strength: float = 0.0):
    model = MLP(n_features=n_features, d_mlp=d_mlp)
    model.handcode_naive_mlp(bias_strength=bias_strength)
    return model


def get_svd_model(n_features: int, d_mlp: int, M: Float[Tensor, "n_features n_features"], flip_weights: bool = True):
    model = MLP(n_features=n_features, d_mlp=d_mlp)
    U, S, V = torch.linalg.svd(M)
    model.w_in.data = U[:, :d_mlp].clone() * S[:d_mlp].clone()
    model.w_out.data = V[:d_mlp, :].clone()
    if flip_weights:
        # Try to make the largest W_in weights all positive, to better match the ReLU
        w_in = model.w_in.data.clone()
        w_out = model.w_out.data.clone()
        for neuron in range(d_mlp):
            argmax_abs_weight = w_in[:, neuron].abs().argmax()
            if w_in[argmax_abs_weight, neuron] < 0:
                w_in[:, neuron] = -w_in[:, neuron]
                w_out[neuron, :] = -w_out[neuron, :]
        model.w_in.data = w_in
        model.w_out.data = w_out
    return model


def get_relu_function():
    def relu_function(x):
        return F.relu(x)

    return relu_function


def get_identity_function():
    def identity_function(x):
        return x

    return identity_function


p = 0.01
n_features = 100
d_mlp = 50
n_steps = 20_000
batch_size_train = 1024
d_embed = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

relu_dataset = NoisyDataset(n_features=n_features, p=p, scale=0.01, exactly_one_active_feature=False)
MplusID = relu_dataset.Mscaled + torch.eye(n_features, device=device)
U, S, V = torch.linalg.svd(MplusID)

half_identity_model = get_half_identity_model(n_features, d_mlp)
half_identity_final_loss = evaluate(half_identity_model, relu_dataset)
print(f"ReLU: half identity: {half_identity_final_loss / p:.3f}")


def semi_nmf(
    target_matrix: np.ndarray,
    d_mlp: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Semi-NMF:  X ≈ U @ V   with U ≥ 0,  V free-sign.
    * U:  (n, k) non-negative
    * V:  (k, m) real-valued

    Simple projected-gradient update:
      - optimise V exactly by least-squares each outer step
      - take a GD step on U and ReLU-project to the positive orthant
    """
    n_features = target_matrix.shape[0]
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
    U_tensor = torch.from_numpy(U).float().to(device)
    V_tensor = torch.from_numpy(V).float().to(device)
    return U_tensor, V_tensor


def get_semi_nmf_model(MplusID: Float[Tensor, "n_features n_features"], d_mlp: int, device: str = "cpu"):
    U, V = semi_nmf(MplusID.cpu().detach().numpy(), d_mlp, device=device)
    model = MLP(n_features=MplusID.shape[0], d_mlp=d_mlp, device=device)
    model.w_in.data = U
    model.w_out.data = V
    return model


nmf_model = get_semi_nmf_model(MplusID, d_mlp, device=device)
nmf_final_loss = evaluate(nmf_model, relu_dataset)
print(f"ReLU: NMF: {nmf_final_loss / p:.3f}")

trained_model = MLP(n_features=n_features, d_mlp=d_mlp)
training_losses = train(trained_model, relu_dataset, batch_size=batch_size_train, steps=n_steps)
trained_final_loss = evaluate(trained_model, relu_dataset)
print(f"ReLU: trained: {trained_final_loss / p:.3f}")

plot_loss_of_input_sparsity(
    [trained_model, half_identity_model, nmf_model],
    [relu_dataset, relu_dataset, relu_dataset],
    labels=["Trained", "Half identity", "NMF"],
    batch_size=10_000_000,
)
plt.show()

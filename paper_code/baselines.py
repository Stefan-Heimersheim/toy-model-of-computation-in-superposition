import torch
import torch.nn.functional as F
from jaxtyping import Float
from mlpinsoup import MLP
from torch import Tensor


def flip_negative_weights(w_in: Float[Tensor, "n_features d_mlp"], w_out: Float[Tensor, "d_mlp n_features"]):
    w_in = w_in.clone()
    w_out = w_out.clone()
    n_features, d_mlp = w_in.shape
    for neuron in range(d_mlp):
        argmax_abs_weight = w_in[:, neuron].abs().argmax()
        if w_in[argmax_abs_weight, neuron] < 0:
            w_in[:, neuron] = -w_in[:, neuron]
            w_out[neuron, :] = -w_out[neuron, :]
    return w_in, w_out


# Baseline hand-coded MLPs


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
        model.w_in.data, model.w_out.data = flip_negative_weights(model.w_in.data, model.w_out.data)
    return model


def semi_nmf(
    X: Float[Tensor, "n m"],
    rank: int,
    max_iter: int = 1000,
    tol: float = 1e-6,
    lr: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Semi-NMF:  X ≈ U @ V   with U ≥ 0,  V free-sign.
    * U:  (n, k) non-negative
    * V:  (k, m) real-valued

    Simple projected-gradient update:
      - optimise V exactly by least-squares each outer step
      - take a GD step on U and ReLU-project to the positive orthant
    """
    n, m = X.shape
    U = torch.abs(torch.randn(n, rank, device=X.device))  # U ≥ 0
    for i in range(max_iter):
        V = torch.linalg.lstsq(U, X).solution
        grad_U = (U @ V - X) @ V.T
        U_next = (U - lr * grad_U).clamp_min_(0.0)
        if torch.norm(U_next - U) / torch.norm(U) < tol:
            U = U_next
            print(f"Semi-NMF converged in {i} iterations")
            break
        U = U_next
    print(f"Semi-NMF exited with {i} iterations")
    return U, V


def get_semi_nmf_model(n_features: int, d_mlp: int, M: Float[Tensor, "n_features n_features"]):
    U, V = semi_nmf(M, rank=d_mlp)
    model = MLP(n_features=n_features, d_mlp=d_mlp)
    model.w_in.data = U
    model.w_out.data = V
    return model


# Ideal baselines (could not be implemented with the MLP)


def get_relu_function():
    def relu_function(x):
        return F.relu(x)

    return relu_function


def get_identity_function():
    def identity_function(x):
        return x

    return identity_function

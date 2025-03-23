"""Utility functions."""
import torch as t
from einops import rearrange, asnumpy,reduce
from jaxtyping import Float
from torch import Tensor
from tqdm.notebook import tqdm
from toy_cis.models import Cis

def threshold_matrix(matrix, threshold=0.001):
    """Sets matrix elements to zero if their absolute value is below the threshold."""
    result = t.clone(matrix)
    result[t.abs(result) < threshold] = 0
    return result


def in_out_response(
    model: Cis, vals: Float[Tensor, ""], device: t.device
) -> Float[Tensor, "feat val feat"]:
    """Compute the input-output response for each feature-val pair, in a single batch."""
    n_feat, n_vals = model.cfg.n_feat, len(vals)
    # input `X` will be: active_feature_idx X values X full_feature_vector
    X = t.zeros(n_feat, n_vals, n_feat, device=device)
    feat_idx = t.arange(n_feat, device=device).unsqueeze(1).expand(n_feat, n_vals)
    val_idx = t.arange(n_vals, device=device).unsqueeze(0).expand(n_feat, n_vals)
    X[feat_idx, val_idx, feat_idx] = vals.unsqueeze(0).expand(n_feat, n_vals)
    # reshape for batch input
    X = rearrange(X, "active_feat val feat_vec -> (active_feat val) 1 feat_vec")
    Y = model.forward(X)
    # reshape for plotting by feature
    Y = rearrange(
        Y, 
        "(active_feat val) 1 feat_vec -> active_feat val feat_vec",
        active_feat=n_feat, 
        val=n_vals
    )
    return Y[t.arange(n_feat), :, t.arange(n_feat)]

def performance_across_sparsities(sparsities, model):
    loss_data = []  # will store mean loss data

    pbar = tqdm(sparsities, desc="Testing over sparsities")
    n_examples = 10000
    for s in pbar:
    
        with t.no_grad():
            # generate examples of inputs
            x, y_true = model.gen_batch_reluPlusX (n_examples, s)
            
            # compute mean loss
            y = model.forward(x)
            active_weight, inactive_weight = 10, 1
            weights = t.where(y_true == 1, active_weight, inactive_weight)
            loss = weights * (y - y_true) ** 2
            
            # store loss at sparsity level
            loss = rearrange(loss, "examples 1 features -> examples features")
            loss = asnumpy(reduce(loss, "examples features -> features", "mean"))
            for feat_idx, val in enumerate(loss):
                loss_data.append({"sparsity": s, "feature_idx": feat_idx, "loss_per_feature": val})
    return loss_data

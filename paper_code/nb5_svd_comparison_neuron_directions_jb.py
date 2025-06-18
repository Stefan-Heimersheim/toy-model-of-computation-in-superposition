import os
import matplotlib.pyplot as plt
import torch
from baselines import get_half_identity_model, get_semi_nmf_model, get_svd_model
from mlpinsoup import MLP, ResidTransposeDataset, evaluate, get_cosine_sim_for_direction, train


def main():
    """Run SVD comparison analysis and generate plots."""
    # Set environment and plotting parameters
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    
    # Parameters
    p = 0.01
    n_features = 100
    d_mlp = 50
    n_steps = 30_000
    batch_size_train = 1024
    d_embed = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print("Setting up dataset and models...")
    
    # Create dataset
    relu_dataset = ResidTransposeDataset(n_features=n_features, d_embed=d_embed, p=p)
    
    # Note: The optimal factor to minimize the SVD loss seems to be 0.5, but using
    # a factor of 1 seems to give nicer SVD directions to visualize.
    MplusID = relu_dataset.Mscaled + 0.5 * torch.eye(n_features, device=device)
    U, S, V = torch.linalg.svd(MplusID)
    
    # Evaluate baseline models
    print("Evaluating baseline models...")
    half_identity_model = get_half_identity_model(n_features, d_mlp)
    half_identity_final_loss = evaluate(half_identity_model, relu_dataset)
    print(f"ReLU: half identity: {half_identity_final_loss / p:.3f}")
    
    svd_model = get_svd_model(n_features, d_mlp, MplusID)
    svd_final_loss = evaluate(svd_model, relu_dataset)
    print(f"ReLU: SVD: {svd_final_loss / p:.3f}")
    
    nmf_model = get_semi_nmf_model(n_features, d_mlp, MplusID)
    nmf_final_loss = evaluate(nmf_model, relu_dataset)
    print(f"ReLU: NMF: {nmf_final_loss / p:.3f}")
      # Train model
    print("Training MLP model...")
    trained_model = MLP(n_features=n_features, d_mlp=d_mlp)
    train(trained_model, relu_dataset, batch_size=batch_size_train, steps=n_steps)
    trained_final_loss = evaluate(trained_model, relu_dataset)
    print(f"ReLU: trained: {trained_final_loss / p:.3f}")
    
    # Get SVD of the mixing matrix M (without adding identity) for heatmap
    print("Computing SVD analysis...")
    M_svd_U, M_svd_S, M_svd_V = torch.linalg.svd(relu_dataset.M)
    
    # For cosine similarity calculations, use M + I as before
    MplusID = relu_dataset.Mscaled.detach() + 1 * torch.eye(n_features, device=relu_dataset.device)
    U, _, _ = torch.linalg.svd(MplusID)
      # Compute cosine similarities
    svd_model_cosine_sims = [get_cosine_sim_for_direction(svd_model, U[:, i]) for i in range(n_features)]
    trained_model_cosine_sims = [get_cosine_sim_for_direction(trained_model, U[:, i]) for i in range(n_features)]
    
    # Create plots
    print("Generating plots...")
    create_plots(
        M_svd_U, M_svd_S, trained_model,
        trained_model_cosine_sims, svd_model_cosine_sims,
        trained_final_loss, svd_final_loss, p
    )
    
    print("Analysis complete! Plot saved to plots/nb5_svd_comparison_neuron_directions_jb.png")


def create_plots(M_svd_U, M_svd_S, trained_model, trained_cosine_sims, svd_cosine_sims, 
                trained_loss, svd_loss, p):
    """Create and save the two-panel figure."""
    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    
    # Left panel: Heatmap of SVD directions vs output weights
    # Compute dot-product magnitudes between SVD directions of M and output weights
    projs = torch.abs(M_svd_U.T @ trained_model.w_out.T)  # [n_features, d_mlp] dot-product magnitudes
    
    # Sort SVD directions by their maximum dot product with any neuron (highest activating at top)
    max_projs_per_direction = projs.max(dim=1)[0]  # Max dot product for each SVD direction
    sort_indices = torch.argsort(max_projs_per_direction, descending=True)
      # Reorder the projection matrix according to activation strength
    projs_sorted = projs[sort_indices, :]
    
    # Create heatmap with sorted directions
    im = ax1.imshow(projs_sorted.detach().cpu().numpy(), aspect="auto", cmap="magma")
    ax1.set_xlabel("MLP neuron")
    ax1.set_ylabel("SVD direction (sorted by max activation)")
    ax1.set_title("SVD directions of $M$ vs MLP output weights\\n(sorted by activation strength)")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Dot product magnitude")
    
    # Right panel: Original cosine similarity plot
    ax2.plot(trained_cosine_sims, label=f"Trained, loss={trained_loss / p:.3f}")
    ax2.plot(svd_cosine_sims, label=f"SVD, loss={svd_loss / p:.3f}")
    ax2.set_title("SVD directions captured by $W_{\\\\rm out} W_{\\\\rm in}$")
    ax2.set_xlabel("Singular vector index $i$")
    ax2.set_ylabel("Cosine similarity of $v_i$ with $W_{\\\\rm out} W_{\\\\rm in} v_i$")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Save the figure
    fig.savefig("plots/nb5_svd_comparison_neuron_directions_jb.png", dpi=150)
    plt.close(fig)  # Close to free memory


if __name__ == "__main__":
    main()

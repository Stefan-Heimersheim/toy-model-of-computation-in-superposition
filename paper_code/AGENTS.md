# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase implementing experiments for a paper on "toy model of computation in superposition". The code explores how MLPs learn to represent sparse features when the number of features exceeds the hidden layer width, forcing features into superposition.

## Core Architecture

### Main Modules
- `mlpinsoup.py`: Core MLP implementation and training infrastructure
  - `MLP`: Simple MLP without biases (ReLU activation)
  - Dataset classes: `CleanDataset`, `NoisyDataset` (supports rank specification), `ResidTransposeDataset`
  - Training and evaluation functions (supports learnable parameters)
  - Plotting utilities for loss vs sparsity analysis

- `baselines.py`: Baseline model implementations
  - Hand-coded models (zero, half-identity, SVD-based)
  - Semi-NMF implementation for matrix factorization
  - Weight processing utilities

### Experiment Scripts
- `nb1_sparsity_regimes.py`: Analysis of different sparsity regimes (main results)
- `nb2_noise_equivalence.py`: Comparison of different noise types  
- `nb3_noise_optimum.py`: Finding optimal noise levels
- `nb4_bias_strength.py`: Bias strength optimization experiments
- `nb5_svd_comparison.py`: SVD vs trained model comparison
- `nb6_mixing_matrix_rank.py`: Performance analysis vs mixing matrix rank with learnable scale

## Key Dependencies
- PyTorch (core ML framework)
- jaxtyping (type hints for tensors)
- einops (tensor operations)
- matplotlib/seaborn (plotting)
- numpy, tqdm

## Code Formatting
- Uses `ruff` for code formatting and linting
- Run `ruff format .` to format all Python files
- Run `ruff check .` to check for linting issues

## Running Experiments

Execute experiment scripts directly:
```bash
python nb1_sparsity_regimes.py
python nb2_noise_equivalence.py
# etc.
```

Each script generates corresponding PNG plots and saves them to the `plots/` directory.

## Model Architecture Details

The core MLP uses:
- Input features → Hidden layer (ReLU) → Output features
- No biases throughout
- Kaiming uniform initialization
- AdamW optimizer with weight decay for training

The research focuses on the regime where `n_features > d_mlp`, forcing the model to learn features in superposition within the hidden layer.
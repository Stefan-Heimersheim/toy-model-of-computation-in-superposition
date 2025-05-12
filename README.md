# Investigating Computation in Superposition in Toy Models

## Background
This repo contains code and experiments exploring **computation in superposition** in toy neural networks, including variants of the "compressed computation" model introduced by Apollo Research in [Braun et al. (2025)](https://www.apolloresearch.ai/research/interpretability-in-parameter-space-minimizing-mechanistic-description-length-with-attribution-based-parameter-decomposition), and "toy models of superposition" introduced by Anthropic in [Elhage et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html).

Neural networks can represent more features than they have neurons by *storing information in superposition* ([Elhage et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html)): this allows networks to compress and reconstruct sparse inputs by distributing feature representations across multiple neurons.

The authors also show a motif by which neural networks can **perform computation in superposition (CiS)**, compute more functions than would be expected if each neuron participated in only one computation. Both phenomena emerge in **sparse input regimes**, where only a few features are active at a time.

Inspired by [Hänni et al. (2024)](https://arxiv.org/abs/2408.05451), we adopt a **stricter definition of computation in superposition**:

> A network exhibits computation in superposition if it performs more *computations* than it has *nonlinearities*.

This avoids conflating CiS with cases where large networks trivially embed solutions using enough parameters or dimensions.

## Our experiments 

We replicate and extend experiments on a toy model introduced by [Apollo Research](https://www.apolloresearch.ai/research/interpretability-in-parameter-space-minimizing-mechanistic-description-length-with-attribution-based-parameter-decomposition), and a simplified variant that exhibits the same behaviour. Our aim is to determine whether this toy model is actually implementing computation in superposition and how, or if another mechanism is at play. 

## Key Findings (so far):

- **CiS-like behaviour emerges**: the model performs better than a monosemantic baseline in sparse-input regimes.

- **Residual stream with noise is critical**: CiS behaviour vanishes when we remove the residual path or the embedding/unembedding layers of the residual of the original Apollo model.

- **Underlying mechanisms**: currently under investigation. 

## Toy model architecture

        Input x
           │
           ├──────────────┐
           ▼              ▼
           │              │
     MLP (ReLUs)    Residual: (Wn @ x)
           │              │
           └──────┬───────┘
                  ▼
              Final Output y


## 📂 Repo Structure

```bash
.
├── notebooks/             # Experiment notebooks
├── toy_cis/               # Training, evaluation, visualization functions
├── test/                  # To be removed? 
├── README.md              # You are here!
└── pyproject.toml         # Environment config
```

## Environment set-up

Recommended with [pixi](https://pixi.sh/latest/tutorials/python).

In the root directory, just run `pixi install --manifest-path ./pyproject.toml` - this will create a conda env named 'toy-cis'.

## 📚 References

* [Elhage et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html): Anthropic's toy model of superposition
* [Cammarata et al. (2020)](https://transformer-circuits.pub/2020/polysemanticity/index.html): Polysemanticity and neuron capacity
* [Hänni et al. (2024)](https://arxiv.org/abs/2408.05451): Mathematical models of computation in superposition
* [Braun et al. (2025)](https://www.apolloresearch.ai/research/interpretability-in-parameter-space-minimizing-mechanistic-description-length-with-attribution-based-parameter-decomposition): Apollo Research's toy model of "compressed computation". 


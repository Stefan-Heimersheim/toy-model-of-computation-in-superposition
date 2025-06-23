# Compressed Computation is (probably) not Computation in Superposition

We investigate the toy model of Compressed Computation (CC), introduced by [Braun et al. (2025)](https://www.apolloresearch.ai/research/interpretability-in-parameter-space-minimizing-mechanistic-description-length-with-attribution-based-parameter-decomposition), which is a model that seemingly computes more non-linear functions (100 target ReLU functions) than it has ReLU neurons (50). Our results cast doubt on whether the mechanism behind this toy model is indeed computing more functions than it has neurons: We find that the model performance solely relies on noisy labels, and that its performance advantage compared to baselines diminishes with lower noise.

Specifically, we show that the Braun et al. (2025) setup can be split into two loss terms: the ReLU task and a noise term that mixes the different input features ("mixing matrix"). We isolate these terms, and show that the optimal total loss increases as we reduce the magnitude of the mixing matrix. This suggests that the loss advantage of the trained model does not originate from a clever algorithm to compute the ReLU functions in superposition (computation in superposition, CiS), but from taking advantage of the noise. Additionally, we find that the directions represented by the trained model mainly lie in the subspace of the positive eigenvalues of the mixing matrix, suggesting that this matrix determines the learned solution. Finally we present a non-trained model derived from the mixing matrix which improves upon previous baselines. This model exhibits a similar performance profile as the trained model, but does not match all its properties.

While we have not been able to fully reverse-engineer the CC model, this work reveals several key mechanisms behind the model. Our results suggest that CC is likely not a suitable toy model of CiS.

## Toy model architecture

<img width="944" alt="model_architecture" src="https://github.com/user-attachments/assets/7858203b-3274-486a-ae3d-f5eaa7d43a29" />

## 📂 Repo Structure

```bash
.
├── paper-code             # Reproduce all experiments and figures in LessWrong post
├── notebooks/             # Additional experiments
├── toy_cis/               # Training, evaluation, visualization functions
├── test/                  # 
├── README.md              # You are here!
└── pyproject.toml         # Environment config
```

## Environment set-up

Recommended with [pixi](https://pixi.sh/latest/tutorials/python).

In the root directory, just run `pixi install --manifest-path ./pyproject.toml` - this will create a conda env named 'toy-cis'.




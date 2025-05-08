$m$ = number of neurons <br>
$n$ = number of features (i.e. input size and output size) <br>
$b$ = batch size

---

We've observed that CiS does not occur in a one-layer, $m$ neuron MLP model trained to perform $relu(x)$ for each of $n$ features where $n > m$; the model cannot learn something better than the "naive loss", where one neuron is dedicated to performing $relu$ for one feature, and all other $n - m$ features are not represented. 

However, if we add a "residual noise term" in the form of a $n \times n$ matrix $W_n$ that gets multiplied by $x$ and added to the hidden layer output, then the model *can* learn CiS and yield a loss lower than the naive loss.

With the noise term, the model, instead of having to learn the target $relu(x)$, gets to learn an easier "residual target" $relu(x) - W_n x$. A key question of course is, what properties of $W_n$ makes the residual target easier to learn?

First we saw that if we make $W_n$  an  $n \times n$ identity with random gaussian noise on the off-diagonals, there's a "goldilocks zone" for the noise within which it helps the model, and outside of which it hurts the model. Intuitively, a goldilocks zone makes sense: if the noise is too low, the model is just trying to learn $relu(x) - x$, i.e. $-min(0, x)$, which is just as hard to learn as $relu(x)$ i.e. $max(0, x)$. And if the noise is too high, we're just trying to learn random noise instead of a cheaper approximation of $relu(x)$. But still, why does *any* noise help?

Then we saw that a symmetric $W_n$ , with the same gaussian noise in each triangle, helps the model learn even better. Adding symmetric noise couples features, so big residual errors line up along a few directions (i.e. the top eigenvectors of $W_n$ have larger eigenvalues than in the asymmetric case), which drops the effective rank of the residual target, making it easier to approximate by the model's weights.

We can think of the neuron weight matrices, with rank $m$, as trying to approximate the greater rank of the target (rank $n$, shape $b \times n$), and since the relative effective ranks of the labels in the different cases we've explored is:

$$
\begin{aligned}
&relu(x) \\
&> relu(x) - W_{n-asymmetric} \hspace{0.25em} x \\
&> relu(x) - W_{n-symmetric} \hspace{0.25em} x
\end{aligned}
$$

it is easier to approximate the latter cases. In this case, the reduced effective rank also corresponds to higher condition numbers, as we get larger eigenvalues.

i.e. ***the residual target*** $r(x) = relu(x) - W_n x$ ***becomes approximately low-rank and is therefore easier to learn than*** $relu(x)$.

We can test this in an additional case, by using a $W_n$ that is even lower effective rank. To do this, we can use rank-r inflation of the top $r$ eigenvalues of $W_n$ to push the residual target's energy into an effective $r$-dimensional subspace, decreasing its effective rank. This looks like: $W_n = I + \alpha U U^T$ , where $U$ is an orthonormal matrix of rank $r < n$ and $\alpha$ is a small noise coefficient.

In code, this looks like:

```python
Wn = t.eye(n_feat).expand(-1, -1)
U = t.randn(n_feat, r, device=device) 
# Batched QR ⇒ UᵀU = I_r 
U, _ = t.linalg.qr(U, mode="reduced")
# Add rank-r outer prod: I + αUUᵀ
self.Wn += alpha * (einsum(U, U, "feat r, feat2 r -> feat feat2"))
# Keep unit diagonal
idx = t.arange(n_feat, device=device)
self.Wn[:, idx, idx] = 1.0

... 

y += einsum(x, self.Wn, "batch inst feat, inst feat feat_out -> batch inst feat_out")
```

In this case, the model's loss is even lower.

One final point is that, although it appears that using a noise matrix that reduces the effective rank of the residual target can indeed allow models to perform CiS, it is also possible for models to "enter another mode" and "abuse" the noise matrix to yield low loss, but by doing something that we would clearly not call CiS (e.g. we see wacky input-output responses).
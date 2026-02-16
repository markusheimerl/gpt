# attention
A self-attention implementation

Consider a self-attention mechanism operating on batched sequences of shape (batch_size × seq_len × d_model). The architecture implements scaled dot-product attention with learnable query, key, value, and output projections. The forward propagation follows:

$$
\begin{align*}
Q &= XW_q \\
K &= XW_k \\
V &= XW_v \\
S &= \frac{QK^T}{\sqrt{d}} \\
A_{ij} &= \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}} \\
Z &= AV \\
Y &= ZW_o
\end{align*}
$$

The query matrix $W_q$, key matrix $W_k$, and value matrix $W_v$ project input sequences into attention subspaces, while the output projection $W_o$ transforms the attended representations. The scaled dot-product attention computes compatibility scores between all sequence positions, applies softmax normalization, and aggregates values according to attention weights, yielding the following backward pass through the chain rule:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_o} &= Z^T(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial Z} &= (\frac{\partial L}{\partial Y})W_o^T \\
\frac{\partial L}{\partial A} &= (\frac{\partial L}{\partial Z})V^T \\
\frac{\partial L}{\partial V} &= A^T(\frac{\partial L}{\partial Z}) \\
\frac{\partial L}{\partial S} &= A \odot \left(\frac{\partial L}{\partial A} - \sum_j \frac{\partial L}{\partial A} \odot A\right) \\
\frac{\partial L}{\partial Q} &= \frac{1}{\sqrt{d}}(\frac{\partial L}{\partial S})K \\
\frac{\partial L}{\partial K} &= \frac{1}{\sqrt{d}}(\frac{\partial L}{\partial S})^TQ \\
\frac{\partial L}{\partial W_q} &= X^T(\frac{\partial L}{\partial Q}) \\
\frac{\partial L}{\partial W_k} &= X^T(\frac{\partial L}{\partial K}) \\
\frac{\partial L}{\partial W_v} &= X^T(\frac{\partial L}{\partial V}) \\
\frac{\partial L}{\partial X} &= (\frac{\partial L}{\partial Q})W_q^T + (\frac{\partial L}{\partial K})W_k^T + (\frac{\partial L}{\partial V})W_v^T
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run -j 6
```
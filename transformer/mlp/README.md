# mlp
A multilayer perceptron implementation

Consider a multilayer perceptron operating on batched inputs of shape (batch_size × input_dim). The architecture consists of a linear transformation followed by swish activation and another linear transformation. The forward propagation follows, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
H &= XW_1 \\
S &= H \odot \sigma(H) \\
Y &= SW_2
\end{align*}
$$

The input transformation matrix $W_1$ maps input features to hidden representations, and the output projection matrix $W_2$ transforms activated hidden states to outputs. The swish activation $H \odot \sigma(H)$ interpolates between linear and nonlinear regimes, yielding the following backward pass through the chain rule:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_2} &= S^T(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial S} &= (\frac{\partial L}{\partial Y})W_2^T \\
\frac{\partial L}{\partial H} &= \frac{\partial L}{\partial S} \odot [\sigma(H) + H \odot \sigma(H) \odot (1-\sigma(H))] \\
\frac{\partial L}{\partial W_1} &= X^T(\frac{\partial L}{\partial H}) \\
\frac{\partial L}{\partial X} &= (\frac{\partial L}{\partial H})W_1^T
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
make run
```
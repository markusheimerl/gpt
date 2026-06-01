# ssm
A diagonal state-space model implementation

Consider a diagonal state-space model operating on batched sequences of shape (batch_size × seq_len × d_model). The architecture consists of an input projection, a per-channel diagonal linear recurrence, a read-out with skip connection, and an output projection. Each output channel $d$ is processed independently by its own length-$N$ diagonal filter parametrised by $(\hat{a}_d, b_d, c_d) \in \mathbb{R}^N$ and a per-channel skip $D_d$. The recurrence coefficient is $a_d = \sigma(\hat{a}_d) \in (0,1)$ so the system is unconditionally stable. The forward propagation follows, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
U &= XW_{in} \\
s_d[t] &= \sigma(\hat{a}_d) \odot s_d[t-1] + b_d \, U[t,d] \\
Z[t,d] &= c_d \cdot s_d[t] + D_d \, U[t,d] \\
Y &= ZW_{out}
\end{align*}
$$

The input transformation matrix $W_{in}$ maps input features into the channels driving the recurrence, and the output projection matrix $W_{out}$ mixes the per-channel read-outs back into the model dimension. Causality is automatic and no positional encoding is needed. This is the data-independent ancestor of S4 / Mamba: the same hidden-state arithmetic without the input-dependent (selective) parameters, with inference cost $O(L \cdot D \cdot N)$ instead of attention's $O(L^2 \cdot D)$. The backward pass runs the recurrence in reverse through the chain rule:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_{out}} &= Z^T(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial Z} &= (\frac{\partial L}{\partial Y})W_{out}^T \\
ds_d[t] &= c_d \cdot \frac{\partial L}{\partial Z[t,d]} + \sigma(\hat{a}_d) \odot ds_d[t+1] \\
\frac{\partial L}{\partial c_d} &= \sum_t \frac{\partial L}{\partial Z[t,d]} \, s_d[t] \\
\frac{\partial L}{\partial b_d} &= \sum_t ds_d[t] \, U[t,d] \\
\frac{\partial L}{\partial \hat{a}_d} &= \left(\sum_t ds_d[t] \, s_d[t-1]\right) \odot \sigma(\hat{a}_d) \odot (1-\sigma(\hat{a}_d)) \\
\frac{\partial L}{\partial D_d} &= \sum_t \frac{\partial L}{\partial Z[t,d]} \, U[t,d] \\
\frac{\partial L}{\partial U[t,d]} &= \sum_n b_d \cdot ds_d[t] + D_d \cdot \frac{\partial L}{\partial Z[t,d]} \\
\frac{\partial L}{\partial W_{in}} &= X^T(\frac{\partial L}{\partial U}) \\
\frac{\partial L}{\partial X} &= (\frac{\partial L}{\partial U})W_{in}^T
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
### Ubuntu
```bash
sudo apt update
sudo apt install -y clang make time libopenblas-dev nvidia-cuda-toolkit git
sudo reboot
git clone https://github.com/markusheimerl/gpt && cd gpt/transformer/ssm/
make run -j 6
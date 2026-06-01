# mingru
A minimal GRU implementation

Consider a minimal gated recurrent unit operating on batched sequences of shape (batch_size × seq_len × d_model). The architecture consists of two input projections producing an update-gate pre-activation and a candidate pre-activation, followed by a per-channel linear recurrence in which the gate interpolates between the previous hidden state and the candidate. Each channel $d$ carries a scalar hidden state $h_d[t]$ updated independently along time. The forward propagation follows, where $\odot$ denotes elementwise multiplication and $g(v) = v + 0.5$ for $v \ge 0$ else $\sigma(v)$ is a strictly positive activation that keeps $\tilde{h}$ log-representable:

$$
\begin{align*}
K &= XW_z \\
V &= XW_h \\
z_d[t] &= \sigma(K[t,d]) \\
\tilde{h}_d[t] &= g(V[t,d]) \\
h_d[t] &= (1 - z_d[t]) \odot h_d[t-1] + z_d[t] \odot \tilde{h}_d[t] \\
Y &= h
\end{align*}
$$

The projection matrix $W_z$ produces the input-dependent update gate that decides how much new information to admit at each step, and $W_h$ produces the candidate hidden state that is mixed in. Causality is automatic and no positional encoding is needed. This is the minimal GRU of Feng et al. (2024), stripped of the hidden-to-hidden recurrences of the classical GRU so that the per-channel recurrence becomes a first-order linear scan with input-dependent coefficients $a_t = 1 - z_t$ and $b_t = z_t \odot \tilde{h}_t$; the resulting form $h_t = a_t \odot h_{t-1} + b_t$ admits a parallel scan along the sequence at training time, with inference cost $O(L \cdot D)$ instead of attention's $O(L^2 \cdot D)$. The backward pass runs the recurrence in reverse through the chain rule:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
dh_d[t] &= \frac{\partial L}{\partial h_d[t]} + (1 - z_d[t+1]) \odot dh_d[t+1] \\
\frac{\partial L}{\partial z_d[t]} &= (\tilde{h}_d[t] - h_d[t-1]) \odot dh_d[t] \\
\frac{\partial L}{\partial \tilde{h}_d[t]} &= z_d[t] \odot dh_d[t] \\
\frac{\partial L}{\partial K[t,d]} &= \frac{\partial L}{\partial z_d[t]} \odot \sigma(K[t,d]) \odot (1 - \sigma(K[t,d])) \\
\frac{\partial L}{\partial V[t,d]} &= \frac{\partial L}{\partial \tilde{h}_d[t]} \odot g'(V[t,d]) \\
\frac{\partial L}{\partial W_z} &= X^T(\frac{\partial L}{\partial K}) \\
\frac{\partial L}{\partial W_h} &= X^T(\frac{\partial L}{\partial V}) \\
\frac{\partial L}{\partial X} &= (\frac{\partial L}{\partial K})W_z^T + (\frac{\partial L}{\partial V})W_h^T
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
git clone https://github.com/markusheimerl/gpt && cd gpt/transformer/mingru/
make run -j 6
```

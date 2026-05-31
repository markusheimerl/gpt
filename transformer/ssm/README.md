# Vanilla diagonal state-space model

A drop-in replacement for the softmax-attention layer using the simplest useful
state-space recurrence:

```
U = X · W_in                                       (input projection)
s_{b,d}[t] = σ(â_d) ⊙ s_{b,d}[t-1] + b_d · U[b,t,d]   (per-channel scan,
                                                       state vector of size N)
Z[b,t,d]  = c_d · s_{b,d}[t] + D_d · U[b,t,d]      (read-out + skip)
Y = Z · W_out                                      (output projection)
```

Each output channel `d` is processed independently by its own length-`N`
diagonal linear filter, parametrised by `(â_d, b_d, c_d) ∈ ℝ^N` and a per-channel
skip `D_d`. The recurrence coefficient is `a_d = σ(â_d) ∈ (0,1)` so the system
is unconditionally stable.

This is the data-independent ancestor of S4 / Mamba: the same hidden-state
arithmetic without the input-dependent (selective) parameters. Causality is
automatic, no positional encoding is needed, and inference is `O(L · D · N)`
instead of attention's `O(L² · D)`.

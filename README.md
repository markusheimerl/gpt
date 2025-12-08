# gpt
A generative pretrained transformer implementation

This project implements an autoregressive sequence model using a transformer architecture. The model processes sequences of byte-pairs (16-bit tokens), learning to predict the next byte-pair given previous context. While this implementation trains on text data, the architecture is agnostic to the content. It can model any byte stream, including, but not limited to, DNA/RNA sequences, compressed data, images, audio, video, or executable binaries.

The architecture begins with a token embedding layer that converts each byte-pair into a continuous vector representation.

The core of the model is a multi-layer transformer that processes the embedded sequences. Each transformer layer consists of two main components: a causal self-attention mechanism and a feed-forward network, both wrapped with residual connections. The causal attention ensures that predictions for each position can only depend on previous positions, which is essential for autoregressive generation. The attention mechanism computes query, key, and value projections, applies rotational positional encoding to the queries and keys to encode relative positions, computes scaled dot-product attention with a causal mask, and projects the result back. The feed-forward network applies two linear transformations with a swish activation, a smooth, non-monotonic function that multiplies its input by its sigmoid, in between.

After processing through all transformer layers, a linear projection maps the final hidden states to logits over the vocabulary (all 65536 possible byte-pair values). These logits are converted to probabilities using the softmax function, and the model is trained to maximize the probability of the correct next byte-pair using cross-entropy loss.

The training process uses the AdamW optimizer, which enhances the standard Adam optimizer by decoupling weight decay from the gradient-based update. AdamW maintains exponential moving averages of both gradients and squared gradients, using these to adapt the learning rate for each parameter individually. The weight decay acts as L2 regularization, encouraging the model to use smaller weights and improving generalization.

The implementation uses BLAS (Basic Linear Algebra Subprograms) for efficient matrix operations, allowing the model to train effectively on modern hardware.

## How to run
```bash
sudo apt update
sudo apt install clang time libopenblas-dev nvidia-cuda-toolkit
git clone --recurse-submodules https://github.com/markusheimerl/gpt
cd gpt/
wget "https://drive.usercontent.google.com/download?confirm=t&id=17lurm_zFxY-o18NoZVwuxgTknqMPQM6S" -O - | gzip -d > corpus.txt
make run -C gpu -j 4
```
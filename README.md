# gpt
A generative pretrained transformer implementation

## How to run
### Ubuntu
```bash
sudo apt update
sudo apt install -y clang make time libopenblas-dev nvidia-cuda-toolkit git curl unzip timidity
sudo reboot
git clone https://github.com/markusheimerl/gpt && cd gpt/

make data           # download MAESTRO v3 (~80 MB) and emit corpus.bin
make train          # train on corpus.bin (writes sample_*.mid periodically)
make infer          # sample from the latest checkpoint into out.mid
make play out.mid   # play any generated .mid via timidity
```

## Architecture
This project implements an autoregressive sequence model using a transformer architecture. The model processes sequences of 8-bit tokens drawn from a hand-designed 256-symbol alphabet of piano performance events, learning to predict the next event given previous context. The training corpus is built from [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) (~1280 expressive virtuoso piano performances captured on Disklaviers); each MIDI file is parsed and rewritten into an event stream over the vocabulary defined in `tokens.h`: 88 `NOTE_ON pitch` tokens (MIDI pitches 21..108, A0..C8), 88 `NOTE_OFF pitch` tokens, 64 `TIME_SHIFT bin` tokens advancing time in 10 ms steps from 10 ms to 640 ms (longer gaps are emitted as multiple shifts), and 16 `VELOCITY bin` tokens setting per-key dynamics for subsequent notes. Because every token denotes a legal musical event, any sampled sequence is by construction a playable performance. While this implementation trains on MIDI event tokens, the architecture is agnostic to the content. It can model any byte stream, including, but not limited to, DNA/RNA sequences, compressed data, images, audio, video, or executable binaries.

The architecture begins with a token embedding layer that converts each event token into a continuous vector representation.

The core of the model is a multi-layer transformer that processes the embedded sequences. Each transformer layer consists of two main components: a causal self-attention mechanism and a feed-forward network, both wrapped with residual connections. The causal attention ensures that predictions for each position can only depend on previous positions, which is essential for autoregressive generation. The attention mechanism computes query, key, and value projections, applies rotational positional encoding to the queries and keys to encode relative positions, computes scaled dot-product attention with a causal mask, and projects the result back. The feed-forward network applies two linear transformations with a swish activation, a smooth, non-monotonic function that multiplies its input by its sigmoid, in between.

After processing through all transformer layers, a linear projection maps the final hidden states to logits over the vocabulary (all 256 event tokens). These logits are converted to probabilities using the softmax function, and the model is trained to maximize the probability of the correct next token using cross-entropy loss. At inference, the model is seeded with a single mid-range `VELOCITY` token and samples tokens autoregressively; a small detokenizer in `tokens.h` then writes a standard format-0 SMF (1 ms per tick) directly to `out.mid`, playable with `timidity` (`make play out.mid`) without any external conversion step.

The training process uses the AdamW optimizer, which enhances the standard Adam optimizer by decoupling weight decay from the gradient-based update. AdamW maintains exponential moving averages of both gradients and squared gradients, using these to adapt the learning rate for each parameter individually. The weight decay acts as L2 regularization, encouraging the model to use smaller weights and improving generalization.

The implementation uses BLAS (Basic Linear Algebra Subprograms) for efficient matrix operations, allowing the model to train effectively on modern hardware. Training runs on the GPU via cuBLASLt with fp16 mixed precision; inference runs on the CPU via OpenBLAS.

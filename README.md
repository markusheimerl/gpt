# gpt
A generative pretrained transformer implementation

This project implements an autoregressive sequence model using a transformer architecture. The model processes sequences of bytes (8-bit tokens), learning to predict the next byte given previous context. While this implementation trains on text data, the architecture is agnostic to the content. It can model any byte stream, including, but not limited to, DNA/RNA sequences, compressed data, images, audio, video, or executable binaries.

The architecture begins with a token embedding layer that converts each byte into a continuous vector representation.

The core of the model is a multi-layer transformer that processes the embedded sequences. Each transformer layer consists of two main components: a causal self-attention mechanism and a feed-forward network, both wrapped with residual connections. The causal attention ensures that predictions for each position can only depend on previous positions, which is essential for autoregressive generation. The attention mechanism computes query, key, and value projections, applies rotational positional encoding to the queries and keys to encode relative positions, computes scaled dot-product attention with a causal mask, and projects the result back. The feed-forward network applies two linear transformations with a swish activation, a smooth, non-monotonic function that multiplies its input by its sigmoid, in between.

After processing through all transformer layers, a linear projection maps the final hidden states to logits over the vocabulary (all 256 possible byte values). These logits are converted to probabilities using the softmax function, and the model is trained to maximize the probability of the correct next byte using cross-entropy loss.

The training process uses the AdamW optimizer, which enhances the standard Adam optimizer by decoupling weight decay from the gradient-based update. AdamW maintains exponential moving averages of both gradients and squared gradients, using these to adapt the learning rate for each parameter individually. The weight decay acts as L2 regularization, encouraging the model to use smaller weights and improving generalization.

The implementation uses BLAS (Basic Linear Algebra Subprograms) for efficient matrix operations, allowing the model to train effectively on modern hardware.

## How to run
### Ubuntu
```bash
sudo apt update
sudo apt install -y clang make time libopenblas-dev nvidia-cuda-toolkit git curl
git clone https://github.com/markusheimerl/gpt && cd gpt/
make data
make run -j 6
make infer
```

## Sample outputs
After one night of training on a laptop RTX 3080 (16 GB), prompted with `"Once upon a time, there was a"`:

```
Once upon a time, there was a big, round ball standing in the park. The ball
was very happy and started to roll all around. It was so much fun!
The ball rolled faster and faster than ever before. It was a lot of fun, but
it was also very tough. As the ball rolled, it started to get dizzy. The ball
did not know why, but it got louder and louder.
Suddenly, the ball rolled out of the bushes and it stopped rolling. The ball
was sad and didn't know what to do. It didn't know how to roll around anymore
because it was a ball. The ball was sor
```

```
Once upon a time, there was a little girl named Sue. She had a big room with
many toys. Sue liked to play in her room every day. She thought it was fun to
wear a pretty dress and shiny shoes.
One day, Sue went to play with her friend Tim. They played with cars and had
lots of fun. But then, something unexpected happened. A small squirrel came to
play with them. The squirrel put on a show for Sue and Tim to say wnvhteon
they came to see.
Sue and Tim liked the squirrel. They thought it was perfect. The squirrel put
on a funny hat and a tiny
```
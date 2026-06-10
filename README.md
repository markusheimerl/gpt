# gpt
A generative pretrained transformer implementation

This project implements an autoregressive sequence model using a transformer architecture as well as a pretraining pipeline. The model processes sequences of bytes (8-bit tokens), learning to predict the next byte given previous context. While this implementation trains on text data, the architecture is agnostic to the content. It can model any byte stream, including, but not limited to, DNA/RNA sequences, compressed data, images, audio, video, or executable binaries.

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
Prompted with `"Once upon a time, there was a"`:

```
markus@thinkpad:~/gpt$ make infer
Loaded: d_model=512  hidden=1024  layers=16  vocab=256  seq_len=1024
Generating 995 tokens (T=0.70, seed=1779612639)
Once upon a time, there was a compassionate little girl named Lily. She loved to play with her friends in the park. One day, she saw a small bird on a tree. The bird was sad because its wing was hurt.
Lily asked the bird, "Do you have any hurt wing?" The bird said, "Yes, I don't have any hurt wing." Lily wanted to help the bird, so she tried to make it feel better. But the bird was too big and her hurt wing still did not want to hurt Lily's wing.
Lily had an idea. She found a long stick and brought it to the bird. The bird said, "Thank you, Lily! You saved me!" The bird felt better and thanked Lily. They played together in the park all day. They were very happy and became best friends.
<|endoftext|>
Once upon a time, there was a little girl named Mia. Mia loved to study with her toys. She had a big box full of toys in her room. One day, Mia found a new toy. The toy was a small doll. The doll had a pretty dress and smiled a little.
Mia took the doll outside to play. She studied hard and felt the dress on her f


markus@thinkpad:~/gpt$ make infer
Loaded: d_model=512  hidden=1024  layers=16  vocab=256  seq_len=1024
Generating 995 tokens (T=0.70, seed=1779612665)
Once upon a time, there was a little boy named Tim. Tim had a big tree in his yard. He loved to run and play in the tree. One day, he saw a perfect bird in his yard. The bird was sad because it could not find its mom.
Tim wanted to help the bird. He kneeled down and looked all around. He saw a little girl named Sue. Sue was playing with a ball. Tim asked her, "How can I be like your bird?" Sue smiled and said, "You can be my friend."
Tim helped the bird get close to Sue. Sue was so happy and thanked Tim. They became good friends and played together in the tree. The bird sang a song and they all lived happily ever after.
<|endoftext|>

Once upon a time, there was a little boy. He was very careful as he walked around a park. One day, he saw an unusual thing called a rabbit. The rabbit hopped over to the thing and asked the other animals if they had seen it. The other animals thought it was a funny sight.
The rabbit and the other animals were very curious. They asked the other animal, "What do you think is a fun
markus@thinkpad:~/gpt$ 
```
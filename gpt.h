#ifndef GPT_H
#define GPT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cblas.h>
#include "transformer/transformer.h"

typedef struct {
    // Token embedding layer
    float* token_embedding;        // [vocab_size x d_model]
    float* token_embedding_grad;   // [vocab_size x d_model]
    
    // Adam parameters for embeddings
    float* token_embedding_m;      // First moment for token embeddings
    float* token_embedding_v;      // Second moment for token embeddings
    float beta1;                   // Exponential decay rate for first moment
    float beta2;                   // Exponential decay rate for second moment
    float epsilon;                 // Small constant for numerical stability
    int t;                         // Time step
    float weight_decay;            // Weight decay parameter for AdamW
    
    // Forward pass buffers
    float* embedded_input;         // [batch_size x seq_len x d_model]
    float* output;                 // [batch_size x seq_len x vocab_size]
    
    // Backward pass buffers
    float* grad_output;            // [batch_size x seq_len x vocab_size]
    
    // Transformer core
    Transformer* transformer;
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int hidden_dim;
    int num_layers;
    int vocab_size;
} GPT;

// Function prototypes
GPT* init_gpt(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size);
void free_gpt(GPT* gpt);
void forward_pass_gpt(GPT* gpt, unsigned short* input_tokens);
float calculate_loss_gpt(GPT* gpt, unsigned short* target_tokens);
void zero_gradients_gpt(GPT* gpt);
void backward_pass_gpt(GPT* gpt, unsigned short* input_tokens);
void update_weights_gpt(GPT* gpt, float learning_rate, int batch_size);
void reset_optimizer_gpt(GPT* gpt);
void save_gpt(GPT* gpt, const char* filename);
GPT* load_gpt(const char* filename, int batch_size, int seq_len);

#endif
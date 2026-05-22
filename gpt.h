#ifndef GPT_H
#define GPT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "transformer/transformer.h"

typedef struct {
    // Token embedding layer
    half* d_token_embedding;       // [vocab_size x d_model]
    half* d_token_embedding_grad;  // [vocab_size x d_model]
    
    // Adam parameters for embeddings
    float* d_token_embedding_m;    // First moment for token embeddings
    float* d_token_embedding_v;    // Second moment for token embeddings
    float beta1;                   // Exponential decay rate for first moment
    float beta2;                   // Exponential decay rate for second moment
    float epsilon;                 // Small constant for numerical stability
    int t;                         // Time step
    float weight_decay;            // Weight decay parameter for AdamW
    
    // Forward pass buffers
    half* d_embedded_input;        // [batch_size x seq_len x d_model]
    half* d_norm_output;           // [batch_size x seq_len x d_model]
    half* d_output;                // [batch_size x seq_len x vocab_size]
    
    // Backward pass buffers
    half* d_grad_output;           // [batch_size x seq_len x vocab_size]
    half* d_grad_norm_output;      // [batch_size x seq_len x d_model]

    // Loss computation buffer
    float* d_loss_result;          // [1]
    
    // Transformer core
    Transformer* transformer;
    
    // cuBLASLt handle and descriptor
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    
    // Matrix layouts
    cublasLtMatrixLayout_t embedding_layout;          // [vocab_size x d_model]
    cublasLtMatrixLayout_t seq_flat_d_model_layout;   // [batch_size * seq_len x d_model]
    cublasLtMatrixLayout_t seq_flat_vocab_layout;     // [batch_size * seq_len x vocab_size]
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int hidden_dim;
    int num_layers;
    int vocab_size;
} GPT;

// Function prototypes
GPT* init_gpt(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_gpt(GPT* gpt);
void forward_pass_gpt(GPT* gpt, unsigned short* d_input_tokens);
float calculate_loss_gpt(GPT* gpt, unsigned short* d_target_tokens);
void zero_gradients_gpt(GPT* gpt);
void backward_pass_gpt(GPT* gpt, unsigned short* d_input_tokens);
void update_weights_gpt(GPT* gpt, float learning_rate, int batch_size);
void reset_optimizer_gpt(GPT* gpt);
void save_gpt(GPT* gpt, const char* filename);
GPT* load_gpt(const char* filename, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle);

#endif
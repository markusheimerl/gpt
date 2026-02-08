#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "attention/attention.h"
#include "mlp/mlp.h"

typedef struct {
    Attention** attention_layers;
    MLP** mlp_layers;
    
    // RMSNorm buffers
    half** d_norm_attn_inputs;  // [num_layers][batch_size x seq_len x d_model]
    half** d_norm_mlp_inputs;   // [num_layers][batch_size x seq_len x d_model]
    
    // cuBLASLt handle
    cublasLtHandle_t cublaslt_handle;
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int hidden_dim;
    int num_layers;
} Transformer;

// Function prototypes
Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, bool is_causal, bool use_rope, cublasLtHandle_t cublaslt_handle);
void free_transformer(Transformer* transformer);
void forward_pass_transformer(Transformer* transformer, half* d_X);
float calculate_loss_transformer(Transformer* transformer, half* d_y);
void zero_gradients_transformer(Transformer* transformer);
void backward_pass_transformer(Transformer* transformer, half* d_X, half* d_grad_X);
void update_weights_transformer(Transformer* transformer, float learning_rate, int batch_size);
void reset_optimizer_transformer(Transformer* transformer);
void serialize_transformer(Transformer* transformer, FILE* file);
Transformer* deserialize_transformer(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle);

#endif
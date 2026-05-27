#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLASLt Error checking macro
#ifndef CHECK_CUBLASLT
#define CHECK_CUBLASLT(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLASLt error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Weights and gradients
    half* d_W_q;      // [d_model x d_model]
    half* d_W_k;      // [d_model x d_model]
    half* d_W_v;      // [d_model x d_model]
    half* d_W_o;      // [d_model x d_model]
    half* d_W_q_grad; // [d_model x d_model]
    half* d_W_k_grad; // [d_model x d_model]
    half* d_W_v_grad; // [d_model x d_model]
    half* d_W_o_grad; // [d_model x d_model]

    // Adam parameters
    float* d_W_q_m;     // First moment for W_q
    float* d_W_q_v;     // Second moment for W_q
    float* d_W_k_m;     // First moment for W_k
    float* d_W_k_v;     // Second moment for W_k
    float* d_W_v_m;     // First moment for W_v
    float* d_W_v_v;     // Second moment for W_v
    float* d_W_o_m;     // First moment for W_o
    float* d_W_o_v;     // Second moment for W_o
    float beta1;        // Exponential decay rate for first moment
    float beta2;        // Exponential decay rate for second moment
    float epsilon;      // Small constant for numerical stability
    int t;              // Time step
    float weight_decay; // Weight decay parameter for AdamW

    // Forward pass buffers
    half* d_Q;           // [batch_size x seq_len x d_model]
    half* d_K;           // [batch_size x seq_len x d_model]
    half* d_V;           // [batch_size x seq_len x d_model]
    half* d_attn_output; // [batch_size x seq_len x d_model]
    half* d_output;      // [batch_size x seq_len x d_model]

    // Backward pass buffers
    half* d_grad_output;      // [batch_size x seq_len x d_model]
    half* d_grad_attn_output; // [batch_size x seq_len x d_model]
    half* d_grad_Q;           // [batch_size x seq_len x d_model]
    half* d_grad_K;           // [batch_size x seq_len x d_model]
    half* d_grad_V;           // [batch_size x seq_len x d_model]

    // Loss computation buffer
    float* d_loss_result;     // [1]

    // Flash-attention softmax stats
    float* d_stats;           // [batch_size x num_heads x seq_len]

    // cuBLASLt handle and descriptor
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;

    // Matrix layouts
    cublasLtMatrixLayout_t weight_layout;   // [d_model x d_model]
    cublasLtMatrixLayout_t seq_flat_layout; // [batch_size * seq_len x d_model]

    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int num_heads;
    int head_dim;
    float scale;
    bool is_causal;
    bool use_rope;
} Attention;

// Function prototypes
Attention* init_attention(int seq_len, int d_model, int num_heads, int batch_size, bool is_causal, bool use_rope, cublasLtHandle_t cublaslt_handle);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, half* d_X);
float calculate_loss_attention(Attention* attn, half* d_y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, half* d_X, half* d_grad_X);
void update_weights_attention(Attention* attn, float learning_rate, int batch_size);
void reset_optimizer_attention(Attention* attn);
void serialize_attention(Attention* attn, FILE* file);
Attention* deserialize_attention(FILE* file, int batch_size, int seq_len, int num_heads, cublasLtHandle_t cublaslt_handle);

#endif
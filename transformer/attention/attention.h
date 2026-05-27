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
    // Weights and gradients [d_model x d_model]
    half *d_W_q, *d_W_k, *d_W_v, *d_W_o;
    half *d_W_q_grad, *d_W_k_grad, *d_W_v_grad, *d_W_o_grad;
    
    // AdamW parameters
    float *d_W_q_m, *d_W_q_v;
    float *d_W_k_m, *d_W_k_v;
    float *d_W_v_m, *d_W_v_v;
    float *d_W_o_m, *d_W_o_v;
    float beta1, beta2, epsilon, weight_decay;
    int t;
    
    // Forward pass buffers [batch_size x seq_len x d_model]
    half *d_Q, *d_K, *d_V;
    half *d_attn_output;  // Output of softmax(QKᵀ/√d)V
    half *d_output;       // Final output after W_o projection
    
    // Backward pass buffers
    half *d_grad_output;       // Alias of d_output
    half *d_grad_attn_output;
    half *d_grad_Q, *d_grad_K, *d_grad_V;

    // Loss accumulator
    float* d_loss_result;

    // cuDNN flash-attention softmax stats (saved by fwd, consumed by bwd)
    float* d_stats;  // [batch_size x num_heads x seq_len]

    // cuBLASLt handle, descriptor, and layouts
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t weight_layout;     // [d_model x d_model]
    cublasLtMatrixLayout_t seq_flat_layout;   // [batch_size * seq_len x d_model]
    
    // Dimensions
    int seq_len, d_model, batch_size, num_heads, head_dim;
    float scale;
    bool is_causal, use_rope;
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

// cuDNN flash-attention C ABI (defined in cudnn_att.cpp)
#ifdef __cplusplus
extern "C" {
#endif
void cudnn_attention_forward(void* Q, void* K, void* V, void* O, void* stats,
                             int B, int NH, int T, int HS, int is_causal);
void cudnn_attention_backward(void* Q, void* K, void* V, void* O, void* dO, void* stats,
                              void* dQ, void* dK, void* dV,
                              int B, int NH, int T, int HS, int is_causal);
void cudnn_attention_destroy(void);
#ifdef __cplusplus
}
#endif

#endif
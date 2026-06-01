#ifndef MINGRU_H
#define MINGRU_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

// Minimal-GRU token mixer (Feng et al. 2024, "Were RNNs All We Needed?").
//   K  = X @ W_z           pre-gate
//   V  = X @ W_h           pre-candidate
//   z  = σ(K)              update gate
//   h̃ = g(V)               candidate, g(v) = v+0.5 if v≥0 else σ(v)  (>0, log-stable)
//   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t      (parallel scan along t)
typedef struct {
    // Weights and gradients
    half* d_W_z;        // [d_model x d_model]
    half* d_W_h;        // [d_model x d_model]
    half* d_W_z_grad;   // [d_model x d_model]
    half* d_W_h_grad;   // [d_model x d_model]

    // Adam moments (fp32)
    float* d_W_z_m;
    float* d_W_z_v;
    float* d_W_h_m;
    float* d_W_h_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;

    // Forward buffers
    half* d_K;          // [batch x seq x d_model]   pre-gate
    half* d_V;          // [batch x seq x d_model]   pre-candidate
    half* d_output;     // [batch x seq x d_model]   scan output H (= layer output)

    // Backward buffers
    half* d_grad_output; // [batch x seq x d_model]  upstream dY
    half* d_grad_K;      // alias of d_K
    half* d_grad_V;      // alias of d_V

    // Loss buffer (only used by the standalone module test)
    float* d_loss_result;

    // cuBLASLt handle / descriptors
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;

    cublasLtMatrixLayout_t weight_layout;    // [d_model x d_model]
    cublasLtMatrixLayout_t seq_flat_layout;  // [batch_size * seq_len x d_model]

    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
} MinGRU;

// Function prototypes
MinGRU* init_mingru(int seq_len, int d_model, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_mingru(MinGRU* m);
void forward_pass_mingru(MinGRU* m, half* d_X);
float calculate_loss_mingru(MinGRU* m, half* d_y);
void zero_gradients_mingru(MinGRU* m);
void backward_pass_mingru(MinGRU* m, half* d_X, half* d_grad_X);
void update_weights_mingru(MinGRU* m, float learning_rate, int batch_size);
void reset_optimizer_mingru(MinGRU* m);
void serialize_mingru(MinGRU* m, FILE* file);
MinGRU* deserialize_mingru(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle);

#endif

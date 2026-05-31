#ifndef SSM_H
#define SSM_H

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

// Vanilla diagonal state-space model layer.
//
//   U = X W_in                                     ([B,L,D])
//   s_{b,d}[t] = a_d ⊙ s_{b,d}[t-1] + b_d * U[b,t,d]   (per-channel, vector len N)
//   Z[b,t,d]   = c_d · s_{b,d}[t] + D_d * U[b,t,d]
//   Y = Z W_out                                    ([B,L,D])
//
// Per-channel parameters a_d, b_d, c_d ∈ ℝ^N (state size N). Recurrence
// coefficient parametrised as a_d = σ(â_d) so values stay in (0,1).
typedef struct {
    // Device weights
    half* d_W_in;     // [d_model x d_model]   input projection
    half* d_W_out;    // [d_model x d_model]   output projection
    half* d_A_raw;    // [d_model x state_dim] raw (σ applied at use)
    half* d_B;        // [d_model x state_dim]
    half* d_C;        // [d_model x state_dim]
    half* d_D;        // [d_model]             per-channel skip

    // Device gradients
    half* d_W_in_grad;
    half* d_W_out_grad;
    half* d_A_raw_grad;
    half* d_B_grad;
    half* d_C_grad;
    half* d_D_grad;

    // Adam state (m, v) for every weight tensor
    float *d_W_in_m,  *d_W_in_v;
    float *d_W_out_m, *d_W_out_v;
    float *d_A_raw_m, *d_A_raw_v;
    float *d_B_m,     *d_B_v;
    float *d_C_m,     *d_C_v;
    float *d_D_m,     *d_D_v;
    float beta1;
    float beta2;
    float epsilon;
    int   t;
    float weight_decay;

    // Forward-pass buffers
    half* d_U;        // [B x L x D]               input-projected sequence
    half* d_Z;        // [B x L x D]               scan output (pre W_out)
    half* d_output;   // [B x L x D]               final output Y
    half* d_states;   // [B x L x D x state_dim]   saved states for backward

    // Backward-pass buffers (aliases of forward buffers)
    half* d_grad_output;   // alias of d_output
    half* d_grad_Z;        // alias of d_Z
    half* d_grad_U;        // alias of d_U

    // Loss accumulator
    float* d_loss_result;

    // cuBLASLt handle and matmul descriptor
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;

    // Matrix layouts
    cublasLtMatrixLayout_t weight_layout;     // [D x D]
    cublasLtMatrixLayout_t seq_flat_layout;   // [B*L x D]

    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int state_dim;
} SSM;

// Function prototypes
SSM* init_ssm(int seq_len, int d_model, int state_dim, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_ssm(SSM* ssm);
void forward_pass_ssm(SSM* ssm, half* d_X);
float calculate_loss_ssm(SSM* ssm, half* d_y);
void zero_gradients_ssm(SSM* ssm);
void backward_pass_ssm(SSM* ssm, half* d_X, half* d_grad_X);
void update_weights_ssm(SSM* ssm, float learning_rate, int batch_size);
void reset_optimizer_ssm(SSM* ssm);
void serialize_ssm(SSM* ssm, FILE* file);
SSM* deserialize_ssm(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle);

#endif

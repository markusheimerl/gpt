#ifndef SSM_H
#define SSM_H

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

// State-space token mixer (single layer, swish dropped → S_t = H_t):
//   H_t = X_t B^T + H_{t-1} A^T
//   Y_t = H_t C^T + X_t D^T
// A: [state_dim x state_dim]   B: [state_dim x d_model]
// C: [d_model   x state_dim]   D: [d_model   x d_model]
typedef struct {
    // Weights and gradients (fp16 storage, fp32 Adam moments)
    half* d_A;        half* d_A_grad;
    half* d_B;        half* d_B_grad;
    half* d_C;        half* d_C_grad;
    half* d_D;        half* d_D_grad;

    // Adam moments (fp32)
    float* d_A_m;  float* d_A_v;
    float* d_B_m;  float* d_B_v;
    float* d_C_m;  float* d_C_v;
    float* d_D_m;  float* d_D_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;

    // Forward / backward buffers
    half* d_H;       // [batch_size x seq_len x state_dim]  (batch-major; saved for backward)
    half* d_output;  // [batch_size x seq_len x d_model]    layer output Y (public, batch-major)
    half* d_grad_output;  // [batch_size x seq_len x d_model]   upstream dY (alias of d_output)
    half* d_grad_H;       // [batch_size x seq_len x state_dim] BPTT scratch

    // Loss buffer (standalone test only)
    float* d_loss_result;

    // cuBLASLt handle / descriptors
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;

    // Matrix layouts (all row-major, fp16). Per-timestep slice layouts use ld = seq_len * <cols>
    // so the same descriptor can address X[:, t, :] / H[:, t, :] via a base-pointer offset.
    cublasLtMatrixLayout_t L_A;        // (state_dim, state_dim)
    cublasLtMatrixLayout_t L_B;        // (state_dim, d_model)
    cublasLtMatrixLayout_t L_C;        // (d_model,   state_dim)
    cublasLtMatrixLayout_t L_D;        // (d_model,   d_model)
    cublasLtMatrixLayout_t L_H_slice;  // (batch_size, state_dim) ld = seq_len * state_dim
    cublasLtMatrixLayout_t L_H_flat;   // (batch_size * seq_len, state_dim)
    cublasLtMatrixLayout_t L_X_flat;   // (batch_size * seq_len, d_model)  (also used for Y, dY, dX)

    // Dimensions
    int seq_len;
    int d_model;
    int state_dim;
    int batch_size;
} SSM;

// Function prototypes
SSM*  init_ssm(int seq_len, int d_model, int state_dim, int batch_size, cublasLtHandle_t cublaslt_handle);
void  free_ssm(SSM* ssm);
void  forward_pass_ssm(SSM* ssm, half* d_X);
float calculate_loss_ssm(SSM* ssm, half* d_y);
void  zero_gradients_ssm(SSM* ssm);
void  backward_pass_ssm(SSM* ssm, half* d_X, half* d_grad_X);
void  update_weights_ssm(SSM* ssm, float learning_rate, int batch_size);
void  reset_optimizer_ssm(SSM* ssm);
void  serialize_ssm(SSM* ssm, FILE* file);
SSM*  deserialize_ssm(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle);

#endif

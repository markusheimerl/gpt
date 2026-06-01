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

typedef struct {
    // Weights and gradients
    half* d_W_in;       // [d_model x d_model]
    half* d_W_out;      // [d_model x d_model]
    half* d_A_raw;      // [d_model x state_dim]   (σ applied at use)
    half* d_B;          // [d_model x state_dim]
    half* d_C;          // [d_model x state_dim]
    half* d_D;          // [d_model]
    half* d_W_in_grad;  // [d_model x d_model]
    half* d_W_out_grad; // [d_model x d_model]
    half* d_A_raw_grad; // [d_model x state_dim]
    half* d_B_grad;     // [d_model x state_dim]
    half* d_C_grad;     // [d_model x state_dim]
    half* d_D_grad;     // [d_model]

    // Adam parameters
    float* d_W_in_m;    // First moment for W_in
    float* d_W_in_v;    // Second moment for W_in
    float* d_W_out_m;   // First moment for W_out
    float* d_W_out_v;   // Second moment for W_out
    float* d_A_raw_m;   // First moment for A_raw
    float* d_A_raw_v;   // Second moment for A_raw
    float* d_B_m;       // First moment for B
    float* d_B_v;       // Second moment for B
    float* d_C_m;       // First moment for C
    float* d_C_v;       // Second moment for C
    float* d_D_m;       // First moment for D
    float* d_D_v;       // Second moment for D
    float beta1;        // Exponential decay rate for first moment
    float beta2;        // Exponential decay rate for second moment
    float epsilon;      // Small constant for numerical stability
    int t;              // Time step
    float weight_decay; // Weight decay parameter for AdamW

    // Forward pass buffers
    half* d_U;       // [batch_size x seq_len x d_model]               input projection
    half* d_Z;       // [batch_size x seq_len x d_model]               scan output (pre W_out)
    half* d_output;  // [batch_size x seq_len x d_model]               final output
    half* d_states;  // [batch_size x seq_len x d_model x state_dim]   saved states

    // Backward pass buffers
    half* d_grad_output;    // [batch_size x seq_len x d_model]
    half* d_grad_Z;         // [batch_size x seq_len x d_model]
    half* d_grad_U;         // [batch_size x seq_len x d_model]

    // Loss computation buffer
    float* d_loss_result;   // [1]

    // cuBLASLt handle and descriptor
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;

    // Matrix layouts
    cublasLtMatrixLayout_t weight_layout;     // [d_model x d_model]
    cublasLtMatrixLayout_t seq_flat_layout;   // [batch_size * seq_len x d_model]

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
#ifndef MLP_H
#define MLP_H

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

// cuBLASLt matrix multiplication macro
#ifndef LT_MATMUL
#define LT_MATMUL(mlp, opA, opB, alpha, A, layA, B, layB, beta, C, layC) do { \
    cublasOperation_t _opA = opA, _opB = opB; \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(mlp->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSA, &_opA, sizeof(_opA))); \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(mlp->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSB, &_opB, sizeof(_opB))); \
    CHECK_CUBLASLT(cublasLtMatmul(mlp->cublaslt_handle, mlp->matmul_desc, \
                                  alpha, A, layA, B, layB, \
                                  beta, C, layC, \
                                  C, layC, NULL, NULL, 0, 0)); \
} while(0)
#endif

typedef struct {
    // Weights and gradients
    half* d_W1;      // [input_dim x hidden_dim]
    half* d_W2;      // [hidden_dim x output_dim]
    half* d_W1_grad; // [input_dim x hidden_dim]
    half* d_W2_grad; // [hidden_dim x output_dim]
    
    // Adam parameters
    float* d_W1_m;      // First moment for W1
    float* d_W1_v;      // Second moment for W1
    float* d_W2_m;      // First moment for W2
    float* d_W2_v;      // Second moment for W2
    float beta1;        // Exponential decay rate for first moment
    float beta2;        // Exponential decay rate for second moment
    float epsilon;      // Small constant for numerical stability
    int t;              // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Forward pass buffers
    half* d_preact;  // [batch_size x hidden_dim]
    half* d_postact; // [batch_size x hidden_dim]
    half* d_output;  // [batch_size x output_dim]

    // Backward pass buffers
    half* d_grad_output;    // [batch_size x output_dim]
    half* d_grad_postact;   // [batch_size x hidden_dim]

    // Loss computation buffer
    float* d_loss_result;   // [1]

    // cuBLASLt handle and descriptor
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    
    // Matrix layouts
    cublasLtMatrixLayout_t W1_layout;           // [input_dim x hidden_dim]
    cublasLtMatrixLayout_t W2_layout;           // [hidden_dim x output_dim]
    cublasLtMatrixLayout_t batch_input_layout;  // [batch_size x input_dim]
    cublasLtMatrixLayout_t batch_hidden_layout; // [batch_size x hidden_dim]
    cublasLtMatrixLayout_t batch_output_layout; // [batch_size x output_dim]
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} MLP;

// Function prototypes
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_mlp(MLP* mlp);
void forward_pass_mlp(MLP* mlp, half* d_X);
float calculate_loss_mlp(MLP* mlp, half* d_y);
void zero_gradients_mlp(MLP* mlp);
void backward_pass_mlp(MLP* mlp, half* d_X, half* d_grad_X);
void update_weights_mlp(MLP* mlp, float learning_rate, int batch_size);
void reset_optimizer_mlp(MLP* mlp);
void serialize_mlp(MLP* mlp, FILE* file);
MLP* deserialize_mlp(FILE* file, int batch_size, cublasLtHandle_t cublaslt_handle);

#endif
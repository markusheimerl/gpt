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
#include "transformer/muon.h"

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
    // Token embedding layer
    half* d_token_embedding;       // [vocab_size x d_model]
    half* d_token_embedding_grad;  // [vocab_size x d_model]

    // Muon optimizer state for the token embedding matrix
    MuonState muon_token_embedding;
    float beta;                    // Momentum decay rate
    float weight_decay;            // Decoupled weight decay coefficient
    int t;                         // Training step counter (for checkpoint resume)
    
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
void forward_pass_gpt(GPT* gpt, unsigned char* d_input_tokens);
float calculate_loss_gpt(GPT* gpt, unsigned char* d_target_tokens);
void zero_gradients_gpt(GPT* gpt);
void backward_pass_gpt(GPT* gpt, unsigned char* d_input_tokens);
void update_weights_gpt(GPT* gpt, float learning_rate, int batch_size);
void reset_optimizer_gpt(GPT* gpt);
void save_gpt(GPT* gpt, const char* filename);
GPT* load_gpt(const char* filename, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle);

#endif
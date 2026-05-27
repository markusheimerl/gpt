#include "attention.h"

// cuBLASLt matrix multiplication macro
#define LT_MATMUL(attn, opA, opB, alpha, A, layA, B, layB, beta, C, layC) do { \
    cublasOperation_t _opA = opA, _opB = opB; \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(attn->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSA, &_opA, sizeof(_opA))); \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(attn->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSB, &_opB, sizeof(_opB))); \
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle, attn->matmul_desc, \
                                  alpha, A, layA, B, layB, \
                                  beta, C, layC, \
                                  C, layC, NULL, NULL, 0, 0)); \
} while(0)

// Initialize the attention layer
Attention* init_attention(int seq_len, int d_model, int num_heads, int batch_size, bool is_causal, bool use_rope, cublasLtHandle_t cublaslt_handle) {
    if (num_heads <= 0 || d_model % num_heads != 0) {
        fprintf(stderr, "d_model (%d) must be divisible by num_heads (%d)\n", d_model, num_heads);
        exit(EXIT_FAILURE);
    }
    
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->batch_size = batch_size;
    attn->num_heads = num_heads;
    attn->head_dim = d_model / num_heads;
    attn->scale = 1.0f / sqrtf(attn->head_dim);
    attn->is_causal = is_causal;
    attn->use_rope = use_rope;
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    // Initialize cuBLASLt
    attn->cublaslt_handle = cublaslt_handle;
    
    size_t weight_size = d_model * d_model;
    size_t seq_batch_size = batch_size * seq_len * d_model;
    
    // Initialize weights on host
    float scale_W = 1.0f / sqrtf(d_model);
    half* h_W_q = (half*)malloc(weight_size * sizeof(half));
    half* h_W_k = (half*)malloc(weight_size * sizeof(half));
    half* h_W_v = (half*)malloc(weight_size * sizeof(half));
    half* h_W_o = (half*)malloc(weight_size * sizeof(half));
    
    for (size_t i = 0; i < weight_size; i++) {
        h_W_q[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W);
        h_W_k[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W);
        h_W_v[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W);
        h_W_o[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W);
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&attn->d_W_q, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_grad, weight_size * sizeof(half)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_v, weight_size * sizeof(float)));
    
    // Allocate forward pass buffers
    CHECK_CUDA(cudaMalloc(&attn->d_Q, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_K, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_V, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_output, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_output, seq_batch_size * sizeof(half)));
    
    // Allocate backward pass buffers (d_grad_output aliases d_output)
    attn->d_grad_output = attn->d_output;
    CHECK_CUDA(cudaMalloc(&attn->d_grad_attn_output, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_Q, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_K, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_V, seq_batch_size * sizeof(half)));

    // cuDNN softmax stats and loss buffer
    CHECK_CUDA(cudaMalloc(&attn->d_stats, (size_t)batch_size * num_heads * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_loss_result, sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_W_q, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_W_k, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_W_v, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_W_o, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(attn->d_W_q_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_q_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_v, 0, weight_size * sizeof(float)));
    
    // Create cuBLASLt matrix multiplication descriptor and layouts
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Weight matrices [d_model x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->weight_layout, CUDA_R_16F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Flattened sequence data [batch_size * seq_len x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->seq_flat_layout, CUDA_R_16F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_flat_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    
    return attn;
}

// Free attention memory
void free_attention(Attention* attn) {
    cublasLtMatmulDescDestroy(attn->matmul_desc);
    cublasLtMatrixLayoutDestroy(attn->weight_layout);
    cublasLtMatrixLayoutDestroy(attn->seq_flat_layout);
    
    cudaFree(attn->d_W_q); cudaFree(attn->d_W_k); cudaFree(attn->d_W_v); cudaFree(attn->d_W_o);
    cudaFree(attn->d_W_q_grad); cudaFree(attn->d_W_k_grad); cudaFree(attn->d_W_v_grad); cudaFree(attn->d_W_o_grad);
    cudaFree(attn->d_W_q_m); cudaFree(attn->d_W_q_v); cudaFree(attn->d_W_k_m); cudaFree(attn->d_W_k_v);
    cudaFree(attn->d_W_v_m); cudaFree(attn->d_W_v_v); cudaFree(attn->d_W_o_m); cudaFree(attn->d_W_o_v);
    cudaFree(attn->d_Q); cudaFree(attn->d_K); cudaFree(attn->d_V);
    cudaFree(attn->d_attn_output); cudaFree(attn->d_output);
    cudaFree(attn->d_grad_attn_output);
    cudaFree(attn->d_grad_Q); cudaFree(attn->d_grad_K); cudaFree(attn->d_grad_V);
    cudaFree(attn->d_stats);
    cudaFree(attn->d_loss_result);
    
    free(attn);
}

// CUDA kernel for RoPE forward pass
__global__ static void rope_forward_kernel_attention(half* Q, half* K, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d_pair = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d_pair >= d_model / 2) return;
    
    int d = d_pair * 2;
    float theta = powf(10000.0f, -((float)d / (float)d_model));
    float angle = (float)t * theta;
    float cos_a = cosf(angle), sin_a = sinf(angle);
    int idx = b * seq_len * d_model + t * d_model + d;
    
    // Rotate Q
    float q0 = __half2float(Q[idx]), q1 = __half2float(Q[idx + 1]);
    Q[idx]     = __float2half(q0 * cos_a - q1 * sin_a);
    Q[idx + 1] = __float2half(q0 * sin_a + q1 * cos_a);
    
    // Rotate K
    float k0 = __half2float(K[idx]), k1 = __half2float(K[idx + 1]);
    K[idx]     = __float2half(k0 * cos_a - k1 * sin_a);
    K[idx + 1] = __float2half(k0 * sin_a + k1 * cos_a);
}

// CUDA kernel for RoPE backward pass
__global__ static void rope_backward_kernel_attention(half* grad_Q, half* grad_K, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d_pair = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d_pair >= d_model / 2) return;
    
    int d = d_pair * 2;
    float theta = powf(10000.0f, -((float)d / (float)d_model));
    float angle = (float)t * theta;
    float cos_a = cosf(angle), sin_a = sinf(angle);
    int idx = b * seq_len * d_model + t * d_model + d;
    
    // Inverse rotate grad_Q
    float gq0 = __half2float(grad_Q[idx]), gq1 = __half2float(grad_Q[idx + 1]);
    grad_Q[idx]     = __float2half( gq0 * cos_a + gq1 * sin_a);
    grad_Q[idx + 1] = __float2half(-gq0 * sin_a + gq1 * cos_a);
    
    // Inverse rotate grad_K
    float gk0 = __half2float(grad_K[idx]), gk1 = __half2float(grad_K[idx + 1]);
    grad_K[idx]     = __float2half( gk0 * cos_a + gk1 * sin_a);
    grad_K[idx + 1] = __float2half(-gk0 * sin_a + gk1 * cos_a);
}

// Forward pass
void forward_pass_attention(Attention* attn, half* d_X) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Q, K, V = XW_q, XW_k, XW_v
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_W_q, attn->weight_layout,
              &beta, attn->d_Q, attn->seq_flat_layout);
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_W_k, attn->weight_layout,
              &beta, attn->d_K, attn->seq_flat_layout);
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_W_v, attn->weight_layout,
              &beta, attn->d_V, attn->seq_flat_layout);
    
    // Apply RoPE to Q and K
    if (attn->use_rope) {
        dim3 grid(attn->batch_size, attn->seq_len);
        rope_forward_kernel_attention<<<grid, attn->d_model / 2>>>(
            attn->d_Q, attn->d_K, attn->batch_size, attn->seq_len, attn->d_model);
    }
    
    // Z = softmax(QKᵀ/√d) V  (cuDNN flash attention)
    cudnn_attention_forward(attn->d_Q, attn->d_K, attn->d_V,
                            attn->d_attn_output, attn->d_stats,
                            attn->batch_size, attn->num_heads, attn->seq_len, attn->head_dim,
                            attn->is_causal ? 1 : 0);
    
    // Y = ZW_o
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout, attn->d_W_o, attn->weight_layout,
              &beta, attn->d_output, attn->seq_flat_layout);
}

// CUDA kernel for computing loss and gradient
__global__ static void compute_loss_and_gradient_kernel_attention(half* grad_output, half* predictions, half* targets, float* loss_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred = __half2float(predictions[idx]);
        float target = __half2float(targets[idx]);
        float diff = pred - target;
        grad_output[idx] = __float2half(diff);
        atomicAdd(loss_result, diff * diff);
    }
}

// Calculate loss
float calculate_loss_attention(Attention* attn, half* d_y) {
    // ∂L/∂Y = Y - Y_true
    int total_elements = attn->batch_size * attn->seq_len * attn->d_model;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    CHECK_CUDA(cudaMemset(attn->d_loss_result, 0, sizeof(float)));
    compute_loss_and_gradient_kernel_attention<<<num_blocks, block_size>>>(
        attn->d_grad_output, attn->d_output, d_y, attn->d_loss_result, total_elements);
    
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, attn->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    return total_loss / total_elements;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;
    CHECK_CUDA(cudaMemset(attn->d_W_q_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_grad, 0, weight_size * sizeof(half)));
}

// Backward pass
void backward_pass_attention(Attention* attn, half* d_X, half* d_grad_X) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // ∂L/∂W_o = Zᵀ(∂L/∂Y);  ∂L/∂Z = (∂L/∂Y)W_oᵀ
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout, attn->d_grad_output, attn->seq_flat_layout,
              &beta, attn->d_W_o_grad, attn->weight_layout);
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              attn->d_grad_output, attn->seq_flat_layout, attn->d_W_o, attn->weight_layout,
              &beta, attn->d_grad_attn_output, attn->seq_flat_layout);
    
    // (∂L/∂Q, ∂L/∂K, ∂L/∂V) via cuDNN flash-attention backward
    cudnn_attention_backward(attn->d_Q, attn->d_K, attn->d_V,
                             attn->d_attn_output, attn->d_grad_attn_output, attn->d_stats,
                             attn->d_grad_Q, attn->d_grad_K, attn->d_grad_V,
                             attn->batch_size, attn->num_heads, attn->seq_len, attn->head_dim,
                             attn->is_causal ? 1 : 0);
    
    // Inverse RoPE on grad_Q and grad_K
    if (attn->use_rope) {
        dim3 grid(attn->batch_size, attn->seq_len);
        rope_backward_kernel_attention<<<grid, attn->d_model / 2>>>(
            attn->d_grad_Q, attn->d_grad_K, attn->batch_size, attn->seq_len, attn->d_model);
    }
    
    // ∂L/∂W_{q,k,v} = Xᵀ(∂L/∂{Q,K,V})
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_grad_Q, attn->seq_flat_layout,
              &beta, attn->d_W_q_grad, attn->weight_layout);
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_grad_K, attn->seq_flat_layout,
              &beta, attn->d_W_k_grad, attn->weight_layout);
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_grad_V, attn->seq_flat_layout,
              &beta, attn->d_W_v_grad, attn->weight_layout);
    
    // ∂L/∂X = (∂L/∂Q)W_qᵀ + (∂L/∂K)W_kᵀ + (∂L/∂V)W_vᵀ
    if (d_grad_X != NULL) {
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_Q, attn->seq_flat_layout, attn->d_W_q, attn->weight_layout,
                  &beta,  d_grad_X, attn->seq_flat_layout);
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_K, attn->seq_flat_layout, attn->d_W_k, attn->weight_layout,
                  &alpha, d_grad_X, attn->seq_flat_layout);
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_V, attn->seq_flat_layout, attn->d_W_v, attn->weight_layout,
                  &alpha, d_grad_X, attn->seq_flat_layout);
    }
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_attention(half* weight, half* grad, float* m, float* v,
                                                     float beta1, float beta2, float epsilon, float learning_rate,
                                                     float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(grad[idx]) / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W);  v = β₂v + (1-β₂)(∂L/∂W)²
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        float w = __half2float(weight[idx]);
        weight[idx] = __float2half(w * (1.0f - learning_rate * weight_decay) - update);
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate, int batch_size) {
    attn->t++;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    int block_size = 256;
    int num_blocks = (weight_size + block_size - 1) / block_size;
    
    half*  weights[] = {attn->d_W_q,      attn->d_W_k,      attn->d_W_v,      attn->d_W_o};
    half*  grads[]   = {attn->d_W_q_grad, attn->d_W_k_grad, attn->d_W_v_grad, attn->d_W_o_grad};
    float* ms[]      = {attn->d_W_q_m,    attn->d_W_k_m,    attn->d_W_v_m,    attn->d_W_o_m};
    float* vs[]      = {attn->d_W_q_v,    attn->d_W_k_v,    attn->d_W_v_v,    attn->d_W_o_v};
    
    for (int w = 0; w < 4; w++) {
        adamw_update_kernel_attention<<<num_blocks, block_size>>>(
            weights[w], grads[w], ms[w], vs[w],
            attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
            alpha_t, weight_size, batch_size);
    }
}

// Reset optimizer state
void reset_optimizer_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;
    
    CHECK_CUDA(cudaMemset(attn->d_W_q_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_q_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_v, 0, weight_size * sizeof(float)));
    
    attn->t = 0;
}

// Serialize attention to file
void serialize_attention(Attention* attn, FILE* file) {
    fwrite(&attn->d_model,   sizeof(int),  1, file);
    fwrite(&attn->is_causal, sizeof(bool), 1, file);
    fwrite(&attn->use_rope,  sizeof(bool), 1, file);
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Weights: copy half from device, convert to float in-place, write
    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_W_q, attn->d_W_q, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k, attn->d_W_k, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v, attn->d_W_v, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o, attn->d_W_o, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    for (int i = weight_size - 1; i >= 0; i--) {
        h_W_q[i] = __half2float(((half*)h_W_q)[i]);
        h_W_k[i] = __half2float(((half*)h_W_k)[i]);
        h_W_v[i] = __half2float(((half*)h_W_v)[i]);
        h_W_o[i] = __half2float(((half*)h_W_o)[i]);
    }
    
    fwrite(h_W_q, sizeof(float), weight_size, file);
    fwrite(h_W_k, sizeof(float), weight_size, file);
    fwrite(h_W_v, sizeof(float), weight_size, file);
    fwrite(h_W_o, sizeof(float), weight_size, file);
    
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    
    // Optimizer state
    fwrite(&attn->t, sizeof(int), 1, file);
    
    float* h_W_q_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_q_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_v = (float*)malloc(weight_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_W_q_m, attn->d_W_q_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_q_v, attn->d_W_q_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k_m, attn->d_W_k_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k_v, attn->d_W_k_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v_m, attn->d_W_v_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v_v, attn->d_W_v_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o_m, attn->d_W_o_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o_v, attn->d_W_o_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_W_q_m, sizeof(float), weight_size, file);
    fwrite(h_W_q_v, sizeof(float), weight_size, file);
    fwrite(h_W_k_m, sizeof(float), weight_size, file);
    fwrite(h_W_k_v, sizeof(float), weight_size, file);
    fwrite(h_W_v_m, sizeof(float), weight_size, file);
    fwrite(h_W_v_v, sizeof(float), weight_size, file);
    fwrite(h_W_o_m, sizeof(float), weight_size, file);
    fwrite(h_W_o_v, sizeof(float), weight_size, file);
    
    free(h_W_q_m); free(h_W_q_v); free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v); free(h_W_o_m); free(h_W_o_v);
}

// Deserialize attention from file
Attention* deserialize_attention(FILE* file, int batch_size, int seq_len, int num_heads, cublasLtHandle_t cublaslt_handle) {
    int d_model;
    bool is_causal, use_rope;
    fread(&d_model,   sizeof(int),  1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope,  sizeof(bool), 1, file);
    
    Attention* attn = init_attention(seq_len, d_model, num_heads, batch_size, is_causal, use_rope, cublaslt_handle);
    
    int weight_size = d_model * d_model;
    
    // Weights: read float, convert to half in-place, copy to device
    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));
    
    fread(h_W_q, sizeof(float), weight_size, file);
    fread(h_W_k, sizeof(float), weight_size, file);
    fread(h_W_v, sizeof(float), weight_size, file);
    fread(h_W_o, sizeof(float), weight_size, file);
    
    for (int i = 0; i < weight_size; i++) {
        ((half*)h_W_q)[i] = __float2half(h_W_q[i]);
        ((half*)h_W_k)[i] = __float2half(h_W_k[i]);
        ((half*)h_W_v)[i] = __float2half(h_W_v[i]);
        ((half*)h_W_o)[i] = __float2half(h_W_o[i]);
    }
    
    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_W_q, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_W_k, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_W_v, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_W_o, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    
    // Optimizer state
    fread(&attn->t, sizeof(int), 1, file);
    
    float* h_W_q_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_q_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_v = (float*)malloc(weight_size * sizeof(float));
    
    fread(h_W_q_m, sizeof(float), weight_size, file);
    fread(h_W_q_v, sizeof(float), weight_size, file);
    fread(h_W_k_m, sizeof(float), weight_size, file);
    fread(h_W_k_v, sizeof(float), weight_size, file);
    fread(h_W_v_m, sizeof(float), weight_size, file);
    fread(h_W_v_v, sizeof(float), weight_size, file);
    fread(h_W_o_m, sizeof(float), weight_size, file);
    fread(h_W_o_v, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_m, h_W_q_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_v, h_W_q_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_m, h_W_k_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_v, h_W_k_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_m, h_W_v_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_v, h_W_v_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_m, h_W_o_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_v, h_W_o_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_W_q_m); free(h_W_q_v); free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v); free(h_W_o_m); free(h_W_o_v);
    
    return attn;
}
#include "gpt.h"

// Initialize the GPT
GPT* init_gpt(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle) {
    GPT* gpt = (GPT*)malloc(sizeof(GPT));
    
    // Store dimensions
    gpt->seq_len = seq_len;
    gpt->d_model = d_model;
    gpt->batch_size = batch_size;
    gpt->hidden_dim = hidden_dim;
    gpt->num_layers = num_layers;
    gpt->vocab_size = 65536;
    gpt->cublaslt_handle = cublaslt_handle;
    
    // Initialize Adam parameters
    gpt->beta1 = 0.9f;
    gpt->beta2 = 0.999f;
    gpt->epsilon = 1e-8f;
    gpt->t = 0;
    gpt->weight_decay = 0.01f;
    
    size_t token_emb_size = gpt->vocab_size * d_model;
    size_t embedded_size = batch_size * seq_len * d_model;
    size_t output_size = (size_t)batch_size * seq_len * gpt->vocab_size;
    
    // Allocate host memory for weight initialization
    half* h_token_embedding = (half*)malloc(token_emb_size * sizeof(half));
    
    // Initialize token embeddings on host
    float token_scale = 1.0f / sqrtf(d_model);
    
    for (size_t i = 0; i < token_emb_size; i++) {
        h_token_embedding[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale);
    }
    
    // Allocate device memory for embeddings and gradients
    CHECK_CUDA(cudaMalloc(&gpt->d_token_embedding, token_emb_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&gpt->d_token_embedding_grad, token_emb_size * sizeof(half)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&gpt->d_token_embedding_m, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpt->d_token_embedding_v, token_emb_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&gpt->d_embedded_input, embedded_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&gpt->d_norm_output, embedded_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&gpt->d_output, output_size * sizeof(half)));
    
    // Alias/Allocate device memory for backward pass buffers
    gpt->d_grad_output = gpt->d_output;
    CHECK_CUDA(cudaMalloc(&gpt->d_grad_norm_output, embedded_size * sizeof(half)));
    
    // Allocate single device float for loss computation
    CHECK_CUDA(cudaMalloc(&gpt->d_loss_result, sizeof(float)));
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(gpt->d_token_embedding, h_token_embedding, token_emb_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(gpt->d_token_embedding_m, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(gpt->d_token_embedding_v, 0, token_emb_size * sizeof(float)));
    
    // Initialize transformer
    gpt->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, true, true, cublaslt_handle);
    
    // Create cuBLASLt matrix multiplication descriptor
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&gpt->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    
    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Create matrix layout descriptors
    // Embedding matrix: [vocab_size x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&gpt->embedding_layout, CUDA_R_16F, gpt->vocab_size, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(gpt->embedding_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Flattened sequence data (d_model): [batch_size * seq_len x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&gpt->seq_flat_d_model_layout, CUDA_R_16F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(gpt->seq_flat_d_model_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Flattened sequence data (vocab): [batch_size * seq_len x vocab_size]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&gpt->seq_flat_vocab_layout, CUDA_R_16F, batch_size * seq_len, gpt->vocab_size, gpt->vocab_size));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(gpt->seq_flat_vocab_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Free host memory
    free(h_token_embedding);
    
    return gpt;
}

// Free GPT memory
void free_gpt(GPT* gpt) {
    // Free transformer
    free_transformer(gpt->transformer);
    
    // Destroy cuBLASLt descriptor
    cublasLtMatmulDescDestroy(gpt->matmul_desc);
    
    // Destroy matrix layouts
    cublasLtMatrixLayoutDestroy(gpt->embedding_layout);
    cublasLtMatrixLayoutDestroy(gpt->seq_flat_d_model_layout);
    cublasLtMatrixLayoutDestroy(gpt->seq_flat_vocab_layout);
    
    // Free device memory
    cudaFree(gpt->d_token_embedding); cudaFree(gpt->d_token_embedding_grad);
    cudaFree(gpt->d_token_embedding_m); cudaFree(gpt->d_token_embedding_v);
    cudaFree(gpt->d_embedded_input);
    cudaFree(gpt->d_norm_output);
    cudaFree(gpt->d_output);
    cudaFree(gpt->d_grad_norm_output);
    
    // Free loss computation buffer
    cudaFree(gpt->d_loss_result);
    
    free(gpt);
}

// CUDA kernel for token embedding lookup
__global__ static void token_embedding_lookup_kernel(half* embedded, half* token_embedding, unsigned short* tokens, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    
    if (b >= batch_size || t >= seq_len) return;
    
    int token = tokens[b * seq_len + t];
    int emb_base = (b * seq_len + t) * d_model;
    int token_base = token * d_model;
    
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        embedded[emb_base + d] = token_embedding[token_base + d];
    }
}

// CUDA kernel for RMSNorm: y = x / sqrt(mean(x^2) + eps)
__global__ static void rmsnorm_forward_kernel(half* output, const half* input, int batch_size, int seq_len, int d_model) {
    extern __shared__ float smem[];
    
    int idx = blockIdx.x;
    if (idx >= batch_size * seq_len) return;
    
    const half* x = &input[idx * d_model];
    half* y = &output[idx * d_model];
    
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp_id = tid >> 5, lane_id = tid & 31, nwarps = nthreads >> 5;
    
    // Parallel sum of squares
    float sum_sq = 0.0f;
    for (int d = tid; d < d_model; d += nthreads) {
        float val = __half2float(x[d]);
        sum_sq += val * val;
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    
    if (warp_id == 0) {
        sum_sq = (lane_id < nwarps) ? smem[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        if (lane_id == 0) smem[0] = sum_sq;
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(smem[0] / d_model + 1e-6f);
    
    // Parallel normalization
    for (int d = tid; d < d_model; d += nthreads) {
        y[d] = __float2half(__half2float(x[d]) * inv_rms);
    }
}

// CUDA kernel for RMSNorm backward pass: ∂L/∂x = (∂L/∂y)/(x / sqrt(mean(x^2) + eps)) - x⊙(Σ_d (∂L/∂y_d)⊙x_d)/(d_model⊙(x / sqrt(mean(x^2) + eps))³)
__global__ static void rmsnorm_backward_kernel(half* grad_input, const half* grad_output, const half* input, int batch_size, int seq_len, int d_model) {
    extern __shared__ float smem[];
    
    int idx = blockIdx.x;
    if (idx >= batch_size * seq_len) return;
    
    const half* x = &input[idx * d_model];
    const half* dy = &grad_output[idx * d_model];
    half* dx = &grad_input[idx * d_model];
    
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp_id = tid >> 5, lane_id = tid & 31, nwarps = nthreads >> 5;
    
    // Parallel computation of sum_sq and sum_dy_x
    float sum_sq = 0.0f, sum_dy_x = 0.0f;
    for (int d = tid; d < d_model; d += nthreads) {
        float x_val = __half2float(x[d]);
        float dy_val = __half2float(dy[d]);
        sum_sq += x_val * x_val;
        sum_dy_x += dy_val * x_val;
    }
    
    // Reduce sum_sq
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < nwarps) ? smem[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        if (lane_id == 0) smem[0] = sum_sq;
    }
    __syncthreads();
    float rms_sq = smem[0] / d_model + 1e-6f;
    float inv_rms = rsqrtf(rms_sq);
    
    // Reduce sum_dy_x
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum_dy_x += __shfl_xor_sync(0xffffffff, sum_dy_x, offset);
    if (lane_id == 0) smem[8 + warp_id] = sum_dy_x;
    __syncthreads();
    if (warp_id == 0) {
        sum_dy_x = (lane_id < nwarps) ? smem[8 + lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sum_dy_x += __shfl_xor_sync(0xffffffff, sum_dy_x, offset);
        if (lane_id == 0) smem[8] = sum_dy_x;
    }
    __syncthreads();
    sum_dy_x = smem[8];
    
    // dx = dy/rms - x * sum_dy_x / (d_model * rms^3)
    //    = dy * inv_rms - x * sum_dy_x * inv_rms^3 / d_model
    float scale = sum_dy_x * inv_rms * inv_rms * inv_rms / d_model;
    
    for (int d = tid; d < d_model; d += nthreads) {
        float x_val = __half2float(x[d]);
        float dy_val = __half2float(dy[d]);
        dx[d] = __float2half(dy_val * inv_rms - x_val * scale);
    }
}

// CUDA kernel for token embedding gradient accumulation
__global__ static void token_embedding_grad_kernel(half* token_embedding_grad, half* grad_embedded, unsigned short* tokens, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    
    if (b >= batch_size || t >= seq_len) return;
    
    int token = tokens[b * seq_len + t];
    int emb_base = (b * seq_len + t) * d_model;
    int token_base = token * d_model;
    
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        float grad = __half2float(grad_embedded[emb_base + d]);
        atomicAdd(&token_embedding_grad[token_base + d], __float2half(grad));
    }
}

// CUDA kernel for softmax and cross-entropy loss computation
__global__ static void softmax_cross_entropy_kernel(float* loss_result, half* grad_logits, half* logits, unsigned short* targets, unsigned char* loss_mask, int batch_size, int seq_len, int vocab_size) {
    extern __shared__ float smem[];
    
    int b = blockIdx.x;
    int t = blockIdx.y;
    
    if (b >= batch_size || t >= seq_len) return;
    
    unsigned char mask = (loss_mask != NULL) ? loss_mask[b * seq_len + t] : 1;
    
    size_t logits_offset = (size_t)b * seq_len * vocab_size + t * vocab_size;
    half* logits_bt = &logits[logits_offset];
    half* grad_logits_bt = &grad_logits[logits_offset];
    
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp_id = tid >> 5, lane_id = tid & 31, nwarps = nthreads >> 5;
    
    // Find max for numerical stability
    float val = -1e30f;
    for (int v = tid; v < vocab_size; v += nthreads) {
        val = fmaxf(val, __half2float(logits_bt[v]));
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane_id < nwarps) ? smem[lane_id] : -1e30f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();
    float max_val = smem[0];
    
    // Compute exp and sum
    val = 0.0f;
    for (int v = tid; v < vocab_size; v += nthreads) {
        float exp_val = expf(__half2float(logits_bt[v]) - max_val);
        grad_logits_bt[v] = __float2half(exp_val);
        val += exp_val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_xor_sync(0xffffffff, val, offset);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane_id < nwarps) ? smem[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_xor_sync(0xffffffff, val, offset);
        if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[0];
    
    // Normalize and set gradient
    int target_token = targets[b * seq_len + t];
    for (int v = tid; v < vocab_size; v += nthreads) {
        float prob = __half2float(grad_logits_bt[v]) * inv_sum;
        if (v == target_token) {
            if (mask) atomicAdd(loss_result, -logf(prob + 1e-10f));
            grad_logits_bt[v] = __float2half((prob - 1.0f) * mask);
        } else {
            grad_logits_bt[v] = __float2half(prob * mask);
        }
    }
}

// Forward pass
void forward_pass_gpt(GPT* gpt, unsigned short* d_input_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 1: Token embedding lookup
    dim3 grid_emb(gpt->batch_size, gpt->seq_len);
    token_embedding_lookup_kernel<<<grid_emb, 256>>>(
        gpt->d_embedded_input, gpt->d_token_embedding, d_input_tokens,
        gpt->batch_size, gpt->seq_len, gpt->d_model
    );
    
    // Step 2: Forward pass through transformer
    forward_pass_transformer(gpt->transformer, gpt->d_embedded_input);
    
    // Step 3: RMSNorm on transformer output
    rmsnorm_forward_kernel<<<gpt->batch_size * gpt->seq_len, 256, 8 * sizeof(float)>>>(
        gpt->d_norm_output,
        gpt->transformer->mlp_layers[gpt->num_layers-1]->d_output,
        gpt->batch_size,
        gpt->seq_len,
        gpt->d_model
    );
    
    // Step 4: Output projection: output = norm_output * token_embedding^T
    LT_MATMUL(gpt, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              gpt->d_norm_output, gpt->seq_flat_d_model_layout,
              gpt->d_token_embedding, gpt->embedding_layout,
              &beta, gpt->d_output, gpt->seq_flat_vocab_layout);
}

// Calculate loss
float calculate_loss_gpt(GPT* gpt, unsigned short* d_target_tokens, unsigned char* d_loss_mask, int num_loss_tokens) {
    // Reset loss accumulator
    CHECK_CUDA(cudaMemset(gpt->d_loss_result, 0, sizeof(float)));
    
    // Compute softmax and cross-entropy loss
    dim3 grid(gpt->batch_size, gpt->seq_len);
    softmax_cross_entropy_kernel<<<grid, 256, 8 * sizeof(float)>>>(
        gpt->d_loss_result, gpt->d_grad_output, gpt->d_output, d_target_tokens, d_loss_mask,
        gpt->batch_size, gpt->seq_len, gpt->vocab_size
    );
    
    // Copy result back to host
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, gpt->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / ((num_loss_tokens > 0) ? num_loss_tokens : (gpt->batch_size * gpt->seq_len));
}

// Zero gradients
void zero_gradients_gpt(GPT* gpt) {
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    
    CHECK_CUDA(cudaMemset(gpt->d_token_embedding_grad, 0, token_emb_size * sizeof(half)));
    
    zero_gradients_transformer(gpt->transformer);
}

// Backward pass
void backward_pass_gpt(GPT* gpt, unsigned short* d_input_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 4 (backward): Backward pass through output projection
    // grad_token_embedding += grad_output^T * norm_output
    LT_MATMUL(gpt, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              gpt->d_grad_output, gpt->seq_flat_vocab_layout,
              gpt->d_norm_output, gpt->seq_flat_d_model_layout,
              &alpha, gpt->d_token_embedding_grad, gpt->embedding_layout);
    
    // grad_norm_output = grad_output * token_embedding
    LT_MATMUL(gpt, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              gpt->d_grad_output, gpt->seq_flat_vocab_layout,
              gpt->d_token_embedding, gpt->embedding_layout,
              &beta, gpt->d_grad_norm_output, gpt->seq_flat_d_model_layout);
    
    // Step 3 (backward): Backward pass through RMSNorm
    rmsnorm_backward_kernel<<<gpt->batch_size * gpt->seq_len, 256, 16 * sizeof(float)>>>(
        gpt->transformer->mlp_layers[gpt->num_layers-1]->d_grad_output,
        gpt->d_grad_norm_output,
        gpt->transformer->mlp_layers[gpt->num_layers-1]->d_output,
        gpt->batch_size,
        gpt->seq_len,
        gpt->d_model
    );
    
    // Step 2 (backward): Backward pass through transformer
    backward_pass_transformer(gpt->transformer, gpt->d_embedded_input, gpt->d_embedded_input);
    
    // Step 1 (backward): Token embedding gradients
    dim3 grid_emb(gpt->batch_size, gpt->seq_len);
    token_embedding_grad_kernel<<<grid_emb, 256>>>(
        gpt->d_token_embedding_grad, gpt->d_embedded_input, d_input_tokens,
        gpt->batch_size, gpt->seq_len, gpt->d_model
    );
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_gpt(half* weight, half* grad, float* m, float* v,
                                               float beta1, float beta2, float epsilon, float learning_rate,
                                               float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(grad[idx]) / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        float w = __half2float(weight[idx]);
        weight[idx] = __float2half(w * (1.0f - learning_rate * weight_decay) - update);
    }
}

// Update weights
void update_weights_gpt(GPT* gpt, float learning_rate, int batch_size) {
    gpt->t++;
    
    float beta1_t = powf(gpt->beta1, gpt->t);
    float beta2_t = powf(gpt->beta2, gpt->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update token embeddings
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    int token_blocks = (token_emb_size + block_size - 1) / block_size;
    adamw_update_kernel_gpt<<<token_blocks, block_size>>>(
        gpt->d_token_embedding, gpt->d_token_embedding_grad, gpt->d_token_embedding_m, gpt->d_token_embedding_v,
        gpt->beta1, gpt->beta2, gpt->epsilon, learning_rate, gpt->weight_decay,
        alpha_t, token_emb_size, batch_size
    );
    
    // Update transformer weights
    update_weights_transformer(gpt->transformer, learning_rate, batch_size);
}

// Reset optimizer state
void reset_optimizer_gpt(GPT* gpt) {
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    
    // Reset Adam moment estimates to zero on device
    CHECK_CUDA(cudaMemset(gpt->d_token_embedding_m, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(gpt->d_token_embedding_v, 0, token_emb_size * sizeof(float)));
    
    // Reset time step
    gpt->t = 0;
    
    // Reset transformer optimizer state
    reset_optimizer_transformer(gpt->transformer);
}

// Serialize GPT to a file
static void serialize_gpt(GPT* gpt, FILE* file) {
    // Write dimensions
    fwrite(&gpt->d_model, sizeof(int), 1, file);
    fwrite(&gpt->hidden_dim, sizeof(int), 1, file);
    fwrite(&gpt->num_layers, sizeof(int), 1, file);
    fwrite(&gpt->vocab_size, sizeof(int), 1, file);
    
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    
    // Allocate host memory for embeddings
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    
    // Copy embeddings from device
    CHECK_CUDA(cudaMemcpy(h_token_embedding, gpt->d_token_embedding, token_emb_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Convert half to float
    for (int i = token_emb_size - 1; i >= 0; i--) h_token_embedding[i] = __half2float(((half*)h_token_embedding)[i]);
    
    // Write embeddings
    fwrite(h_token_embedding, sizeof(float), token_emb_size, file);
    
    free(h_token_embedding);
    
    // Write optimizer state
    fwrite(&gpt->t, sizeof(int), 1, file);
    
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_token_embedding_m, gpt->d_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_token_embedding_v, gpt->d_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fwrite(h_token_embedding_v, sizeof(float), token_emb_size, file);
    
    free(h_token_embedding_m); free(h_token_embedding_v);
    
    // Serialize transformer
    serialize_transformer(gpt->transformer, file);
}

// Deserialize GPT from a file
static GPT* deserialize_gpt(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle) {
    // Read dimensions
    int d_model, hidden_dim, num_layers, vocab_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&vocab_size, sizeof(int), 1, file);
    
    // Initialize GPT
    GPT* gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    
    int token_emb_size = vocab_size * d_model;
    
    // Allocate host memory for embeddings
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    
    // Read embeddings
    fread(h_token_embedding, sizeof(float), token_emb_size, file);
    
    // Convert float to half
    for (int i = 0; i < token_emb_size; i++) ((half*)h_token_embedding)[i] = __float2half(h_token_embedding[i]);
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(gpt->d_token_embedding, h_token_embedding, token_emb_size * sizeof(half), cudaMemcpyHostToDevice));
    
    free(h_token_embedding);
    
    // Read optimizer state
    fread(&gpt->t, sizeof(int), 1, file);
    
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    
    fread(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fread(h_token_embedding_v, sizeof(float), token_emb_size, file);
    
    CHECK_CUDA(cudaMemcpy(gpt->d_token_embedding_m, h_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpt->d_token_embedding_v, h_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_token_embedding_m); free(h_token_embedding_v);
    
    // Free the initialized transformer
    free_transformer(gpt->transformer);
    
    // Deserialize transformer
    gpt->transformer = deserialize_transformer(file, batch_size, seq_len, cublaslt_handle);
    
    return gpt;
}

// Save GPT to a file
void save_gpt(GPT* gpt, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    serialize_gpt(gpt, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load GPT from a file
GPT* load_gpt(const char* filename, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    GPT* gpt = deserialize_gpt(file, batch_size, seq_len, cublaslt_handle);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return gpt;
}
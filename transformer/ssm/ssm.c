#include "ssm.h"

// cuBLASLt matrix multiplication macro
#define LT_MATMUL(ssm, opA, opB, alpha, A, layA, B, layB, beta, C, layC) do { \
    cublasOperation_t _opA = opA, _opB = opB; \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ssm->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSA, &_opA, sizeof(_opA))); \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ssm->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSB, &_opB, sizeof(_opB))); \
    CHECK_CUBLASLT(cublasLtMatmul(ssm->cublaslt_handle, ssm->matmul_desc, \
                                  alpha, A, layA, B, layB, \
                                  beta, C, layC, \
                                  C, layC, NULL, NULL, 0, 0)); \
} while(0)

// Threads-per-block for the scan kernels (one warp; state_dim must be <= 32)
#define SSM_WARP 32

// Initialize the SSM
SSM* init_ssm(int seq_len, int d_model, int state_dim, int batch_size, cublasLtHandle_t cublaslt_handle) {
    if (state_dim <= 0 || state_dim > SSM_WARP) {
        fprintf(stderr, "state_dim (%d) must be in [1, %d]\n", state_dim, SSM_WARP);
        exit(EXIT_FAILURE);
    }

    SSM* ssm = (SSM*)malloc(sizeof(SSM));

    // Store dimensions
    ssm->seq_len = seq_len;
    ssm->d_model = d_model;
    ssm->batch_size = batch_size;
    ssm->state_dim = state_dim;

    // Initialize Adam parameters
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->t = 0;
    ssm->weight_decay = 0.01f;

    // Initialize cuBLASLt
    ssm->cublaslt_handle = cublaslt_handle;

    size_t weight_size = (size_t)d_model * d_model;
    size_t small_size = (size_t)d_model * state_dim;
    size_t seq_batch = (size_t)batch_size * seq_len * d_model;
    size_t states_size = (size_t)batch_size * seq_len * d_model * state_dim;

    // Allocate host memory for weight initialization
    half* h_W_in  = (half*)malloc(weight_size * sizeof(half));
    half* h_W_out = (half*)malloc(weight_size * sizeof(half));
    half* h_A_raw = (half*)malloc(small_size  * sizeof(half));
    half* h_B     = (half*)malloc(small_size  * sizeof(half));
    half* h_C     = (half*)malloc(small_size  * sizeof(half));
    half* h_D     = (half*)malloc(d_model     * sizeof(half));

    // Initialize weights on host
    float w_scale = 1.0f / sqrtf((float)d_model);
    for (size_t i = 0; i < weight_size; i++) {
        h_W_in[i]  = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale);
        h_W_out[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale);
    }
    // a_d = σ(â). Init â ≈ 3.0 gives a ≈ 0.95 → long memory by default.
    float bc_scale = 1.0f / sqrtf((float)state_dim);
    for (size_t i = 0; i < small_size; i++) {
        h_A_raw[i] = __float2half(3.0f + ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.1f);
        h_B[i]     = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * bc_scale);
        h_C[i]     = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * bc_scale);
    }
    // D = 1: SSM block starts as an identity-like passthrough through W_out∘W_in.
    for (int i = 0; i < d_model; i++) h_D[i] = __float2half(1.0f);

    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_W_in,       weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_out,      weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_raw,      small_size  * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B,          small_size  * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C,          small_size  * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D,          d_model     * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_in_grad,  weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_out_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_raw_grad, small_size  * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_grad,     small_size  * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad,     small_size  * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad,     d_model     * sizeof(half)));

    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&ssm->d_W_in_m,  weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_in_v,  weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_out_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_W_out_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_raw_m, small_size  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_raw_v, small_size  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_m,     small_size  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_v,     small_size  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_m,     small_size  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_v,     small_size  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_m,     d_model     * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_v,     d_model     * sizeof(float)));

    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&ssm->d_U,      seq_batch   * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_Z,      seq_batch   * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_output, seq_batch   * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_states, states_size * sizeof(half)));

    // Alias device memory for backward pass buffers
    ssm->d_grad_output = ssm->d_output;
    ssm->d_grad_Z      = ssm->d_Z;
    ssm->d_grad_U      = ssm->d_U;

    // Allocate single device float for loss computation
    CHECK_CUDA(cudaMalloc(&ssm->d_loss_result, sizeof(float)));

    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(ssm->d_W_in,  h_W_in,  weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_out, h_W_out, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_raw, h_A_raw, small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B,     h_B,     small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C,     h_C,     small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D,     h_D,     d_model     * sizeof(half), cudaMemcpyHostToDevice));

    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(ssm->d_W_in_m,  0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_in_v,  0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_out_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_out_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_raw_m, 0, small_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_raw_v, 0, small_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m,     0, small_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v,     0, small_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m,     0, small_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v,     0, small_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m,     0, d_model     * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v,     0, d_model     * sizeof(float)));

    // Create cuBLASLt matrix multiplication descriptor
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&ssm->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));

    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    // Create matrix layout descriptors
    // W_in, W_out and their gradients: [d_model x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ssm->weight_layout, CUDA_R_16F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(ssm->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // U, Z, output and their gradients: [batch_size * seq_len x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ssm->seq_flat_layout, CUDA_R_16F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(ssm->seq_flat_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // Free host memory
    free(h_W_in); free(h_W_out); free(h_A_raw); free(h_B); free(h_C); free(h_D);

    return ssm;
}

// Free SSM memory
void free_ssm(SSM* ssm) {
    // Destroy cuBLASLt descriptor
    cublasLtMatmulDescDestroy(ssm->matmul_desc);

    // Destroy matrix layouts
    cublasLtMatrixLayoutDestroy(ssm->weight_layout);
    cublasLtMatrixLayoutDestroy(ssm->seq_flat_layout);

    // Free device memory
    cudaFree(ssm->d_W_in); cudaFree(ssm->d_W_out);
    cudaFree(ssm->d_A_raw); cudaFree(ssm->d_B); cudaFree(ssm->d_C); cudaFree(ssm->d_D);
    cudaFree(ssm->d_W_in_grad); cudaFree(ssm->d_W_out_grad);
    cudaFree(ssm->d_A_raw_grad); cudaFree(ssm->d_B_grad); cudaFree(ssm->d_C_grad); cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_W_in_m); cudaFree(ssm->d_W_in_v);
    cudaFree(ssm->d_W_out_m); cudaFree(ssm->d_W_out_v);
    cudaFree(ssm->d_A_raw_m); cudaFree(ssm->d_A_raw_v);
    cudaFree(ssm->d_B_m); cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m); cudaFree(ssm->d_C_v);
    cudaFree(ssm->d_D_m); cudaFree(ssm->d_D_v);
    cudaFree(ssm->d_U); cudaFree(ssm->d_Z);
    cudaFree(ssm->d_output); cudaFree(ssm->d_states);

    // Free loss computation buffer
    cudaFree(ssm->d_loss_result);

    free(ssm);
}

// CUDA kernel for the per-channel forward scan
__global__ static void ssm_forward_kernel(half* Z, half* H, half* U,
                                           half* A_raw, half* B_w, half* C_w, half* D_w,
                                           int batch_size, int seq_len, int d_model, int state_dim) {
    int b = blockIdx.y;
    int d = blockIdx.x;
    int n = threadIdx.x;
    if (b >= batch_size || d >= d_model) return;

    bool active = (n < state_dim);
    float a = 0.0f, bn = 0.0f, cn = 0.0f, state = 0.0f;
    if (active) {
        a  = 1.0f / (1.0f + expf(-__half2float(A_raw[d * state_dim + n])));
        bn = __half2float(B_w[d * state_dim + n]);
        cn = __half2float(C_w[d * state_dim + n]);
    }
    float dskip = __half2float(D_w[d]);

    // s_n[t] = σ(â)s_n[t-1] + b_n·u[t] ; Z[t] = Σ_n c_n·s_n[t] + D·u[t]
    for (int t = 0; t < seq_len; t++) {
        float u = __half2float(U[((size_t)b * seq_len + t) * d_model + d]);
        if (active) {
            state = a * state + bn * u;
            H[(((size_t)b * seq_len + t) * d_model + d) * state_dim + n] = __float2half(state);
        }
        float partial = active ? cn * state : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (n == 0) Z[((size_t)b * seq_len + t) * d_model + d] = __float2half(partial + dskip * u);
    }
}

// Forward pass
void forward_pass_ssm(SSM* ssm, half* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // U = XW_in
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, ssm->seq_flat_layout,
              ssm->d_W_in, ssm->weight_layout,
              &beta, ssm->d_U, ssm->seq_flat_layout);

    // Per-channel linear recurrence → Z
    dim3 grid(ssm->d_model, ssm->batch_size);
    ssm_forward_kernel<<<grid, SSM_WARP>>>(
        ssm->d_Z, ssm->d_states, ssm->d_U,
        ssm->d_A_raw, ssm->d_B, ssm->d_C, ssm->d_D,
        ssm->batch_size, ssm->seq_len, ssm->d_model, ssm->state_dim);

    // Y = ZW_out
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              ssm->d_Z, ssm->seq_flat_layout,
              ssm->d_W_out, ssm->weight_layout,
              &beta, ssm->d_output, ssm->seq_flat_layout);
}

// CUDA kernel for computing loss and gradient
__global__ static void compute_loss_and_gradient_kernel_ssm(half* grad_output, half* predictions, half* targets, float* loss_result, int size) {
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
float calculate_loss_ssm(SSM* ssm, half* d_y) {
    // ∂L/∂Y = Y - Y_true
    int total_elements = ssm->batch_size * ssm->seq_len * ssm->d_model;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    // Reset loss accumulator to zero
    CHECK_CUDA(cudaMemset(ssm->d_loss_result, 0, sizeof(float)));

    // Compute gradient and accumulate loss
    compute_loss_and_gradient_kernel_ssm<<<num_blocks, block_size>>>(
        ssm->d_grad_output, ssm->d_output, d_y, ssm->d_loss_result, total_elements
    );

    // Copy result back to host
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, ssm->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));

    return total_loss / total_elements;
}

// Zero gradients
void zero_gradients_ssm(SSM* ssm) {
    int weight_size = ssm->d_model * ssm->d_model;
    int small_size = ssm->d_model * ssm->state_dim;

    CHECK_CUDA(cudaMemset(ssm->d_W_in_grad,  0, weight_size  * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_W_out_grad, 0, weight_size  * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_A_raw_grad, 0, small_size   * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad,     0, small_size   * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad,     0, small_size   * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad,     0, ssm->d_model * sizeof(half)));
}

// CUDA kernel for the per-channel backward scan
__global__ static void ssm_backward_kernel(half* dU, half* dA_raw, half* dB, half* dC, half* dD,
                                            half* U, half* dZ, half* H,
                                            half* A_raw, half* B_w, half* C_w, half* D_w,
                                            int batch_size, int seq_len, int d_model, int state_dim) {
    int b = blockIdx.y;
    int d = blockIdx.x;
    int n = threadIdx.x;
    if (b >= batch_size || d >= d_model) return;

    bool active = (n < state_dim);
    float a_raw_v = active ? __half2float(A_raw[d * state_dim + n]) : 0.0f;
    float a  = active ? 1.0f / (1.0f + expf(-a_raw_v))      : 0.0f;
    float bn = active ? __half2float(B_w[d * state_dim + n]) : 0.0f;
    float cn = active ? __half2float(C_w[d * state_dim + n]) : 0.0f;
    float dskip = __half2float(D_w[d]);

    float ds = 0.0f;
    float da_acc = 0.0f, db_acc = 0.0f, dc_acc = 0.0f, dD_acc = 0.0f;

    // ds_n[t] = c_n·dy + a_n·ds_n[t+1] ; du[t] = Σ_n b_n·ds_n[t] + D·dy
    for (int t = seq_len - 1; t >= 0; t--) {
        size_t flat = ((size_t)b * seq_len + t) * d_model + d;
        float dy = __half2float(dZ[flat]);
        float u  = __half2float(U[flat]);
        float s_t   = active ? __half2float(H[flat * state_dim + n]) : 0.0f;
        float s_tm1 = (active && t > 0) ? __half2float(H[(flat - d_model) * state_dim + n]) : 0.0f;

        ds = cn * dy + a * ds;

        dc_acc += dy * s_t;
        db_acc += ds * u;
        da_acc += ds * s_tm1;
        dD_acc += dy * u;

        float partial = active ? bn * ds : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (n == 0) dU[flat] = __float2half(partial + dskip * dy);
    }

    // dâ_n = da_n·σ(â)·(1-σ(â)) ; accumulate across batches via atomicAdd
    if (active) {
        float da_raw = da_acc * a * (1.0f - a);
        atomicAdd(&dA_raw[d * state_dim + n], __float2half(da_raw));
        atomicAdd(&dB[d * state_dim + n],     __float2half(db_acc));
        atomicAdd(&dC[d * state_dim + n],     __float2half(dc_acc));
    }
    if (n == 0) atomicAdd(&dD[d], __float2half(dD_acc));
}

// Backward pass
void backward_pass_ssm(SSM* ssm, half* d_X, half* d_grad_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // ∂L/∂W_out = Zᵀ(∂L/∂Y)
    LT_MATMUL(ssm, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              ssm->d_Z, ssm->seq_flat_layout,
              ssm->d_grad_output, ssm->seq_flat_layout,
              &alpha, ssm->d_W_out_grad, ssm->weight_layout);

    // ∂L/∂Z = (∂L/∂Y)W_outᵀ
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              ssm->d_grad_output, ssm->seq_flat_layout,
              ssm->d_W_out, ssm->weight_layout,
              &beta, ssm->d_grad_Z, ssm->seq_flat_layout);

    // Per-channel scan backward → dU and (dA_raw, dB, dC, dD)
    dim3 grid(ssm->d_model, ssm->batch_size);
    ssm_backward_kernel<<<grid, SSM_WARP>>>(
        ssm->d_grad_U, ssm->d_A_raw_grad, ssm->d_B_grad, ssm->d_C_grad, ssm->d_D_grad,
        ssm->d_U, ssm->d_grad_Z, ssm->d_states,
        ssm->d_A_raw, ssm->d_B, ssm->d_C, ssm->d_D,
        ssm->batch_size, ssm->seq_len, ssm->d_model, ssm->state_dim);

    // ∂L/∂W_in = Xᵀ(∂L/∂U)
    LT_MATMUL(ssm, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, ssm->seq_flat_layout,
              ssm->d_grad_U, ssm->seq_flat_layout,
              &alpha, ssm->d_W_in_grad, ssm->weight_layout);

    if (d_grad_X != NULL) {
        // ∂L/∂X = (∂L/∂U)W_inᵀ
        LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  ssm->d_grad_U, ssm->seq_flat_layout,
                  ssm->d_W_in, ssm->weight_layout,
                  &beta, d_grad_X, ssm->seq_flat_layout);
    }
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_ssm(half* weight, half* grad, float* m, float* v,
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

// Update weights using AdamW
void update_weights_ssm(SSM* ssm, float learning_rate, int batch_size) {
    ssm->t++;

    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);

    int block_size = 256;

    int weight_size = ssm->d_model * ssm->d_model;
    int small_size = ssm->d_model * ssm->state_dim;
    int d_size = ssm->d_model;

    // Update W_in weights
    int W_in_blocks = (weight_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<W_in_blocks, block_size>>>(
        ssm->d_W_in, ssm->d_W_in_grad, ssm->d_W_in_m, ssm->d_W_in_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, weight_size, batch_size
    );

    // Update W_out weights
    int W_out_blocks = (weight_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<W_out_blocks, block_size>>>(
        ssm->d_W_out, ssm->d_W_out_grad, ssm->d_W_out_m, ssm->d_W_out_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, weight_size, batch_size
    );

    // Update A_raw weights
    int A_blocks = (small_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<A_blocks, block_size>>>(
        ssm->d_A_raw, ssm->d_A_raw_grad, ssm->d_A_raw_m, ssm->d_A_raw_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, small_size, batch_size
    );

    // Update B weights
    int B_blocks = (small_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<B_blocks, block_size>>>(
        ssm->d_B, ssm->d_B_grad, ssm->d_B_m, ssm->d_B_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, small_size, batch_size
    );

    // Update C weights
    int C_blocks = (small_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<C_blocks, block_size>>>(
        ssm->d_C, ssm->d_C_grad, ssm->d_C_m, ssm->d_C_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, small_size, batch_size
    );

    // Update D weights
    int D_blocks = (d_size + block_size - 1) / block_size;
    adamw_update_kernel_ssm<<<D_blocks, block_size>>>(
        ssm->d_D, ssm->d_D_grad, ssm->d_D_m, ssm->d_D_v,
        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay,
        alpha_t, d_size, batch_size
    );
}

// Reset optimizer state
void reset_optimizer_ssm(SSM* ssm) {
    int weight_size = ssm->d_model * ssm->d_model;
    int small_size = ssm->d_model * ssm->state_dim;

    // Reset Adam moment estimates to zero on device
    CHECK_CUDA(cudaMemset(ssm->d_W_in_m,  0, weight_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_in_v,  0, weight_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_out_m, 0, weight_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_out_v, 0, weight_size  * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_raw_m, 0, small_size   * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_raw_v, 0, small_size   * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m,     0, small_size   * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v,     0, small_size   * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m,     0, small_size   * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v,     0, small_size   * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m,     0, ssm->d_model * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v,     0, ssm->d_model * sizeof(float)));

    // Reset time step
    ssm->t = 0;
}

// Serialize SSM to a file
void serialize_ssm(SSM* ssm, FILE* file) {
    // Write dimensions
    fwrite(&ssm->d_model, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);

    int weight_size = ssm->d_model * ssm->d_model;
    int small_size = ssm->d_model * ssm->state_dim;
    int d_size = ssm->d_model;

    // Allocate host buffers for weights
    float* h_W_in  = (float*)malloc(weight_size * sizeof(float));
    float* h_W_out = (float*)malloc(weight_size * sizeof(float));
    float* h_A_raw = (float*)malloc(small_size  * sizeof(float));
    float* h_B     = (float*)malloc(small_size  * sizeof(float));
    float* h_C     = (float*)malloc(small_size  * sizeof(float));
    float* h_D     = (float*)malloc(d_size      * sizeof(float));

    // Copy weights from device
    CHECK_CUDA(cudaMemcpy(h_W_in,  ssm->d_W_in,  weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_out, ssm->d_W_out, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_A_raw, ssm->d_A_raw, small_size  * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B,     ssm->d_B,     small_size  * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C,     ssm->d_C,     small_size  * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D,     ssm->d_D,     d_size      * sizeof(half), cudaMemcpyDeviceToHost));

    // Convert half to float
    for (int i = weight_size - 1; i >= 0; i--) h_W_in[i]  = __half2float(((half*)h_W_in)[i]);
    for (int i = weight_size - 1; i >= 0; i--) h_W_out[i] = __half2float(((half*)h_W_out)[i]);
    for (int i = small_size  - 1; i >= 0; i--) h_A_raw[i] = __half2float(((half*)h_A_raw)[i]);
    for (int i = small_size  - 1; i >= 0; i--) h_B[i]     = __half2float(((half*)h_B)[i]);
    for (int i = small_size  - 1; i >= 0; i--) h_C[i]     = __half2float(((half*)h_C)[i]);
    for (int i = d_size      - 1; i >= 0; i--) h_D[i]     = __half2float(((half*)h_D)[i]);

    // Write weights
    fwrite(h_W_in,  sizeof(float), weight_size, file);
    fwrite(h_W_out, sizeof(float), weight_size, file);
    fwrite(h_A_raw, sizeof(float), small_size,  file);
    fwrite(h_B,     sizeof(float), small_size,  file);
    fwrite(h_C,     sizeof(float), small_size,  file);
    fwrite(h_D,     sizeof(float), d_size,      file);

    free(h_W_in); free(h_W_out); free(h_A_raw); free(h_B); free(h_C); free(h_D);

    // Write optimizer state
    fwrite(&ssm->t, sizeof(int), 1, file);

    // Allocate host buffers for optimizer state
    float* h_W_in_m  = (float*)malloc(weight_size * sizeof(float));
    float* h_W_in_v  = (float*)malloc(weight_size * sizeof(float));
    float* h_W_out_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_out_v = (float*)malloc(weight_size * sizeof(float));
    float* h_A_raw_m = (float*)malloc(small_size  * sizeof(float));
    float* h_A_raw_v = (float*)malloc(small_size  * sizeof(float));
    float* h_B_m     = (float*)malloc(small_size  * sizeof(float));
    float* h_B_v     = (float*)malloc(small_size  * sizeof(float));
    float* h_C_m     = (float*)malloc(small_size  * sizeof(float));
    float* h_C_v     = (float*)malloc(small_size  * sizeof(float));
    float* h_D_m     = (float*)malloc(d_size      * sizeof(float));
    float* h_D_v     = (float*)malloc(d_size      * sizeof(float));

    // Copy optimizer state from device
    CHECK_CUDA(cudaMemcpy(h_W_in_m,  ssm->d_W_in_m,  weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_in_v,  ssm->d_W_in_v,  weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_out_m, ssm->d_W_out_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_out_v, ssm->d_W_out_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_A_raw_m, ssm->d_A_raw_m, small_size  * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_A_raw_v, ssm->d_A_raw_v, small_size  * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B_m,     ssm->d_B_m,     small_size  * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B_v,     ssm->d_B_v,     small_size  * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_m,     ssm->d_C_m,     small_size  * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_v,     ssm->d_C_v,     small_size  * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D_m,     ssm->d_D_m,     d_size      * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D_v,     ssm->d_D_v,     d_size      * sizeof(float), cudaMemcpyDeviceToHost));

    // Write optimizer state
    fwrite(h_W_in_m,  sizeof(float), weight_size, file);
    fwrite(h_W_in_v,  sizeof(float), weight_size, file);
    fwrite(h_W_out_m, sizeof(float), weight_size, file);
    fwrite(h_W_out_v, sizeof(float), weight_size, file);
    fwrite(h_A_raw_m, sizeof(float), small_size,  file);
    fwrite(h_A_raw_v, sizeof(float), small_size,  file);
    fwrite(h_B_m,     sizeof(float), small_size,  file);
    fwrite(h_B_v,     sizeof(float), small_size,  file);
    fwrite(h_C_m,     sizeof(float), small_size,  file);
    fwrite(h_C_v,     sizeof(float), small_size,  file);
    fwrite(h_D_m,     sizeof(float), d_size,      file);
    fwrite(h_D_v,     sizeof(float), d_size,      file);

    // Free host buffers
    free(h_W_in_m);  free(h_W_in_v);
    free(h_W_out_m); free(h_W_out_v);
    free(h_A_raw_m); free(h_A_raw_v);
    free(h_B_m);     free(h_B_v);
    free(h_C_m);     free(h_C_v);
    free(h_D_m);     free(h_D_v);
}

// Deserialize SSM from a file
SSM* deserialize_ssm(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle) {
    // Read dimensions
    int d_model, state_dim;
    fread(&d_model, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);

    // Initialize SSM
    SSM* ssm = init_ssm(seq_len, d_model, state_dim, batch_size, cublaslt_handle);

    int weight_size = d_model * d_model;
    int small_size = d_model * state_dim;
    int d_size = d_model;

    // Allocate host buffers for weights
    float* h_W_in  = (float*)malloc(weight_size * sizeof(float));
    float* h_W_out = (float*)malloc(weight_size * sizeof(float));
    float* h_A_raw = (float*)malloc(small_size  * sizeof(float));
    float* h_B     = (float*)malloc(small_size  * sizeof(float));
    float* h_C     = (float*)malloc(small_size  * sizeof(float));
    float* h_D     = (float*)malloc(d_size      * sizeof(float));

    // Read weights
    fread(h_W_in,  sizeof(float), weight_size, file);
    fread(h_W_out, sizeof(float), weight_size, file);
    fread(h_A_raw, sizeof(float), small_size,  file);
    fread(h_B,     sizeof(float), small_size,  file);
    fread(h_C,     sizeof(float), small_size,  file);
    fread(h_D,     sizeof(float), d_size,      file);

    // Convert float to half
    for (int i = 0; i < weight_size; i++) ((half*)h_W_in)[i]  = __float2half(h_W_in[i]);
    for (int i = 0; i < weight_size; i++) ((half*)h_W_out)[i] = __float2half(h_W_out[i]);
    for (int i = 0; i < small_size;  i++) ((half*)h_A_raw)[i] = __float2half(h_A_raw[i]);
    for (int i = 0; i < small_size;  i++) ((half*)h_B)[i]     = __float2half(h_B[i]);
    for (int i = 0; i < small_size;  i++) ((half*)h_C)[i]     = __float2half(h_C[i]);
    for (int i = 0; i < d_size;      i++) ((half*)h_D)[i]     = __float2half(h_D[i]);

    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(ssm->d_W_in,  h_W_in,  weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_out, h_W_out, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_raw, h_A_raw, small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B,     h_B,     small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C,     h_C,     small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D,     h_D,     d_size      * sizeof(half), cudaMemcpyHostToDevice));

    free(h_W_in); free(h_W_out); free(h_A_raw); free(h_B); free(h_C); free(h_D);

    // Read optimizer state
    fread(&ssm->t, sizeof(int), 1, file);

    // Allocate host buffers for optimizer state
    float* h_W_in_m  = (float*)malloc(weight_size * sizeof(float));
    float* h_W_in_v  = (float*)malloc(weight_size * sizeof(float));
    float* h_W_out_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_out_v = (float*)malloc(weight_size * sizeof(float));
    float* h_A_raw_m = (float*)malloc(small_size  * sizeof(float));
    float* h_A_raw_v = (float*)malloc(small_size  * sizeof(float));
    float* h_B_m     = (float*)malloc(small_size  * sizeof(float));
    float* h_B_v     = (float*)malloc(small_size  * sizeof(float));
    float* h_C_m     = (float*)malloc(small_size  * sizeof(float));
    float* h_C_v     = (float*)malloc(small_size  * sizeof(float));
    float* h_D_m     = (float*)malloc(d_size      * sizeof(float));
    float* h_D_v     = (float*)malloc(d_size      * sizeof(float));

    // Read optimizer state
    fread(h_W_in_m,  sizeof(float), weight_size, file);
    fread(h_W_in_v,  sizeof(float), weight_size, file);
    fread(h_W_out_m, sizeof(float), weight_size, file);
    fread(h_W_out_v, sizeof(float), weight_size, file);
    fread(h_A_raw_m, sizeof(float), small_size,  file);
    fread(h_A_raw_v, sizeof(float), small_size,  file);
    fread(h_B_m,     sizeof(float), small_size,  file);
    fread(h_B_v,     sizeof(float), small_size,  file);
    fread(h_C_m,     sizeof(float), small_size,  file);
    fread(h_C_v,     sizeof(float), small_size,  file);
    fread(h_D_m,     sizeof(float), d_size,      file);
    fread(h_D_v,     sizeof(float), d_size,      file);

    // Copy optimizer state to device
    CHECK_CUDA(cudaMemcpy(ssm->d_W_in_m,  h_W_in_m,  weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_in_v,  h_W_in_v,  weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_out_m, h_W_out_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_out_v, h_W_out_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_raw_m, h_A_raw_m, small_size  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_raw_v, h_A_raw_v, small_size  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_m,     h_B_m,     small_size  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_v,     h_B_v,     small_size  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_m,     h_C_m,     small_size  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_v,     h_C_v,     small_size  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_m,     h_D_m,     d_size      * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_v,     h_D_v,     d_size      * sizeof(float), cudaMemcpyHostToDevice));

    // Free host buffers
    free(h_W_in_m);  free(h_W_in_v);
    free(h_W_out_m); free(h_W_out_v);
    free(h_A_raw_m); free(h_A_raw_v);
    free(h_B_m);     free(h_B_v);
    free(h_C_m);     free(h_C_v);
    free(h_D_m);     free(h_D_v);

    return ssm;
}
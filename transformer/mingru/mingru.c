#include "mingru.h"

// cuBLASLt matrix multiplication macro
#define LT_MATMUL(m, opA, opB, alpha, A, layA, B, layB, beta, C, layC) do { \
    cublasOperation_t _opA = opA, _opB = opB; \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(m->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSA, &_opA, sizeof(_opA))); \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(m->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSB, &_opB, sizeof(_opB))); \
    CHECK_CUBLASLT(cublasLtMatmul(m->cublaslt_handle, m->matmul_desc, \
                                  alpha, A, layA, B, layB, \
                                  beta, C, layC, \
                                  C, layC, NULL, NULL, 0, 0)); \
} while(0)

// One warp per CUDA block; each thread owns one channel d.
#define MINGRU_WARP 32

MinGRU* init_mingru(int seq_len, int d_model, int batch_size, cublasLtHandle_t cublaslt_handle) {
    MinGRU* m = (MinGRU*)malloc(sizeof(MinGRU));

    m->seq_len    = seq_len;
    m->d_model    = d_model;
    m->batch_size = batch_size;

    m->beta1        = 0.9f;
    m->beta2        = 0.999f;
    m->epsilon      = 1e-8f;
    m->t            = 0;
    m->weight_decay = 0.01f;

    m->cublaslt_handle = cublaslt_handle;

    size_t weight_size = (size_t)d_model * d_model;
    size_t seq_batch   = (size_t)batch_size * seq_len * d_model;

    // Host-side weight init
    half* h_W_z = (half*)malloc(weight_size * sizeof(half));
    half* h_W_h = (half*)malloc(weight_size * sizeof(half));

    float w_scale = 1.0f / sqrtf((float)d_model);
    for (size_t i = 0; i < weight_size; i++) {
        h_W_z[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale);
        h_W_h[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale);
    }

    // Device weights and grads
    CHECK_CUDA(cudaMalloc(&m->d_W_z,      weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&m->d_W_h,      weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&m->d_W_z_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&m->d_W_h_grad, weight_size * sizeof(half)));

    // Adam moments
    CHECK_CUDA(cudaMalloc(&m->d_W_z_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&m->d_W_z_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&m->d_W_h_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&m->d_W_h_v, weight_size * sizeof(float)));

    // Forward / backward activation buffers
    CHECK_CUDA(cudaMalloc(&m->d_K,           seq_batch * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&m->d_V,           seq_batch * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&m->d_output,      seq_batch * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&m->d_grad_output, seq_batch * sizeof(half)));

    // Alias gradient buffers onto K, V (overwritten after scan backward consumes them)
    m->d_grad_K = m->d_K;
    m->d_grad_V = m->d_V;

    CHECK_CUDA(cudaMalloc(&m->d_loss_result, sizeof(float)));

    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(m->d_W_z, h_W_z, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m->d_W_h, h_W_h, weight_size * sizeof(half), cudaMemcpyHostToDevice));

    // Zero Adam moments
    CHECK_CUDA(cudaMemset(m->d_W_z_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(m->d_W_z_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(m->d_W_h_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(m->d_W_h_v, 0, weight_size * sizeof(float)));

    // cuBLASLt descriptors
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&m->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&m->weight_layout, CUDA_R_16F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(m->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&m->seq_flat_layout, CUDA_R_16F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(m->seq_flat_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    free(h_W_z); free(h_W_h);
    return m;
}

void free_mingru(MinGRU* m) {
    cublasLtMatmulDescDestroy(m->matmul_desc);
    cublasLtMatrixLayoutDestroy(m->weight_layout);
    cublasLtMatrixLayoutDestroy(m->seq_flat_layout);

    cudaFree(m->d_W_z); cudaFree(m->d_W_h);
    cudaFree(m->d_W_z_grad); cudaFree(m->d_W_h_grad);
    cudaFree(m->d_W_z_m); cudaFree(m->d_W_z_v);
    cudaFree(m->d_W_h_m); cudaFree(m->d_W_h_v);
    cudaFree(m->d_K); cudaFree(m->d_V);
    cudaFree(m->d_output); cudaFree(m->d_grad_output);
    cudaFree(m->d_loss_result);
    free(m);
}

// Candidate-state activation: tanh, bounded in [-1, 1].
// Derivative: 1 - tanh(v)^2.
__device__ static inline float mingru_g(float v) {
    return tanhf(v);
}

// Per-channel forward scan: h_t = (1-z_t) h_{t-1} + z_t h̃_t.
// grid = (ceil(d_model/WARP), batch_size); block = WARP threads, one channel per thread.
// State is carried in an fp32 register.
__global__ static void mingru_forward_kernel(half* __restrict__ H,
                                              const half* __restrict__ K,
                                              const half* __restrict__ V,
                                              int batch_size, int seq_len, int d_model) {
    int b = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || d >= d_model) return;

    float state = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        size_t flat = ((size_t)b * seq_len + t) * d_model + d;
        float k = __half2float(K[flat]);
        float v = __half2float(V[flat]);
        float z = 1.0f / (1.0f + expf(-k));
        float h_tilde = mingru_g(v);
        state = (1.0f - z) * state + z * h_tilde;
        H[flat] = __float2half(state);
    }
}

// Reverse scan: given dH, recompute z, h̃ from K, V and produce dK, dV.
//   dh_t (total) = dH_upstream[t] + (1-z_{t+1}) * dh_{t+1}
//   dz_t  = (h̃_t - h_{t-1}) * dh_t           → dK = dz * z * (1-z)
//   dh̃_t  = z_t * dh_t                        → dV = dh̃ * g'(v)
__global__ static void mingru_backward_kernel(half* __restrict__ dK,
                                               half* __restrict__ dV,
                                               const half* __restrict__ K,
                                               const half* __restrict__ V,
                                               const half* __restrict__ H,
                                               const half* __restrict__ dH,
                                               int batch_size, int seq_len, int d_model) {
    int b = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || d >= d_model) return;

    float ds = 0.0f;
    for (int t = seq_len - 1; t >= 0; t--) {
        size_t flat = ((size_t)b * seq_len + t) * d_model + d;
        float k = __half2float(K[flat]);
        float v = __half2float(V[flat]);
        float z = 1.0f / (1.0f + expf(-k));
        float h_tilde = mingru_g(v);
        float h_tm1 = (t > 0) ? __half2float(H[flat - d_model]) : 0.0f;

        float dh = __half2float(dH[flat]) + ds;
        float dz       = (h_tilde - h_tm1) * dh;
        float dh_tilde = z * dh;

        float dk = dz * z * (1.0f - z);
        // g'(v) = 1 - tanh(v)^2; h_tilde == tanh(v).
        float dv = dh_tilde * (1.0f - h_tilde * h_tilde);

        dK[flat] = __float2half(dk);
        dV[flat] = __float2half(dv);

        ds = (1.0f - z) * dh;
    }
}

void forward_pass_mingru(MinGRU* m, half* d_X) {
    const float alpha = 1.0f, beta = 0.0f;

    // K = X @ W_z, V = X @ W_h
    LT_MATMUL(m, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, m->seq_flat_layout,
              m->d_W_z, m->weight_layout,
              &beta, m->d_K, m->seq_flat_layout);
    LT_MATMUL(m, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, m->seq_flat_layout,
              m->d_W_h, m->weight_layout,
              &beta, m->d_V, m->seq_flat_layout);

    // H = parallel_scan(K, V)
    dim3 grid((m->d_model + MINGRU_WARP - 1) / MINGRU_WARP, m->batch_size);
    mingru_forward_kernel<<<grid, MINGRU_WARP>>>(
        m->d_output, m->d_K, m->d_V,
        m->batch_size, m->seq_len, m->d_model);
}

__global__ static void compute_loss_and_gradient_kernel_mingru(half* grad_output, half* predictions, half* targets, float* loss_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred   = __half2float(predictions[idx]);
        float target = __half2float(targets[idx]);
        float diff = pred - target;
        grad_output[idx] = __float2half(diff);
        atomicAdd(loss_result, diff * diff);
    }
}

float calculate_loss_mingru(MinGRU* m, half* d_y) {
    int total_elements = m->batch_size * m->seq_len * m->d_model;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    CHECK_CUDA(cudaMemset(m->d_loss_result, 0, sizeof(float)));

    compute_loss_and_gradient_kernel_mingru<<<num_blocks, block_size>>>(
        m->d_grad_output, m->d_output, d_y, m->d_loss_result, total_elements
    );

    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, m->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    return total_loss / total_elements;
}

void zero_gradients_mingru(MinGRU* m) {
    int weight_size = m->d_model * m->d_model;
    CHECK_CUDA(cudaMemset(m->d_W_z_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(m->d_W_h_grad, 0, weight_size * sizeof(half)));
}

void backward_pass_mingru(MinGRU* m, half* d_X, half* d_grad_X) {
    const float alpha = 1.0f, beta = 0.0f;

    // dH = d_grad_output (output IS H — no W_out projection in minimal minGRU).
    // Reverse scan → dK (overwrites K), dV (overwrites V).
    dim3 grid((m->d_model + MINGRU_WARP - 1) / MINGRU_WARP, m->batch_size);
    mingru_backward_kernel<<<grid, MINGRU_WARP>>>(
        m->d_grad_K, m->d_grad_V,
        m->d_K, m->d_V, m->d_output, m->d_grad_output,
        m->batch_size, m->seq_len, m->d_model);

    // dW_z += Xᵀ @ dK, dW_h += Xᵀ @ dV
    LT_MATMUL(m, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, m->seq_flat_layout,
              m->d_grad_K, m->seq_flat_layout,
              &alpha, m->d_W_z_grad, m->weight_layout);
    LT_MATMUL(m, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, m->seq_flat_layout,
              m->d_grad_V, m->seq_flat_layout,
              &alpha, m->d_W_h_grad, m->weight_layout);

    if (d_grad_X != NULL) {
        // dX = dK @ W_zᵀ + dV @ W_hᵀ
        LT_MATMUL(m, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  m->d_grad_K, m->seq_flat_layout,
                  m->d_W_z, m->weight_layout,
                  &beta, d_grad_X, m->seq_flat_layout);
        LT_MATMUL(m, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  m->d_grad_V, m->seq_flat_layout,
                  m->d_W_h, m->weight_layout,
                  &alpha, d_grad_X, m->seq_flat_layout);
    }
}

__global__ static void adamw_update_kernel_mingru(half* weight, half* grad, float* mom, float* vel,
                                                   float beta1, float beta2, float epsilon, float learning_rate,
                                                   float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(grad[idx]) / batch_size;
        mom[idx] = beta1 * mom[idx] + (1.0f - beta1) * g;
        vel[idx] = beta2 * vel[idx] + (1.0f - beta2) * g * g;
        float update = alpha_t * mom[idx] / (sqrtf(vel[idx]) + epsilon);
        float w = __half2float(weight[idx]);
        weight[idx] = __float2half(w * (1.0f - learning_rate * weight_decay) - update);
    }
}

void update_weights_mingru(MinGRU* m, float learning_rate, int batch_size) {
    m->t++;
    float beta1_t = powf(m->beta1, m->t);
    float beta2_t = powf(m->beta2, m->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);

    int block_size = 256;
    int weight_size = m->d_model * m->d_model;
    int blocks = (weight_size + block_size - 1) / block_size;

    adamw_update_kernel_mingru<<<blocks, block_size>>>(
        m->d_W_z, m->d_W_z_grad, m->d_W_z_m, m->d_W_z_v,
        m->beta1, m->beta2, m->epsilon, learning_rate, m->weight_decay,
        alpha_t, weight_size, batch_size);

    adamw_update_kernel_mingru<<<blocks, block_size>>>(
        m->d_W_h, m->d_W_h_grad, m->d_W_h_m, m->d_W_h_v,
        m->beta1, m->beta2, m->epsilon, learning_rate, m->weight_decay,
        alpha_t, weight_size, batch_size);
}

void reset_optimizer_mingru(MinGRU* m) {
    int weight_size = m->d_model * m->d_model;
    CHECK_CUDA(cudaMemset(m->d_W_z_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(m->d_W_z_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(m->d_W_h_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(m->d_W_h_v, 0, weight_size * sizeof(float)));
    m->t = 0;
}

void serialize_mingru(MinGRU* m, FILE* file) {
    fwrite(&m->d_model, sizeof(int), 1, file);

    int weight_size = m->d_model * m->d_model;

    float* h_W_z = (float*)malloc(weight_size * sizeof(float));
    float* h_W_h = (float*)malloc(weight_size * sizeof(float));

    CHECK_CUDA(cudaMemcpy(h_W_z, m->d_W_z, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_h, m->d_W_h, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = weight_size - 1; i >= 0; i--) h_W_z[i] = __half2float(((half*)h_W_z)[i]);
    for (int i = weight_size - 1; i >= 0; i--) h_W_h[i] = __half2float(((half*)h_W_h)[i]);

    fwrite(h_W_z, sizeof(float), weight_size, file);
    fwrite(h_W_h, sizeof(float), weight_size, file);
    free(h_W_z); free(h_W_h);

    fwrite(&m->t, sizeof(int), 1, file);

    float* h_W_z_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_z_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_h_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_h_v = (float*)malloc(weight_size * sizeof(float));

    CHECK_CUDA(cudaMemcpy(h_W_z_m, m->d_W_z_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_z_v, m->d_W_z_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_h_m, m->d_W_h_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_h_v, m->d_W_h_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));

    fwrite(h_W_z_m, sizeof(float), weight_size, file);
    fwrite(h_W_z_v, sizeof(float), weight_size, file);
    fwrite(h_W_h_m, sizeof(float), weight_size, file);
    fwrite(h_W_h_v, sizeof(float), weight_size, file);

    free(h_W_z_m); free(h_W_z_v); free(h_W_h_m); free(h_W_h_v);
}

MinGRU* deserialize_mingru(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle) {
    int d_model;
    fread(&d_model, sizeof(int), 1, file);

    MinGRU* m = init_mingru(seq_len, d_model, batch_size, cublaslt_handle);

    int weight_size = d_model * d_model;

    float* h_W_z = (float*)malloc(weight_size * sizeof(float));
    float* h_W_h = (float*)malloc(weight_size * sizeof(float));

    fread(h_W_z, sizeof(float), weight_size, file);
    fread(h_W_h, sizeof(float), weight_size, file);

    for (int i = 0; i < weight_size; i++) ((half*)h_W_z)[i] = __float2half(h_W_z[i]);
    for (int i = 0; i < weight_size; i++) ((half*)h_W_h)[i] = __float2half(h_W_h[i]);

    CHECK_CUDA(cudaMemcpy(m->d_W_z, h_W_z, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m->d_W_h, h_W_h, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    free(h_W_z); free(h_W_h);

    fread(&m->t, sizeof(int), 1, file);

    float* h_W_z_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_z_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_h_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_h_v = (float*)malloc(weight_size * sizeof(float));

    fread(h_W_z_m, sizeof(float), weight_size, file);
    fread(h_W_z_v, sizeof(float), weight_size, file);
    fread(h_W_h_m, sizeof(float), weight_size, file);
    fread(h_W_h_v, sizeof(float), weight_size, file);

    CHECK_CUDA(cudaMemcpy(m->d_W_z_m, h_W_z_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m->d_W_z_v, h_W_z_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m->d_W_h_m, h_W_h_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m->d_W_h_v, h_W_h_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));

    free(h_W_z_m); free(h_W_z_v); free(h_W_h_m); free(h_W_h_v);
    return m;
}

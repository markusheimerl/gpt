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

// Threads-per-block for scan kernels (one warp; state_dim must be <= 32)
#define SSM_WARP 32

// ───────────────────────────────────────────────────────────────────────────
//  Initialisation / teardown
// ───────────────────────────────────────────────────────────────────────────

SSM* init_ssm(int seq_len, int d_model, int state_dim, int batch_size, cublasLtHandle_t cublaslt_handle) {
    if (state_dim <= 0 || state_dim > SSM_WARP) {
        fprintf(stderr, "state_dim (%d) must be in [1, %d]\n", state_dim, SSM_WARP);
        exit(EXIT_FAILURE);
    }

    SSM* ssm = (SSM*)malloc(sizeof(SSM));

    ssm->seq_len    = seq_len;
    ssm->d_model    = d_model;
    ssm->batch_size = batch_size;
    ssm->state_dim  = state_dim;

    ssm->beta1        = 0.9f;
    ssm->beta2        = 0.999f;
    ssm->epsilon      = 1e-8f;
    ssm->t            = 0;
    ssm->weight_decay = 0.01f;

    ssm->cublaslt_handle = cublaslt_handle;

    size_t weight_size   = (size_t)d_model * d_model;
    size_t seq_batch     = (size_t)batch_size * seq_len * d_model;
    size_t small_size    = (size_t)d_model * state_dim;
    size_t states_size   = (size_t)batch_size * seq_len * d_model * state_dim;

    // ── Host initialisation ───────────────────────────────────────────────
    half* h_W_in  = (half*)malloc(weight_size * sizeof(half));
    half* h_W_out = (half*)malloc(weight_size * sizeof(half));
    half* h_A_raw = (half*)malloc(small_size  * sizeof(half));
    half* h_B     = (half*)malloc(small_size  * sizeof(half));
    half* h_C     = (half*)malloc(small_size  * sizeof(half));
    half* h_D     = (half*)malloc(d_model     * sizeof(half));

    float w_scale = 1.0f / sqrtf((float)d_model);
    for (size_t i = 0; i < weight_size; i++) {
        h_W_in[i]  = __float2half(((float)rand() / RAND_MAX * 2.0f - 1.0f) * w_scale);
        h_W_out[i] = __float2half(((float)rand() / RAND_MAX * 2.0f - 1.0f) * w_scale);
    }
    // a_d = σ(â). Init â ≈ 3.0 gives a ≈ 0.95 → long memory by default.
    float bc_scale = 1.0f / sqrtf((float)state_dim);
    for (size_t i = 0; i < small_size; i++) {
        h_A_raw[i] = __float2half(3.0f + ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f);
        h_B[i]     = __float2half(((float)rand() / RAND_MAX * 2.0f - 1.0f) * bc_scale);
        h_C[i]     = __float2half(((float)rand() / RAND_MAX * 2.0f - 1.0f) * bc_scale);
    }
    // D = 1: SSM block starts as an identity-like passthrough through W_out∘W_in.
    for (int i = 0; i < d_model; i++) h_D[i] = __float2half(1.0f);

    // ── Device allocations ────────────────────────────────────────────────
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

    CHECK_CUDA(cudaMalloc(&ssm->d_U,       seq_batch   * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_Z,       seq_batch   * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_output,  seq_batch   * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_states,  states_size * sizeof(half)));

    ssm->d_grad_output = ssm->d_output;
    ssm->d_grad_Z      = ssm->d_Z;
    ssm->d_grad_U      = ssm->d_U;

    CHECK_CUDA(cudaMalloc(&ssm->d_loss_result, sizeof(float)));

    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(ssm->d_W_in,  h_W_in,  weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_W_out, h_W_out, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_raw, h_A_raw, small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B,     h_B,     small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C,     h_C,     small_size  * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D,     h_D,     d_model     * sizeof(half), cudaMemcpyHostToDevice));

    // Zero Adam state
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

    // cuBLASLt descriptors / layouts
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&ssm->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ssm->weight_layout, CUDA_R_16F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(ssm->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ssm->seq_flat_layout, CUDA_R_16F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(ssm->seq_flat_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    free(h_W_in); free(h_W_out); free(h_A_raw); free(h_B); free(h_C); free(h_D);

    return ssm;
}

void free_ssm(SSM* ssm) {
    cublasLtMatmulDescDestroy(ssm->matmul_desc);
    cublasLtMatrixLayoutDestroy(ssm->weight_layout);
    cublasLtMatrixLayoutDestroy(ssm->seq_flat_layout);

    cudaFree(ssm->d_W_in);  cudaFree(ssm->d_W_out);
    cudaFree(ssm->d_A_raw); cudaFree(ssm->d_B); cudaFree(ssm->d_C); cudaFree(ssm->d_D);

    cudaFree(ssm->d_W_in_grad);  cudaFree(ssm->d_W_out_grad);
    cudaFree(ssm->d_A_raw_grad); cudaFree(ssm->d_B_grad); cudaFree(ssm->d_C_grad); cudaFree(ssm->d_D_grad);

    cudaFree(ssm->d_W_in_m);  cudaFree(ssm->d_W_in_v);
    cudaFree(ssm->d_W_out_m); cudaFree(ssm->d_W_out_v);
    cudaFree(ssm->d_A_raw_m); cudaFree(ssm->d_A_raw_v);
    cudaFree(ssm->d_B_m);     cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m);     cudaFree(ssm->d_C_v);
    cudaFree(ssm->d_D_m);     cudaFree(ssm->d_D_v);

    cudaFree(ssm->d_U); cudaFree(ssm->d_Z); cudaFree(ssm->d_output); cudaFree(ssm->d_states);

    cudaFree(ssm->d_loss_result);

    free(ssm);
}

// ───────────────────────────────────────────────────────────────────────────
//  Forward scan
// ───────────────────────────────────────────────────────────────────────────
//
// One block per (batch b, channel d). One warp (32 threads) per block; thread
// n owns state component s_n (for n < state_dim).
//
//   s_n[t] = σ(â_{d,n}) · s_n[t-1] + B_{d,n} · U[b,t,d]
//   Z[b,t,d] = Σ_n C_{d,n} · s_n[t]  +  D_d · U[b,t,d]
//
// All states s_n[t] are saved into d_states for the backward pass.
__global__ static void ssm_forward_kernel(
    const half* __restrict__ U,      // [B, L, D]
    const half* __restrict__ A_raw,  // [D, N]
    const half* __restrict__ B_w,    // [D, N]
    const half* __restrict__ C_w,    // [D, N]
    const half* __restrict__ D_w,    // [D]
    half* __restrict__ Z,            // [B, L, D]
    half* __restrict__ H,            // [B, L, D, N]
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

    for (int t = 0; t < seq_len; t++) {
        float u = __half2float(U[((size_t)b * seq_len + t) * d_model + d]);
        if (active) {
            state = a * state + bn * u;
            H[(((size_t)b * seq_len + t) * d_model + d) * state_dim + n] = __float2half(state);
        }
        float partial = active ? cn * state : 0.0f;
        // warp reduce
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (n == 0) Z[((size_t)b * seq_len + t) * d_model + d] = __float2half(partial + dskip * u);
    }
}

void forward_pass_ssm(SSM* ssm, half* d_X) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Step 1: U = X W_in   ([B*L x D])
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, ssm->seq_flat_layout,
              ssm->d_W_in, ssm->weight_layout,
              &beta, ssm->d_U, ssm->seq_flat_layout);

    // Step 2: per-channel linear recurrence
    dim3 grid(ssm->d_model, ssm->batch_size);
    ssm_forward_kernel<<<grid, SSM_WARP>>>(
        ssm->d_U, ssm->d_A_raw, ssm->d_B, ssm->d_C, ssm->d_D,
        ssm->d_Z, ssm->d_states,
        ssm->batch_size, ssm->seq_len, ssm->d_model, ssm->state_dim);

    // Step 3: Y = Z W_out
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              ssm->d_Z, ssm->seq_flat_layout,
              ssm->d_W_out, ssm->weight_layout,
              &beta, ssm->d_output, ssm->seq_flat_layout);
}

// ───────────────────────────────────────────────────────────────────────────
//  Loss / zero-grad
// ───────────────────────────────────────────────────────────────────────────

__global__ static void compute_loss_and_gradient_kernel_ssm(
    half* grad_output, half* predictions, half* targets, float* loss_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred   = __half2float(predictions[idx]);
        float target = __half2float(targets[idx]);
        float diff   = pred - target;
        grad_output[idx] = __float2half(diff);
        atomicAdd(loss_result, diff * diff);
    }
}

float calculate_loss_ssm(SSM* ssm, half* d_y) {
    int total = ssm->batch_size * ssm->seq_len * ssm->d_model;
    int block = 256;
    int nblk  = (total + block - 1) / block;
    CHECK_CUDA(cudaMemset(ssm->d_loss_result, 0, sizeof(float)));
    compute_loss_and_gradient_kernel_ssm<<<nblk, block>>>(
        ssm->d_grad_output, ssm->d_output, d_y, ssm->d_loss_result, total);
    float loss;
    CHECK_CUDA(cudaMemcpy(&loss, ssm->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    return loss / total;
}

void zero_gradients_ssm(SSM* ssm) {
    size_t ws = (size_t)ssm->d_model * ssm->d_model;
    size_t ss = (size_t)ssm->d_model * ssm->state_dim;
    CHECK_CUDA(cudaMemset(ssm->d_W_in_grad,  0, ws            * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_W_out_grad, 0, ws            * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_A_raw_grad, 0, ss            * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad,     0, ss            * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad,     0, ss            * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad,     0, ssm->d_model  * sizeof(half)));
}

// ───────────────────────────────────────────────────────────────────────────
//  Backward scan
// ───────────────────────────────────────────────────────────────────────────
//
// Per channel d, per state dim n, the recurrence
//    s_n[t]  = a_n · s_n[t-1] + b_n · u[t]
//    y[t]   = Σ_n c_n · s_n[t] + D · u[t]
// has backward (going from t=L-1 down to 0):
//    ds_n[t] = c_n · dy[t] + a_n · ds_n[t+1]   (with ds_n[L] = 0)
//    du[t]   = Σ_n b_n · ds_n[t] + D · dy[t]
//    dc_n   += dy[t] · s_n[t]
//    db_n   += ds_n[t] · u[t]
//    da_n   += ds_n[t] · s_n[t-1]              (s_n[-1] = 0)
//    dD     += dy[t] · u[t]
// then dâ_n = da_n · σ(â) · (1-σ(â)).
__global__ static void ssm_backward_kernel(
    const half* __restrict__ U,        // [B, L, D]
    const half* __restrict__ dZ,       // [B, L, D]   (= dY-projected back through W_out)
    const half* __restrict__ H,        // [B, L, D, N]
    const half* __restrict__ A_raw,    // [D, N]
    const half* __restrict__ B_w,      // [D, N]
    const half* __restrict__ C_w,      // [D, N]
    const half* __restrict__ D_w,      // [D]
    half* __restrict__ dU,             // [B, L, D]   out
    half* __restrict__ dA_raw,         // [D, N]      out (atomic)
    half* __restrict__ dB,             // [D, N]      out (atomic)
    half* __restrict__ dC,             // [D, N]      out (atomic)
    half* __restrict__ dD,             // [D]         out (atomic)
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

    float ds = 0.0f;          // ds_n[t+1] carried backward in time
    float da_acc = 0.0f, db_acc = 0.0f, dc_acc = 0.0f, dD_acc = 0.0f;

    for (int t = seq_len - 1; t >= 0; t--) {
        size_t flat = ((size_t)b * seq_len + t) * d_model + d;
        float dy = __half2float(dZ[flat]);
        float u  = __half2float(U[flat]);
        float s_t        = active ? __half2float(H[flat * state_dim + n]) : 0.0f;
        float s_tm1      = (active && t > 0) ? __half2float(H[(flat - d_model) * state_dim + n]) : 0.0f;

        // ds_n[t] = c_n · dy + a_n · ds_n[t+1]
        ds = cn * dy + a * ds;

        // Parameter gradients
        dc_acc += dy * s_t;
        db_acc += ds * u;
        da_acc += ds * s_tm1;          // s_n[-1] = 0 handles t==0
        dD_acc += dy * u;

        // du[t] = Σ_n b_n · ds_n[t] + D · dy
        float partial = active ? bn * ds : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (n == 0) dU[flat] = __float2half(partial + dskip * dy);
    }

    // Accumulate parameter gradients across batches via atomicAdd on half.
    if (active) {
        float da_raw = da_acc * a * (1.0f - a);    // chain through σ
        atomicAdd(&dA_raw[d * state_dim + n], __float2half(da_raw));
        atomicAdd(&dB[d * state_dim + n],     __float2half(db_acc));
        atomicAdd(&dC[d * state_dim + n],     __float2half(dc_acc));
    }
    if (n == 0) atomicAdd(&dD[d], __float2half(dD_acc));
}

void backward_pass_ssm(SSM* ssm, half* d_X, half* d_grad_X) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Step 3 (back): W_out_grad += Zᵀ · dY  ;  dZ = dY · W_outᵀ
    LT_MATMUL(ssm, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              ssm->d_Z, ssm->seq_flat_layout,
              ssm->d_grad_output, ssm->seq_flat_layout,
              &alpha, ssm->d_W_out_grad, ssm->weight_layout);

    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              ssm->d_grad_output, ssm->seq_flat_layout,
              ssm->d_W_out, ssm->weight_layout,
              &beta, ssm->d_grad_Z, ssm->seq_flat_layout);

    // Step 2 (back): per-channel scan backward producing dU and (dA_raw, dB, dC, dD).
    dim3 grid(ssm->d_model, ssm->batch_size);
    ssm_backward_kernel<<<grid, SSM_WARP>>>(
        ssm->d_U, ssm->d_grad_Z, ssm->d_states,
        ssm->d_A_raw, ssm->d_B, ssm->d_C, ssm->d_D,
        ssm->d_grad_U,
        ssm->d_A_raw_grad, ssm->d_B_grad, ssm->d_C_grad, ssm->d_D_grad,
        ssm->batch_size, ssm->seq_len, ssm->d_model, ssm->state_dim);

    // Step 1 (back): W_in_grad += Xᵀ · dU  ;  dX = dU · W_inᵀ
    LT_MATMUL(ssm, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, ssm->seq_flat_layout,
              ssm->d_grad_U, ssm->seq_flat_layout,
              &alpha, ssm->d_W_in_grad, ssm->weight_layout);

    if (d_grad_X != NULL) {
        LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  ssm->d_grad_U, ssm->seq_flat_layout,
                  ssm->d_W_in, ssm->weight_layout,
                  &beta, d_grad_X, ssm->seq_flat_layout);
    }
}

// ───────────────────────────────────────────────────────────────────────────
//  AdamW
// ───────────────────────────────────────────────────────────────────────────

__global__ static void adamw_update_kernel_ssm(half* weight, half* grad, float* m, float* v,
                                               float beta1, float beta2, float epsilon, float learning_rate,
                                               float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(grad[idx]) / batch_size;
        m[idx]  = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx]  = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        float w = __half2float(weight[idx]);
        weight[idx] = __float2half(w * (1.0f - learning_rate * weight_decay) - update);
    }
}

static void adamw_update_tensor(half* w, half* g, float* m, float* v, int size,
                                float beta1, float beta2, float eps, float lr,
                                float wd, float alpha_t, int batch_size) {
    int block = 256;
    int nblk = (size + block - 1) / block;
    adamw_update_kernel_ssm<<<nblk, block>>>(w, g, m, v, beta1, beta2, eps, lr, wd, alpha_t, size, batch_size);
}

void update_weights_ssm(SSM* ssm, float learning_rate, int batch_size) {
    ssm->t++;
    float beta1_t = powf(ssm->beta1, ssm->t);
    float beta2_t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);

    int ws = ssm->d_model * ssm->d_model;
    int ss = ssm->d_model * ssm->state_dim;
    int ds = ssm->d_model;

    adamw_update_tensor(ssm->d_W_in,  ssm->d_W_in_grad,  ssm->d_W_in_m,  ssm->d_W_in_v,  ws,
                        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay, alpha_t, batch_size);
    adamw_update_tensor(ssm->d_W_out, ssm->d_W_out_grad, ssm->d_W_out_m, ssm->d_W_out_v, ws,
                        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay, alpha_t, batch_size);
    adamw_update_tensor(ssm->d_A_raw, ssm->d_A_raw_grad, ssm->d_A_raw_m, ssm->d_A_raw_v, ss,
                        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay, alpha_t, batch_size);
    adamw_update_tensor(ssm->d_B,     ssm->d_B_grad,     ssm->d_B_m,     ssm->d_B_v,     ss,
                        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay, alpha_t, batch_size);
    adamw_update_tensor(ssm->d_C,     ssm->d_C_grad,     ssm->d_C_m,     ssm->d_C_v,     ss,
                        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay, alpha_t, batch_size);
    adamw_update_tensor(ssm->d_D,     ssm->d_D_grad,     ssm->d_D_m,     ssm->d_D_v,     ds,
                        ssm->beta1, ssm->beta2, ssm->epsilon, learning_rate, ssm->weight_decay, alpha_t, batch_size);
}

void reset_optimizer_ssm(SSM* ssm) {
    size_t ws = (size_t)ssm->d_model * ssm->d_model;
    size_t ss = (size_t)ssm->d_model * ssm->state_dim;
    CHECK_CUDA(cudaMemset(ssm->d_W_in_m,  0, ws * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_in_v,  0, ws * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_out_m, 0, ws * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_W_out_v, 0, ws * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_raw_m, 0, ss * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_raw_v, 0, ss * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m,     0, ss * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v,     0, ss * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m,     0, ss * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v,     0, ss * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m,     0, ssm->d_model * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v,     0, ssm->d_model * sizeof(float)));
    ssm->t = 0;
}

// ───────────────────────────────────────────────────────────────────────────
//  Serialisation
// ───────────────────────────────────────────────────────────────────────────

static void write_half_buffer(FILE* f, const half* d_src, size_t n) {
    float* tmp = (float*)malloc(n * sizeof(float));
    half*  h   = (half*) malloc(n * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h, d_src, n * sizeof(half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) tmp[i] = __half2float(h[i]);
    fwrite(tmp, sizeof(float), n, f);
    free(tmp); free(h);
}

static void write_float_buffer(FILE* f, const float* d_src, size_t n) {
    float* tmp = (float*)malloc(n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(tmp, d_src, n * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(tmp, sizeof(float), n, f);
    free(tmp);
}

static void read_half_buffer(FILE* f, half* d_dst, size_t n) {
    float* tmp = (float*)malloc(n * sizeof(float));
    half*  h   = (half*) malloc(n * sizeof(half));
    fread(tmp, sizeof(float), n, f);
    for (size_t i = 0; i < n; i++) h[i] = __float2half(tmp[i]);
    CHECK_CUDA(cudaMemcpy(d_dst, h, n * sizeof(half), cudaMemcpyHostToDevice));
    free(tmp); free(h);
}

static void read_float_buffer(FILE* f, float* d_dst, size_t n) {
    float* tmp = (float*)malloc(n * sizeof(float));
    fread(tmp, sizeof(float), n, f);
    CHECK_CUDA(cudaMemcpy(d_dst, tmp, n * sizeof(float), cudaMemcpyHostToDevice));
    free(tmp);
}

void serialize_ssm(SSM* ssm, FILE* f) {
    fwrite(&ssm->d_model,   sizeof(int), 1, f);
    fwrite(&ssm->state_dim, sizeof(int), 1, f);

    size_t ws = (size_t)ssm->d_model * ssm->d_model;
    size_t ss = (size_t)ssm->d_model * ssm->state_dim;
    size_t ds = ssm->d_model;

    write_half_buffer(f, ssm->d_W_in,  ws);
    write_half_buffer(f, ssm->d_W_out, ws);
    write_half_buffer(f, ssm->d_A_raw, ss);
    write_half_buffer(f, ssm->d_B,     ss);
    write_half_buffer(f, ssm->d_C,     ss);
    write_half_buffer(f, ssm->d_D,     ds);

    fwrite(&ssm->t, sizeof(int), 1, f);
    write_float_buffer(f, ssm->d_W_in_m,  ws); write_float_buffer(f, ssm->d_W_in_v,  ws);
    write_float_buffer(f, ssm->d_W_out_m, ws); write_float_buffer(f, ssm->d_W_out_v, ws);
    write_float_buffer(f, ssm->d_A_raw_m, ss); write_float_buffer(f, ssm->d_A_raw_v, ss);
    write_float_buffer(f, ssm->d_B_m,     ss); write_float_buffer(f, ssm->d_B_v,     ss);
    write_float_buffer(f, ssm->d_C_m,     ss); write_float_buffer(f, ssm->d_C_v,     ss);
    write_float_buffer(f, ssm->d_D_m,     ds); write_float_buffer(f, ssm->d_D_v,     ds);
}

SSM* deserialize_ssm(FILE* f, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle) {
    int d_model, state_dim;
    fread(&d_model,   sizeof(int), 1, f);
    fread(&state_dim, sizeof(int), 1, f);

    SSM* ssm = init_ssm(seq_len, d_model, state_dim, batch_size, cublaslt_handle);

    size_t ws = (size_t)d_model * d_model;
    size_t ss = (size_t)d_model * state_dim;
    size_t ds = d_model;

    read_half_buffer(f, ssm->d_W_in,  ws);
    read_half_buffer(f, ssm->d_W_out, ws);
    read_half_buffer(f, ssm->d_A_raw, ss);
    read_half_buffer(f, ssm->d_B,     ss);
    read_half_buffer(f, ssm->d_C,     ss);
    read_half_buffer(f, ssm->d_D,     ds);

    fread(&ssm->t, sizeof(int), 1, f);
    read_float_buffer(f, ssm->d_W_in_m,  ws); read_float_buffer(f, ssm->d_W_in_v,  ws);
    read_float_buffer(f, ssm->d_W_out_m, ws); read_float_buffer(f, ssm->d_W_out_v, ws);
    read_float_buffer(f, ssm->d_A_raw_m, ss); read_float_buffer(f, ssm->d_A_raw_v, ss);
    read_float_buffer(f, ssm->d_B_m,     ss); read_float_buffer(f, ssm->d_B_v,     ss);
    read_float_buffer(f, ssm->d_C_m,     ss); read_float_buffer(f, ssm->d_C_v,     ss);
    read_float_buffer(f, ssm->d_D_m,     ds); read_float_buffer(f, ssm->d_D_v,     ds);

    return ssm;
}

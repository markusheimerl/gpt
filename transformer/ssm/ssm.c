#include "ssm.h"

// cuBLASLt row-major matmul wrapper: C = alpha · op(A) op(B) + beta · C
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

static cublasLtMatrixLayout_t make_layout(int rows, int cols, int ld) {
    cublasLtMatrixLayout_t lay;
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&lay, CUDA_R_16F, rows, cols, ld));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(lay, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    return lay;
}

SSM* init_ssm(int seq_len, int d_model, int state_dim, int batch_size, cublasLtHandle_t cublaslt_handle) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));

    ssm->seq_len    = seq_len;
    ssm->d_model    = d_model;
    ssm->state_dim  = state_dim;
    ssm->batch_size = batch_size;

    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->t = 0;
    ssm->weight_decay = 0.01f;

    ssm->cublaslt_handle = cublaslt_handle;

    size_t A_size = (size_t)state_dim * state_dim;
    size_t B_size = (size_t)state_dim * d_model;
    size_t C_size = (size_t)d_model   * state_dim;
    size_t D_size = (size_t)d_model   * d_model;
    size_t seq_flat = (size_t)batch_size * seq_len * d_model;
    size_t H_buf  = (size_t)seq_len * batch_size * state_dim;

    // Host-side weight init (reference scales: A small, B/C/D ~1.5/√fan-in)
    half* h_A = (half*)malloc(A_size * sizeof(half));
    half* h_B = (half*)malloc(B_size * sizeof(half));
    half* h_C = (half*)malloc(C_size * sizeof(half));
    half* h_D = (half*)malloc(D_size * sizeof(half));

    float sA = 0.5f / sqrtf((float)state_dim);
    float sB = 1.5f / sqrtf((float)d_model);
    float sC = 1.5f / sqrtf((float)state_dim);
    float sD = 1.5f / sqrtf((float)d_model);

    for (size_t i = 0; i < A_size; i++) h_A[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * sA);
    for (size_t i = 0; i < B_size; i++) h_B[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * sB);
    for (size_t i = 0; i < C_size; i++) h_C[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * sC);
    for (size_t i = 0; i < D_size; i++) h_D[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * sD);

    // Device alloc: weights & grads
    CHECK_CUDA(cudaMalloc(&ssm->d_A,      A_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B,      B_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C,      C_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D,      D_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_grad, A_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_grad, B_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad, C_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, D_size * sizeof(half)));

    // Adam moments (fp32)
    CHECK_CUDA(cudaMalloc(&ssm->d_A_m, A_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_v, A_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_m, B_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_v, B_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_m, C_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_v, C_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_m, D_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_v, D_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, A_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, A_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, B_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, B_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, C_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, C_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, D_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, D_size * sizeof(float)));

    // Forward / backward scratch
    CHECK_CUDA(cudaMalloc(&ssm->d_H,      H_buf    * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_output, seq_flat * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&ssm->d_grad_H, H_buf    * sizeof(half)));
    ssm->d_grad_output = ssm->d_output;  // upstream dY overwrites Y in-place (matches MLP/MinGRU convention)

    CHECK_CUDA(cudaMalloc(&ssm->d_loss_result, sizeof(float)));

    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, h_A, A_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, h_B, B_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, h_C, C_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, h_D, D_size * sizeof(half), cudaMemcpyHostToDevice));
    free(h_A); free(h_B); free(h_C); free(h_D);

    // Matmul descriptor: TF32 compute on fp16 storage (consistent with the rest of the pipeline)
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&ssm->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));

    // Layouts (row-major, fp16). Per-timestep slices use ld = seq_len*d_model so the same
    // descriptor addresses any X[:, t, :] sub-matrix via a base-pointer offset.
    ssm->L_A       = make_layout(state_dim,             state_dim, state_dim);
    ssm->L_B       = make_layout(state_dim,             d_model,   d_model);
    ssm->L_C       = make_layout(d_model,               state_dim, state_dim);
    ssm->L_D       = make_layout(d_model,               d_model,   d_model);
    ssm->L_H_slice = make_layout(batch_size,            state_dim, seq_len * state_dim);
    ssm->L_H_flat  = make_layout(batch_size * seq_len,  state_dim, state_dim);
    ssm->L_X_flat  = make_layout(batch_size * seq_len,  d_model,   d_model);

    return ssm;
}

void free_ssm(SSM* ssm) {
    cublasLtMatmulDescDestroy(ssm->matmul_desc);
    cublasLtMatrixLayoutDestroy(ssm->L_A);
    cublasLtMatrixLayoutDestroy(ssm->L_B);
    cublasLtMatrixLayoutDestroy(ssm->L_C);
    cublasLtMatrixLayoutDestroy(ssm->L_D);
    cublasLtMatrixLayoutDestroy(ssm->L_H_slice);
    cublasLtMatrixLayoutDestroy(ssm->L_H_flat);
    cublasLtMatrixLayoutDestroy(ssm->L_X_flat);

    cudaFree(ssm->d_A); cudaFree(ssm->d_B); cudaFree(ssm->d_C); cudaFree(ssm->d_D);
    cudaFree(ssm->d_A_grad); cudaFree(ssm->d_B_grad); cudaFree(ssm->d_C_grad); cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_A_m); cudaFree(ssm->d_A_v);
    cudaFree(ssm->d_B_m); cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m); cudaFree(ssm->d_C_v);
    cudaFree(ssm->d_D_m); cudaFree(ssm->d_D_v);

    cudaFree(ssm->d_H);
    cudaFree(ssm->d_output);
    cudaFree(ssm->d_grad_H);
    cudaFree(ssm->d_loss_result);

    free(ssm);
}

// Sub-matrix slice of a batch-major [B, T, N] tensor: row b of slice t lives at offset
// b*(T*N) + t*N, so the slice pointer is just `buf + t*N` and the matmul layout uses
// ld = T*N (see L_H_slice in init). X/Y/dY/dX are never sliced per-t -- their per-t work
// is fused into single (B*T, *) matmuls below.
static inline half* H_slice(half* H, int t, int N) { return H + (size_t)t * N; }

void forward_pass_ssm(SSM* ssm, half* d_X) {
    const float one = 1.0f, zero = 0.0f;
    int N = ssm->state_dim, T = ssm->seq_len;

    // (1) Input projection over all (B, T) tokens at once: H = X @ B^T
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_T, &one,
              d_X,      ssm->L_X_flat,
              ssm->d_B, ssm->L_B,
              &zero, ssm->d_H, ssm->L_H_flat);

    // (2) Sequential recurrence (the only step that crosses time): H_t += H_{t-1} @ A^T
    for (int t = 1; t < T; t++) {
        LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_T, &one,
                  H_slice(ssm->d_H, t - 1, N), ssm->L_H_slice,
                  ssm->d_A,                    ssm->L_A,
                  &one, H_slice(ssm->d_H, t, N), ssm->L_H_slice);
    }

    // (3) Output projection + skip: Y = H @ C^T + X @ D^T
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_T, &one,
              ssm->d_H, ssm->L_H_flat,
              ssm->d_C, ssm->L_C,
              &zero, ssm->d_output, ssm->L_X_flat);
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_T, &one,
              d_X,      ssm->L_X_flat,
              ssm->d_D, ssm->L_D,
              &one, ssm->d_output, ssm->L_X_flat);
}

// MSE: writes grad_output = pred - target, accumulates sum of squared diffs
__global__ static void compute_loss_and_gradient_kernel_ssm(half* grad_output, half* predictions, half* targets, float* loss_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred   = __half2float(predictions[idx]);
        float target = __half2float(targets[idx]);
        float diff = pred - target;
        grad_output[idx] = __float2half(diff);
        atomicAdd(loss_result, diff * diff);
    }
}

float calculate_loss_ssm(SSM* ssm, half* d_y) {
    int total = ssm->batch_size * ssm->seq_len * ssm->d_model;
    int block = 256, blocks = (total + block - 1) / block;
    CHECK_CUDA(cudaMemset(ssm->d_loss_result, 0, sizeof(float)));
    compute_loss_and_gradient_kernel_ssm<<<blocks, block>>>(
        ssm->d_grad_output, ssm->d_output, d_y, ssm->d_loss_result, total);
    float loss;
    CHECK_CUDA(cudaMemcpy(&loss, ssm->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    return loss / total;
}

void zero_gradients_ssm(SSM* ssm) {
    int M = ssm->d_model, N = ssm->state_dim;
    CHECK_CUDA(cudaMemset(ssm->d_A_grad, 0, (size_t)N * N * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad, 0, (size_t)N * M * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad, 0, (size_t)M * N * sizeof(half)));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, (size_t)M * M * sizeof(half)));
}

void backward_pass_ssm(SSM* ssm, half* d_X, half* d_grad_X) {
    const float one = 1.0f, zero = 0.0f;
    int N = ssm->state_dim, T = ssm->seq_len;

    // (1) Output-side weight grads (fully batched over B*T):
    //     dC += dY^T @ H ,  dD += dY^T @ X
    LT_MATMUL(ssm, CUBLAS_OP_T, CUBLAS_OP_N, &one,
              ssm->d_grad_output, ssm->L_X_flat,
              ssm->d_H,           ssm->L_H_flat,
              &one, ssm->d_C_grad, ssm->L_C);
    LT_MATMUL(ssm, CUBLAS_OP_T, CUBLAS_OP_N, &one,
              ssm->d_grad_output, ssm->L_X_flat,
              d_X,                ssm->L_X_flat,
              &one, ssm->d_D_grad, ssm->L_D);

    // (2) Seed dH from output path (batched): dH = dY @ C
    LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_N, &one,
              ssm->d_grad_output, ssm->L_X_flat,
              ssm->d_C,           ssm->L_C,
              &zero, ssm->d_grad_H, ssm->L_H_flat);

    // (3) Sequential BPTT: at step t (descending) propagate dH_t back through the recurrence.
    //     dA  += dH_t^T @ H_{t-1}
    //     dH_{t-1} += dH_t @ A     (adds BPTT term to the dY-seeded value already in dH_{t-1})
    for (int t = T - 1; t >= 1; t--) {
        half* dH_t    = H_slice(ssm->d_grad_H, t,     N);
        half* H_prev  = H_slice(ssm->d_H,      t - 1, N);
        half* dH_prev = H_slice(ssm->d_grad_H, t - 1, N);

        LT_MATMUL(ssm, CUBLAS_OP_T, CUBLAS_OP_N, &one,
                  dH_t,   ssm->L_H_slice,
                  H_prev, ssm->L_H_slice,
                  &one, ssm->d_A_grad, ssm->L_A);

        LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_N, &one,
                  dH_t,     ssm->L_H_slice,
                  ssm->d_A, ssm->L_A,
                  &one, dH_prev, ssm->L_H_slice);
    }

    // (4) Input-side weight grad (batched): dB += dH^T @ X
    LT_MATMUL(ssm, CUBLAS_OP_T, CUBLAS_OP_N, &one,
              ssm->d_grad_H, ssm->L_H_flat,
              d_X,           ssm->L_X_flat,
              &one, ssm->d_B_grad, ssm->L_B);

    // (5) Optional dX (batched): dX = dH @ B + dY @ D
    if (d_grad_X != NULL) {
        LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_N, &one,
                  ssm->d_grad_H, ssm->L_H_flat,
                  ssm->d_B,      ssm->L_B,
                  &zero, d_grad_X, ssm->L_X_flat);
        LT_MATMUL(ssm, CUBLAS_OP_N, CUBLAS_OP_N, &one,
                  ssm->d_grad_output, ssm->L_X_flat,
                  ssm->d_D,           ssm->L_D,
                  &one, d_grad_X, ssm->L_X_flat);
    }
}

// AdamW
__global__ static void adamw_update_kernel_ssm(half* weight, half* grad, float* m, float* v,
                                               float beta1, float beta2, float epsilon, float learning_rate,
                                               float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(grad[idx]) / batch_size;
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        float w = __half2float(weight[idx]);
        weight[idx] = __float2half(w * (1.0f - learning_rate * weight_decay) - update);
    }
}

static inline void run_adam(SSM* ssm, half* w, half* g, float* m, float* v,
                            int size, float lr, float alpha_t, int batch_size) {
    int block = 256, blocks = (size + block - 1) / block;
    adamw_update_kernel_ssm<<<blocks, block>>>(w, g, m, v,
        ssm->beta1, ssm->beta2, ssm->epsilon, lr, ssm->weight_decay,
        alpha_t, size, batch_size);
}

void update_weights_ssm(SSM* ssm, float learning_rate, int batch_size) {
    ssm->t++;
    float b1t = powf(ssm->beta1, ssm->t);
    float b2t = powf(ssm->beta2, ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - b2t) / (1.0f - b1t);

    int M = ssm->d_model, N = ssm->state_dim;
    run_adam(ssm, ssm->d_A, ssm->d_A_grad, ssm->d_A_m, ssm->d_A_v, N * N, learning_rate, alpha_t, batch_size);
    run_adam(ssm, ssm->d_B, ssm->d_B_grad, ssm->d_B_m, ssm->d_B_v, N * M, learning_rate, alpha_t, batch_size);
    run_adam(ssm, ssm->d_C, ssm->d_C_grad, ssm->d_C_m, ssm->d_C_v, M * N, learning_rate, alpha_t, batch_size);
    run_adam(ssm, ssm->d_D, ssm->d_D_grad, ssm->d_D_m, ssm->d_D_v, M * M, learning_rate, alpha_t, batch_size);
}

void reset_optimizer_ssm(SSM* ssm) {
    int M = ssm->d_model, N = ssm->state_dim;
    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, (size_t)N * N * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, (size_t)N * N * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, (size_t)N * M * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, (size_t)N * M * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, (size_t)M * N * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, (size_t)M * N * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, (size_t)M * M * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, (size_t)M * M * sizeof(float)));
    ssm->t = 0;
}

// Serialize: weights as fp32 on disk for portability, optimizer state as native fp32.
static void write_half_as_float(FILE* f, half* d_ptr, size_t n) {
    half*  h_half  = (half*)malloc(n * sizeof(half));
    float* h_float = (float*)malloc(n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_half, d_ptr, n * sizeof(half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) h_float[i] = __half2float(h_half[i]);
    fwrite(h_float, sizeof(float), n, f);
    free(h_half); free(h_float);
}

static void read_float_as_half(FILE* f, half* d_ptr, size_t n) {
    float* h_float = (float*)malloc(n * sizeof(float));
    half*  h_half  = (half*)malloc(n * sizeof(half));
    fread(h_float, sizeof(float), n, f);
    for (size_t i = 0; i < n; i++) h_half[i] = __float2half(h_float[i]);
    CHECK_CUDA(cudaMemcpy(d_ptr, h_half, n * sizeof(half), cudaMemcpyHostToDevice));
    free(h_float); free(h_half);
}

static void write_dev_float(FILE* f, float* d_ptr, size_t n) {
    float* h = (float*)malloc(n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h, sizeof(float), n, f);
    free(h);
}

static void read_dev_float(FILE* f, float* d_ptr, size_t n) {
    float* h = (float*)malloc(n * sizeof(float));
    fread(h, sizeof(float), n, f);
    CHECK_CUDA(cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

void serialize_ssm(SSM* ssm, FILE* file) {
    fwrite(&ssm->d_model,   sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);

    size_t A = (size_t)ssm->state_dim * ssm->state_dim;
    size_t Bs = (size_t)ssm->state_dim * ssm->d_model;
    size_t Cs = (size_t)ssm->d_model   * ssm->state_dim;
    size_t D = (size_t)ssm->d_model   * ssm->d_model;

    write_half_as_float(file, ssm->d_A, A);
    write_half_as_float(file, ssm->d_B, Bs);
    write_half_as_float(file, ssm->d_C, Cs);
    write_half_as_float(file, ssm->d_D, D);

    fwrite(&ssm->t, sizeof(int), 1, file);
    write_dev_float(file, ssm->d_A_m, A); write_dev_float(file, ssm->d_A_v, A);
    write_dev_float(file, ssm->d_B_m, Bs); write_dev_float(file, ssm->d_B_v, Bs);
    write_dev_float(file, ssm->d_C_m, Cs); write_dev_float(file, ssm->d_C_v, Cs);
    write_dev_float(file, ssm->d_D_m, D); write_dev_float(file, ssm->d_D_v, D);
}

SSM* deserialize_ssm(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle) {
    int d_model, state_dim;
    fread(&d_model,   sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);

    SSM* ssm = init_ssm(seq_len, d_model, state_dim, batch_size, cublaslt_handle);

    size_t A = (size_t)state_dim * state_dim;
    size_t Bs = (size_t)state_dim * d_model;
    size_t Cs = (size_t)d_model   * state_dim;
    size_t D = (size_t)d_model   * d_model;

    read_float_as_half(file, ssm->d_A, A);
    read_float_as_half(file, ssm->d_B, Bs);
    read_float_as_half(file, ssm->d_C, Cs);
    read_float_as_half(file, ssm->d_D, D);

    fread(&ssm->t, sizeof(int), 1, file);
    read_dev_float(file, ssm->d_A_m, A); read_dev_float(file, ssm->d_A_v, A);
    read_dev_float(file, ssm->d_B_m, Bs); read_dev_float(file, ssm->d_B_v, Bs);
    read_dev_float(file, ssm->d_C_m, Cs); read_dev_float(file, ssm->d_C_v, Cs);
    read_dev_float(file, ssm->d_D_m, D); read_dev_float(file, ssm->d_D_v, D);
    return ssm;
}

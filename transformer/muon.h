#ifndef MUON_H
#define MUON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA / cuBLASLt error checking. These macros may already be defined by the
// including translation unit; guard against redefinition.
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

// Muon optimizer (Keller Jordan, 2024):
//   MomentumOrthogonalized via Newton-schUlz iteratiON.
// For each 2-D weight matrix W with gradient G we keep an fp32 momentum
// buffer M.  The update each step is:
//   M  <- beta * M + G
//   X  <- newton_schulz5(M / ||M||_F)              // quintic iteration
//   W  <- W - lr * ( sqrt(max(R,C)/min(R,C)) * X + weight_decay * W )
// The Newton-Schulz polynomial X <- a*X + (b*A + c*A^2) @ X with
// A = X @ X^T orthogonalises X by driving its singular values to 1.

#define MUON_NS_STEPS 5
#define MUON_NS_A      3.4445f
#define MUON_NS_B     (-4.7750f)
#define MUON_NS_C      2.0315f
#define MUON_NORM_EPS  1e-7f

typedef struct {
    float* d_M;     // fp32 momentum buffer, shape (rows, cols)
    float* d_X;     // fp32 working buffer, shape (rows, cols)
    float* d_X2;    // fp32 ping-pong buffer, shape (rows, cols)
    float* d_A;     // fp32 scratch, shape (m, m) where m = min(rows, cols)
    float* d_A2;    // fp32 scratch, shape (m, m)
    float* d_norm;  // fp32 scalar (sum of squares for Frobenius norm)
    int rows;
    int cols;
    // fp32 row-major layouts for cuBLASLt
    cublasLtMatrixLayout_t X_layout;  // (rows, cols)
    cublasLtMatrixLayout_t A_layout;  // (m,    m)
} MuonState;

// ===== Kernels ==============================================================

__global__ static void muon_momentum_kernel(float* M, float* X, const half* G,
                                            float beta, float inv_batch, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(G[i]) * inv_batch;
        float m = beta * M[i] + g;
        M[i] = m;
        X[i] = m;
    }
}

__global__ static void muon_frob_sq_kernel(const float* X, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = (i < n) ? X[i] : 0.0f;
    sdata[tid] = v * v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ static void muon_normalize_kernel(float* X, const float* sum_sq,
                                             float eps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float inv = 1.0f / (sqrtf(*sum_sq) + eps);
        X[i] = X[i] * inv;
    }
}

__global__ static void muon_mix_AB_kernel(float* A, const float* A2,
                                          float b, float c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = b * A[i] + c * A2[i];
}

__global__ static void muon_copy_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ static void muon_apply_update_kernel(half* W, const float* X,
                                                float scale, float lr,
                                                float weight_decay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float w = __half2float(W[i]);
        w = w - lr * (scale * X[i] + weight_decay * w);
        W[i] = __float2half(w);
    }
}

// ===== State management =====================================================

static inline void muon_state_init(MuonState* s, int rows, int cols) {
    s->rows = rows;
    s->cols = cols;
    int n = rows * cols;
    int m = (rows < cols) ? rows : cols;

    CHECK_CUDA(cudaMalloc(&s->d_M,    n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s->d_X,    n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s->d_X2,   n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s->d_A,    m * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s->d_A2,   m * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s->d_norm, sizeof(float)));
    CHECK_CUDA(cudaMemset(s->d_M, 0, n * sizeof(float)));

    // Row-major fp32 layouts (leading dim = number of columns for row-major).
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&s->X_layout, CUDA_R_32F, rows, cols, cols));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(s->X_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&s->A_layout, CUDA_R_32F, m, m, m));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(s->A_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

static inline void muon_state_free(MuonState* s) {
    cudaFree(s->d_M);
    cudaFree(s->d_X);
    cudaFree(s->d_X2);
    cudaFree(s->d_A);
    cudaFree(s->d_A2);
    cudaFree(s->d_norm);
    cublasLtMatrixLayoutDestroy(s->X_layout);
    cublasLtMatrixLayoutDestroy(s->A_layout);
}

static inline void muon_state_reset(MuonState* s) {
    CHECK_CUDA(cudaMemset(s->d_M, 0, s->rows * s->cols * sizeof(float)));
}

static inline void muon_state_serialize(MuonState* s, FILE* file) {
    int n = s->rows * s->cols;
    float* h = (float*)malloc(n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h, s->d_M, n * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h, sizeof(float), n, file);
    free(h);
}

static inline void muon_state_deserialize(MuonState* s, FILE* file) {
    int n = s->rows * s->cols;
    float* h = (float*)malloc(n * sizeof(float));
    size_t got = fread(h, sizeof(float), n, file);
    (void)got;
    CHECK_CUDA(cudaMemcpy(s->d_M, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

// Internal helper that sets transpose flags and dispatches an fp32 matmul.
static inline void muon_matmul_(cublasLtHandle_t handle,
                                cublasLtMatmulDesc_t desc,
                                cublasOperation_t opA, cublasOperation_t opB,
                                const float* alpha,
                                const float* A, cublasLtMatrixLayout_t lA,
                                const float* B, cublasLtMatrixLayout_t lB,
                                const float* beta,
                                float* C, cublasLtMatrixLayout_t lC) {
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &opA, sizeof(opA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &opB, sizeof(opB)));
    CHECK_CUBLASLT(cublasLtMatmul(handle, desc,
                                  alpha, A, lA, B, lB,
                                  beta,  C, lC, C, lC,
                                  NULL, NULL, 0, 0));
}

// ===== Muon update step =====================================================

static inline void muon_step(cublasLtHandle_t handle,
                             cublasLtMatmulDesc_t desc,
                             half* d_W, half* d_G,
                             MuonState* s,
                             float beta, float lr,
                             float weight_decay, int batch_size) {
    int rows = s->rows, cols = s->cols;
    int n      = rows * cols;
    int m_dim  = (rows < cols) ? rows : cols;
    int k_dim  = (rows < cols) ? cols : rows;
    int block  = 256;

    // 1) Momentum buffer update and seed working buffer X <- M.
    muon_momentum_kernel<<<(n + block - 1) / block, block>>>(
        s->d_M, s->d_X, d_G, beta, 1.0f / (float)batch_size, n);

    // 2) Normalize X by its Frobenius norm so Newton-Schulz is well-scaled.
    CHECK_CUDA(cudaMemsetAsync(s->d_norm, 0, sizeof(float)));
    muon_frob_sq_kernel<<<(n + block - 1) / block, block>>>(s->d_X, s->d_norm, n);
    muon_normalize_kernel<<<(n + block - 1) / block, block>>>(
        s->d_X, s->d_norm, MUON_NORM_EPS, n);

    // 3) Newton-Schulz iteration.  Treat X as the "wide" matrix (m_dim x k_dim)
    //    by adapting transpose flags when rows > cols.
    const float one      = 1.0f;
    const float zero     = 0.0f;
    const float a_coef   = MUON_NS_A;
    cublasOperation_t opN = CUBLAS_OP_N, opT = CUBLAS_OP_T;
    cublasOperation_t xxt_opA = (rows <= cols) ? opN : opT;
    cublasOperation_t xxt_opB = (rows <= cols) ? opT : opN;

    float* d_X  = s->d_X;
    float* d_X2 = s->d_X2;

    for (int step = 0; step < MUON_NS_STEPS; ++step) {
        // A = X' @ X'^T    (m_dim x m_dim)
        muon_matmul_(handle, desc, xxt_opA, xxt_opB,
                     &one, d_X, s->X_layout, d_X, s->X_layout,
                     &zero, s->d_A, s->A_layout);

        // A2 = A @ A
        muon_matmul_(handle, desc, opN, opN,
                     &one, s->d_A, s->A_layout, s->d_A, s->A_layout,
                     &zero, s->d_A2, s->A_layout);

        // A <- b*A + c*A2  (so A now holds the polynomial "B" matrix)
        int mm = m_dim * m_dim;
        muon_mix_AB_kernel<<<(mm + block - 1) / block, block>>>(
            s->d_A, s->d_A2, MUON_NS_B, MUON_NS_C, mm);

        // X_new = a*X + B @ X' (preserves X's (rows, cols) shape).
        // Use a ping-pong buffer because cuBLASLt requires the C operand to
        // be distinct from A and B.  Seed X2 with X so beta * C contributes
        // the a*X term.
        muon_copy_kernel<<<(n + block - 1) / block, block>>>(d_X2, d_X, n);
        if (rows <= cols) {
            // X2 = 1.0 * (A @ X) + a_coef * X2
            muon_matmul_(handle, desc, opN, opN,
                         &one, s->d_A, s->A_layout, d_X, s->X_layout,
                         &a_coef, d_X2, s->X_layout);
        } else {
            // X2 = 1.0 * (X @ A^T) + a_coef * X2
            muon_matmul_(handle, desc, opN, opT,
                         &one, d_X, s->X_layout, s->d_A, s->A_layout,
                         &a_coef, d_X2, s->X_layout);
        }
        float* tmp = d_X; d_X = d_X2; d_X2 = tmp;
    }

    // Persist the swap so subsequent calls see a consistent layout.
    s->d_X = d_X;
    s->d_X2 = d_X2;

    // 4) Apply update: W <- W - lr * ( sqrt(k/m) * X + weight_decay * W ).
    float scale = sqrtf((float)k_dim / (float)m_dim);
    muon_apply_update_kernel<<<(n + block - 1) / block, block>>>(
        d_W, d_X, scale, lr, weight_decay, n);
}

#endif // MUON_H

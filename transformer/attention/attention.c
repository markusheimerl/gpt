#include "attention.h"
#include <mma.h>

// ============================================================================
// Flash attention (FA2-style, WMMA tensor cores).
// Layout: half tensors with logical (B,NH,T,HS) over [B,T,NH,HS] physical memory.
// d_stats stores log-sum-exp per (B,NH,T) for the backward pass.
// ============================================================================
static constexpr int BM = 64, BN = 64, NWARPS = 4, WSIZE = 32, RPW = BM / NWARPS;
static constexpr int HS = 64, HT = HS / 16, NT = BN / 16;

// Cooperative 16B-vectorized tile load: copies a [BM x HS] tile from global memory
__device__ static inline void load_tile(half* dst, const half* base, int t0, int T, int NH) {
    const int LPR = HS / 8;
    const int LT  = BM * LPR;
    const int LPT = LT / (NWARPS * WSIZE);
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < LPT; i++) {
        int ln = tid + i * (NWARPS * WSIZE);
        int r  = ln / LPR;
        int d8 = (ln % LPR) * 8;
        int t  = t0 + r;
        if (t < T) *(int4*)(dst + ln*8) = *(const int4*)(base + t*NH*HS + d8);
        else       *(int4*)(dst + ln*8) = {0,0,0,0};
    }
}

// Flash attention forward kernel
__global__ static void flash_fwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                        const half* __restrict__ V, half* __restrict__ O,
                                        float* __restrict__ stats, int T, int NH, int is_causal, float scale) {
    const int b = blockIdx.z, h = blockIdx.y, mb = blockIdx.x;
    const int m_start = mb*BM;
    if (m_start >= T) return;
    const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
    const int bh = b*T*NH*HS + h*HS;

    extern __shared__ unsigned char smem[];
    half*  Qs = (half*)smem;
    half*  Ks = Qs + BM*HS;
    half*  Vs = Ks + BN*HS;
    half*  Ps = Vs + BN*HS;
    float* Sf = (float*)(Ps + BM*BN);
    float* mi = Sf + BM*BN;
    float* li = mi + BM;
    float* al = li + BM;

    if (tid < BM) { mi[tid] = -1e30f; li[tid] = 0.f; }
    load_tile(Qs, Q + bh, m_start, T, NH);
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> O_frag[HT];
    #pragma unroll
    for (int k = 0; k < HT; k++) nvcuda::wmma::fill_fragment(O_frag[k], 0.f);

    const int n_end = is_causal ? min(T, m_start+BM) : T;
    for (int n_start = 0; n_start < n_end; n_start += BN) {
        load_tile(Ks, K + bh, n_start, T, NH);
        load_tile(Vs, V + bh, n_start, T, NH);
        __syncthreads();

        // S = Q K^T
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> S_frag[NT];
        #pragma unroll
        for (int j = 0; j < NT; j++) nvcuda::wmma::fill_fragment(S_frag[j], 0.f);
        #pragma unroll
        for (int k = 0; k < HT; k++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::row_major> a;
            nvcuda::wmma::load_matrix_sync(a, Qs + warp*RPW*HS + k*16, HS);
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::col_major> bf;
                nvcuda::wmma::load_matrix_sync(bf, Ks + j*16*HS + k*16, HS);
                nvcuda::wmma::mma_sync(S_frag[j], a, bf, S_frag[j]);
            }
        }
        #pragma unroll
        for (int j = 0; j < NT; j++)
            nvcuda::wmma::store_matrix_sync(Sf + warp*RPW*BN + j*16, S_frag[j], BN, nvcuda::wmma::mem_row_major);
        __syncwarp();

        // Online softmax: mask, find row max, exponentiate, update running (m, l, alpha)
        if (lane < RPW) {
            int gr = warp*RPW + lane, t_q = m_start + gr;
            float rmax = -1e30f;
            #pragma unroll
            for (int c = 0; c < BN; c++) {
                int t_k = n_start + c;
                float s = Sf[gr*BN+c]*scale;
                bool ok = (t_k<T) & (t_q<T) & (!is_causal | (t_k<=t_q));
                if (!ok) s = -1e30f;
                Sf[gr*BN+c] = s;
                if (s > rmax) rmax = s;
            }
            float m_old = mi[gr], m_new = fmaxf(m_old, rmax);
            float alpha = (m_old == -1e30f) ? 0.f : expf(m_old - m_new);
            float rs = 0.f;
            #pragma unroll
            for (int c = 0; c < BN; c++) {
                float p = (rmax == -1e30f) ? 0.f : expf(Sf[gr*BN+c] - m_new);
                Ps[gr*BN+c] = __float2half(p);
                rs += p;
            }
            mi[gr] = m_new; li[gr] = li[gr]*alpha + rs; al[gr] = alpha;
        }
        __syncthreads();

        // Rescale running output accumulator: O *= alpha
        float* O_sc = Sf;
        #pragma unroll
        for (int k = 0; k < HT; k++)
            nvcuda::wmma::store_matrix_sync(O_sc + warp*RPW*HS + k*16, O_frag[k], HS, nvcuda::wmma::mem_row_major);
        __syncwarp();
        if (lane < RPW) {
            int gr = warp*RPW + lane;
            float a = al[gr];
            #pragma unroll
            for (int d = 0; d < HS; d++) O_sc[gr*HS + d] *= a;
        }
        __syncwarp();
        #pragma unroll
        for (int k = 0; k < HT; k++)
            nvcuda::wmma::load_matrix_sync(O_frag[k], O_sc + warp*RPW*HS + k*16, HS, nvcuda::wmma::mem_row_major);

        // O += P V
        #pragma unroll
        for (int k = 0; k < NT; k++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::row_major> a;
            nvcuda::wmma::load_matrix_sync(a, Ps + warp*RPW*BN + k*16, BN);
            #pragma unroll
            for (int j = 0; j < HT; j++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::row_major> bf;
                nvcuda::wmma::load_matrix_sync(bf, Vs + k*16*HS + j*16, HS);
                nvcuda::wmma::mma_sync(O_frag[j], a, bf, O_frag[j]);
            }
        }
        __syncthreads();
    }

    // Normalize by row sum and write final output; save LSE for backward
    float* O_sc = Sf;
    #pragma unroll
    for (int k = 0; k < HT; k++)
        nvcuda::wmma::store_matrix_sync(O_sc + warp*RPW*HS + k*16, O_frag[k], HS, nvcuda::wmma::mem_row_major);
    __syncwarp();
    if (lane < RPW) {
        int gr = warp*RPW + lane, t = m_start + gr;
        if (t < T) {
            float l = li[gr], inv = (l>0.f) ? 1.f/l : 0.f;
            #pragma unroll
            for (int d = 0; d < HS; d++)
                O[bh + t*NH*HS + d] = __float2half(O_sc[gr*HS + d] * inv);
            if (stats) stats[b*NH*T + h*T + t] = (l>0.f) ? (mi[gr] + logf(l)) : -1e30f;
        }
    }
}

// Flash attention backward kernel
__global__ static void flash_bwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                        const half* __restrict__ V, const half* __restrict__ O,
                                        const half* __restrict__ dO, const float* __restrict__ stats,
                                        half* __restrict__ dQ, half* __restrict__ dK, half* __restrict__ dV,
                                        int T, int NH, int is_causal, float scale) {
    const int b = blockIdx.z, h = blockIdx.y, mb = blockIdx.x;
    const int m_start = mb*BM;
    if (m_start >= T) return;
    const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
    const int bh = b*T*NH*HS + h*HS;
    const float* lse = stats + b*NH*T + h*T;

    extern __shared__ unsigned char smem[];
    half*  Qs  = (half*)smem;
    half*  dOs = Qs + BM*HS;
    half*  Ks  = dOs + BM*HS;
    half*  Vs  = Ks + BN*HS;
    half*  Ps  = Vs + BN*HS;
    half*  dSs = Ps + BM*BN;
    float* Sf  = (float*)(dSs + BM*BN);
    float* Di  = Sf + BM*BN;
    float* Li  = Di + BM;

    __shared__ float dV_sc[BM*HS];
    __shared__ float dK_sc[BM*HS];
    __shared__ float dQ_sc[BM*HS];

    load_tile(Qs,  Q  + bh, m_start, T, NH);
    load_tile(dOs, dO + bh, m_start, T, NH);
    if (tid < BM) {
        int t = m_start + tid;
        Li[tid] = (t<T) ? lse[t] : -1e30f;
        Di[tid] = 0.f;
    }
    __syncthreads();

    // D[i] = sum_d O[i,d] * dO[i,d]
    if (lane < RPW) {
        int gr = warp*RPW + lane;
        float s = 0.f;
        int t = m_start + gr;
        if (t < T) {
            #pragma unroll
            for (int d = 0; d < HS; d++) {
                float oo = __half2float(O [bh + t*NH*HS + d]);
                float dd = __half2float(dO[bh + t*NH*HS + d]);
                s += oo * dd;
            }
        }
        Di[gr] = s;
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> dQ_frag[HT];
    #pragma unroll
    for (int k = 0; k < HT; k++) nvcuda::wmma::fill_fragment(dQ_frag[k], 0.f);

    const int n_end = is_causal ? min(T, m_start+BM) : T;
    for (int n_start = 0; n_start < n_end; n_start += BN) {
        load_tile(Ks, K + bh, n_start, T, NH);
        load_tile(Vs, V + bh, n_start, T, NH);
        __syncthreads();

        // S = Q K^T
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> S_frag[NT];
        #pragma unroll
        for (int j = 0; j < NT; j++) nvcuda::wmma::fill_fragment(S_frag[j], 0.f);
        #pragma unroll
        for (int k = 0; k < HT; k++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::row_major> a;
            nvcuda::wmma::load_matrix_sync(a, Qs + warp*RPW*HS + k*16, HS);
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::col_major> bf;
                nvcuda::wmma::load_matrix_sync(bf, Ks + j*16*HS + k*16, HS);
                nvcuda::wmma::mma_sync(S_frag[j], a, bf, S_frag[j]);
            }
        }
        #pragma unroll
        for (int j = 0; j < NT; j++)
            nvcuda::wmma::store_matrix_sync(Sf + warp*RPW*BN + j*16, S_frag[j], BN, nvcuda::wmma::mem_row_major);
        __syncwarp();

        // Recompute P = softmax(S * scale - LSE)
        if (lane < RPW) {
            int gr = warp*RPW + lane, t_q = m_start + gr;
            float L = Li[gr];
            #pragma unroll
            for (int c = 0; c < BN; c++) {
                int t_k = n_start + c;
                bool ok = (t_k<T) & (t_q<T) & (!is_causal | (t_k<=t_q));
                float p = ok ? expf(Sf[gr*BN+c]*scale - L) : 0.f;
                Ps[gr*BN+c] = __float2half(p);
            }
        }
        __syncthreads();

        // dV += P^T dO  (atomic accumulation into global dV)
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> dV_frag[HT];
        #pragma unroll
        for (int k = 0; k < HT; k++) nvcuda::wmma::fill_fragment(dV_frag[k], 0.f);
        #pragma unroll
        for (int k = 0; k < (BM/16); k++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::col_major> a;
            nvcuda::wmma::load_matrix_sync(a, Ps + k*16*BN + warp*RPW, BN);
            #pragma unroll
            for (int j = 0; j < HT; j++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::row_major> bf;
                nvcuda::wmma::load_matrix_sync(bf, dOs + k*16*HS + j*16, HS);
                nvcuda::wmma::mma_sync(dV_frag[j], a, bf, dV_frag[j]);
            }
        }
        #pragma unroll
        for (int j = 0; j < HT; j++)
            nvcuda::wmma::store_matrix_sync(dV_sc + warp*16*HS + j*16, dV_frag[j], HS, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 16*HS; i += WSIZE) {
            int r = i/HS, d = i%HS, t_k = n_start + warp*16 + r;
            if (t_k < T) atomicAdd((half*)&dV[bh + t_k*NH*HS + d], __float2half(dV_sc[warp*16*HS + i]));
        }
        __syncthreads();

        // dP = dO V^T
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> dP_frag[NT];
        #pragma unroll
        for (int j = 0; j < NT; j++) nvcuda::wmma::fill_fragment(dP_frag[j], 0.f);
        #pragma unroll
        for (int k = 0; k < HT; k++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::row_major> a;
            nvcuda::wmma::load_matrix_sync(a, dOs + warp*RPW*HS + k*16, HS);
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::col_major> bf;
                nvcuda::wmma::load_matrix_sync(bf, Vs + j*16*HS + k*16, HS);
                nvcuda::wmma::mma_sync(dP_frag[j], a, bf, dP_frag[j]);
            }
        }
        #pragma unroll
        for (int j = 0; j < NT; j++)
            nvcuda::wmma::store_matrix_sync(Sf + warp*RPW*BN + j*16, dP_frag[j], BN, nvcuda::wmma::mem_row_major);
        __syncwarp();

        // dS = P * (dP - D)
        if (lane < RPW) {
            int gr = warp*RPW + lane, t_q = m_start + gr;
            float D = Di[gr];
            #pragma unroll
            for (int c = 0; c < BN; c++) {
                int t_k = n_start + c;
                bool ok = (t_k<T) & (t_q<T) & (!is_causal | (t_k<=t_q));
                float p = __half2float(Ps[gr*BN+c]);
                float ds = ok ? p * (Sf[gr*BN+c] - D) : 0.f;
                dSs[gr*BN+c] = __float2half(ds);
            }
        }
        __syncthreads();

        // dQ += dS K  (accumulated across n tiles in registers)
        #pragma unroll
        for (int k = 0; k < NT; k++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::row_major> a;
            nvcuda::wmma::load_matrix_sync(a, dSs + warp*RPW*BN + k*16, BN);
            #pragma unroll
            for (int j = 0; j < HT; j++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::row_major> bf;
                nvcuda::wmma::load_matrix_sync(bf, Ks + k*16*HS + j*16, HS);
                nvcuda::wmma::mma_sync(dQ_frag[j], a, bf, dQ_frag[j]);
            }
        }

        // dK += dS^T Q * scale  (atomic accumulation into global dK)
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> dK_frag[HT];
        #pragma unroll
        for (int k = 0; k < HT; k++) nvcuda::wmma::fill_fragment(dK_frag[k], 0.f);
        #pragma unroll
        for (int k = 0; k < (BM/16); k++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::col_major> a;
            nvcuda::wmma::load_matrix_sync(a, dSs + k*16*BN + warp*RPW, BN);
            #pragma unroll
            for (int j = 0; j < HT; j++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::row_major> bf;
                nvcuda::wmma::load_matrix_sync(bf, Qs + k*16*HS + j*16, HS);
                nvcuda::wmma::mma_sync(dK_frag[j], a, bf, dK_frag[j]);
            }
        }
        #pragma unroll
        for (int j = 0; j < HT; j++)
            nvcuda::wmma::store_matrix_sync(dK_sc + warp*16*HS + j*16, dK_frag[j], HS, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 16*HS; i += WSIZE) {
            int r = i/HS, d = i%HS, t_k = n_start + warp*16 + r;
            if (t_k < T) atomicAdd((half*)&dK[bh + t_k*NH*HS + d], __float2half(dK_sc[warp*16*HS + i] * scale));
        }
        __syncthreads();
    }

    // Write dQ * scale
    #pragma unroll
    for (int k = 0; k < HT; k++)
        nvcuda::wmma::store_matrix_sync(dQ_sc + warp*RPW*HS + k*16, dQ_frag[k], HS, nvcuda::wmma::mem_row_major);
    __syncwarp();
    if (lane < RPW) {
        int gr = warp*RPW + lane, t = m_start + gr;
        if (t < T) {
            #pragma unroll
            for (int d = 0; d < HS; d++)
                dQ[bh + t*NH*HS + d] = __float2half(dQ_sc[gr*HS + d] * scale);
        }
    }
}

// Launch flash attention forward
static void flash_attention_forward(const half* Q, const half* K, const half* V, half* O, float* stats,
                                    int B, int NH, int T, int head_dim, int is_causal) {
    if (head_dim != HS) { fprintf(stderr, "flash attn: head_dim=%d not supported (compiled for %d)\n", head_dim, HS); exit(1); }
    float scale = 1.0f / sqrtf((float)HS);
    size_t smem = BM*HS*sizeof(half) + 2*BN*HS*sizeof(half) + BM*BN*sizeof(half)
                + BM*BN*sizeof(float) + 3*BM*sizeof(float);
    cudaFuncSetAttribute(flash_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((T + BM - 1) / BM, NH, B);
    flash_fwd_kernel<<<grid, NWARPS*WSIZE, smem>>>(Q, K, V, O, stats, T, NH, is_causal, scale);
}

// Launch flash attention backward
static void flash_attention_backward(const half* Q, const half* K, const half* V, const half* O,
                                     const half* dO, const float* stats,
                                     half* dQ, half* dK, half* dV,
                                     int B, int NH, int T, int head_dim, int is_causal) {
    if (head_dim != HS) { fprintf(stderr, "flash attn: head_dim=%d not supported (compiled for %d)\n", head_dim, HS); exit(1); }
    float scale = 1.0f / sqrtf((float)HS);
    size_t smem = 2*BM*HS*sizeof(half) + 2*BN*HS*sizeof(half) + 2*BM*BN*sizeof(half)
                + BM*BN*sizeof(float) + 2*BM*sizeof(float);
    CHECK_CUDA(cudaMemsetAsync(dK, 0, (size_t)B*NH*T*HS*sizeof(half)));
    CHECK_CUDA(cudaMemsetAsync(dV, 0, (size_t)B*NH*T*HS*sizeof(half)));
    cudaFuncSetAttribute(flash_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((T + BM - 1) / BM, NH, B);
    flash_bwd_kernel<<<grid, NWARPS*WSIZE, smem>>>(Q, K, V, O, dO, stats, dQ, dK, dV, T, NH, is_causal, scale);
}

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

    // Allocate host memory for weight initialization
    half* h_W_q = (half*)malloc(weight_size * sizeof(half));
    half* h_W_k = (half*)malloc(weight_size * sizeof(half));
    half* h_W_v = (half*)malloc(weight_size * sizeof(half));
    half* h_W_o = (half*)malloc(weight_size * sizeof(half));

    // Initialize weights on host
    float scale_W = 1.0f / sqrtf(d_model);

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

    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&attn->d_Q, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_K, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_V, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_output, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_output, seq_batch_size * sizeof(half)));

    // Alias/Allocate device memory for backward pass buffers
    attn->d_grad_output = attn->d_output;
    CHECK_CUDA(cudaMalloc(&attn->d_grad_attn_output, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_Q, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_K, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_V, seq_batch_size * sizeof(half)));

    // Allocate device memory for flash-attention softmax stats
    CHECK_CUDA(cudaMalloc(&attn->d_stats, (size_t)batch_size * num_heads * seq_len * sizeof(float)));

    // Allocate single device float for loss computation
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

    // Create cuBLASLt matrix multiplication descriptor
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));

    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    // Create matrix layout descriptors
    // W_q, W_k, W_v, W_o and their gradients: [d_model x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->weight_layout, CUDA_R_16F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // X, Q, K, V, attn_output, output and their gradients: [batch_size * seq_len x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->seq_flat_layout, CUDA_R_16F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_flat_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // Free host memory
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);

    return attn;
}

// Free attention memory
void free_attention(Attention* attn) {
    // Destroy cuBLASLt descriptor
    cublasLtMatmulDescDestroy(attn->matmul_desc);

    // Destroy matrix layouts
    cublasLtMatrixLayoutDestroy(attn->weight_layout);
    cublasLtMatrixLayoutDestroy(attn->seq_flat_layout);

    // Free device memory
    cudaFree(attn->d_W_q); cudaFree(attn->d_W_k); cudaFree(attn->d_W_v); cudaFree(attn->d_W_o);
    cudaFree(attn->d_W_q_grad); cudaFree(attn->d_W_k_grad); cudaFree(attn->d_W_v_grad); cudaFree(attn->d_W_o_grad);
    cudaFree(attn->d_W_q_m); cudaFree(attn->d_W_q_v);
    cudaFree(attn->d_W_k_m); cudaFree(attn->d_W_k_v);
    cudaFree(attn->d_W_v_m); cudaFree(attn->d_W_v_v);
    cudaFree(attn->d_W_o_m); cudaFree(attn->d_W_o_v);
    cudaFree(attn->d_Q); cudaFree(attn->d_K); cudaFree(attn->d_V);
    cudaFree(attn->d_attn_output); cudaFree(attn->d_output);
    cudaFree(attn->d_grad_attn_output);
    cudaFree(attn->d_grad_Q); cudaFree(attn->d_grad_K); cudaFree(attn->d_grad_V);
    cudaFree(attn->d_stats);

    // Free loss computation buffer
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

    float q0 = __half2float(Q[idx]), q1 = __half2float(Q[idx + 1]);
    Q[idx]     = __float2half(q0 * cos_a - q1 * sin_a);
    Q[idx + 1] = __float2half(q0 * sin_a + q1 * cos_a);

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

    float gq0 = __half2float(grad_Q[idx]), gq1 = __half2float(grad_Q[idx + 1]);
    grad_Q[idx]     = __float2half( gq0 * cos_a + gq1 * sin_a);
    grad_Q[idx + 1] = __float2half(-gq0 * sin_a + gq1 * cos_a);

    float gk0 = __half2float(grad_K[idx]), gk1 = __half2float(grad_K[idx + 1]);
    grad_K[idx]     = __float2half( gk0 * cos_a + gk1 * sin_a);
    grad_K[idx + 1] = __float2half(-gk0 * sin_a + gk1 * cos_a);
}

// Forward pass
void forward_pass_attention(Attention* attn, half* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Q = XW_q
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_W_q, attn->weight_layout,
              &beta, attn->d_Q, attn->seq_flat_layout);

    // K = XW_k
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_W_k, attn->weight_layout,
              &beta, attn->d_K, attn->seq_flat_layout);

    // V = XW_v
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_W_v, attn->weight_layout,
              &beta, attn->d_V, attn->seq_flat_layout);

    // Apply rotary positional embeddings to Q and K
    if (attn->use_rope) {
        dim3 grid(attn->batch_size, attn->seq_len);
        rope_forward_kernel_attention<<<grid, attn->d_model / 2>>>(
            attn->d_Q, attn->d_K, attn->batch_size, attn->seq_len, attn->d_model
        );
    }

    // Z = softmax(QK^T/√d) V
    flash_attention_forward(
        attn->d_Q, attn->d_K, attn->d_V,
        attn->d_attn_output, attn->d_stats,
        attn->batch_size, attn->num_heads, attn->seq_len, attn->head_dim,
        attn->is_causal ? 1 : 0
    );

    // Y = ZW_o
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout,
              attn->d_W_o, attn->weight_layout,
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

    // Reset loss accumulator to zero
    CHECK_CUDA(cudaMemset(attn->d_loss_result, 0, sizeof(float)));

    // Compute gradient and accumulate loss
    compute_loss_and_gradient_kernel_attention<<<num_blocks, block_size>>>(
        attn->d_grad_output, attn->d_output, d_y, attn->d_loss_result, total_elements
    );

    // Copy result back to host
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
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // ∂L/∂W_o = Z^T(∂L/∂Y)
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout,
              attn->d_grad_output, attn->seq_flat_layout,
              &alpha, attn->d_W_o_grad, attn->weight_layout);

    // ∂L/∂Z = (∂L/∂Y)W_o^T
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              attn->d_grad_output, attn->seq_flat_layout,
              attn->d_W_o, attn->weight_layout,
              &beta, attn->d_grad_attn_output, attn->seq_flat_layout);

    // ∂L/∂Q, ∂L/∂K, ∂L/∂V via flash attention backward
    flash_attention_backward(
        attn->d_Q, attn->d_K, attn->d_V,
        attn->d_attn_output, attn->d_grad_attn_output, attn->d_stats,
        attn->d_grad_Q, attn->d_grad_K, attn->d_grad_V,
        attn->batch_size, attn->num_heads, attn->seq_len, attn->head_dim,
        attn->is_causal ? 1 : 0
    );

    // Apply rotary positional embeddings backward to grad_Q and grad_K
    if (attn->use_rope) {
        dim3 grid(attn->batch_size, attn->seq_len);
        rope_backward_kernel_attention<<<grid, attn->d_model / 2>>>(
            attn->d_grad_Q, attn->d_grad_K, attn->batch_size, attn->seq_len, attn->d_model
        );
    }

    // ∂L/∂W_q = X^T(∂L/∂Q)
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_grad_Q, attn->seq_flat_layout,
              &alpha, attn->d_W_q_grad, attn->weight_layout);

    // ∂L/∂W_k = X^T(∂L/∂K)
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_grad_K, attn->seq_flat_layout,
              &alpha, attn->d_W_k_grad, attn->weight_layout);

    // ∂L/∂W_v = X^T(∂L/∂V)
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_grad_V, attn->seq_flat_layout,
              &alpha, attn->d_W_v_grad, attn->weight_layout);

    if (d_grad_X != NULL) {
        // ∂L/∂X = (∂L/∂Q)W_q^T + (∂L/∂K)W_k^T + (∂L/∂V)W_v^T
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_Q, attn->seq_flat_layout,
                  attn->d_W_q, attn->weight_layout,
                  &beta, d_grad_X, attn->seq_flat_layout);

        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_K, attn->seq_flat_layout,
                  attn->d_W_k, attn->weight_layout,
                  &alpha, d_grad_X, attn->seq_flat_layout);

        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_V, attn->seq_flat_layout,
                  attn->d_W_v, attn->weight_layout,
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
void update_weights_attention(Attention* attn, float learning_rate, int batch_size) {
    attn->t++;

    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);

    int block_size = 256;

    int weight_size = attn->d_model * attn->d_model;
    int weight_blocks = (weight_size + block_size - 1) / block_size;

    // Update W_q weights
    adamw_update_kernel_attention<<<weight_blocks, block_size>>>(
        attn->d_W_q, attn->d_W_q_grad, attn->d_W_q_m, attn->d_W_q_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, batch_size
    );

    // Update W_k weights
    adamw_update_kernel_attention<<<weight_blocks, block_size>>>(
        attn->d_W_k, attn->d_W_k_grad, attn->d_W_k_m, attn->d_W_k_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, batch_size
    );

    // Update W_v weights
    adamw_update_kernel_attention<<<weight_blocks, block_size>>>(
        attn->d_W_v, attn->d_W_v_grad, attn->d_W_v_m, attn->d_W_v_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, batch_size
    );

    // Update W_o weights
    adamw_update_kernel_attention<<<weight_blocks, block_size>>>(
        attn->d_W_o, attn->d_W_o_grad, attn->d_W_o_m, attn->d_W_o_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, batch_size
    );
}

// Reset optimizer state
void reset_optimizer_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;

    // Reset Adam moment estimates to zero on device
    CHECK_CUDA(cudaMemset(attn->d_W_q_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_q_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_v, 0, weight_size * sizeof(float)));

    // Reset time step
    attn->t = 0;
}

// Serialize attention to a file
void serialize_attention(Attention* attn, FILE* file) {
    // Write dimensions
    fwrite(&attn->d_model, sizeof(int), 1, file);
    fwrite(&attn->is_causal, sizeof(bool), 1, file);
    fwrite(&attn->use_rope, sizeof(bool), 1, file);

    int weight_size = attn->d_model * attn->d_model;

    // Allocate host buffers for weights
    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));

    // Copy weights from device
    CHECK_CUDA(cudaMemcpy(h_W_q, attn->d_W_q, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k, attn->d_W_k, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v, attn->d_W_v, weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o, attn->d_W_o, weight_size * sizeof(half), cudaMemcpyDeviceToHost));

    // Convert half to float
    for (int i = weight_size - 1; i >= 0; i--) h_W_q[i] = __half2float(((half*)h_W_q)[i]);
    for (int i = weight_size - 1; i >= 0; i--) h_W_k[i] = __half2float(((half*)h_W_k)[i]);
    for (int i = weight_size - 1; i >= 0; i--) h_W_v[i] = __half2float(((half*)h_W_v)[i]);
    for (int i = weight_size - 1; i >= 0; i--) h_W_o[i] = __half2float(((half*)h_W_o)[i]);

    // Write weights
    fwrite(h_W_q, sizeof(float), weight_size, file);
    fwrite(h_W_k, sizeof(float), weight_size, file);
    fwrite(h_W_v, sizeof(float), weight_size, file);
    fwrite(h_W_o, sizeof(float), weight_size, file);

    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);

    // Write optimizer state
    fwrite(&attn->t, sizeof(int), 1, file);

    // Allocate host buffers for optimizer state
    float* h_W_q_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_q_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_v = (float*)malloc(weight_size * sizeof(float));

    // Copy optimizer state from device
    CHECK_CUDA(cudaMemcpy(h_W_q_m, attn->d_W_q_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_q_v, attn->d_W_q_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k_m, attn->d_W_k_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k_v, attn->d_W_k_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v_m, attn->d_W_v_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v_v, attn->d_W_v_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o_m, attn->d_W_o_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o_v, attn->d_W_o_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Write optimizer state
    fwrite(h_W_q_m, sizeof(float), weight_size, file);
    fwrite(h_W_q_v, sizeof(float), weight_size, file);
    fwrite(h_W_k_m, sizeof(float), weight_size, file);
    fwrite(h_W_k_v, sizeof(float), weight_size, file);
    fwrite(h_W_v_m, sizeof(float), weight_size, file);
    fwrite(h_W_v_v, sizeof(float), weight_size, file);
    fwrite(h_W_o_m, sizeof(float), weight_size, file);
    fwrite(h_W_o_v, sizeof(float), weight_size, file);

    // Free host buffers
    free(h_W_q_m); free(h_W_q_v);
    free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v);
    free(h_W_o_m); free(h_W_o_v);
}

// Deserialize attention from a file
Attention* deserialize_attention(FILE* file, int batch_size, int seq_len, int num_heads, cublasLtHandle_t cublaslt_handle) {
    // Read dimensions
    int d_model;
    bool is_causal, use_rope;
    fread(&d_model, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope, sizeof(bool), 1, file);

    // Initialize attention
    Attention* attn = init_attention(seq_len, d_model, num_heads, batch_size, is_causal, use_rope, cublaslt_handle);

    int weight_size = d_model * d_model;

    // Allocate host buffers for weights
    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));

    // Read weights
    fread(h_W_q, sizeof(float), weight_size, file);
    fread(h_W_k, sizeof(float), weight_size, file);
    fread(h_W_v, sizeof(float), weight_size, file);
    fread(h_W_o, sizeof(float), weight_size, file);

    // Convert float to half
    for (int i = 0; i < weight_size; i++) ((half*)h_W_q)[i] = __float2half(h_W_q[i]);
    for (int i = 0; i < weight_size; i++) ((half*)h_W_k)[i] = __float2half(h_W_k[i]);
    for (int i = 0; i < weight_size; i++) ((half*)h_W_v)[i] = __float2half(h_W_v[i]);
    for (int i = 0; i < weight_size; i++) ((half*)h_W_o)[i] = __float2half(h_W_o[i]);

    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_W_q, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_W_k, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_W_v, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_W_o, weight_size * sizeof(half), cudaMemcpyHostToDevice));

    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);

    // Read optimizer state
    fread(&attn->t, sizeof(int), 1, file);

    // Allocate host buffers for optimizer state
    float* h_W_q_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_q_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_v = (float*)malloc(weight_size * sizeof(float));

    // Read optimizer state
    fread(h_W_q_m, sizeof(float), weight_size, file);
    fread(h_W_q_v, sizeof(float), weight_size, file);
    fread(h_W_k_m, sizeof(float), weight_size, file);
    fread(h_W_k_v, sizeof(float), weight_size, file);
    fread(h_W_v_m, sizeof(float), weight_size, file);
    fread(h_W_v_v, sizeof(float), weight_size, file);
    fread(h_W_o_m, sizeof(float), weight_size, file);
    fread(h_W_o_v, sizeof(float), weight_size, file);

    // Copy optimizer state to device
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_m, h_W_q_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_v, h_W_q_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_m, h_W_k_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_v, h_W_k_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_m, h_W_v_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_v, h_W_v_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_m, h_W_o_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_v, h_W_o_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // Free host buffers
    free(h_W_q_m); free(h_W_q_v);
    free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v);
    free(h_W_o_m); free(h_W_o_v);

    return attn;
}
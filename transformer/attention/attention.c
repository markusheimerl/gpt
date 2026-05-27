#include "attention.h"
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// Flash attention (FA2-style, WMMA tensor cores).
// Layout: half tensors with logical (B,NH,T,HS) over [B,T,NH,HS] physical memory.
// d_stats stores log-sum-exp per (B,NH,T) for the backward pass.
// ============================================================================
static constexpr int BM = 64, BN = 64, NWARPS = 4, WSIZE = 32, RPW = BM / NWARPS;

template<int ROWS, int HS>
__device__ __forceinline__ void load_tile(half* dst, const half* base,
                                          int t0, int T, int NH) {
    constexpr int LPR = HS / 8;
    constexpr int LT  = ROWS * LPR;
    constexpr int LPT = LT / (NWARPS * WSIZE);
    static_assert(LT % (NWARPS * WSIZE) == 0, "tile must divide into 16B lines");
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

template<int HS>
__global__ __launch_bounds__(NWARPS*WSIZE, 2)
static void flash_fwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                             const half* __restrict__ V, half* __restrict__ O,
                             float* __restrict__ stats, int T, int NH, int is_causal, float scale) {
    static_assert(HS % 16 == 0 && HS <= BN && HS % 8 == 0, "");
    constexpr int HT = HS/16, NT = BN/16;

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
    float* mi = Sf + BM*BN, *li = mi + BM, *al = li + BM;

    if (tid < BM) { mi[tid] = -1e30f; li[tid] = 0.f; }
    load_tile<BM, HS>(Qs, Q + bh, m_start, T, NH);
    __syncthreads();

    wmma::fragment<wmma::accumulator,16,16,16,float> O_frag[HT];
    #pragma unroll
    for (int k=0;k<HT;k++) wmma::fill_fragment(O_frag[k], 0.f);

    const int n_end = is_causal ? min(T, m_start+BM) : T;
    for (int n_start = 0; n_start < n_end; n_start += BN) {
        load_tile<BN, HS>(Ks, K + bh, n_start, T, NH);
        load_tile<BN, HS>(Vs, V + bh, n_start, T, NH);
        __syncthreads();

        wmma::fragment<wmma::accumulator,16,16,16,float> S_frag[NT];
        #pragma unroll
        for (int j=0;j<NT;j++) wmma::fill_fragment(S_frag[j], 0.f);
        #pragma unroll
        for (int k=0;k<HT;k++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::load_matrix_sync(a, Qs + warp*RPW*HS + k*16, HS);
            #pragma unroll
            for (int j=0;j<NT;j++) {
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
                wmma::load_matrix_sync(bf, Ks + j*16*HS + k*16, HS);
                wmma::mma_sync(S_frag[j], a, bf, S_frag[j]);
            }
        }
        #pragma unroll
        for (int j=0;j<NT;j++)
            wmma::store_matrix_sync(Sf + warp*RPW*BN + j*16, S_frag[j], BN, wmma::mem_row_major);
        __syncwarp();

        if (lane < RPW) {
            int gr = warp*RPW + lane, t_q = m_start + gr;
            float rmax = -1e30f;
            #pragma unroll
            for (int c=0;c<BN;c++) {
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
            for (int c=0;c<BN;c++) {
                float p = (rmax == -1e30f) ? 0.f : expf(Sf[gr*BN+c] - m_new);
                Ps[gr*BN+c] = __float2half(p);
                rs += p;
            }
            mi[gr] = m_new; li[gr] = li[gr]*alpha + rs; al[gr] = alpha;
        }
        __syncthreads();

        float* O_sc = Sf;
        #pragma unroll
        for (int k=0;k<HT;k++)
            wmma::store_matrix_sync(O_sc + warp*RPW*HS + k*16, O_frag[k], HS, wmma::mem_row_major);
        __syncwarp();
        if (lane < RPW) {
            int gr = warp*RPW + lane; float a = al[gr];
            #pragma unroll
            for (int d=0; d<HS; d++) O_sc[gr*HS + d] *= a;
        }
        __syncwarp();
        #pragma unroll
        for (int k=0;k<HT;k++)
            wmma::load_matrix_sync(O_frag[k], O_sc + warp*RPW*HS + k*16, HS, wmma::mem_row_major);

        #pragma unroll
        for (int k=0;k<NT;k++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::load_matrix_sync(a, Ps + warp*RPW*BN + k*16, BN);
            #pragma unroll
            for (int j=0;j<HT;j++) {
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
                wmma::load_matrix_sync(bf, Vs + k*16*HS + j*16, HS);
                wmma::mma_sync(O_frag[j], a, bf, O_frag[j]);
            }
        }
        __syncthreads();
    }

    float* O_sc = Sf;
    #pragma unroll
    for (int k=0;k<HT;k++)
        wmma::store_matrix_sync(O_sc + warp*RPW*HS + k*16, O_frag[k], HS, wmma::mem_row_major);
    __syncwarp();
    if (lane < RPW) {
        int gr = warp*RPW + lane, t = m_start + gr;
        if (t < T) {
            float l = li[gr], inv = (l>0.f) ? 1.f/l : 0.f;
            #pragma unroll
            for (int d=0; d<HS; d++)
                O[bh + t*NH*HS + d] = __float2half(O_sc[gr*HS + d] * inv);
            if (stats) stats[b*NH*T + h*T + t] = (l>0.f) ? (mi[gr] + logf(l)) : -1e30f;
        }
    }
}

template<int HS>
__global__ __launch_bounds__(NWARPS*WSIZE, 2)
static void flash_bwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                             const half* __restrict__ V, const half* __restrict__ O,
                             const half* __restrict__ dO, const float* __restrict__ stats,
                             half* __restrict__ dQ, half* __restrict__ dK, half* __restrict__ dV,
                             int T, int NH, int is_causal, float scale) {
    static_assert(HS % 16 == 0 && HS <= BN && HS % 8 == 0, "");
    constexpr int HT = HS/16, NT = BN/16;

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

    load_tile<BM, HS>(Qs,  Q  + bh, m_start, T, NH);
    load_tile<BM, HS>(dOs, dO + bh, m_start, T, NH);
    if (tid < BM) {
        int t = m_start + tid;
        Li[tid] = (t<T) ? lse[t] : -1e30f;
        Di[tid] = 0.f;
    }
    __syncthreads();
    if (lane < RPW) {
        int gr = warp*RPW + lane;
        float s = 0.f;
        int t = m_start + gr;
        if (t < T) {
            #pragma unroll
            for (int d=0; d<HS; d++) {
                float oo = __half2float(O [bh + t*NH*HS + d]);
                float dd = __half2float(dO[bh + t*NH*HS + d]);
                s += oo * dd;
            }
        }
        Di[gr] = s;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator,16,16,16,float> dQ_frag[HT];
    #pragma unroll
    for (int k=0;k<HT;k++) wmma::fill_fragment(dQ_frag[k], 0.f);

    const int n_end = is_causal ? min(T, m_start+BM) : T;
    for (int n_start = 0; n_start < n_end; n_start += BN) {
        load_tile<BN, HS>(Ks, K + bh, n_start, T, NH);
        load_tile<BN, HS>(Vs, V + bh, n_start, T, NH);
        __syncthreads();

        wmma::fragment<wmma::accumulator,16,16,16,float> S_frag[NT];
        #pragma unroll
        for (int j=0;j<NT;j++) wmma::fill_fragment(S_frag[j], 0.f);
        #pragma unroll
        for (int k=0;k<HT;k++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::load_matrix_sync(a, Qs + warp*RPW*HS + k*16, HS);
            #pragma unroll
            for (int j=0;j<NT;j++) {
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
                wmma::load_matrix_sync(bf, Ks + j*16*HS + k*16, HS);
                wmma::mma_sync(S_frag[j], a, bf, S_frag[j]);
            }
        }
        #pragma unroll
        for (int j=0;j<NT;j++)
            wmma::store_matrix_sync(Sf + warp*RPW*BN + j*16, S_frag[j], BN, wmma::mem_row_major);
        __syncwarp();

        if (lane < RPW) {
            int gr = warp*RPW + lane, t_q = m_start + gr;
            float L = Li[gr];
            #pragma unroll
            for (int c=0;c<BN;c++) {
                int t_k = n_start + c;
                bool ok = (t_k<T) & (t_q<T) & (!is_causal | (t_k<=t_q));
                float p = ok ? expf(Sf[gr*BN+c]*scale - L) : 0.f;
                Ps[gr*BN+c] = __float2half(p);
            }
        }
        __syncthreads();

        {
            wmma::fragment<wmma::accumulator,16,16,16,float> dV_frag[HT];
            #pragma unroll
            for (int k=0;k<HT;k++) wmma::fill_fragment(dV_frag[k], 0.f);
            #pragma unroll
            for (int k=0;k<(BM/16);k++) {
                wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major> a;
                wmma::load_matrix_sync(a, Ps + k*16*BN + warp*RPW, BN);
                #pragma unroll
                for (int j=0;j<HT;j++) {
                    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
                    wmma::load_matrix_sync(bf, dOs + k*16*HS + j*16, HS);
                    wmma::mma_sync(dV_frag[j], a, bf, dV_frag[j]);
                }
            }
            __shared__ float dV_sc[BM*HS];
            #pragma unroll
            for (int j=0;j<HT;j++)
                wmma::store_matrix_sync(dV_sc + warp*16*HS + j*16, dV_frag[j], HS, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 16*HS; i += WSIZE) {
                int r = i/HS, d = i%HS, t_k = n_start + warp*16 + r;
                if (t_k < T) atomicAdd((half*)&dV[bh + t_k*NH*HS + d], __float2half(dV_sc[warp*16*HS + i]));
            }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator,16,16,16,float> dP_frag[NT];
        #pragma unroll
        for (int j=0;j<NT;j++) wmma::fill_fragment(dP_frag[j], 0.f);
        #pragma unroll
        for (int k=0;k<HT;k++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::load_matrix_sync(a, dOs + warp*RPW*HS + k*16, HS);
            #pragma unroll
            for (int j=0;j<NT;j++) {
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
                wmma::load_matrix_sync(bf, Vs + j*16*HS + k*16, HS);
                wmma::mma_sync(dP_frag[j], a, bf, dP_frag[j]);
            }
        }
        #pragma unroll
        for (int j=0;j<NT;j++)
            wmma::store_matrix_sync(Sf + warp*RPW*BN + j*16, dP_frag[j], BN, wmma::mem_row_major);
        __syncwarp();

        if (lane < RPW) {
            int gr = warp*RPW + lane, t_q = m_start + gr;
            float D = Di[gr];
            #pragma unroll
            for (int c=0;c<BN;c++) {
                int t_k = n_start + c;
                bool ok = (t_k<T) & (t_q<T) & (!is_causal | (t_k<=t_q));
                float p = __half2float(Ps[gr*BN+c]);
                float ds = ok ? p * (Sf[gr*BN+c] - D) : 0.f;
                dSs[gr*BN+c] = __float2half(ds);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k=0;k<NT;k++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::load_matrix_sync(a, dSs + warp*RPW*BN + k*16, BN);
            #pragma unroll
            for (int j=0;j<HT;j++) {
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
                wmma::load_matrix_sync(bf, Ks + k*16*HS + j*16, HS);
                wmma::mma_sync(dQ_frag[j], a, bf, dQ_frag[j]);
            }
        }

        {
            wmma::fragment<wmma::accumulator,16,16,16,float> dK_frag[HT];
            #pragma unroll
            for (int k=0;k<HT;k++) wmma::fill_fragment(dK_frag[k], 0.f);
            #pragma unroll
            for (int k=0;k<(BM/16);k++) {
                wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major> a;
                wmma::load_matrix_sync(a, dSs + k*16*BN + warp*RPW, BN);
                #pragma unroll
                for (int j=0;j<HT;j++) {
                    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
                    wmma::load_matrix_sync(bf, Qs + k*16*HS + j*16, HS);
                    wmma::mma_sync(dK_frag[j], a, bf, dK_frag[j]);
                }
            }
            __shared__ float dK_sc[BM*HS];
            #pragma unroll
            for (int j=0;j<HT;j++)
                wmma::store_matrix_sync(dK_sc + warp*16*HS + j*16, dK_frag[j], HS, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 16*HS; i += WSIZE) {
                int r = i/HS, d = i%HS, t_k = n_start + warp*16 + r;
                if (t_k < T) atomicAdd((half*)&dK[bh + t_k*NH*HS + d], __float2half(dK_sc[warp*16*HS + i] * scale));
            }
        }
        __syncthreads();
    }

    __shared__ float dQ_sc[BM*HS];
    #pragma unroll
    for (int k=0;k<HT;k++)
        wmma::store_matrix_sync(dQ_sc + warp*RPW*HS + k*16, dQ_frag[k], HS, wmma::mem_row_major);
    __syncwarp();
    if (lane < RPW) {
        int gr = warp*RPW + lane, t = m_start + gr;
        if (t < T) {
            #pragma unroll
            for (int d=0; d<HS; d++)
                dQ[bh + t*NH*HS + d] = __float2half(dQ_sc[gr*HS + d] * scale);
        }
    }
}

template<int HS> static inline size_t fwd_smem() {
    return BM*HS*sizeof(half) + 2*BN*HS*sizeof(half) + BM*BN*sizeof(half)
         + BM*BN*sizeof(float) + 3*BM*sizeof(float);
}
template<int HS> static inline size_t bwd_smem() {
    return 2*BM*HS*sizeof(half) + 2*BN*HS*sizeof(half) + 2*BM*BN*sizeof(half)
         + BM*BN*sizeof(float) + 2*BM*sizeof(float);
}

#define LAUNCH_FWD(HS_VAL) do { \
    size_t sm = fwd_smem<HS_VAL>(); \
    cudaFuncSetAttribute(flash_fwd_kernel<HS_VAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); \
    flash_fwd_kernel<HS_VAL><<<grid, NWARPS*WSIZE, sm>>>(Q,K,V,O,stats,T,NH,is_causal,scale); \
} while(0)

#define LAUNCH_BWD(HS_VAL) do { \
    size_t sm = bwd_smem<HS_VAL>(); \
    cudaFuncSetAttribute(flash_bwd_kernel<HS_VAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); \
    flash_bwd_kernel<HS_VAL><<<grid, NWARPS*WSIZE, sm>>>(Q,K,V,O,dO,stats,dQ,dK,dV,T,NH,is_causal,scale); \
} while(0)

static void flash_attention_forward(const half* Q, const half* K, const half* V, half* O, float* stats,
                                     int B, int NH, int T, int HS, int is_causal) {
    float scale = 1.0f / sqrtf((float)HS);
    dim3 grid((T + BM - 1) / BM, NH, B);
    switch (HS) {
        case 16:  LAUNCH_FWD(16);  break;
        case 32:  LAUNCH_FWD(32);  break;
        case 48:  LAUNCH_FWD(48);  break;
        case 64:  LAUNCH_FWD(64);  break;
        default:  fprintf(stderr, "flash attn: unsupported HS=%d\n", HS); exit(1);
    }
}

static void flash_attention_backward(const half* Q, const half* K, const half* V, const half* O,
                                      const half* dO, const float* stats,
                                      half* dQ, half* dK, half* dV,
                                      int B, int NH, int T, int HS, int is_causal) {
    float scale = 1.0f / sqrtf((float)HS);
    CHECK_CUDA(cudaMemsetAsync(dK, 0, (size_t)B*NH*T*HS*sizeof(half)));
    CHECK_CUDA(cudaMemsetAsync(dV, 0, (size_t)B*NH*T*HS*sizeof(half)));
    dim3 grid((T + BM - 1) / BM, NH, B);
    switch (HS) {
        case 16:  LAUNCH_BWD(16);  break;
        case 32:  LAUNCH_BWD(32);  break;
        case 48:  LAUNCH_BWD(48);  break;
        case 64:  LAUNCH_BWD(64);  break;
        default:  fprintf(stderr, "flash attn: unsupported HS=%d\n", HS); exit(1);
    }
}

// ============================================================================
// cuBLASLt matrix multiplication macro
// ============================================================================
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
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->batch_size = batch_size;
    attn->num_heads = num_heads;
    attn->head_dim = d_model / num_heads;
    attn->scale = 1.0f / sqrtf(attn->head_dim);
    attn->is_causal = is_causal;
    attn->use_rope = use_rope;

    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;

    attn->cublaslt_handle = cublaslt_handle;

    size_t weight_size = d_model * d_model;
    size_t seq_batch_size = batch_size * seq_len * d_model;

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

    CHECK_CUDA(cudaMalloc(&attn->d_W_q, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_grad, weight_size * sizeof(half)));

    CHECK_CUDA(cudaMalloc(&attn->d_W_q_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_v, weight_size * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&attn->d_Q, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_K, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_V, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_output, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_output, seq_batch_size * sizeof(half)));

    attn->d_grad_output = attn->d_output;
    CHECK_CUDA(cudaMalloc(&attn->d_grad_attn_output, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_Q, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_K, seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_V, seq_batch_size * sizeof(half)));

    CHECK_CUDA(cudaMalloc(&attn->d_stats, (size_t)batch_size * num_heads * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_loss_result, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_W_q, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_W_k, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_W_v, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_W_o, weight_size * sizeof(half), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(attn->d_W_q_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_q_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_v, 0, weight_size * sizeof(float)));

    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->weight_layout, CUDA_R_16F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->seq_flat_layout, CUDA_R_16F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_flat_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);

    return attn;
}

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

    if (attn->use_rope) {
        dim3 grid(attn->batch_size, attn->seq_len);
        rope_forward_kernel_attention<<<grid, attn->d_model / 2>>>(
            attn->d_Q, attn->d_K, attn->batch_size, attn->seq_len, attn->d_model);
    }

    // Z = softmax(QK^T/√d) V
    flash_attention_forward(attn->d_Q, attn->d_K, attn->d_V,
                            attn->d_attn_output, attn->d_stats,
                            attn->batch_size, attn->num_heads, attn->seq_len, attn->head_dim,
                            attn->is_causal ? 1 : 0);

    // Y = ZW_o
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout, attn->d_W_o, attn->weight_layout,
              &beta, attn->d_output, attn->seq_flat_layout);
}

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

float calculate_loss_attention(Attention* attn, half* d_y) {
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

void zero_gradients_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;
    CHECK_CUDA(cudaMemset(attn->d_W_q_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_grad, 0, weight_size * sizeof(half)));
}

void backward_pass_attention(Attention* attn, half* d_X, half* d_grad_X) {
    const float alpha = 1.0f, beta = 0.0f;

    // ∂L/∂W_o = Z^T(∂L/∂Y);  ∂L/∂Z = (∂L/∂Y)W_o^T
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout, attn->d_grad_output, attn->seq_flat_layout,
              &beta, attn->d_W_o_grad, attn->weight_layout);
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              attn->d_grad_output, attn->seq_flat_layout, attn->d_W_o, attn->weight_layout,
              &beta, attn->d_grad_attn_output, attn->seq_flat_layout);

    // (∂L/∂Q, ∂L/∂K, ∂L/∂V)
    flash_attention_backward(attn->d_Q, attn->d_K, attn->d_V,
                             attn->d_attn_output, attn->d_grad_attn_output, attn->d_stats,
                             attn->d_grad_Q, attn->d_grad_K, attn->d_grad_V,
                             attn->batch_size, attn->num_heads, attn->seq_len, attn->head_dim,
                             attn->is_causal ? 1 : 0);

    if (attn->use_rope) {
        dim3 grid(attn->batch_size, attn->seq_len);
        rope_backward_kernel_attention<<<grid, attn->d_model / 2>>>(
            attn->d_grad_Q, attn->d_grad_K, attn->batch_size, attn->seq_len, attn->d_model);
    }

    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_grad_Q, attn->seq_flat_layout,
              &beta, attn->d_W_q_grad, attn->weight_layout);
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_grad_K, attn->seq_flat_layout,
              &beta, attn->d_W_k_grad, attn->weight_layout);
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout, attn->d_grad_V, attn->seq_flat_layout,
              &beta, attn->d_W_v_grad, attn->weight_layout);

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

__global__ static void adamw_update_kernel_attention(half* weight, half* grad, float* m, float* v,
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

Attention* deserialize_attention(FILE* file, int batch_size, int seq_len, int num_heads, cublasLtHandle_t cublaslt_handle) {
    int d_model;
    bool is_causal, use_rope;
    fread(&d_model,   sizeof(int),  1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope,  sizeof(bool), 1, file);

    Attention* attn = init_attention(seq_len, d_model, num_heads, batch_size, is_causal, use_rope, cublaslt_handle);

    int weight_size = d_model * d_model;

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
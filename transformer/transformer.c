#include "transformer.h"

// Initialise the transformer (minGRU token-mixer + MLP feed-forward per layer).
Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle) {
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));

    transformer->seq_len     = seq_len;
    transformer->d_model     = d_model;
    transformer->batch_size  = batch_size;
    transformer->hidden_dim  = hidden_dim;
    transformer->num_layers  = num_layers;
    transformer->cublaslt_handle = cublaslt_handle;

    size_t norm_buffer_size = (size_t)batch_size * seq_len * d_model * sizeof(half);
    transformer->d_norm_mixer_inputs = (half**)malloc(num_layers * sizeof(half*));
    transformer->d_norm_mlp_inputs   = (half**)malloc(num_layers * sizeof(half*));
    for (int i = 0; i < num_layers; i++) {
        CHECK_CUDA(cudaMalloc(&transformer->d_norm_mixer_inputs[i], norm_buffer_size));
        CHECK_CUDA(cudaMalloc(&transformer->d_norm_mlp_inputs[i],   norm_buffer_size));
    }

    transformer->mingru_layers = (MinGRU**)malloc(num_layers * sizeof(MinGRU*));
    transformer->mlp_layers    = (MLP**)malloc(num_layers * sizeof(MLP*));
    for (int i = 0; i < num_layers; i++) {
        transformer->mingru_layers[i] = init_mingru(seq_len, d_model, batch_size, cublaslt_handle);
        transformer->mlp_layers[i]    = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublaslt_handle);
    }
    return transformer;
}

void free_transformer(Transformer* transformer) {
    for (int i = 0; i < transformer->num_layers; i++) {
        free_mingru(transformer->mingru_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
    }
    free(transformer->mingru_layers);
    free(transformer->mlp_layers);

    for (int i = 0; i < transformer->num_layers; i++) {
        cudaFree(transformer->d_norm_mixer_inputs[i]);
        cudaFree(transformer->d_norm_mlp_inputs[i]);
    }
    free(transformer->d_norm_mixer_inputs);
    free(transformer->d_norm_mlp_inputs);

    free(transformer);
}

// In-place residual: a += b
__global__ static void residual_add_kernel(half* a, half* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

// RMSNorm: y = x / sqrt(mean(x²) + eps)
__global__ void rmsnorm_forward_kernel(half* output, const half* input, int batch_size, int seq_len, int d_model) {
    extern __shared__ float smem[];

    int idx = blockIdx.x;
    if (idx >= batch_size * seq_len) return;

    const half* x = &input[idx * d_model];
    half* y = &output[idx * d_model];

    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp_id = tid >> 5, lane_id = tid & 31, nwarps = nthreads >> 5;

    float sum_sq = 0.0f;
    for (int d = tid; d < d_model; d += nthreads) {
        float v = __half2float(x[d]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < nwarps) ? smem[lane_id] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
        if (lane_id == 0) smem[0] = sum_sq;
    }
    __syncthreads();

    float inv_rms = rsqrtf(smem[0] / d_model + 1e-6f);
    for (int d = tid; d < d_model; d += nthreads) {
        y[d] = __float2half(__half2float(x[d]) * inv_rms);
    }
}

// RMSNorm backward
__global__ void rmsnorm_backward_kernel(half* grad_input, const half* grad_output, const half* input, int batch_size, int seq_len, int d_model) {
    extern __shared__ float smem[];

    int idx = blockIdx.x;
    if (idx >= batch_size * seq_len) return;

    const half* x  = &input[idx * d_model];
    const half* dy = &grad_output[idx * d_model];
    half*       dx = &grad_input[idx * d_model];

    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp_id = tid >> 5, lane_id = tid & 31, nwarps = nthreads >> 5;

    float sum_sq = 0.0f, sum_dy_x = 0.0f;
    for (int d = tid; d < d_model; d += nthreads) {
        float xv = __half2float(x[d]);
        float dv = __half2float(dy[d]);
        sum_sq   += xv * xv;
        sum_dy_x += dv * xv;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < nwarps) ? smem[lane_id] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
        if (lane_id == 0) smem[0] = sum_sq;
    }
    __syncthreads();
    float rms_sq = smem[0] / d_model + 1e-6f;
    float inv_rms = rsqrtf(rms_sq);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum_dy_x += __shfl_xor_sync(0xffffffff, sum_dy_x, off);
    if (lane_id == 0) smem[8 + warp_id] = sum_dy_x;
    __syncthreads();
    if (warp_id == 0) {
        sum_dy_x = (lane_id < nwarps) ? smem[8 + lane_id] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum_dy_x += __shfl_xor_sync(0xffffffff, sum_dy_x, off);
        if (lane_id == 0) smem[8] = sum_dy_x;
    }
    __syncthreads();
    sum_dy_x = smem[8];

    float scale = sum_dy_x * inv_rms * inv_rms * inv_rms / d_model;
    for (int d = tid; d < d_model; d += nthreads) {
        float xv = __half2float(x[d]);
        float dv = __half2float(dy[d]);
        dx[d] = __float2half(dv * inv_rms - xv * scale);
    }
}

void forward_pass_transformer(Transformer* transformer, half* d_X) {
    int total = transformer->batch_size * transformer->seq_len * transformer->d_model;
    int blocks = (total + 255) / 256;

    for (int layer = 0; layer < transformer->num_layers; layer++) {
        half* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;

        // RMSNorm → minGRU → residual
        rmsnorm_forward_kernel<<<transformer->batch_size * transformer->seq_len, 256, 8 * sizeof(float)>>>(
            transformer->d_norm_mixer_inputs[layer],
            layer_input,
            transformer->batch_size, transformer->seq_len, transformer->d_model);

        forward_pass_mingru(transformer->mingru_layers[layer], transformer->d_norm_mixer_inputs[layer]);

        residual_add_kernel<<<blocks, 256>>>(
            transformer->mingru_layers[layer]->d_output, layer_input, total);

        // RMSNorm → MLP → residual
        rmsnorm_forward_kernel<<<transformer->batch_size * transformer->seq_len, 256, 8 * sizeof(float)>>>(
            transformer->d_norm_mlp_inputs[layer],
            transformer->mingru_layers[layer]->d_output,
            transformer->batch_size, transformer->seq_len, transformer->d_model);

        forward_pass_mlp(transformer->mlp_layers[layer], transformer->d_norm_mlp_inputs[layer]);

        residual_add_kernel<<<blocks, 256>>>(
            transformer->mlp_layers[layer]->d_output,
            transformer->mingru_layers[layer]->d_output,
            total);
    }
}

float calculate_loss_transformer(Transformer* transformer, half* d_y) {
    return calculate_loss_mlp(transformer->mlp_layers[transformer->num_layers - 1], d_y);
}

void zero_gradients_transformer(Transformer* transformer) {
    for (int i = 0; i < transformer->num_layers; i++) {
        zero_gradients_mingru(transformer->mingru_layers[i]);
        zero_gradients_mlp(transformer->mlp_layers[i]);
    }
}

void backward_pass_transformer(Transformer* transformer, half* d_X, half* d_grad_X) {
    int total = transformer->batch_size * transformer->seq_len * transformer->d_model;
    int blocks = (total + 255) / 256;

    for (int layer = transformer->num_layers - 1; layer >= 0; layer--) {
        half* layer_input      = (layer == 0) ? d_X      : transformer->mlp_layers[layer-1]->d_output;
        half* layer_grad_input = (layer == 0) ? d_grad_X : transformer->mlp_layers[layer-1]->d_grad_output;

        // Back through MLP (in-place: writes into d_norm_mlp_inputs)
        backward_pass_mlp(
            transformer->mlp_layers[layer],
            transformer->d_norm_mlp_inputs[layer],
            transformer->d_norm_mlp_inputs[layer]);

        // Back through pre-MLP RMSNorm → grad w.r.t. (mixer_out + residual)
        rmsnorm_backward_kernel<<<transformer->batch_size * transformer->seq_len, 256, 16 * sizeof(float)>>>(
            transformer->mingru_layers[layer]->d_grad_output,
            transformer->d_norm_mlp_inputs[layer],
            transformer->mingru_layers[layer]->d_output,
            transformer->batch_size, transformer->seq_len, transformer->d_model);

        // Residual: grad flows through both branches; add MLP-direct grad
        residual_add_kernel<<<blocks, 256>>>(
            transformer->mingru_layers[layer]->d_grad_output,
            transformer->mlp_layers[layer]->d_grad_output,
            total);

        // Back through minGRU (writes grad w.r.t. norm_mixer_inputs)
        backward_pass_mingru(
            transformer->mingru_layers[layer],
            transformer->d_norm_mixer_inputs[layer],
            transformer->d_norm_mixer_inputs[layer]);

        if (layer_grad_input != NULL) {
            rmsnorm_backward_kernel<<<transformer->batch_size * transformer->seq_len, 256, 16 * sizeof(float)>>>(
                layer_grad_input,
                transformer->d_norm_mixer_inputs[layer],
                layer_input,
                transformer->batch_size, transformer->seq_len, transformer->d_model);

            residual_add_kernel<<<blocks, 256>>>(
                layer_grad_input,
                transformer->mingru_layers[layer]->d_grad_output,
                total);
        }
    }
}

void update_weights_transformer(Transformer* transformer, float learning_rate, int batch_size) {
    for (int i = 0; i < transformer->num_layers; i++) {
        update_weights_mingru(transformer->mingru_layers[i], learning_rate, batch_size);
        update_weights_mlp(transformer->mlp_layers[i], learning_rate, batch_size);
    }
}

void reset_optimizer_transformer(Transformer* transformer) {
    for (int i = 0; i < transformer->num_layers; i++) {
        reset_optimizer_mingru(transformer->mingru_layers[i]);
        reset_optimizer_mlp(transformer->mlp_layers[i]);
    }
}

void serialize_transformer(Transformer* transformer, FILE* file) {
    fwrite(&transformer->d_model,    sizeof(int), 1, file);
    fwrite(&transformer->hidden_dim, sizeof(int), 1, file);
    fwrite(&transformer->num_layers, sizeof(int), 1, file);
    for (int i = 0; i < transformer->num_layers; i++) {
        serialize_mingru(transformer->mingru_layers[i], file);
        serialize_mlp(transformer->mlp_layers[i], file);
    }
}

Transformer* deserialize_transformer(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle) {
    int d_model, hidden_dim, num_layers;
    fread(&d_model,    sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);

    Transformer* transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);

    for (int i = 0; i < num_layers; i++) {
        free_mingru(transformer->mingru_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
        transformer->mingru_layers[i] = deserialize_mingru(file, batch_size, seq_len, cublaslt_handle);
        transformer->mlp_layers[i]    = deserialize_mlp(file, batch_size * seq_len, cublaslt_handle);
    }
    return transformer;
}

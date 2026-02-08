#include "transformer.h"

// Initialize the transformer
Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, bool is_causal, bool use_rope, cublasLtHandle_t cublaslt_handle) {
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    
    // Store dimensions and handle
    transformer->seq_len = seq_len;
    transformer->d_model = d_model;
    transformer->batch_size = batch_size;
    transformer->hidden_dim = hidden_dim;
    transformer->num_layers = num_layers;
    transformer->cublaslt_handle = cublaslt_handle;
    
    // Allocate RMSNorm buffer arrays
    size_t norm_buffer_size = batch_size * seq_len * d_model * sizeof(half);
    transformer->d_norm_attn_inputs = (half**)malloc(num_layers * sizeof(half*));
    transformer->d_norm_mlp_inputs = (half**)malloc(num_layers * sizeof(half*));
    
    for (int i = 0; i < num_layers; i++) {
        CHECK_CUDA(cudaMalloc(&transformer->d_norm_attn_inputs[i], norm_buffer_size));
        CHECK_CUDA(cudaMalloc(&transformer->d_norm_mlp_inputs[i], norm_buffer_size));
    }
    
    // Allocate arrays for layer components
    transformer->attention_layers = (Attention**)malloc(num_layers * sizeof(Attention*));
    transformer->mlp_layers = (MLP**)malloc(num_layers * sizeof(MLP*));
    
    // Initialize all layers
    for (int i = 0; i < num_layers; i++) {
        transformer->attention_layers[i] = init_attention(seq_len, d_model, 8, batch_size, is_causal, use_rope, cublaslt_handle);
        transformer->mlp_layers[i] = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublaslt_handle);
    }
    
    return transformer;
}

// Free transformer memory
void free_transformer(Transformer* transformer) {
    // Free all layers
    for (int i = 0; i < transformer->num_layers; i++) {
        free_attention(transformer->attention_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
    }
    
    // Free layer arrays
    free(transformer->attention_layers);
    free(transformer->mlp_layers);
    
    // Free RMSNorm buffers
    for (int i = 0; i < transformer->num_layers; i++) {
        cudaFree(transformer->d_norm_attn_inputs[i]);
        cudaFree(transformer->d_norm_mlp_inputs[i]);
    }
    free(transformer->d_norm_attn_inputs);
    free(transformer->d_norm_mlp_inputs);
    
    free(transformer);
}

// CUDA kernel for in-place residual connection: a += b
__global__ static void residual_add_kernel(half* a, half* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

// CUDA kernel for RMSNorm: y = x / sqrt(mean(x^2) + eps)
__global__ void rmsnorm_forward_kernel(half* output, const half* input, int batch_size, int seq_len, int d_model) {
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
__global__ void rmsnorm_backward_kernel(half* grad_input, const half* grad_output, const half* input, int batch_size, int seq_len, int d_model) {
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
    
    // Reduce sum_dy_x (use smem[8..15] to avoid overwriting smem[0])
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

// Forward pass
void forward_pass_transformer(Transformer* transformer, half* d_X) {
    // Process each layer sequentially
    for (int layer = 0; layer < transformer->num_layers; layer++) {
        half* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;
        
        // Step 1: RMSNorm before attention
        rmsnorm_forward_kernel<<<transformer->batch_size * transformer->seq_len, 256, 8 * sizeof(float)>>>(
            transformer->d_norm_attn_inputs[layer],
            layer_input,
            transformer->batch_size,
            transformer->seq_len,
            transformer->d_model
        );
        
        // Step 2: Attention layer
        forward_pass_attention(transformer->attention_layers[layer], transformer->d_norm_attn_inputs[layer]);
        
        // Step 3: First residual connection - attention_output += input
        residual_add_kernel<<<(transformer->batch_size * transformer->seq_len * transformer->d_model + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_output, 
            layer_input, 
            transformer->batch_size * transformer->seq_len * transformer->d_model
        );
        
        // Step 4: RMSNorm before MLP
        rmsnorm_forward_kernel<<<transformer->batch_size * transformer->seq_len, 256, 8 * sizeof(float)>>>(
            transformer->d_norm_mlp_inputs[layer],
            transformer->attention_layers[layer]->d_output,
            transformer->batch_size,
            transformer->seq_len,
            transformer->d_model
        );
        
        // Step 5: MLP layer
        forward_pass_mlp(transformer->mlp_layers[layer], transformer->d_norm_mlp_inputs[layer]);
        
        // Step 6: Second residual connection - mlp_output += attention_output
        residual_add_kernel<<<(transformer->batch_size * transformer->seq_len * transformer->d_model + 255) / 256, 256>>>(
            transformer->mlp_layers[layer]->d_output, 
            transformer->attention_layers[layer]->d_output, 
            transformer->batch_size * transformer->seq_len * transformer->d_model
        );
    }
}

// Calculate loss
float calculate_loss_transformer(Transformer* transformer, half* d_y) {
    return calculate_loss_mlp(transformer->mlp_layers[transformer->num_layers - 1], d_y);
}

// Zero gradients
void zero_gradients_transformer(Transformer* transformer) {
    for (int i = 0; i < transformer->num_layers; i++) {
        zero_gradients_attention(transformer->attention_layers[i]);
        zero_gradients_mlp(transformer->mlp_layers[i]);
    }
}

// Backward pass
void backward_pass_transformer(Transformer* transformer, half* d_X, half* d_grad_X) {
    // Process layers in reverse order
    for (int layer = transformer->num_layers - 1; layer >= 0; layer--) {
        half* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;
        half* layer_grad_input = (layer == 0) ? d_grad_X : transformer->mlp_layers[layer-1]->d_grad_output;
        
        // Step 1: Backward through MLP
        backward_pass_mlp(
            transformer->mlp_layers[layer], 
            transformer->d_norm_mlp_inputs[layer], 
            transformer->d_norm_mlp_inputs[layer]
        );
        
        // Step 2: Backward through RMSNorm before MLP
        rmsnorm_backward_kernel<<<transformer->batch_size * transformer->seq_len, 256, 16 * sizeof(float)>>>(
            transformer->attention_layers[layer]->d_grad_output, 
            transformer->d_norm_mlp_inputs[layer], 
            transformer->attention_layers[layer]->d_output, 
            transformer->batch_size, 
            transformer->seq_len, 
            transformer->d_model
        );
        
        // Step 3: Add gradient from second residual connection (mlp_output += attention_output)
        residual_add_kernel<<<(transformer->batch_size * transformer->seq_len * transformer->d_model + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_grad_output, 
            transformer->mlp_layers[layer]->d_grad_output, 
            transformer->batch_size * transformer->seq_len * transformer->d_model
        );
        
        // Step 4: Backward through attention
        backward_pass_attention(
            transformer->attention_layers[layer], 
            transformer->d_norm_attn_inputs[layer], 
            transformer->d_norm_attn_inputs[layer]
        );
        
        // Step 5: Backward through RMSNorm before attention
        if (layer_grad_input != NULL) {
            rmsnorm_backward_kernel<<<transformer->batch_size * transformer->seq_len, 256, 16 * sizeof(float)>>>(
                layer_grad_input, 
                transformer->d_norm_attn_inputs[layer], 
                layer_input, 
                transformer->batch_size, 
                transformer->seq_len, 
                transformer->d_model
            );
            
            // Step 6: Add gradient from first residual connection (attention_output += input)
            residual_add_kernel<<<(transformer->batch_size * transformer->seq_len * transformer->d_model + 255) / 256, 256>>>(
                layer_grad_input, 
                transformer->attention_layers[layer]->d_grad_output, 
                transformer->batch_size * transformer->seq_len * transformer->d_model
            );
        }
    }
}

// Update weights for all components
void update_weights_transformer(Transformer* transformer, float learning_rate, int batch_size) {
    for (int i = 0; i < transformer->num_layers; i++) {
        update_weights_attention(transformer->attention_layers[i], learning_rate, batch_size);
        update_weights_mlp(transformer->mlp_layers[i], learning_rate, batch_size);
    }
}

// Reset optimizer state
void reset_optimizer_transformer(Transformer* transformer) {
    for (int i = 0; i < transformer->num_layers; i++) {
        reset_optimizer_attention(transformer->attention_layers[i]);
        reset_optimizer_mlp(transformer->mlp_layers[i]);
    }
}

// Serialize transformer to a file
void serialize_transformer(Transformer* transformer, FILE* file) {
    // Write dimensions
    fwrite(&transformer->d_model, sizeof(int), 1, file);
    fwrite(&transformer->hidden_dim, sizeof(int), 1, file);
    fwrite(&transformer->num_layers, sizeof(int), 1, file);
    
    // Write attention flags
    fwrite(&transformer->attention_layers[0]->is_causal, sizeof(bool), 1, file);
    fwrite(&transformer->attention_layers[0]->use_rope, sizeof(bool), 1, file);
    
    // Serialize all layers
    for (int i = 0; i < transformer->num_layers; i++) {
        serialize_attention(transformer->attention_layers[i], file);
        serialize_mlp(transformer->mlp_layers[i], file);
    }
}

// Deserialize transformer from a file
Transformer* deserialize_transformer(FILE* file, int batch_size, int seq_len, cublasLtHandle_t cublaslt_handle) {
    // Read dimensions
    int d_model, hidden_dim, num_layers;
    bool is_causal, use_rope;
    fread(&d_model, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope, sizeof(bool), 1, file);
    
    // Initialize transformer
    Transformer* transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, is_causal, use_rope, cublaslt_handle);
    
    // Deserialize all layers
    for (int i = 0; i < num_layers; i++) {
        // Free the initialized components
        free_attention(transformer->attention_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
        
        // Deserialize the saved components
        transformer->attention_layers[i] = deserialize_attention(file, batch_size, seq_len, 8, cublaslt_handle);
        transformer->mlp_layers[i] = deserialize_mlp(file, batch_size * seq_len, cublaslt_handle);
    }
    
    return transformer;
}
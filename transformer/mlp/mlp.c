#include "mlp.h"

// Initialize the MLP
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size, cublasLtHandle_t cublaslt_handle) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    
    // Store dimensions
    mlp->input_dim = input_dim;
    mlp->hidden_dim = hidden_dim;
    mlp->output_dim = output_dim;
    mlp->batch_size = batch_size;
    
    // Initialize Adam parameters
    mlp->beta1 = 0.9f;
    mlp->beta2 = 0.999f;
    mlp->epsilon = 1e-8f;
    mlp->t = 0;
    mlp->weight_decay = 0.01f;
    
    // Initialize cuBLASLt
    mlp->cublaslt_handle = cublaslt_handle;
    
    size_t w1_size = input_dim * hidden_dim;
    size_t w2_size = hidden_dim * output_dim;
    size_t hidden_buffer_size = batch_size * hidden_dim;
    size_t output_buffer_size = batch_size * output_dim;
    
    // Allocate host memory for weight initialization
    half* h_W1 = (half*)malloc(w1_size * sizeof(half));
    half* h_W2 = (half*)malloc(w2_size * sizeof(half));
    
    // Initialize weights on host
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 1.0f / sqrtf(hidden_dim);
    
    for (size_t i = 0; i < w1_size; i++) {
        h_W1[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1);
    }
    
    for (size_t i = 0; i < w2_size; i++) {
        h_W2[i] = __float2half(((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2);
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&mlp->d_W1, w1_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2, w2_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_grad, w1_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_grad, w2_size * sizeof(half)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_m, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_v, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_m, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_v, w2_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&mlp->d_preact, hidden_buffer_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&mlp->d_postact, hidden_buffer_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&mlp->d_output, output_buffer_size * sizeof(half)));
    
    // Alias device memory for backward pass buffers
    mlp->d_grad_postact = mlp->d_postact;
    mlp->d_grad_output = mlp->d_output;
    
    // Allocate single device float for loss computation
    CHECK_CUDA(cudaMalloc(&mlp->d_loss_result, sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(mlp->d_W1, h_W1, w1_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2, h_W2, w2_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(mlp->d_W1_m, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W1_v, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_m, 0, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_v, 0, w2_size * sizeof(float)));
    
    // Create cuBLASLt matrix multiplication descriptor
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&mlp->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    
    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Create matrix layout descriptors
    // W1 and W1_grad: [input_dim x hidden_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->W1_layout, CUDA_R_16F, input_dim, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->W1_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // W2 and W2_grad: [hidden_dim x output_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->W2_layout, CUDA_R_16F, hidden_dim, output_dim, output_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->W2_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // X and grad_X: [batch_size x input_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->batch_input_layout, CUDA_R_16F, batch_size, input_dim, input_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->batch_input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // H, S, grad_hidden, layer_preact, layer_postact: [batch_size x hidden_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->batch_hidden_layout, CUDA_R_16F, batch_size, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->batch_hidden_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Y, grad_output, layer_output: [batch_size x output_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->batch_output_layout, CUDA_R_16F, batch_size, output_dim, output_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->batch_output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Free host memory
    free(h_W1); free(h_W2);
    
    return mlp;
}

// Free MLP memory
void free_mlp(MLP* mlp) {
    // Destroy cuBLASLt descriptor
    cublasLtMatmulDescDestroy(mlp->matmul_desc);
    
    // Destroy matrix layouts
    cublasLtMatrixLayoutDestroy(mlp->W1_layout);
    cublasLtMatrixLayoutDestroy(mlp->W2_layout);
    cublasLtMatrixLayoutDestroy(mlp->batch_input_layout);
    cublasLtMatrixLayoutDestroy(mlp->batch_hidden_layout);
    cublasLtMatrixLayoutDestroy(mlp->batch_output_layout);
    
    // Free device memory
    cudaFree(mlp->d_W1); cudaFree(mlp->d_W2);
    cudaFree(mlp->d_W1_grad); cudaFree(mlp->d_W2_grad);
    cudaFree(mlp->d_W1_m); cudaFree(mlp->d_W1_v);
    cudaFree(mlp->d_W2_m); cudaFree(mlp->d_W2_v);
    cudaFree(mlp->d_preact); cudaFree(mlp->d_postact);
    cudaFree(mlp->d_output);
    
    // Free loss computation buffer
    cudaFree(mlp->d_loss_result);
    
    free(mlp);
}

// CUDA kernel for Swish activation
__global__ static void swish_forward_kernel_mlp(half* output, half* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = __half2float(pre_activation[idx]);
        output[idx] = __float2half(h / (1.0f + expf(-h)));
    }
}

// CUDA kernel for Swish derivative
__global__ static void swish_backward_kernel_mlp(half* grad_hidden, half* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = __half2float(pre_activation[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-h));
        float grad = __half2float(grad_hidden[idx]);
        grad_hidden[idx] = __float2half(grad * (sigmoid + h * sigmoid * (1.0f - sigmoid)));
    }
}

// Forward pass
void forward_pass_mlp(MLP* mlp, half* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // H = XW₁
    LT_MATMUL(mlp, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, mlp->batch_input_layout,
              mlp->d_W1, mlp->W1_layout,
              &beta, mlp->d_preact, mlp->batch_hidden_layout);

    // S = H⊙σ(H)
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_postact,
        mlp->d_preact,
        mlp->batch_size * mlp->hidden_dim
    );

    // Y = SW₂
    LT_MATMUL(mlp, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              mlp->d_postact, mlp->batch_hidden_layout,
              mlp->d_W2, mlp->W2_layout,
              &beta, mlp->d_output, mlp->batch_output_layout);
}

// CUDA kernel for computing loss and gradient
__global__ static void compute_loss_and_gradient_kernel_mlp(half* grad_output, half* predictions, half* targets, float* loss_result, int size) {
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
float calculate_loss_mlp(MLP* mlp, half* d_y) {
    // ∂L/∂Y = Y - Y_true
    int total_elements = mlp->batch_size * mlp->output_dim;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Reset loss accumulator to zero
    CHECK_CUDA(cudaMemset(mlp->d_loss_result, 0, sizeof(float)));
    
    // Compute gradient and accumulate loss
    compute_loss_and_gradient_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_grad_output, mlp->d_output, d_y, mlp->d_loss_result, total_elements
    );
    
    // Copy result back to host
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, mlp->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / total_elements;
}

// Zero gradients
void zero_gradients_mlp(MLP* mlp) {
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    CHECK_CUDA(cudaMemset(mlp->d_W1_grad, 0, w1_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_grad, 0, w2_size * sizeof(half)));
}

// Backward pass
void backward_pass_mlp(MLP* mlp, half* d_X, half* d_grad_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // ∂L/∂W₂ = Sᵀ(∂L/∂Y)
    LT_MATMUL(mlp, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              mlp->d_postact, mlp->batch_hidden_layout,
              mlp->d_grad_output, mlp->batch_output_layout,
              &alpha, mlp->d_W2_grad, mlp->W2_layout);

    // ∂L/∂S = (∂L/∂Y)W₂ᵀ
    LT_MATMUL(mlp, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              mlp->d_grad_output, mlp->batch_output_layout,
              mlp->d_W2, mlp->W2_layout,
              &beta, mlp->d_grad_postact, mlp->batch_hidden_layout);

    // ∂L/∂H = ∂L/∂S⊙[σ(H)+H⊙σ(H)⊙(1-σ(H))]
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_backward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_grad_postact,
        mlp->d_preact,
        mlp->batch_size * mlp->hidden_dim
    );

    // ∂L/∂W₁ = Xᵀ(∂L/∂H)
    LT_MATMUL(mlp, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, mlp->batch_input_layout,
              mlp->d_grad_postact, mlp->batch_hidden_layout,
              &alpha, mlp->d_W1_grad, mlp->W1_layout);
    
    if (d_grad_X != NULL) {
        // ∂L/∂X = (∂L/∂H)W₁ᵀ
        LT_MATMUL(mlp, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  mlp->d_grad_postact, mlp->batch_hidden_layout,
                  mlp->d_W1, mlp->W1_layout,
                  &beta, d_grad_X, mlp->batch_input_layout);
    }
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_mlp(half* weight, half* grad, float* m, float* v,
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
void update_weights_mlp(MLP* mlp, float learning_rate, int batch_size) {
    mlp->t++;
    
    float beta1_t = powf(mlp->beta1, mlp->t);
    float beta2_t = powf(mlp->beta2, mlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    // Update W₁ weights
    int W1_blocks = (w1_size + block_size - 1) / block_size;
    adamw_update_kernel_mlp<<<W1_blocks, block_size>>>(
        mlp->d_W1, mlp->d_W1_grad, mlp->d_W1_m, mlp->d_W1_v,
        mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
        alpha_t, w1_size, batch_size
    );
    
    // Update W₂ weights
    int W2_blocks = (w2_size + block_size - 1) / block_size;
    adamw_update_kernel_mlp<<<W2_blocks, block_size>>>(
        mlp->d_W2, mlp->d_W2_grad, mlp->d_W2_m, mlp->d_W2_v,
        mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
        alpha_t, w2_size, batch_size
    );
}

// Reset optimizer state
void reset_optimizer_mlp(MLP* mlp) {
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    // Reset Adam moment estimates to zero on device
    CHECK_CUDA(cudaMemset(mlp->d_W1_m, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W1_v, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_m, 0, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_v, 0, w2_size * sizeof(float)));
    
    // Reset time step
    mlp->t = 0;
}

// Serialize MLP to a file
void serialize_mlp(MLP* mlp, FILE* file) {
    // Write dimensions
    fwrite(&mlp->input_dim, sizeof(int), 1, file);
    fwrite(&mlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&mlp->output_dim, sizeof(int), 1, file);
    
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    // Allocate host buffers for weights
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    
    // Copy weights from device
    CHECK_CUDA(cudaMemcpy(h_W1, mlp->d_W1, w1_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2, mlp->d_W2, w2_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Convert half to float
    for (int i = w1_size - 1; i >= 0; i--) h_W1[i] = __half2float(((half*)h_W1)[i]);
    for (int i = w2_size - 1; i >= 0; i--) h_W2[i] = __half2float(((half*)h_W2)[i]);
    
    // Write weights
    fwrite(h_W1, sizeof(float), w1_size, file);
    fwrite(h_W2, sizeof(float), w2_size, file);
    
    free(h_W1); free(h_W2);
    
    // Write optimizer state
    fwrite(&mlp->t, sizeof(int), 1, file);
    
    // Allocate host buffers for optimizer state
    float* h_W1_m = (float*)malloc(w1_size * sizeof(float));
    float* h_W1_v = (float*)malloc(w1_size * sizeof(float));
    float* h_W2_m = (float*)malloc(w2_size * sizeof(float));
    float* h_W2_v = (float*)malloc(w2_size * sizeof(float));
    
    // Copy optimizer state from device
    CHECK_CUDA(cudaMemcpy(h_W1_m, mlp->d_W1_m, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W1_v, mlp->d_W1_v, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_m, mlp->d_W2_m, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_v, mlp->d_W2_v, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write optimizer state
    fwrite(h_W1_m, sizeof(float), w1_size, file);
    fwrite(h_W1_v, sizeof(float), w1_size, file);
    fwrite(h_W2_m, sizeof(float), w2_size, file);
    fwrite(h_W2_v, sizeof(float), w2_size, file);
    
    // Free host buffers
    free(h_W1_m); free(h_W1_v);
    free(h_W2_m); free(h_W2_v);
}

// Deserialize MLP from a file
MLP* deserialize_mlp(FILE* file, int batch_size, cublasLtHandle_t cublaslt_handle) {
    // Read dimensions
    int input_dim, hidden_dim, output_dim;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    
    // Initialize MLP
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, cublaslt_handle);
    
    int w1_size = input_dim * hidden_dim;
    int w2_size = hidden_dim * output_dim;
    
    // Allocate host buffers for weights
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    
    // Read weights
    fread(h_W1, sizeof(float), w1_size, file);
    fread(h_W2, sizeof(float), w2_size, file);
    
    // Convert float to half
    for (int i = 0; i < w1_size; i++) ((half*)h_W1)[i] = __float2half(h_W1[i]);
    for (int i = 0; i < w2_size; i++) ((half*)h_W2)[i] = __float2half(h_W2[i]);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(mlp->d_W1, h_W1, w1_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2, h_W2, w2_size * sizeof(half), cudaMemcpyHostToDevice));
    
    free(h_W1); free(h_W2);
    
    // Read optimizer state
    fread(&mlp->t, sizeof(int), 1, file);
    
    // Allocate host buffers for optimizer state
    float* h_W1_m = (float*)malloc(w1_size * sizeof(float));
    float* h_W1_v = (float*)malloc(w1_size * sizeof(float));
    float* h_W2_m = (float*)malloc(w2_size * sizeof(float));
    float* h_W2_v = (float*)malloc(w2_size * sizeof(float));
    
    // Read optimizer state
    fread(h_W1_m, sizeof(float), w1_size, file);
    fread(h_W1_v, sizeof(float), w1_size, file);
    fread(h_W2_m, sizeof(float), w2_size, file);
    fread(h_W2_v, sizeof(float), w2_size, file);
    
    // Copy optimizer state to device
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_m, h_W1_m, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_v, h_W1_v, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_m, h_W2_m, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_v, h_W2_v, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free host buffers
    free(h_W1_m); free(h_W1_v);
    free(h_W2_m); free(h_W2_v);
    
    return mlp;
}
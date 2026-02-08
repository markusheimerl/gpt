#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <cblas.h>
#include "transformer.h"

void generate_data(float** X, float** y, int seq_len, int num_samples, int d_model,
                   float range_min, float range_max) {
    // Row-major layout: [num_samples x seq_len x d_model]
    const int total = num_samples * seq_len * d_model;
    
    *X = (float*)malloc(total * sizeof(float));
    *y = (float*)malloc(total * sizeof(float));
    
    // Fill X with random data
    float range = range_max - range_min;
    for (int i = 0; i < total; i++) {
        (*X)[i] = range_min + ((float)rand() / (float)RAND_MAX) * range;
    }
    
    // Create attention matrix A: [seq_len × seq_len] - this is shared across all samples
    float* A = (float*)malloc(seq_len * seq_len * sizeof(float));
    float a_scale = 1.0f / sqrtf(seq_len);
    
    for (int i = 0; i < seq_len * seq_len; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * a_scale;
    }
    
    // Row-wise softmax on A to create proper attention weights
    for (int i = 0; i < seq_len; i++) {
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; j++) {
            float v = A[i * seq_len + j];
            if (v > max_val) max_val = v;
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float exp_val = expf(A[i * seq_len + j] - max_val);
            A[i * seq_len + j] = exp_val;
            sum += exp_val;
        }
        
        for (int j = 0; j < seq_len; j++) {
            A[i * seq_len + j] /= sum;
        }
    }
    
    // Create MLP-like transformation matrix W: [d_model × d_model] - this is also shared
    float* W = (float*)malloc(d_model * d_model * sizeof(float));
    float w_scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < d_model * d_model; i++) {
        W[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale;
    }
    
    // Allocate temporary buffer for attention output
    float* attn_output = (float*)malloc(total * sizeof(float));
    
    // Step 1: Apply attention transformation for each batch: attn_output = A * X
    for (int b = 0; b < num_samples; b++) {
        float* X_b = &(*X)[b * seq_len * d_model];
        float* attn_b = &attn_output[b * seq_len * d_model];
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, d_model, seq_len,
                    1.0f, A, seq_len,
                    X_b, d_model,
                    0.0f, attn_b, d_model);
    }
    
    // Step 2: First residual connection: attn_output += X
    cblas_saxpy(total, 1.0f, *X, 1, attn_output, 1);
    
    // Step 3: Apply MLP transformation: y = attn_output * W
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                num_samples * seq_len, d_model, d_model,
                1.0f, attn_output, d_model,
                W, d_model,
                0.0f, *y, d_model);
    
    // Step 4: Second residual connection: y += attn_output
    cblas_saxpy(total, 1.0f, attn_output, 1, *y, 1);
    
    // Add small noise to make the problem more realistic
    float noise_scale = range * 0.001f;
    for (int i = 0; i < total; i++) {
        float noise = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * noise_scale;
        (*y)[i] += noise;
    }
    
    free(A);
    free(W);
    free(attn_output);
    
    printf("Generated transformer data: %d samples, seq_len %d, d_model %d\n", 
           num_samples, seq_len, d_model);
}

void save_data(float* X, float* y, int seq_len, int num_samples, int d_model,
               const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: cannot write %s\n", filename);
        return;
    }
    
    // Header: batch_id, seq_pos, then features, then targets
    fprintf(f, "batch_id,seq_pos,");
    for (int d = 0; d < d_model; d++) {
        fprintf(f, "x_d%d,", d);
    }
    for (int d = 0; d < d_model; d++) {
        fprintf(f, "y_d%d%s", d, d == d_model-1 ? "\n" : ",");
    }
    
    // Data: one row per (batch, sequence_position)
    for (int b = 0; b < num_samples; b++) {
        for (int t = 0; t < seq_len; t++) {
            fprintf(f, "%d,%d,", b, t);
            
            // X features for this (batch, position)
            for (int d = 0; d < d_model; d++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                fprintf(f, "%.6f,", X[idx]);
            }
            
            // Y features for this (batch, position)
            for (int d = 0; d < d_model; d++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                fprintf(f, "%.6f%s", y[idx], d == d_model-1 ? "\n" : ",");
            }
        }
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}

int main() {
    srand(time(NULL));

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Parameters
    const int seq_len = 128;
    const int d_model = 64;
    const int hidden_dim = 256;
    const int num_samples = 1024;
    const int batch_size = 32;
    const int num_layers = 3;
    
    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, seq_len, num_samples, d_model, -5.0f, 5.0f);

    // Convert synthetic data to half precision
    half *h_X = (half*)malloc(num_samples * seq_len * d_model * sizeof(half));
    half *h_y = (half*)malloc(num_samples * seq_len * d_model * sizeof(half));
    for (int i = 0; i < num_samples * seq_len * d_model; i++) h_X[i] = __float2half(X[i]);
    for (int i = 0; i < num_samples * seq_len * d_model; i++) h_y[i] = __float2half(y[i]);
    
    // Initialize transformer
    Transformer* transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, false, false, cublaslt_handle);
    
    // Training parameters
    const int num_epochs = 50;
    const float learning_rate = 0.0003f;
    const int num_batches = num_samples / batch_size;
    
    // Allocate device memory for batch data
    half *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_size * seq_len * d_model * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_size * seq_len * d_model * sizeof(half)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Calculate batch offset
            int batch_offset = batch * batch_size * seq_len * d_model;

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_X, &h_X[batch_offset], batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &h_y[batch_offset], batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_transformer(transformer, d_X);
            
            // Calculate loss
            float loss = calculate_loss_transformer(transformer, d_y);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_transformer(transformer);
            backward_pass_transformer(transformer, d_X, NULL);
            
            // Update weights
            update_weights_transformer(transformer, learning_rate, batch_size);
        }
        
        epoch_loss /= num_batches;

        // Print progress
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_transformer.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_transformer_data.csv", localtime(&now));

    // Save model
    FILE* model_file = fopen(model_fname, "wb");
    serialize_transformer(transformer, model_file);
    fclose(model_file);
    printf("Model saved to %s\n", model_fname);
    
    // Save data
    save_data(X, y, seq_len, num_samples, d_model, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    model_file = fopen(model_fname, "rb");
    Transformer* loaded_transformer = deserialize_transformer(model_file, batch_size, seq_len, cublaslt_handle);
    fclose(model_file);
    printf("Model loaded from %s\n", model_fname);

    // Forward pass with loaded model on first batch
    CHECK_CUDA(cudaMemcpy(d_X, h_X, batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));
    forward_pass_transformer(loaded_transformer, d_X);
    
    // Copy predictions back to host
    half* h_output = (half*)malloc(batch_size * seq_len * d_model * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_output, loaded_transformer->mlp_layers[loaded_transformer->num_layers - 1]->d_output, batch_size * seq_len * d_model * sizeof(half), cudaMemcpyDeviceToHost));

    // Evaluate model performance on first batch
    printf("Feature\tR²\t\tMAE\t\tSample Predictions\n");
    printf("-------\t--------\t--------\t--------------------------------\n");

    for (int d = 0; d < d_model; d++) {
        // Calculate mean for R² across all positions and batches for this feature
        float y_mean = 0.0f;
        int total_elements = batch_size * seq_len;
        
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                y_mean += __half2float(h_y[idx]);
            }
        }
        y_mean /= total_elements;
        
        // Calculate R² and MAE for this feature
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                float pred = __half2float(h_output[idx]);
                float actual = __half2float(h_y[idx]);
                float diff = pred - actual;
                
                ss_res += diff * diff;
                ss_tot += (actual - y_mean) * (actual - y_mean);
                mae += fabs(diff);
            }
        }
        
        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= total_elements;
        
        // Print summary with sample predictions from first batch, first few positions
        printf("d%d\t%.6f\t%.3f\t\t", d, r2, mae);
        for (int sample = 0; sample < 3; sample++) {
            // Show predictions from batch 0, positions 0, 1, 2
            int idx = 0 * seq_len * d_model + sample * d_model + d;
            float pred = __half2float(h_output[idx]);
            float actual = __half2float(h_y[idx]);
            printf("%.2f/%.2f ", pred, actual);
        }
        printf("\n");
    }
    
    // Cleanup
    free(X);
    free(y);
    free(h_X);
    free(h_y);
    free(h_output);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_y));
    free_transformer(transformer);
    free_transformer(loaded_transformer);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}
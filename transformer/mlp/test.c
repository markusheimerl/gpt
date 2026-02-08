#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <cblas.h>
#include "mlp.h"

void generate_data(float** X, float** y, int num_samples, int input_dim, int output_dim,
                   float range_min, float range_max) {
    int total_x = num_samples * input_dim;
    int total_y = num_samples * output_dim;
    
    *X = (float*)malloc(total_x * sizeof(float));
    *y = (float*)malloc(total_y * sizeof(float));
    
    // Generate input data in row-major format: [num_samples × input_dim]
    for (int sample = 0; sample < num_samples; sample++) {
        for (int feature = 0; feature < input_dim; feature++) {
            (*X)[sample * input_dim + feature] = range_min + 
                ((float)rand() / (float)RAND_MAX) * (range_max - range_min);
        }
    }
    
    // Create random transformation matrix W: [input_dim × output_dim]
    float* W = (float*)malloc(input_dim * output_dim * sizeof(float));
    float w_scale = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < input_dim * output_dim; i++) {
        W[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale;
    }
    
    // Transform input data: y = X * W
    // Using row-major BLAS: y[num_samples × output_dim] = X[num_samples × input_dim] * W[input_dim × output_dim]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                num_samples, output_dim, input_dim,
                1.0f, *X, input_dim,
                W, output_dim,
                0.0f, *y, output_dim);
    
    // Add noise to outputs
    for (int i = 0; i < total_y; i++) {
        float noise = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale;
        (*y)[i] += noise;
    }
    
    // Free temporary transformation matrix
    free(W);
    
    printf("Generated regression data: %d samples, %d inputs, %d outputs (with %.3f noise)\n", 
           num_samples, input_dim, output_dim, w_scale);
}

void save_data(float* X, float* y, int num_samples, int input_dim, int output_dim,
               const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) { 
        printf("Error: cannot write %s\n", filename); 
        return; 
    }
    
    // Header
    for (int i = 0; i < input_dim; i++) {
        fprintf(f, "x%d,", i);
    }
    for (int i = 0; i < output_dim; i++) {
        fprintf(f, "y%d%s", i, i == output_dim-1 ? "\n" : ",");
    }
    
    // Data
    for (int s = 0; s < num_samples; s++) {
        for (int i = 0; i < input_dim; i++) {
            fprintf(f, "%.6f,", X[s * input_dim + i]);
        }
        for (int i = 0; i < output_dim; i++) {
            fprintf(f, "%.6f%s", y[s * output_dim + i], i == output_dim-1 ? "\n" : ",");
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
    const int input_dim = 16;
    const int hidden_dim = 256;
    const int output_dim = 16;
    const int num_samples = 524288;
    const int batch_size = 512;
    
    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, num_samples, input_dim, output_dim, -30.0f, 30.0f);

    // Convert synthetic data to half precision
    half *h_X = (half*)malloc(num_samples * input_dim * sizeof(half));
    half *h_y = (half*)malloc(num_samples * output_dim * sizeof(half));
    for (int i = 0; i < num_samples * input_dim; i++) h_X[i] = __float2half(X[i]);
    for (int i = 0; i < num_samples * output_dim; i++) h_y[i] = __float2half(y[i]);

    // Initialize network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, cublaslt_handle);
    
    // Training parameters
    const int num_epochs = 10;
    const float learning_rate = 0.0003f;
    const int num_batches = num_samples / batch_size;
    
    // Allocate device memory for batch data
    half *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_size * input_dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_size * output_dim * sizeof(half)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_X, &h_X[batch * batch_size * input_dim], batch_size * input_dim * sizeof(half), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &h_y[batch * batch_size * output_dim], batch_size * output_dim * sizeof(half), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_mlp(mlp, d_X);
            
            // Calculate loss
            float loss = calculate_loss_mlp(mlp, d_y);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_mlp(mlp);
            backward_pass_mlp(mlp, d_X, NULL);
            
            // Update weights
            update_weights_mlp(mlp, learning_rate, batch_size);
        }
        
        epoch_loss /= num_batches;

        // Print progress
        if (epoch % 2 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    // Save model
    FILE* model_file = fopen(model_fname, "wb");
    serialize_mlp(mlp, model_file);
    fclose(model_file);
    printf("Model saved to %s\n", model_fname);
    
    // Save data
    save_data(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    model_file = fopen(model_fname, "rb");
    MLP* loaded_mlp = deserialize_mlp(model_file, batch_size, cublaslt_handle);
    fclose(model_file);
    printf("Model loaded from %s\n", model_fname);
    
    // Forward pass with loaded model on first batch
    CHECK_CUDA(cudaMemcpy(d_X, h_X, batch_size * input_dim * sizeof(half), cudaMemcpyHostToDevice));
    forward_pass_mlp(loaded_mlp, d_X);
    
    // Copy predictions back to host
    half* h_output = (half*)malloc(batch_size * output_dim * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_output, loaded_mlp->d_output, batch_size * output_dim * sizeof(half), cudaMemcpyDeviceToHost));

    // Evaluate model performance on first batch
    printf("Output\tR²\t\tMAE\t\tSample Predictions\n");
    printf("------\t--------\t--------\t--------------------------------\n");

    for (int i = 0; i < output_dim; i++) {
        // Calculate mean for R²
        float y_mean = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            y_mean += __half2float(h_y[j * output_dim + i]);
        }
        y_mean /= batch_size;
        
        // Calculate R² and MAE
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            float pred = __half2float(h_output[j * output_dim + i]);
            float actual = __half2float(h_y[j * output_dim + i]);
            float diff = pred - actual;
            
            ss_res += diff * diff;
            ss_tot += (actual - y_mean) * (actual - y_mean);
            mae += fabs(diff);
        }
        
        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= batch_size;
        
        // Print summary
        printf("y%d\t%.6f\t%.3f\t\t", i, r2, mae);
        for (int j = 0; j < 3; j++) {
            float pred = __half2float(h_output[j * output_dim + i]);
            float actual = __half2float(h_y[j * output_dim + i]);
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
    free_mlp(mlp);
    free_mlp(loaded_mlp);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}
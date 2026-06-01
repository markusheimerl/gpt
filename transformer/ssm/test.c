#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <cblas.h>
#include "ssm.h"

void generate_data(float** X, float** y, int seq_len, int num_samples, int d_model,
                   float range_min, float range_max) {
    int total = num_samples * seq_len * d_model;

    *X = (float*)malloc(total * sizeof(float));
    *y = (float*)malloc(total * sizeof(float));

    // Generate input data in row-major format: [num_samples × seq_len × d_model]
    float range = range_max - range_min;
    for (int i = 0; i < total; i++) {
        (*X)[i] = range_min + ((float)rand() / (float)RAND_MAX) * range;
    }

    // Per-channel decay coefficient r in [0.7, 0.95]
    float* r = (float*)malloc(d_model * sizeof(float));
    for (int d = 0; d < d_model; d++) {
        r[d] = 0.7f + 0.25f * ((float)rand() / (float)RAND_MAX);
    }

    // Causal per-channel linear filter (representable by a diagonal SSM)
    for (int b = 0; b < num_samples; b++) {
        for (int d = 0; d < d_model; d++) {
            float state = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                state = r[d] * state + (*X)[idx];
                (*y)[idx] = state;
            }
        }
    }

    free(r);

    // Add noise to outputs
    float noise_scale = range * 0.001f;
    for (int i = 0; i < total; i++) {
        (*y)[i] += ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * noise_scale;
    }

    printf("Generated SSM data: %d samples, seq_len %d, d_model %d (with %.4f noise)\n",
           num_samples, seq_len, d_model, noise_scale);
}

int main() {
    srand(time(NULL));

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Parameters
    const int seq_len = 128;
    const int d_model = 64;
    const int state_dim = 8;
    const int num_samples = 1024;
    const int batch_size = 32;

    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, seq_len, num_samples, d_model, -2.0f, 2.0f);

    // Convert synthetic data to half precision
    half *h_X = (half*)malloc(num_samples * seq_len * d_model * sizeof(half));
    half *h_y = (half*)malloc(num_samples * seq_len * d_model * sizeof(half));
    for (int i = 0; i < num_samples * seq_len * d_model; i++) h_X[i] = __float2half(X[i]);
    for (int i = 0; i < num_samples * seq_len * d_model; i++) h_y[i] = __float2half(y[i]);

    // Initialize network
    SSM* ssm = init_ssm(seq_len, d_model, state_dim, batch_size, cublaslt_handle);

    // Training parameters
    const int num_epochs = 50;
    const float learning_rate = 0.001f;
    const int num_batches = num_samples / batch_size;

    // Allocate device memory for batch data
    half *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_size * seq_len * d_model * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_size * seq_len * d_model * sizeof(half)));

    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int off = batch * batch_size * seq_len * d_model;

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_X, &h_X[off], batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &h_y[off], batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));

            // Forward pass
            forward_pass_ssm(ssm, d_X);

            // Calculate loss
            float loss = calculate_loss_ssm(ssm, d_y);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_ssm(ssm);
            backward_pass_ssm(ssm, d_X, NULL);

            // Update weights
            update_weights_ssm(ssm, learning_rate, batch_size);
        }

        epoch_loss /= num_batches;

        // Print progress
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }

    // Get timestamp for filename
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_ssm.bin", localtime(&now));

    // Save model
    FILE* model_file = fopen(model_fname, "wb");
    serialize_ssm(ssm, model_file);
    fclose(model_file);
    printf("Model saved to %s\n", model_fname);

    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    model_file = fopen(model_fname, "rb");
    SSM* loaded_ssm = deserialize_ssm(model_file, batch_size, seq_len, cublaslt_handle);
    fclose(model_file);
    printf("Model loaded from %s\n", model_fname);

    // Forward pass with loaded model on first batch
    CHECK_CUDA(cudaMemcpy(d_X, h_X, batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));
    forward_pass_ssm(loaded_ssm, d_X);

    // Copy predictions back to host
    half* h_output = (half*)malloc(batch_size * seq_len * d_model * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_output, loaded_ssm->d_output, batch_size * seq_len * d_model * sizeof(half), cudaMemcpyDeviceToHost));

    // Evaluate model performance on first batch
    printf("Feature\tR²\t\tMAE\t\tSample Predictions\n");
    printf("-------\t--------\t--------\t--------------------------------\n");

    for (int d = 0; d < d_model; d++) {
        // Calculate mean for R²
        float y_mean = 0.0f;
        int N = batch_size * seq_len;
        for (int b = 0; b < batch_size; b++)
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                y_mean += __half2float(h_y[idx]);
            }
        y_mean /= N;

        // Calculate R² and MAE
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int b = 0; b < batch_size; b++)
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                float pred = __half2float(h_output[idx]);
                float actual = __half2float(h_y[idx]);
                float diff = pred - actual;

                ss_res += diff * diff;
                ss_tot += (actual - y_mean) * (actual - y_mean);
                mae += fabsf(diff);
            }

        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= N;

        // Print summary
        printf("d%d\t%.6f\t%.3f\t\t", d, r2, mae);
        for (int s = 0; s < 3; s++) {
            int idx = s * d_model + d;
            printf("%.2f/%.2f ", __half2float(h_output[idx]), __half2float(h_y[idx]));
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
    free_ssm(ssm);
    free_ssm(loaded_ssm);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));

    return 0;
}
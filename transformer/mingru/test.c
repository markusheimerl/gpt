#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <cblas.h>
#include "mingru.h"

// Synthetic data: per-channel causal exponential filter, representable as the
// fixed point of a minGRU layer.
void generate_data(float** X, float** y, int seq_len, int num_samples, int d_model,
                   float range_min, float range_max) {
    int total = num_samples * seq_len * d_model;

    *X = (float*)malloc(total * sizeof(float));
    *y = (float*)malloc(total * sizeof(float));

    float range = range_max - range_min;
    for (int i = 0; i < total; i++) {
        (*X)[i] = range_min + ((float)rand() / (float)RAND_MAX) * range;
    }

    float* r = (float*)malloc(d_model * sizeof(float));
    for (int d = 0; d < d_model; d++) {
        r[d] = 0.7f + 0.25f * ((float)rand() / (float)RAND_MAX);
    }

    for (int b = 0; b < num_samples; b++) {
        for (int d = 0; d < d_model; d++) {
            float state = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                state = r[d] * state + (1.0f - r[d]) * (*X)[idx];
                (*y)[idx] = state;
            }
        }
    }

    free(r);

    float noise_scale = range * 0.001f;
    for (int i = 0; i < total; i++) {
        (*y)[i] += ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * noise_scale;
    }

    printf("Generated minGRU data: %d samples, seq_len %d, d_model %d (with %.4f noise)\n",
           num_samples, seq_len, d_model, noise_scale);
}

int main() {
    srand(time(NULL));

    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    const int seq_len = 128;
    const int d_model = 64;
    const int num_samples = 1024;
    const int batch_size = 32;

    float *X, *y;
    generate_data(&X, &y, seq_len, num_samples, d_model, -2.0f, 2.0f);

    half *h_X = (half*)malloc(num_samples * seq_len * d_model * sizeof(half));
    half *h_y = (half*)malloc(num_samples * seq_len * d_model * sizeof(half));
    for (int i = 0; i < num_samples * seq_len * d_model; i++) h_X[i] = __float2half(X[i]);
    for (int i = 0; i < num_samples * seq_len * d_model; i++) h_y[i] = __float2half(y[i]);

    MinGRU* m = init_mingru(seq_len, d_model, batch_size, cublaslt_handle);

    const int num_epochs = 50;
    const float learning_rate = 0.001f;
    const int num_batches = num_samples / batch_size;

    half *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_size * seq_len * d_model * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_size * seq_len * d_model * sizeof(half)));

    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        for (int batch = 0; batch < num_batches; batch++) {
            int off = batch * batch_size * seq_len * d_model;
            CHECK_CUDA(cudaMemcpy(d_X, &h_X[off], batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &h_y[off], batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));

            forward_pass_mingru(m, d_X);
            float loss = calculate_loss_mingru(m, d_y);
            epoch_loss += loss;

            if (epoch == num_epochs) continue;
            zero_gradients_mingru(m);
            backward_pass_mingru(m, d_X, NULL);
            update_weights_mingru(m, learning_rate, batch_size);
        }
        epoch_loss /= num_batches;
        if (epoch % 5 == 0) printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
    }

    // Round-trip serialization
    char fname[64];
    time_t now = time(NULL);
    strftime(fname, sizeof(fname), "%Y%m%d_%H%M%S_mingru.bin", localtime(&now));
    FILE* fp = fopen(fname, "wb"); serialize_mingru(m, fp); fclose(fp);
    fp = fopen(fname, "rb"); MinGRU* loaded = deserialize_mingru(fp, batch_size, seq_len, cublaslt_handle); fclose(fp);
    printf("Round-trip save/load: %s\n", fname);

    CHECK_CUDA(cudaMemcpy(d_X, h_X, batch_size * seq_len * d_model * sizeof(half), cudaMemcpyHostToDevice));
    forward_pass_mingru(loaded, d_X);

    half* h_out = (half*)malloc(batch_size * seq_len * d_model * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_out, loaded->d_output, batch_size * seq_len * d_model * sizeof(half), cudaMemcpyDeviceToHost));

    float ss_res = 0.0f, ss_tot = 0.0f, y_mean = 0.0f;
    int N = batch_size * seq_len * d_model;
    for (int i = 0; i < N; i++) y_mean += __half2float(h_y[i]);
    y_mean /= N;
    for (int i = 0; i < N; i++) {
        float diff = __half2float(h_out[i]) - __half2float(h_y[i]);
        ss_res += diff * diff;
        float c = __half2float(h_y[i]) - y_mean;
        ss_tot += c * c;
    }
    printf("Loaded model R² = %.6f\n", 1.0f - ss_res / ss_tot);

    free(X); free(y); free(h_X); free(h_y); free(h_out);
    cudaFree(d_X); cudaFree(d_y);
    free_mingru(m); free_mingru(loaded);
    cublasLtDestroy(cublaslt_handle);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <cuda_fp16.h>
#include "gpt.h"

// Get the total size of a file
size_t get_file_size(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fclose(f);
    return size;
}

// Create shuffled sequence indices for entire corpus
size_t* create_shuffled_indices(size_t total_sequences) {
    size_t* indices = (size_t*)malloc(total_sequences * sizeof(size_t));
    
    // Initialize sequentially
    for (size_t i = 0; i < total_sequences; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    for (size_t i = total_sequences - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    return indices;
}

// Sample sequences using shuffled indices
void sample_sequences(const char* filename, size_t* indices, int seq_len, unsigned char* input_tokens, unsigned char* target_tokens, size_t num_sequences) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    
    unsigned char* buffer = (unsigned char*)malloc((seq_len + 1) * sizeof(unsigned char));
    
    for (size_t i = 0; i < num_sequences; i++) {
        fseek(f, indices[i] * seq_len, SEEK_SET);
        
        if (fread(buffer, 1, seq_len + 1, f) < (size_t)(seq_len + 1)) break;
        
        for (int j = 0; j < seq_len; j++) {
            input_tokens[i * seq_len + j] = buffer[j];
            target_tokens[i * seq_len + j] = buffer[j + 1];
        }
    }
    
    free(buffer);
    fclose(f);
}

GPT* gpt = NULL;

// Signal handler to save model on Ctrl+C
void handle_sigint(int signum) {
    if (gpt) {
        char filename[64];
        time_t now = time(NULL);
        strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_gpt.bin", localtime(&now));
        save_gpt(gpt, filename);
    }
    exit(128 + signum);
}

// Generate text autoregressively from a prompt
void generate_text(GPT* gpt, float temperature, unsigned char* d_input_tokens, const char* bos, int gen_len) {
    // Start with zero-initialized sequence
    unsigned char* h_tokens = (unsigned char*)calloc(gpt->seq_len, sizeof(unsigned char));
    
    // Set beginning of sequence (prompt)
    int bos_len = (int)strlen(bos);
    for (int i = 0; i < bos_len; i++) {
        h_tokens[i] = (unsigned char)bos[i];
    }
    
    printf("\"%s", bos);
    fflush(stdout);
    
    float* h_logits = (float*)malloc(gpt->vocab_size * sizeof(float));
    half* h_logits_half = (half*)malloc(gpt->vocab_size * sizeof(half));
    
    // Generate tokens one at a time
    for (int pos = bos_len - 1; pos < gen_len; pos++) {
        // Copy current sequence to device
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_tokens, gpt->seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Forward pass to get logits
        forward_pass_gpt(gpt, d_input_tokens);
        
        // Copy logits for current position back to host
        CHECK_CUDA(cudaMemcpy(h_logits_half, &gpt->d_output[pos * gpt->vocab_size], gpt->vocab_size * sizeof(half), cudaMemcpyDeviceToHost));
        
        // Convert to float
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] = __half2float(h_logits_half[v]);
        }
        
        // Apply temperature scaling and find max for numerical stability
        float max_logit = -1e30f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        // Compute softmax probabilities
        float sum_exp = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / (float)RAND_MAX;
        unsigned char next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = (unsigned char)v;
                break;
            }
        }
        
        // Add sampled token to sequence
        h_tokens[pos + 1] = next_token;
        printf("%c", (char)next_token);
        fflush(stdout);
    }
    
    printf("\"\n");
    free(h_tokens);
    free(h_logits);
    free(h_logits_half);
}

int main(int argc, char* argv[]) {
    const char* corpus_path = argv[1];
    const char* checkpoint_path = (argc > 2) ? argv[2] : NULL;
    
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLAS
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Model hyperparameters
    const int seq_len = 1024;
    const int num_layers = 17;
    const int batch_size = 32;
    const int d_model = num_layers * 32;
    const int hidden_dim = d_model * 2;
    float learning_rate = 0.0001f;
    const int accum_steps = 1;
    
    // Initialize or load model
    if (checkpoint_path) {
        gpt = load_gpt(checkpoint_path, batch_size, seq_len, cublaslt_handle);
    } else {
        gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    {
        size_t d = gpt->d_model;
        size_t N = gpt->transformer->state_dim;
        size_t hid = gpt->transformer->mlp_layers[0]->hidden_dim;
        size_t ssm_per_layer = 2 * d * d + 3 * d * N + d;
        size_t mlp_per_layer = d * hid + hid * d;
        printf("Parameters: ~%.1fM\n",
            (float)(gpt->vocab_size * d + gpt->transformer->num_layers * (ssm_per_layer + mlp_per_layer)) / 1e6f);
    }
    
    // Create shuffled indices for random sampling without replacement
    size_t total_sequences = (get_file_size(corpus_path) - 1) / seq_len;
    size_t* shuffled_indices = create_shuffled_indices(total_sequences);
    
    // Allocate host buffers for sequences
    size_t sequences_per_chunk = (128 * 1024 * 1024) / seq_len;
    unsigned char* input_tokens = (unsigned char*)malloc(sequences_per_chunk * seq_len * sizeof(unsigned char));
    unsigned char* target_tokens = (unsigned char*)malloc(sequences_per_chunk * seq_len * sizeof(unsigned char));
    
    // Allocate device buffers
    unsigned char *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned char)));
    
    // Calculate starting position based on training progress
    size_t start_chunk = 0;
    int start_batch = 0;

    if (checkpoint_path && ((size_t)gpt->t * accum_steps) > 0) {
        start_chunk = ((size_t)gpt->t * accum_steps) / (sequences_per_chunk / batch_size);
        start_batch = (int)(((size_t)gpt->t * accum_steps) % (sequences_per_chunk / batch_size));
        if (start_chunk >= (total_sequences / sequences_per_chunk)) {
            start_chunk = (total_sequences / sequences_per_chunk);
            start_batch = 0;
        }
        printf("Resuming from batch %zu (chunk %zu, batch %d)\n", ((size_t)gpt->t * accum_steps), start_chunk, start_batch);
    }

    // Training loop: process corpus in chunks with random sampling
    for (size_t chunk_idx = start_chunk; chunk_idx < (total_sequences / sequences_per_chunk); chunk_idx++) {
        // Sample next chunk of sequences from shuffled corpus
        sample_sequences(corpus_path, &shuffled_indices[chunk_idx * sequences_per_chunk], seq_len, input_tokens, target_tokens, sequences_per_chunk);
        
        // Determine starting batch for this chunk
        int batch_start = (chunk_idx == start_chunk) ? start_batch : 0;

        // Train on all batches in this chunk
        for (int batch = batch_start; batch < (int)(sequences_per_chunk / batch_size); batch++) {
            struct timespec start; clock_gettime(CLOCK_MONOTONIC, &start);
            
            // Copy batch to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch * batch_size * seq_len], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch * batch_size * seq_len], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_gpt(gpt, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_gpt(gpt, d_target_tokens);
            if (loss >= 6.0) raise(SIGINT);
            
            // Backward pass
            if (batch % accum_steps == 0) zero_gradients_gpt(gpt);
            backward_pass_gpt(gpt, d_input_tokens);
            
            // Update weights with cosine learning rate schedule
            if ((batch + 1) % accum_steps == 0) {
                float lr = learning_rate * fminf(1.0f, (float)(chunk_idx * (sequences_per_chunk / batch_size) + batch) / 100.0f) * (0.1f + 0.45f * (1.0f + cosf(M_PI * ((float)(chunk_idx * (sequences_per_chunk / batch_size) + batch) / (float)(total_sequences / batch_size)))));
                update_weights_gpt(gpt, lr, batch_size * accum_steps);

                CHECK_CUDA(cudaDeviceSynchronize()); struct timespec end; clock_gettime(CLOCK_MONOTONIC, &end);
                printf("Chunk [%zu/%zu], Batch [%d/%d], Loss: %.6f, LR: %.7f, dt: %.2fms, tok/s: %.0f, bpb: %.4f, ETA: %.1fh\n",
                    chunk_idx, total_sequences / sequences_per_chunk, batch, (int)(sequences_per_chunk / batch_size),
                    loss, lr, ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6),
                    (batch_size * seq_len) / ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9),
                    loss / log(2.0),
                    ((double)total_sequences / batch_size - (chunk_idx * (sequences_per_chunk / batch_size) + batch) - 1) * ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) / 3600.0);
            }
        }
        
        // Generate sample text
        const char* prompts[] = {
            "Once upon a time, there was a",
            "Lily and Tom were playing in the",
            "One day, a little girl named",
            "The big dog ran to the",
            "\"Hello!\" said the",
            "Once upon a time, in a",
            "Tim was very",
        };
        int num_prompts = sizeof(prompts) / sizeof(prompts[0]);
        printf("\n");
        for (int p = 0; p < num_prompts; p++) {
            printf("--- Sample %d/%d ---\n", p + 1, num_prompts);
            generate_text(gpt, 0.001f, d_input_tokens, prompts[p], 256);
            printf("\n");
        }
        
        // Save checkpoint
        save_gpt(gpt, "checkpoint_gpt.bin");
    }
    
    // Save final model with timestamp
    char filename[64];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_gpt.bin", localtime(&now));
    save_gpt(gpt, filename);
    
    // Cleanup
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free(shuffled_indices);
    free_gpt(gpt);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>
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

// Read next text chunk from open corpus file, tokenize via SentencePiece,
// create non-overlapping input/target sequence pairs.
// Returns number of complete sequences created.
size_t read_tokenize_chunk(FILE* corpus, int seq_len, unsigned short* input_tokens,
                           unsigned short* target_tokens, size_t max_sequences) {
    size_t text_buf_size = max_sequences * seq_len * 4;
    char* text = (char*)malloc(text_buf_size);
    if (!text) return 0;

    size_t n = fread(text, 1, text_buf_size, corpus);
    if (n == 0) { free(text); return 0; }

    // Back up to last newline so we don't split a line
    size_t orig_n = n;
    while (n > 0 && text[n - 1] != '\n') n--;
    if (n == 0) n = orig_n;
    else fseek(corpus, -(long)(orig_n - n), SEEK_CUR);

    // Write text to temp file for spm_encode
    char tmp[] = "/tmp/gpt_tok_XXXXXX";
    int fd = mkstemp(tmp);
    if (fd < 0) { free(text); return 0; }
    FILE* tmp_fp = fdopen(fd, "w");
    fwrite(text, 1, n, tmp_fp);
    fclose(tmp_fp);
    free(text);

    // Run spm_encode to get token IDs
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "./sentencepiece/build/src/spm_encode "
             "--model=./sentencepiece/spm_custom.model --output_format=id < %s", tmp);
    FILE* fp = popen(cmd, "r");
    if (!fp) { unlink(tmp); return 0; }

    size_t max_tokens = max_sequences * (seq_len + 1);
    unsigned short* tokens = (unsigned short*)malloc(max_tokens * sizeof(unsigned short));
    size_t num_tokens = 0;
    while (num_tokens < max_tokens && fscanf(fp, "%hu", &tokens[num_tokens]) == 1)
        num_tokens++;
    pclose(fp);
    unlink(tmp);

    // Create non-overlapping input[i..i+seq_len-1] / target[i+1..i+seq_len] pairs
    size_t num_seq = 0;
    for (size_t i = 0; i + seq_len < num_tokens && num_seq < max_sequences; i += seq_len) {
        memcpy(&input_tokens[num_seq * seq_len], &tokens[i], seq_len * sizeof(unsigned short));
        memcpy(&target_tokens[num_seq * seq_len], &tokens[i + 1], seq_len * sizeof(unsigned short));
        num_seq++;
    }

    free(tokens);
    return num_seq;
}

// Shuffle sequence pairs in-place (Fisher-Yates)
void shuffle_sequences(unsigned short* input_tokens, unsigned short* target_tokens,
                       size_t num_sequences, int seq_len) {
    size_t row_bytes = seq_len * sizeof(unsigned short);
    unsigned short* tmp = (unsigned short*)malloc(row_bytes);
    for (size_t i = num_sequences - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        if (i != j) {
            memcpy(tmp, &input_tokens[i * seq_len], row_bytes);
            memcpy(&input_tokens[i * seq_len], &input_tokens[j * seq_len], row_bytes);
            memcpy(&input_tokens[j * seq_len], tmp, row_bytes);
            memcpy(tmp, &target_tokens[i * seq_len], row_bytes);
            memcpy(&target_tokens[i * seq_len], &target_tokens[j * seq_len], row_bytes);
            memcpy(&target_tokens[j * seq_len], tmp, row_bytes);
        }
    }
    free(tmp);
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
void generate_text(GPT* gpt, float temperature, unsigned short* d_input_tokens, const char* prompt, int gen_len) {
    // Encode prompt via spm_encode
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "echo '%s' | ./sentencepiece/build/src/spm_encode "
             "--model=./sentencepiece/spm_custom.model --output_format=id", prompt);
    FILE* fp = popen(cmd, "r");
    if (!fp) return;

    unsigned short* h_tokens = (unsigned short*)calloc(gpt->seq_len, sizeof(unsigned short));
    int prompt_len = 0;
    while (fscanf(fp, "%hu", &h_tokens[prompt_len]) == 1 && prompt_len < gpt->seq_len)
        prompt_len++;
    pclose(fp);
    if (prompt_len == 0) { free(h_tokens); return; }

    float* h_logits = (float*)malloc(gpt->vocab_size * sizeof(float));
    half* h_logits_half = (half*)malloc(gpt->vocab_size * sizeof(half));

    // Generate tokens one at a time
    for (int pos = prompt_len - 1; pos < gen_len && pos < gpt->seq_len - 1; pos++) {
        // Copy current sequence to device
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_tokens, gpt->seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));

        // Forward pass to get logits
        forward_pass_gpt(gpt, d_input_tokens);

        // Copy logits for current position back to host
        CHECK_CUDA(cudaMemcpy(h_logits_half, &gpt->d_output[pos * gpt->vocab_size], gpt->vocab_size * sizeof(half), cudaMemcpyDeviceToHost));

        // Apply temperature scaling and find max for numerical stability
        float max_logit = -1e30f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] = __half2float(h_logits_half[v]) / temperature;
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
        unsigned short next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }

        // Add sampled token to sequence
        h_tokens[pos + 1] = next_token;
    }

    // Decode via spm_decode
    char temp[] = "/tmp/gpt_XXXXXX";
    int fd = mkstemp(temp);
    FILE* tf = fdopen(fd, "w");
    for (int i = 0; i < gen_len && i < gpt->seq_len; i++)
        fprintf(tf, "%hu ", h_tokens[i]);
    fclose(tf);

    char dcmd[512];
    snprintf(dcmd, sizeof(dcmd),
             "printf '\"' && ./sentencepiece/build/src/spm_decode "
             "--model=./sentencepiece/spm_custom.model --input_format=id < %s "
             "&& printf '\"\n'", temp);
    system(dcmd);
    unlink(temp);

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
    const int seq_len = 512;
    const int num_layers = 21;
    const int batch_size = 22;
    const int d_model = num_layers * 64;
    const int hidden_dim = d_model * 4;
    float learning_rate = 0.0001f;
    const int accum_steps = 1;
    
    // Initialize or load model
    if (checkpoint_path) {
        gpt = load_gpt(checkpoint_path, batch_size, seq_len, cublaslt_handle);
    } else {
        gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    printf("Parameters: ~%.1fM\n", (float)(gpt->vocab_size * gpt->d_model + gpt->transformer->num_layers * ((size_t)4 * gpt->d_model * gpt->d_model + gpt->d_model * gpt->transformer->mlp_layers[0]->hidden_dim + gpt->transformer->mlp_layers[0]->hidden_dim * gpt->d_model)) / 1e6f);
    
    // Corpus stats (estimate ~4 text bytes per token)
    size_t corpus_size = get_file_size(corpus_path);
    size_t sequences_per_chunk = (128 * 1024 * 1024) / (seq_len * 2);
    size_t text_per_chunk = sequences_per_chunk * seq_len * 4;
    size_t total_sequences = corpus_size / 4 / seq_len;
    size_t total_chunks = corpus_size / text_per_chunk;
    if (total_chunks == 0) total_chunks = 1;
    
    // Allocate host buffers for sequences
    unsigned short* input_tokens = (unsigned short*)malloc(sequences_per_chunk * seq_len * sizeof(unsigned short));
    unsigned short* target_tokens = (unsigned short*)malloc(sequences_per_chunk * seq_len * sizeof(unsigned short));
    
    // Allocate device buffers
    unsigned short *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned short)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned short)));
    
    // Open corpus for sequential reading
    FILE* corpus_file = fopen(corpus_path, "r");
    if (!corpus_file) { fprintf(stderr, "Cannot open %s\n", corpus_path); return 1; }

    // Calculate starting position based on training progress
    size_t start_chunk = 0;
    int start_batch = 0;

    if (checkpoint_path && ((size_t)gpt->t * accum_steps) > 0) {
        start_chunk = ((size_t)gpt->t * accum_steps) / (sequences_per_chunk / batch_size);
        start_batch = (int)(((size_t)gpt->t * accum_steps) % (sequences_per_chunk / batch_size));
        if (start_chunk >= total_chunks) {
            start_chunk = total_chunks;
            start_batch = 0;
        }
        // Seek to approximate position in text file
        fseek(corpus_file, start_chunk * text_per_chunk, SEEK_SET);
        printf("Resuming from batch %zu (chunk %zu, batch %d)\n", ((size_t)gpt->t * accum_steps), start_chunk, start_batch);
    }

    size_t chunk_idx = start_chunk;

    // Training loop: process corpus in chunks
    while (!feof(corpus_file)) {
        // Read next text chunk and tokenize on-the-fly
        size_t num_sequences = read_tokenize_chunk(corpus_file, seq_len, input_tokens, target_tokens, sequences_per_chunk);
        if (num_sequences < (size_t)batch_size) break;

        // Sanity check: verify target is input shifted by 1 (before shuffling)
        int shift_ok = 1;
        for (int j = 0; j < seq_len - 1; j++) {
            if (target_tokens[j] != input_tokens[j + 1]) { shift_ok = 0; break; }
        }
        if (!shift_ok) { printf("FATAL: input/target shift-by-1 check FAILED\n"); raise(SIGINT); }

        // Sanity check: decode first 20 tokens to verify tokenization round-trip
        {
            char tmp_sc[] = "/tmp/gpt_sc_XXXXXX";
            int fd_sc = mkstemp(tmp_sc);
            FILE* fp_sc = fdopen(fd_sc, "w");
            for (int j = 0; j < 20 && j < seq_len; j++) fprintf(fp_sc, "%hu ", input_tokens[j]);
            fclose(fp_sc);
            char sc_cmd[512];
            snprintf(sc_cmd, sizeof(sc_cmd),
                     "./sentencepiece/build/src/spm_decode "
                     "--model=./sentencepiece/spm_custom.model --input_format=id < %s", tmp_sc);
            FILE* fp_dec = popen(sc_cmd, "r");
            if (fp_dec) {
                char dec_buf[512];
                size_t dec_len = fread(dec_buf, 1, sizeof(dec_buf) - 1, fp_dec);
                pclose(fp_dec);
                while (dec_len > 0 && (dec_buf[dec_len-1] == '\n' || dec_buf[dec_len-1] == '\r')) dec_len--;
                dec_buf[dec_len] = '\0';
                printf("  Chunk %zu/%zu: %zu seqs, shift OK, first 20 tokens: \"%s\"\n", chunk_idx, total_chunks, num_sequences, dec_buf);
            }
            unlink(tmp_sc);
        }

        // Shuffle sequences within this chunk
        shuffle_sequences(input_tokens, target_tokens, num_sequences, seq_len);

        // Determine starting batch for this chunk
        int batch_start = (chunk_idx == start_chunk) ? start_batch : 0;

        // Train on all batches in this chunk
        for (int batch = batch_start; batch < (int)(num_sequences / batch_size); batch++) {
            struct timespec start; clock_gettime(CLOCK_MONOTONIC, &start);
            
            // Copy batch to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch * batch_size * seq_len], batch_size * seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch * batch_size * seq_len], batch_size * seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_gpt(gpt, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_gpt(gpt, d_target_tokens);
            if (loss >= 12.0) raise(SIGINT);
            
            // Backward pass
            if (batch % accum_steps == 0) zero_gradients_gpt(gpt);
            backward_pass_gpt(gpt, d_input_tokens);
            
            // Update weights with cosine learning rate schedule
            if ((batch + 1) % accum_steps == 0) {
                float lr = learning_rate * fminf(1.0f, (float)(chunk_idx * (sequences_per_chunk / batch_size) + batch) / 1000.0f) * (0.1f + 0.45f * (1.0f + cosf(M_PI * ((float)(chunk_idx * (sequences_per_chunk / batch_size) + batch) / (float)(total_sequences / batch_size)))));
                update_weights_gpt(gpt, lr, batch_size * accum_steps);

                CHECK_CUDA(cudaDeviceSynchronize()); struct timespec end; clock_gettime(CLOCK_MONOTONIC, &end);
                printf("Chunk [%zu/%zu], Batch [%d/%d], Loss: %.6f, LR: %.7f, dt: %.2fms, tok/s: %.0f, bpb: %.4f, ETA: %.1fh\n",
                    chunk_idx, total_chunks, batch, (int)(num_sequences / batch_size),
                    loss, lr, ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6),
                    (batch_size * seq_len) / ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9),
                    loss / log(2.0) / 2.0,
                    ((double)total_sequences / batch_size - (chunk_idx * (sequences_per_chunk / batch_size) + batch) - 1) * ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) / 3600.0);
            }
        }
        
        // Generate sample text
        printf("\n--- Sample ---\n");
        generate_text(gpt, 0.001f, d_input_tokens, "<|bos|>The capital of France is", 64);
        generate_text(gpt, 0.001f, d_input_tokens, "<|bos|>The chemical symbol of gold is", 64);
        generate_text(gpt, 0.001f, d_input_tokens, "<|bos|>If yesterday was Friday, then tomorrow will be", 64);
        generate_text(gpt, 0.001f, d_input_tokens, "<|bos|>The opposite of hot is", 64);
        generate_text(gpt, 0.001f, d_input_tokens, "<|bos|>The planets of the solar system are:", 64);
        generate_text(gpt, 0.001f, d_input_tokens, "<|bos|>My favorite color is", 64);
        generate_text(gpt, 0.001f, d_input_tokens, "<|bos|>If 5*x + 3 = 13, then x is", 64);
        printf("--- End ---\n\n");
        
        // Save checkpoint
        save_gpt(gpt, "checkpoint_gpt.bin");
        chunk_idx++;
    }
    
    fclose(corpus_file);

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
    free_gpt(gpt);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}
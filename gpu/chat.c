#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <cuda_fp16.h>
#include "gpt.h"

static GPT* global_gpt = NULL;
static cublasLtHandle_t global_handle;
static unsigned short* global_d_input_tokens = NULL;
static unsigned short* global_h_tokens = NULL;
static float* global_h_logits = NULL;
static half* global_h_logits_half = NULL;

void cleanup_and_exit(int signum) {
    (void)signum;
    printf("\n\nExiting...\n");
    
    // Cleanup
    if (global_h_tokens) free(global_h_tokens);
    if (global_h_logits) free(global_h_logits);
    if (global_h_logits_half) free(global_h_logits_half);
    if (global_d_input_tokens) cudaFree(global_d_input_tokens);
    if (global_gpt) free_gpt(global_gpt);
    cublasLtDestroy(global_handle);
    
    exit(0);
}

void generate_response(GPT* gpt, const char* question, unsigned short* h_tokens, unsigned short* d_input_tokens, 
                       float* h_logits, half* h_logits_half) {
    const int seq_len = gpt->seq_len;
    
    // Build the prompt
    char prompt[4096];
    snprintf(prompt, sizeof(prompt), "<|bos|><|user_start|>%s<|user_end|><|assistant_start|>", question);
    size_t prompt_len = strlen(prompt);
    int prompt_token_count = (prompt_len + 1) / 2;
    
    // Clear token buffer
    memset(h_tokens, 0, seq_len * sizeof(unsigned short));
    
    // Encode prompt into tokens
    for (int i = 0; i < prompt_token_count; i++) {
        h_tokens[i] = (unsigned short)((unsigned char)prompt[i * 2] << 8) | 
                      ((size_t)(i * 2 + 1) < prompt_len ? (unsigned char)prompt[i * 2 + 1] : ' ');
    }
    
    const float temperature = 0.7f;
    const int max_new_tokens = 256;
    const char* end_marker = "<|assistant_end|>";
    const size_t end_marker_len = strlen(end_marker);
    
    // Buffer for generated text (to check for end marker before printing)
    char output_buffer[2048];
    int output_len = 0;
    int printed_len = 0;
    
    // Generate tokens
    int pos_start = prompt_token_count - 1;
    int done = 0;
    
    for (int pos = pos_start; pos < pos_start + max_new_tokens && pos < seq_len - 1 && !done; pos++) {
        // Forward pass
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_tokens, seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));
        forward_pass_gpt(gpt, d_input_tokens);
        
        // Get logits for current position
        CHECK_CUDA(cudaMemcpy(h_logits_half, &gpt->d_output[pos * gpt->vocab_size], 
                             gpt->vocab_size * sizeof(half), cudaMemcpyDeviceToHost));
        
        // Convert to float and apply temperature
        float max_logit = -1e30f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] = __half2float(h_logits_half[v]) / temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        
        // Sample
        float r = (float)rand() / (float)RAND_MAX;
        unsigned short next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            cumsum += h_logits[v] / sum_exp;
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }
        
        h_tokens[pos + 1] = next_token;
        
        // Add to output buffer
        if (output_len < (int)sizeof(output_buffer) - 3) {
            output_buffer[output_len++] = (char)(next_token >> 8);
            output_buffer[output_len++] = (char)(next_token & 0xFF);
            output_buffer[output_len] = '\0';
        }
        
        // Check if we have the end marker in our buffer
        char* marker_pos = strstr(output_buffer, end_marker);
        if (marker_pos) {
            // Found end marker - print everything up to it
            int pos_in_buffer = marker_pos - output_buffer;
            while (printed_len < pos_in_buffer) {
                putchar(output_buffer[printed_len]);
                printed_len++;
            }
            done = 1;
            break;
        }
        
        // Print any complete characters that are safe (not potentially part of end marker)
        // Keep at least end_marker_len characters in buffer to check for end marker
        while (printed_len < output_len - (int)end_marker_len && printed_len < output_len) {
            putchar(output_buffer[printed_len]);
            fflush(stdout);
            printed_len++;
        }
    }
    
    // Print any remaining characters if we didn't find the end marker
    if (!done) {
        while (printed_len < output_len) {
            putchar(output_buffer[printed_len]);
            printed_len++;
        }
    }
    
    printf("\n");
}

int main(void) {
    srand(time(NULL));
    signal(SIGINT, cleanup_and_exit);
    
    // Initialize cuBLAS
    CHECK_CUBLASLT(cublasLtCreate(&global_handle));
    
    // Load the latest model
    const char* model_file = "checkpoint_gpt.bin";
    FILE* test = fopen(model_file, "rb");
    if (!test) {
        FILE* pipe = popen("ls -t *_gpt.bin 2>/dev/null | head -n1", "r");
        if (pipe) {
            static char latest[256];
            if (fgets(latest, sizeof(latest), pipe)) {
                latest[strcspn(latest, "\n")] = 0;
                if (strlen(latest) > 0) model_file = latest;
            }
            pclose(pipe);
        }
    } else {
        fclose(test);
    }
    
    // Load model with batch_size=1 for inference
    const int seq_len = 512;
    printf("Loading model from %s...\n", model_file);
    global_gpt = load_gpt(model_file, 1, seq_len, global_handle);
    if (!global_gpt) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded. Ready!\n");
    
    // Allocate buffers once
    global_h_tokens = (unsigned short*)calloc(seq_len, sizeof(unsigned short));
    CHECK_CUDA(cudaMalloc(&global_d_input_tokens, seq_len * sizeof(unsigned short)));
    
    global_h_logits = (float*)malloc(global_gpt->vocab_size * sizeof(float));
    global_h_logits_half = (half*)malloc(global_gpt->vocab_size * sizeof(half));
    
    // Interactive mode
    char question[4096];
    
    while (1) {
        printf("\n\033[1;36m?\033[0m ");
        fflush(stdout);
        
        if (!fgets(question, sizeof(question), stdin)) {
            break;  // EOF or error
        }
        
        // Remove newline
        question[strcspn(question, "\n")] = 0;
        
        // Skip empty questions
        if (strlen(question) == 0) continue;
        
        printf("\033[1;32m>\033[0m ");
        fflush(stdout);
        
        generate_response(global_gpt, question, global_h_tokens, global_d_input_tokens, 
                         global_h_logits, global_h_logits_half);
    }
    
    // Normal cleanup
    cleanup_and_exit(0);
    return 0;
}
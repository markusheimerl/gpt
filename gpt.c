#include "gpt.h"

// Initialize the GPT
GPT* init_gpt(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size) {
    GPT* gpt = (GPT*)malloc(sizeof(GPT));
    
    // Store dimensions
    gpt->seq_len = seq_len;
    gpt->d_model = d_model;
    gpt->batch_size = batch_size;
    gpt->hidden_dim = hidden_dim;
    gpt->num_layers = num_layers;
    gpt->vocab_size = 65536;
    
    // Initialize Adam parameters
    gpt->beta1 = 0.9f;
    gpt->beta2 = 0.999f;
    gpt->epsilon = 1e-8f;
    gpt->t = 0;
    gpt->weight_decay = 0.01f;
    
    size_t token_emb_size = gpt->vocab_size * d_model;
    size_t embedded_size = batch_size * seq_len * d_model;
    size_t output_size = batch_size * seq_len * gpt->vocab_size;
    
    // Allocate memory for embeddings and gradients
    gpt->token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    gpt->token_embedding_grad = (float*)malloc(token_emb_size * sizeof(float));
    
    // Allocate memory for Adam parameters
    gpt->token_embedding_m = (float*)calloc(token_emb_size, sizeof(float));
    gpt->token_embedding_v = (float*)calloc(token_emb_size, sizeof(float));
    
    // Allocate memory for forward pass buffers
    gpt->embedded_input = (float*)malloc(embedded_size * sizeof(float));
    gpt->norm_output = (float*)malloc(embedded_size * sizeof(float));
    gpt->output = (float*)malloc(output_size * sizeof(float));
    
    // Alias/Allocate memory for backward pass buffers
    gpt->grad_output = gpt->output;
    gpt->grad_norm_output = (float*)malloc(embedded_size * sizeof(float));
    
    // Initialize token embeddings
    float token_scale = 1.0f / sqrtf(d_model);
    
    for (size_t i = 0; i < token_emb_size; i++) {
        gpt->token_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    // Initialize transformer
    gpt->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, true, true);
    
    return gpt;
}

// Free GPT memory
void free_gpt(GPT* gpt) {
    // Free transformer
    free_transformer(gpt->transformer);
    
    // Free memory
    free(gpt->token_embedding); free(gpt->token_embedding_grad);
    free(gpt->token_embedding_m); free(gpt->token_embedding_v);
    free(gpt->embedded_input);
    free(gpt->norm_output);
    free(gpt->output);
    free(gpt->grad_norm_output);
    
    free(gpt);
}

// Token embedding lookup
static void token_embedding_lookup(float* embedded, float* token_embedding, unsigned short* tokens, int batch_size, int seq_len, int d_model) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int token_idx = b * seq_len + t;
            int token = tokens[token_idx];
            int emb_offset = b * seq_len * d_model + t * d_model;
            int token_emb_offset = token * d_model;
            
            for (int d = 0; d < d_model; d++) {
                embedded[emb_offset + d] = token_embedding[token_emb_offset + d];
            }
        }
    }
}

// RMSNorm forward: y = x / sqrt(mean(x^2) + eps)
extern void rmsnorm_forward(float* output, const float* input, int batch_size, int seq_len, int d_model);

// RMSNorm backward: ∂L/∂x = (∂L/∂y)/(x / sqrt(mean(x^2) + eps)) - x⊙(Σ_d (∂L/∂y_d)⊙x_d)/(d_model⊙(x / sqrt(mean(x^2) + eps))³)
extern void rmsnorm_backward(float* grad_input, const float* grad_output, const float* input, int batch_size, int seq_len, int d_model);

// Softmax and cross-entropy loss computation
static float softmax_cross_entropy(float* grad_logits, float* logits, unsigned short* targets, int batch_size, int seq_len, int vocab_size) {
    float total_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int logits_offset = b * seq_len * vocab_size + t * vocab_size;
            float* logits_bt = &logits[logits_offset];
            float* grad_logits_bt = &grad_logits[logits_offset];
            
            // Find max for numerical stability
            float max_logit = -1e30f;
            for (int v = 0; v < vocab_size; v++) {
                if (logits_bt[v] > max_logit) max_logit = logits_bt[v];
            }
            
            // Compute softmax probabilities
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab_size; v++) {
                float exp_val = expf(logits_bt[v] - max_logit);
                grad_logits_bt[v] = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize to get probabilities
            for (int v = 0; v < vocab_size; v++) {
                grad_logits_bt[v] /= sum_exp;
            }
            
            // Compute cross-entropy loss and gradient
            int target_token = targets[b * seq_len + t];
            float target_prob = grad_logits_bt[target_token];
            
            // Add cross-entropy loss
            total_loss += -logf(target_prob + 1e-10f);
            
            // Set gradient: softmax - one_hot
            grad_logits_bt[target_token] -= 1.0f;
        }
    }
    
    return total_loss;
}

// Token embedding gradient accumulation
static void token_embedding_grad_accumulation(float* token_embedding_grad, float* grad_embedded, unsigned short* tokens, int batch_size, int seq_len, int d_model) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int token_idx = b * seq_len + t;
            int token = tokens[token_idx];
            int emb_offset = b * seq_len * d_model + t * d_model;
            int token_emb_offset = token * d_model;
            
            for (int d = 0; d < d_model; d++) {
                token_embedding_grad[token_emb_offset + d] += grad_embedded[emb_offset + d];
            }
        }
    }
}

// Forward pass
void forward_pass_gpt(GPT* gpt, unsigned short* input_tokens) {
    // Step 1: Token embedding lookup
    token_embedding_lookup(gpt->embedded_input, gpt->token_embedding, input_tokens,
                          gpt->batch_size, gpt->seq_len, gpt->d_model);
    
    // Step 2: Forward pass through transformer
    forward_pass_transformer(gpt->transformer, gpt->embedded_input);
    
    // Step 3: RMSNorm on transformer output
    rmsnorm_forward(gpt->norm_output,
                    gpt->transformer->mlp_layers[gpt->num_layers-1]->output,
                    gpt->batch_size,
                    gpt->seq_len,
                    gpt->d_model);
    
    // Step 4: Output projection: output = norm_output * token_embedding^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                gpt->batch_size * gpt->seq_len, gpt->vocab_size, gpt->d_model,
                1.0f, gpt->norm_output, gpt->d_model,
                gpt->token_embedding, gpt->d_model,
                0.0f, gpt->output, gpt->vocab_size);
}

// Calculate loss
float calculate_loss_gpt(GPT* gpt, unsigned short* target_tokens) {
    // Compute softmax and cross-entropy loss
    float total_loss = softmax_cross_entropy(gpt->grad_output, gpt->output, 
                                            target_tokens, gpt->batch_size, gpt->seq_len, gpt->vocab_size);
    
    return total_loss / (gpt->batch_size * gpt->seq_len);
}

// Zero gradients
void zero_gradients_gpt(GPT* gpt) {
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    
    memset(gpt->token_embedding_grad, 0, token_emb_size * sizeof(float));
    
    zero_gradients_transformer(gpt->transformer);
}

// Backward pass
void backward_pass_gpt(GPT* gpt, unsigned short* input_tokens) {
    // Step 4 (backward): Backward pass through output projection
    // grad_token_embedding += grad_output^T * norm_output
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                gpt->vocab_size, gpt->d_model, gpt->batch_size * gpt->seq_len,
                1.0f, gpt->grad_output, gpt->vocab_size,
                gpt->norm_output, gpt->d_model,
                1.0f, gpt->token_embedding_grad, gpt->d_model);
    
    // grad_norm_output = grad_output * token_embedding
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                gpt->batch_size * gpt->seq_len, gpt->d_model, gpt->vocab_size,
                1.0f, gpt->grad_output, gpt->vocab_size,
                gpt->token_embedding, gpt->d_model,
                0.0f, gpt->grad_norm_output, gpt->d_model);
    
    // Step 3 (backward): Backward pass through RMSNorm
    rmsnorm_backward(gpt->transformer->mlp_layers[gpt->num_layers-1]->grad_output,
                     gpt->grad_norm_output,
                     gpt->transformer->mlp_layers[gpt->num_layers-1]->output,
                     gpt->batch_size,
                     gpt->seq_len,
                     gpt->d_model);
    
    // Step 2 (backward): Backward pass through transformer
    backward_pass_transformer(gpt->transformer, gpt->embedded_input, gpt->embedded_input);
    
    // Step 1 (backward): Token embedding gradients
    token_embedding_grad_accumulation(gpt->token_embedding_grad, gpt->embedded_input, input_tokens,
                                     gpt->batch_size, gpt->seq_len, gpt->d_model);
}

// Update weights
void update_weights_gpt(GPT* gpt, float learning_rate, int batch_size) {
    gpt->t++;
    
    float beta1_t = powf(gpt->beta1, gpt->t);
    float beta2_t = powf(gpt->beta2, gpt->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    
    // Update token embeddings
    for (int i = 0; i < token_emb_size; i++) {
        float grad = gpt->token_embedding_grad[i] / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        gpt->token_embedding_m[i] = gpt->beta1 * gpt->token_embedding_m[i] + (1.0f - gpt->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        gpt->token_embedding_v[i] = gpt->beta2 * gpt->token_embedding_v[i] + (1.0f - gpt->beta2) * grad * grad;
        
        float update = alpha_t * gpt->token_embedding_m[i] / (sqrtf(gpt->token_embedding_v[i]) + gpt->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        gpt->token_embedding[i] = gpt->token_embedding[i] * (1.0f - learning_rate * gpt->weight_decay) - update;
    }
    
    // Update transformer weights
    update_weights_transformer(gpt->transformer, learning_rate, batch_size);
}

// Reset optimizer state
void reset_optimizer_gpt(GPT* gpt) {
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    
    // Reset Adam moment estimates to zero
    memset(gpt->token_embedding_m, 0, token_emb_size * sizeof(float));
    memset(gpt->token_embedding_v, 0, token_emb_size * sizeof(float));
    
    // Reset time step
    gpt->t = 0;
    
    // Reset transformer optimizer state
    reset_optimizer_transformer(gpt->transformer);
}

// Serialize GPT to a file
static void serialize_gpt(GPT* gpt, FILE* file) {
    // Write dimensions
    fwrite(&gpt->d_model, sizeof(int), 1, file);
    fwrite(&gpt->hidden_dim, sizeof(int), 1, file);
    fwrite(&gpt->num_layers, sizeof(int), 1, file);
    fwrite(&gpt->vocab_size, sizeof(int), 1, file);
    
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    
    // Write embeddings
    fwrite(gpt->token_embedding, sizeof(float), token_emb_size, file);
    
    // Write optimizer state
    fwrite(&gpt->t, sizeof(int), 1, file);
    fwrite(gpt->token_embedding_m, sizeof(float), token_emb_size, file);
    fwrite(gpt->token_embedding_v, sizeof(float), token_emb_size, file);
    
    // Serialize transformer
    serialize_transformer(gpt->transformer, file);
}

// Deserialize GPT from a file
static GPT* deserialize_gpt(FILE* file, int batch_size, int seq_len) {
    // Read dimensions
    int d_model, hidden_dim, num_layers, vocab_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&vocab_size, sizeof(int), 1, file);
    
    // Initialize GPT
    GPT* gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, batch_size);
    
    int token_emb_size = vocab_size * d_model;
    
    // Read embeddings
    fread(gpt->token_embedding, sizeof(float), token_emb_size, file);
    
    // Read optimizer state
    fread(&gpt->t, sizeof(int), 1, file);
    fread(gpt->token_embedding_m, sizeof(float), token_emb_size, file);
    fread(gpt->token_embedding_v, sizeof(float), token_emb_size, file);
    
    // Free the initialized transformer
    free_transformer(gpt->transformer);
    
    // Deserialize transformer
    gpt->transformer = deserialize_transformer(file, batch_size, seq_len);
    
    return gpt;
}

// Save GPT to a file
void save_gpt(GPT* gpt, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    serialize_gpt(gpt, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load GPT from a file
GPT* load_gpt(const char* filename, int batch_size, int seq_len) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    GPT* gpt = deserialize_gpt(file, batch_size, seq_len);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return gpt;
}
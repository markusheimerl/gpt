#include "data.h"

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
void sample_sequences(const char* filename, size_t* indices, int seq_len, unsigned short* input_tokens, unsigned short* target_tokens, size_t num_sequences) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    
    unsigned char* buffer = (unsigned char*)malloc((seq_len + 1) * 2 * sizeof(unsigned char));
    
    for (size_t i = 0; i < num_sequences; i++) {
        fseek(f, indices[i] * seq_len * 2, SEEK_SET);
        
        if (fread(buffer, 1, (seq_len + 1) * 2, f) < (size_t)((seq_len + 1) * 2)) break;
        
        for (int j = 0; j < seq_len; j++) {
            input_tokens[i * seq_len + j] = (unsigned short)((buffer[j * 2] << 8) | buffer[j * 2 + 1]);
            target_tokens[i * seq_len + j] = (unsigned short)((buffer[(j + 1) * 2] << 8) | buffer[(j + 1) * 2 + 1]);
        }
    }
    
    free(buffer);
    fclose(f);
}

// Count lines in a text file
size_t count_lines(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return 0;
    size_t count = 0;
    int c, last = '\n';
    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') count++;
        last = c;
    }
    if (last != '\n' && count > 0) count++;
    fclose(f);
    return count;
}

// Load midtraining sequences from text file with loss masking
void load_midtraining_sequences(const char* filename, int seq_len, unsigned short* input_tokens, unsigned short* target_tokens, unsigned char* loss_mask, size_t start_line, size_t num_sequences) {
    FILE* f = fopen(filename, "r");
    if (!f) return;
    
    char* line = NULL;
    size_t line_cap = 0;
    
    // Skip to start line
    for (size_t i = 0; i < start_line; i++) getline(&line, &line_cap, f);
    
    // Process sequences
    for (size_t i = 0; i < num_sequences; i++) {
        ssize_t len = getline(&line, &line_cap, f);
        if (len <= 0) break;
        if (line[len - 1] == '\n') len--;
        
        // Find assistant response region
        char* start_pos = strstr(line, "<|assistant_start|>");
        char* end_pos = strstr(line, "<|assistant_end|>");
        int start_char = start_pos ? (start_pos - line + 19) : -1;
        int end_char = end_pos ? (end_pos - line + 17) : -1;
        
        // Tokenize
        for (int t = 0; t < seq_len; t++) {
            int c1_idx = t * 2, c2_idx = t * 2 + 1;
            unsigned char c1 = (c1_idx < len) ? line[c1_idx] : ' ';
            unsigned char c2 = (c2_idx < len) ? line[c2_idx] : ' ';
            input_tokens[i * seq_len + t] = (c1 << 8) | c2;
            
            int tc1_idx = (t + 1) * 2, tc2_idx = (t + 1) * 2 + 1;
            unsigned char tc1 = (tc1_idx < len) ? line[tc1_idx] : ' ';
            unsigned char tc2 = (tc2_idx < len) ? line[tc2_idx] : ' ';
            target_tokens[i * seq_len + t] = (tc1 << 8) | tc2;
            
            // Mask: only compute loss for assistant response (not padding)
            loss_mask[i * seq_len + t] = (start_char >= 0 && end_char >= 0 && tc1_idx >= start_char && tc1_idx < end_char && tc1_idx < len) ? 1 : 0;
        }
    }
    
    if (line) free(line);
    fclose(f);
}
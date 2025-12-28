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

// Get the number of lines in a file
size_t get_line_count(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    
    size_t count = 0;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') count++;
    }
    
    fclose(f);
    return count;
}

// Get the file position of each line start
size_t* get_line_positions(const char* filename, size_t* line_count) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    
    // Count lines first
    *line_count = get_line_count(filename);
    
    if (*line_count == 0) {
        fclose(f);
        return NULL;
    }
    
    // Allocate array for positions
    size_t* positions = (size_t*)malloc((*line_count) * sizeof(size_t));
    
    // Record position of each line
    rewind(f);
    positions[0] = 0;
    size_t idx = 1;
    size_t pos = 0;
    int ch;
    
    while ((ch = fgetc(f)) != EOF) {
        pos++;
        if (ch == '\n' && idx < *line_count) {
            positions[idx++] = pos;
        }
    }
    
    fclose(f);
    return positions;
}

// Sample sequences line-by-line with padding to seq_len
void sample_sequences_line_padded(const char* filename, size_t* line_positions, size_t* line_indices, 
                                   int seq_len, unsigned short* input_tokens, unsigned short* target_tokens, 
                                   size_t num_sequences) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    
    unsigned char* line_buffer = (unsigned char*)malloc(seq_len * 4);  // Generous buffer
    
    for (size_t i = 0; i < num_sequences; i++) {
        size_t line_idx = line_indices[i];
        fseek(f, line_positions[line_idx], SEEK_SET);
        
        // Read until newline or EOF
        size_t line_len = 0;
        int ch;
        while ((ch = fgetc(f)) != EOF && ch != '\n' && line_len < (size_t)(seq_len * 4)) {
            line_buffer[line_len++] = (unsigned char)ch;
        }
        
        // Convert to tokens with padding
        for (int j = 0; j < seq_len; j++) {
            unsigned short input_token, target_token;
            
            // Input token at position j*2
            if ((size_t)j * 2 < line_len) {
                unsigned char c1 = line_buffer[j * 2];
                unsigned char c2 = ((size_t)j * 2 + 1 < line_len) ? line_buffer[j * 2 + 1] : ' ';
                input_token = (c1 << 8) | c2;
            } else {
                input_token = 0x2020;  // "  " (space-space token)
            }
            
            // Target token at position (j+1)*2-1, which is j*2+1 (shifted by 1 char)
            if ((size_t)j * 2 + 1 < line_len) {
                unsigned char c1 = line_buffer[j * 2 + 1];
                unsigned char c2 = ((size_t)j * 2 + 2 < line_len) ? line_buffer[j * 2 + 2] : ' ';
                target_token = (c1 << 8) | c2;
            } else {
                target_token = 0x2020;  // "  " (space-space token)
            }
            
            input_tokens[i * seq_len + j] = input_token;
            target_tokens[i * seq_len + j] = target_token;
        }
    }
    
    free(line_buffer);
    fclose(f);
}
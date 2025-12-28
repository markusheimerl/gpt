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

// Sample sequences line-by-line with padding to seq_len and create loss mask
void sample_sequences_line_padded(const char* filename, size_t* line_positions, size_t* line_indices, 
                                   int seq_len, unsigned short* input_tokens, unsigned short* target_tokens,
                                   unsigned char* loss_mask, size_t num_sequences) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    
    unsigned char* line_buffer = (unsigned char*)malloc(seq_len * 4);  // Generous buffer
    
    const char* start_marker = "<|assistant_start|>";
    const char* end_marker = "<|assistant_end|>";
    int start_marker_len = strlen(start_marker);
    int end_marker_len = strlen(end_marker);
    
    for (size_t i = 0; i < num_sequences; i++) {
        size_t line_idx = line_indices[i];
        fseek(f, line_positions[line_idx], SEEK_SET);
        
        // Read until newline or EOF
        size_t line_len = 0;
        int ch;
        while ((ch = fgetc(f)) != EOF && ch != '\n' && line_len < (size_t)(seq_len * 4)) {
            line_buffer[line_len++] = (unsigned char)ch;
        }
        
        // Find markers in the line
        int mask_start_char = -1;
        int mask_end_char = -1;
        
        // Search for start marker
        for (size_t pos = 0; pos + start_marker_len <= line_len; pos++) {
            if (memcmp(&line_buffer[pos], start_marker, start_marker_len) == 0) {
                mask_start_char = (int)(pos + start_marker_len);
                break;
            }
        }
        
        // Search for end marker
        for (size_t pos = 0; pos + end_marker_len <= line_len; pos++) {
            if (memcmp(&line_buffer[pos], end_marker, end_marker_len) == 0) {
                mask_end_char = (int)(pos + end_marker_len - 1);  // inclusive
                break;
            }
        }
        
        // Convert to tokens with padding and create mask
        for (int j = 0; j < seq_len; j++) {
            unsigned short input_token, target_token;
            size_t char_pos_1 = (size_t)(j * 2);
            size_t char_pos_2 = (size_t)(j * 2 + 1);
            size_t char_pos_3 = (size_t)(j * 2 + 2);
            
            // Input token at position j*2
            if (char_pos_1 < line_len) {
                unsigned char c1 = line_buffer[char_pos_1];
                unsigned char c2 = (char_pos_2 < line_len) ? line_buffer[char_pos_2] : ' ';
                input_token = (c1 << 8) | c2;
            } else {
                input_token = 0x2020;  // "  " (space-space token)
            }
            
            // Target token at position j*2+1 (shifted by 1 char)
            if (char_pos_2 < line_len) {
                unsigned char c1 = line_buffer[char_pos_2];
                unsigned char c2 = (char_pos_3 < line_len) ? line_buffer[char_pos_3] : ' ';
                target_token = (c1 << 8) | c2;
            } else {
                target_token = 0x2020;  // "  " (space-space token)
            }
            
            input_tokens[i * seq_len + j] = input_token;
            target_tokens[i * seq_len + j] = target_token;
            
            // Determine if this target token position should contribute to loss
            // Target token j corresponds to characters at positions 2j+1 and 2j+2
            int target_char_start = j * 2 + 1;
            int target_char_end = j * 2 + 2;
            
            // Include in loss if:
            // 1. Both markers were found
            // 2. Target token falls within the masked region
            // 3. Target token is not padding
            if (mask_start_char >= 0 && mask_end_char >= 0 &&
                target_char_start >= mask_start_char && 
                target_char_end <= mask_end_char &&
                target_token != 0x2020) {
                loss_mask[i * seq_len + j] = 1;
            } else {
                loss_mask[i * seq_len + j] = 0;
            }
        }
    }
    
    free(line_buffer);
    fclose(f);
}
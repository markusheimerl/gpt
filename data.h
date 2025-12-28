#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Get the total size of a file
size_t get_file_size(const char* filename);

// Create shuffled sequence indices for entire corpus
size_t* create_shuffled_indices(size_t total_sequences);

// Sample sequences using shuffled indices
void sample_sequences(const char* filename, size_t* indices, int seq_len, unsigned short* input_tokens, unsigned short* target_tokens, size_t num_sequences);

size_t count_lines(const char* filename);
void load_midtraining_sequences(const char* filename, int seq_len, unsigned short* input_tokens, unsigned short* target_tokens, unsigned char* loss_mask, size_t start_line, size_t num_sequences);

#endif
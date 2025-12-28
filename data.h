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

// Get the number of lines in a file
size_t get_line_count(const char* filename);

// Get the file position of each line start
size_t* get_line_positions(const char* filename, size_t* line_count);

// Sample sequences line-by-line with padding to seq_len
void sample_sequences_line_padded(const char* filename, size_t* line_positions, size_t* line_indices, 
                                   int seq_len, unsigned short* input_tokens, unsigned short* target_tokens, 
                                   size_t num_sequences);

#endif
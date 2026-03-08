#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static void trim_mlp(FILE* fin, FILE* fout) {
    int input_dim, hidden_dim, output_dim;
    fread(&input_dim,  sizeof(int), 1, fin);
    fread(&hidden_dim, sizeof(int), 1, fin);
    fread(&output_dim, sizeof(int), 1, fin);
    fwrite(&input_dim,  sizeof(int), 1, fout);
    fwrite(&hidden_dim, sizeof(int), 1, fout);
    fwrite(&output_dim, sizeof(int), 1, fout);

    size_t w1 = (size_t)input_dim  * hidden_dim;
    size_t w2 = (size_t)hidden_dim * output_dim;

    float* W1 = malloc(w1 * sizeof(float));
    float* W2 = malloc(w2 * sizeof(float));
    fread(W1, sizeof(float), w1, fin);
    fread(W2, sizeof(float), w2, fin);
    fwrite(W1, sizeof(float), w1, fout);
    fwrite(W2, sizeof(float), w2, fout);
    free(W1); free(W2);

    /* skip: t, W1_m, W1_v, W2_m, W2_v */
    int t; fread(&t, sizeof(int), 1, fin);
    fseek(fin, (long)((w1 + w1 + w2 + w2) * sizeof(float)), SEEK_CUR);
}

static void trim_attention(FILE* fin, FILE* fout) {
    int d_model;
    bool is_causal, use_rope;
    fread(&d_model,   sizeof(int),  1, fin);
    fread(&is_causal, sizeof(bool), 1, fin);
    fread(&use_rope,  sizeof(bool), 1, fin);
    fwrite(&d_model,   sizeof(int),  1, fout);
    fwrite(&is_causal, sizeof(bool), 1, fout);
    fwrite(&use_rope,  sizeof(bool), 1, fout);

    size_t w = (size_t)d_model * d_model;

    float* W_q = malloc(w * sizeof(float));
    float* W_k = malloc(w * sizeof(float));
    float* W_v = malloc(w * sizeof(float));
    float* W_o = malloc(w * sizeof(float));
    fread(W_q, sizeof(float), w, fin);
    fread(W_k, sizeof(float), w, fin);
    fread(W_v, sizeof(float), w, fin);
    fread(W_o, sizeof(float), w, fin);
    fwrite(W_q, sizeof(float), w, fout);
    fwrite(W_k, sizeof(float), w, fout);
    fwrite(W_v, sizeof(float), w, fout);
    fwrite(W_o, sizeof(float), w, fout);
    free(W_q); free(W_k); free(W_v); free(W_o);

    /* skip: t, W_q_m, W_q_v, W_k_m, W_k_v, W_v_m, W_v_v, W_o_m, W_o_v */
    int t; fread(&t, sizeof(int), 1, fin);
    fseek(fin, (long)(8 * w * sizeof(float)), SEEK_CUR);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.bin>\n"
                        "Output: <model>_trim.bin\n", argv[0]);
        return 1;
    }

    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror("open input"); return 1; }

    char outpath[1024];
    const char* dot = strrchr(argv[1], '.');
    if (dot && strcmp(dot, ".bin") == 0)
        snprintf(outpath, sizeof(outpath), "%.*s_trim.bin",
                 (int)(dot - argv[1]), argv[1]);
    else
        snprintf(outpath, sizeof(outpath), "%s_trim.bin", argv[1]);

    FILE* fout = fopen(outpath, "wb");
    if (!fout) { perror("open output"); fclose(fin); return 1; }

    printf("Input : %s\n", argv[1]);
    printf("Output: %s\n\n", outpath);

    /* ── GPT header ── */
    int d_model, hidden_dim, num_layers, vocab_size;
    fread(&d_model,    sizeof(int), 1, fin);
    fread(&hidden_dim, sizeof(int), 1, fin);
    fread(&num_layers, sizeof(int), 1, fin);
    fread(&vocab_size, sizeof(int), 1, fin);
    fwrite(&d_model,    sizeof(int), 1, fout);
    fwrite(&hidden_dim, sizeof(int), 1, fout);
    fwrite(&num_layers, sizeof(int), 1, fout);
    fwrite(&vocab_size, sizeof(int), 1, fout);
    printf("d_model=%d  hidden_dim=%d  num_layers=%d  vocab_size=%d\n\n",
           d_model, hidden_dim, num_layers, vocab_size);

    /* ── Token embedding ── */
    printf("Token embedding...\n");
    size_t emb = (size_t)vocab_size * d_model;
    float* E = malloc(emb * sizeof(float));
    fread(E,  sizeof(float), emb, fin);
    fwrite(E, sizeof(float), emb, fout);
    free(E);
    /* skip: t, m, v */
    int t; fread(&t, sizeof(int), 1, fin);
    fseek(fin, (long)(2 * emb * sizeof(float)), SEEK_CUR);

    /* ── Transformer header ── */
    int tf_d_model, tf_hidden_dim, tf_num_layers;
    bool is_causal, use_rope;
    fread(&tf_d_model,    sizeof(int),  1, fin);
    fread(&tf_hidden_dim, sizeof(int),  1, fin);
    fread(&tf_num_layers, sizeof(int),  1, fin);
    fread(&is_causal,     sizeof(bool), 1, fin);
    fread(&use_rope,      sizeof(bool), 1, fin);
    fwrite(&tf_d_model,    sizeof(int),  1, fout);
    fwrite(&tf_hidden_dim, sizeof(int),  1, fout);
    fwrite(&tf_num_layers, sizeof(int),  1, fout);
    fwrite(&is_causal,     sizeof(bool), 1, fout);
    fwrite(&use_rope,      sizeof(bool), 1, fout);

    /* ── Layers ── */
    for (int i = 0; i < num_layers; i++) {
        printf("Layer %d/%d\n", i + 1, num_layers);
        trim_attention(fin, fout);
        trim_mlp(fin, fout);
    }

    fclose(fin); fclose(fout);
    printf("\nDone.\n");
    return 0;
}
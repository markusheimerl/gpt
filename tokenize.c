#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPM_TRAIN    "sentencepiece/build/src/spm_train"
#define SPM_ENCODE   "sentencepiece/build/src/spm_encode"
#define SPM_DECODE   "sentencepiece/build/src/spm_decode"
#define SAMPLE_FILE  "spm_sample.txt"
#define MODEL_PREFIX "spm_model"
#define VOCAB_SIZE   32000
#define DEFAULT_LINES 10000

static int write_sample(const char *src, int max_lines) {
    FILE *in  = fopen(src, "r");
    FILE *out = fopen(SAMPLE_FILE, "w");
    if (!in || !out) { if (in) fclose(in); if (out) fclose(out); return -1; }
    char *line = NULL; size_t cap = 0; ssize_t len; int n = 0;
    while (n < max_lines && (len = getline(&line, &cap, in)) > 0)
        { fwrite(line, 1, len, out); n++; }
    free(line); fclose(in); fclose(out);
    printf("Wrote %d lines to %s\n", n, SAMPLE_FILE);
    return n;
}

static void round_trip(const char *text) {
    char cmd[512];

    /* write input to temp file to avoid any shell quoting issues */
    FILE *f = fopen("/tmp/spm_in.txt", "w");
    if (!f) return;
    fprintf(f, "%s\n", text);
    fclose(f);

    /* encode → ids */
    snprintf(cmd, sizeof(cmd), "%s --model=%s.model --output_format=id < /tmp/spm_in.txt",
             SPM_ENCODE, MODEL_PREFIX);
    f = popen(cmd, "r");
    if (!f) return;
    char ids[65536] = {0};
    fgets(ids, sizeof(ids), f);
    pclose(f);
    ids[strcspn(ids, "\n")] = '\0';

    /* count tokens */
    int n = 1;
    for (const char *p = ids; *p; p++) n += (*p == ' ');

    /* decode ids → text */
    FILE *tmp = fopen("/tmp/spm_ids.txt", "w");
    if (!tmp) return;
    fprintf(tmp, "%s\n", ids);
    fclose(tmp);

    snprintf(cmd, sizeof(cmd), "%s --model=%s.model --input_format=id < /tmp/spm_ids.txt",
             SPM_DECODE, MODEL_PREFIX);
    f = popen(cmd, "r");
    if (!f) return;
    char decoded[65536] = {0};
    fgets(decoded, sizeof(decoded), f);
    pclose(f);
    decoded[strcspn(decoded, "\n")] = '\0';

    printf("in:  %s\nids: [%s] len=%d\nout: %s\n\n", text, ids, n, decoded);
}

int main(int argc, char *argv[]) {
    const char *corpus = argc > 1 ? argv[1] : "corpus.txt";
    int lines = argc > 2 ? atoi(argv[2]) : DEFAULT_LINES;

    if (write_sample(corpus, lines) <= 0) { fprintf(stderr, "Failed to read %s\n", corpus); return 1; }

    printf("Training BPE tokenizer (vocab=%d)...\n", VOCAB_SIZE);
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "%s --input=%s --model_prefix=%s --vocab_size=%d"
        " --model_type=bpe --character_coverage=1.0"
        " --unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3",
        SPM_TRAIN, SAMPLE_FILE, MODEL_PREFIX, VOCAB_SIZE);
    if (system(cmd) != 0) { fprintf(stderr, "Training failed\n"); return 1; }
    printf("Saved %s.model\n\n", MODEL_PREFIX);

    const char *tests[] = {
        "<|bos|>The capital of France is Paris.",
        "Hello world, this is a tokenizer test.",
        "The quick brown fox jumps over the lazy dog.",
        NULL
    };
    for (int i = 0; tests[i]; i++) round_trip(tests[i]);

    return 0;
}
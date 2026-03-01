#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPM_TRAIN    "sentencepiece/build/src/spm_train"
#define SPM_ENCODE   "sentencepiece/build/src/spm_encode"
#define SPM_DECODE   "sentencepiece/build/src/spm_decode"
#define MODEL_PREFIX "spm_model"
#define VOCAB_SIZE   65536  /* 2^16 */

static void round_trip(const char *text) {
    char cmd[512];

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

    printf("Training BPE tokenizer (vocab=%d) on %s...\n", VOCAB_SIZE, corpus);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "%s"
        " --input=%s"
        " --model_prefix=%s"
        " --vocab_size=%d"
        " --model_type=bpe"
        " --character_coverage=1.0"
        " --input_sentence_size=5000000"
        " --shuffle_input_sentence=true"
        " --unk_id=0"
        " --bos_id=-1"
        " --eos_id=-1"
        " --pad_id=-1"
        " --user_defined_symbols='<|bos|>,<|user|>,<|assistant|>'",
        SPM_TRAIN, corpus, MODEL_PREFIX, VOCAB_SIZE);

    if (system(cmd) != 0) { fprintf(stderr, "Training failed\n"); return 1; }
    printf("Saved %s.model\n\n", MODEL_PREFIX);

    const char *tests[] = {
        "<|bos|><|user|>\nWhat is the capital of France?",
        "<|bos|><|assistant|>\nThe capital of France is Paris.",
        "<|bos|>The quick brown fox jumps over the lazy dog.",
        "<|bos|>Hello world, this is a tokenizer test.",
        "<|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars.",
        NULL
    };
    for (int i = 0; tests[i]; i++) round_trip(tests[i]);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>

#define SPM_TRAIN    "sentencepiece/build/src/spm_train"
#define SPM_ENCODE   "sentencepiece/build/src/spm_encode"
#define SPM_DECODE   "sentencepiece/build/src/spm_decode"
#define MODEL_PREFIX "spm_model"
#define VOCAB_SIZE   65536
#define WRITE_BUF_TOKENS (1 << 20)

// ============================================================================
// Tokenizer training and round-trip test
// ============================================================================

static void round_trip(const char *text) {
    char cmd[512];

    FILE *f = fopen("/tmp/spm_in.txt", "w");
    if (!f) return;
    fprintf(f, "%s\n", text);
    fclose(f);

    snprintf(cmd, sizeof(cmd), "%s --model=%s.model --output_format=id < /tmp/spm_in.txt",
             SPM_ENCODE, MODEL_PREFIX);
    f = popen(cmd, "r");
    if (!f) return;
    char ids[65536] = {0};
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n")] = '\0';
        if (*ids && *line) strcat(ids, " ");
        strcat(ids, line);
    }
    pclose(f);

    int n = (*ids) ? 1 : 0;
    for (const char *p = ids; *p; p++) n += (*p == ' ');

    FILE *tmp = fopen("/tmp/spm_ids.txt", "w");
    if (!tmp) return;
    fprintf(tmp, "%s\n", ids);
    fclose(tmp);

    snprintf(cmd, sizeof(cmd), "%s --model=%s.model --input_format=id < /tmp/spm_ids.txt",
             SPM_DECODE, MODEL_PREFIX);
    f = popen(cmd, "r");
    if (!f) return;
    char decoded[65536] = {0};
    while (fgets(line, sizeof(line), f)) strcat(decoded, line);
    pclose(f);
    decoded[strcspn(decoded, "\n")] = '\0';

    printf("in:  %s\nids: [%s] len=%d\nout: %s\n\n", text, ids, n, decoded);
}

static void train_tokenizer(const char *corpus) {
    printf("Training BPE tokenizer (vocab=%d) on %s...\n", VOCAB_SIZE, corpus);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "%s"
        " --input=%s"
        " --model_prefix=%s"
        " --vocab_size=%d"
        " --model_type=bpe"
        " --character_coverage=1.0"
        " --input_sentence_size=100000"
        " --shuffle_input_sentence=true"
        " --unk_id=0"
        " --bos_id=-1"
        " --eos_id=-1"
        " --pad_id=-1"
        " --user_defined_symbols='<|bos|>,<|user|>,<|assistant|>'",
        SPM_TRAIN, corpus, MODEL_PREFIX, VOCAB_SIZE);

    if (system(cmd) != 0) { fprintf(stderr, "Training failed\n"); exit(1); }
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
}

// ============================================================================
// Corpus tokenization
// ============================================================================

static void fmt_eta(double secs, char* out, size_t sz) {
    int h = (int)(secs / 3600), m = (int)((secs - h*3600) / 60), s = (int)secs % 60;
    if (h > 0) snprintf(out, sz, "%dh%02dm%02ds", h, m, s);
    else        snprintf(out, sz, "%dm%02ds", m, s);
}

static off_t next_line_start(const char* path, off_t approx, off_t file_size) {
    if (approx <= 0)         return 0;
    if (approx >= file_size) return file_size;
    FILE* f = fopen(path, "rb");
    if (!f) return approx;
    fseeko(f, approx, SEEK_SET);
    while (!feof(f) && fgetc(f) != '\n');
    off_t pos = ftello(f);
    fclose(f);
    return pos >= file_size ? file_size : pos;
}

static void run_worker(const char* inp, off_t start, off_t end, const char* out_bin) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "dd if='%s' bs=65536 skip=%zu count=%zu iflag=skip_bytes,count_bytes 2>/dev/null"
        " | %s --model=%s.model --output_format=id",
        inp, (size_t)start, (size_t)(end - start), SPM_ENCODE, MODEL_PREFIX);

    FILE* spm = popen(cmd, "r");
    if (!spm) { perror("popen"); _exit(1); }

    FILE* fout = fopen(out_bin, "wb");
    if (!fout) { perror("fopen worker out"); pclose(spm); _exit(1); }

    unsigned char* wbuf = malloc(WRITE_BUF_TOKENS * 2);
    size_t wpos = 0;
    char*  line = NULL;
    size_t cap  = 0;

    while (getline(&line, &cap, spm) > 0) {
        char *p = line, *ep;
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            if (!*p) break;
            long id = strtol(p, &ep, 10);
            if (ep == p) break;
            p = ep;
            unsigned short tok = (unsigned short)(id & 0xFFFF);
            wbuf[wpos * 2]     = (tok >> 8) & 0xFF;
            wbuf[wpos * 2 + 1] =  tok       & 0xFF;
            if (++wpos == WRITE_BUF_TOKENS) { fwrite(wbuf, 2, wpos, fout); wpos = 0; }
        }
    }
    if (wpos) fwrite(wbuf, 2, wpos, fout);

    free(wbuf); free(line);
    pclose(spm);
    fclose(fout);
    _exit(0);
}

static void tokenize_corpus(const char* inp, int nw) {
    if (nw < 1) nw = 1;
    if (nw > 64) nw = 64;

    char out[512];
    snprintf(out, sizeof(out), "%s.bin", inp);

    struct stat st;
    if (stat(inp, &st) != 0) { perror("stat"); exit(1); }
    off_t fsz = st.st_size;

    printf("Input  : %s (%.2f GB)\n", inp,  (double)fsz / 1073741824.0);
    printf("Output : %s\n", out);
    printf("Workers: %d\n\n", nw);
    fflush(stdout);

    off_t* bounds = malloc((nw + 1) * sizeof(off_t));
    bounds[0] = 0;
    for (int i = 1; i < nw; i++)
        bounds[i] = next_line_start(inp, (off_t)((double)fsz * i / nw), fsz);
    bounds[nw] = fsz;

    char** tmps = malloc(nw * sizeof(char*));
    for (int i = 0; i < nw; i++) {
        tmps[i] = malloc(strlen(out) + 32);
        sprintf(tmps[i], "%s.chunk%d", out, i);
    }

    pid_t* pids = malloc(nw * sizeof(pid_t));
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < nw; i++) {
        printf("Worker %d: [%.3f GB, %.3f GB)\n", i,
               (double)bounds[i]   / 1073741824.0,
               (double)bounds[i+1] / 1073741824.0);
        pids[i] = fork();
        if (pids[i] < 0) { perror("fork"); exit(1); }
        if (pids[i] == 0)
            run_worker(inp, bounds[i], bounds[i+1], tmps[i]);
    }
    printf("\nAll %d workers running...\n\n", nw); fflush(stdout);

    int  ndone = 0;
    int* done  = calloc(nw, sizeof(int));

    while (ndone < nw) {
        int status;
        pid_t finished = waitpid(-1, &status, WNOHANG);

        if (finished > 0) {
            for (int i = 0; i < nw; i++) {
                if (pids[i] == finished && !done[i]) {
                    done[i] = 1; ndone++;
                    struct timespec tn; clock_gettime(CLOCK_MONOTONIC, &tn);
                    double el = (tn.tv_sec - t0.tv_sec) + (tn.tv_nsec - t0.tv_nsec) * 1e-9;
                    printf("\nWorker %d finished  [%d/%d done]  %.1fs elapsed\n",
                           i, ndone, nw, el);
                    fflush(stdout);
                    break;
                }
            }
        } else if (finished == 0) {
            struct timespec tn; clock_gettime(CLOCK_MONOTONIC, &tn);
            double el = (tn.tv_sec - t0.tv_sec) + (tn.tv_nsec - t0.tv_nsec) * 1e-9;

            size_t written = 0;
            for (int i = 0; i < nw; i++) {
                struct stat ts;
                if (stat(tmps[i], &ts) == 0) written += (size_t)ts.st_size;
            }

            size_t toks = written / 2;
            double tps  = el > 1.0 ? (double)toks / el : 0.0;
            double est_frac = (double)(toks * 4) / (double)fsz;
            if (est_frac > 1.0) est_frac = 1.0;
            double eta = (est_frac > 0.001 && el > 2.0) ? el / est_frac - el : 0.0;
            char eta_str[32];
            if (est_frac <= 0.001 || el <= 2.0)
                snprintf(eta_str, sizeof(eta_str), "calculating...");
            else
                fmt_eta(eta, eta_str, sizeof(eta_str));

            printf("\r  [%d/%d done] | %5.1f%% | %.2f GB | %.0f ktok/s | ETA: %-16s",
                   ndone, nw, est_frac * 100.0,
                   (double)written / 1073741824.0,
                   tps / 1000.0, eta_str);
            fflush(stdout);
            sleep(1);
        } else break;
    }
    free(done);

    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("\n\nAll workers done in %.1fs. Merging %d chunks → %s ...\n",
           elapsed, nw, out);
    fflush(stdout);

    FILE* fout = fopen(out, "wb");
    if (!fout) { perror("fopen output"); exit(1); }

    size_t total_toks = 0;
    unsigned char* mbuf = malloc(WRITE_BUF_TOKENS * 2);

    for (int i = 0; i < nw; i++) {
        FILE* fin = fopen(tmps[i], "rb");
        if (!fin) { fprintf(stderr, "Cannot open chunk %s\n", tmps[i]); continue; }
        size_t n;
        while ((n = fread(mbuf, 2, WRITE_BUF_TOKENS, fin)) > 0) {
            fwrite(mbuf, 2, n, fout);
            total_toks += n;
        }
        fclose(fin);
        unlink(tmps[i]);
        printf("  Merged chunk %d\n", i); fflush(stdout);
    }
    free(mbuf);
    fclose(fout);

    printf("\nDone.\n");
    printf("  Tokens  : %zu\n", total_toks);
    printf("  Output  : %s  (%.3f GB)\n", out, (double)(total_toks * 2) / 1073741824.0);
    printf("  Time    : %.1fs\n", elapsed);

    const int seq_len = 512;
    size_t seqs = total_toks > (size_t)seq_len ? (total_toks - seq_len) / seq_len : 0;
    printf("  Usable sequences (seq_len=%d): %zu\n", seq_len, seqs);

    for (int i = 0; i < nw; i++) free(tmps[i]);
    free(tmps); free(bounds); free(pids);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    const char* corpus = argc > 1 ? argv[1] : "corpus.txt";
    int nw = argc > 2 ? atoi(argv[2]) : (int)sysconf(_SC_NPROCESSORS_ONLN);

    train_tokenizer(corpus);
    tokenize_corpus(corpus, nw);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SPM_ENCODE   "sentencepiece/build/src/spm_encode"
#define MODEL_PREFIX "spm_model"
#define WRITE_BUF_TOKENS (1 << 20)

static size_t count_lines(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { perror("fopen"); return 0; }

    size_t n = 0;
    char buf[1 << 20];
    size_t bytes;
    while ((bytes = fread(buf, 1, sizeof(buf), f)) > 0)
        for (size_t i = 0; i < bytes; i++)
            if (buf[i] == '\n') n++;

    fclose(f);
    return n;
}

static void fmt_eta(double secs, char* out, size_t sz) {
    int h = (int)(secs / 3600);
    int m = (int)((secs - h * 3600) / 60);
    int s = (int)(secs) % 60;
    if (h > 0) snprintf(out, sz, "%dh%02dm%02ds", h, m, s);
    else        snprintf(out, sz, "%dm%02ds", m, s);
}

int main(int argc, char* argv[]) {
    const char* input_file = argc > 1 ? argv[1] : "corpus.txt";

    char output_file[512];
    snprintf(output_file, sizeof(output_file), "%s.bin", input_file);

    printf("Counting lines in %s ...\n", input_file); fflush(stdout);
    size_t total_lines_expected = count_lines(input_file);
    printf("  %zu lines\n", total_lines_expected);

    printf("Tokenizing: %s -> %s\n", input_file, output_file); fflush(stdout);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s --model=%s.model --output_format=id < \"%s\"",
             SPM_ENCODE, MODEL_PREFIX, input_file);

    FILE* spm = popen(cmd, "r");
    if (!spm) { perror("popen"); return 1; }

    FILE* out = fopen(output_file, "wb");
    if (!out) { perror("fopen"); pclose(spm); return 1; }

    unsigned char* wbuf = (unsigned char*)malloc(WRITE_BUF_TOKENS * 2);
    if (!wbuf) { fputs("OOM\n", stderr); fclose(out); pclose(spm); return 1; }

    char*   line     = NULL;
    size_t  line_cap = 0;
    ssize_t line_len;
    size_t  wbuf_pos     = 0;
    size_t  total_tokens = 0;
    size_t  total_lines  = 0;

    struct timespec t0, tnow;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while ((line_len = getline(&line, &line_cap, spm)) > 0) {
        char* ptr = line;
        char* end;

        while (*ptr) {
            while (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')
                ptr++;
            if (!*ptr) break;

            long id = strtol(ptr, &end, 10);
            if (end == ptr) break;
            ptr = end;

            unsigned short tok = (unsigned short)(id & 0xFFFF);
            wbuf[wbuf_pos * 2]     = (tok >> 8) & 0xFF;
            wbuf[wbuf_pos * 2 + 1] =  tok       & 0xFF;
            wbuf_pos++;
            total_tokens++;

            if (wbuf_pos == WRITE_BUF_TOKENS) {
                if (fwrite(wbuf, 2, WRITE_BUF_TOKENS, out) != WRITE_BUF_TOKENS) {
                    perror("fwrite"); goto cleanup;
                }
                wbuf_pos = 0;
            }
        }

        total_lines++;

        if (total_lines % 10000 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            double elapsed = (tnow.tv_sec  - t0.tv_sec) +
                             (tnow.tv_nsec - t0.tv_nsec) * 1e-9;
            double frac    = total_lines_expected > 0
                             ? (double)total_lines / (double)total_lines_expected
                             : 0.0;
            double eta_sec = frac > 0.0 ? elapsed / frac - elapsed : 0.0;

            char eta_str[32];
            fmt_eta(eta_sec, eta_str, sizeof(eta_str));

            printf("\r  %5.1f%% | lines: %7zu/%zu | tokens: %10zu | "
                   "%.0f tok/s | %.3f GB | ETA: %-12s",
                   frac * 100.0,
                   total_lines, total_lines_expected,
                   total_tokens,
                   total_tokens / elapsed,
                   (double)(total_tokens * 2) / (1024.0 * 1024.0 * 1024.0),
                   eta_str);
            fflush(stdout);
        }
    }

    if (wbuf_pos > 0) {
        if (fwrite(wbuf, 2, wbuf_pos, out) != wbuf_pos)
            perror("fwrite");
    }

cleanup:
    free(line);
    free(wbuf);
    fclose(out);

    int rc = pclose(spm);
    if (rc != 0)
        fprintf(stderr, "\nspm_encode exited with status %d\n", rc);

    printf("\n\nDone.\n");
    printf("  Lines  : %zu\n",  total_lines);
    printf("  Tokens : %zu\n",  total_tokens);
    printf("  Output : %s  (%.3f GB)\n", output_file,
           (double)(total_tokens * 2) / (1024.0 * 1024.0 * 1024.0));

    const int seq_len = 512;
    size_t sequences = total_tokens > (size_t)seq_len
                       ? (total_tokens - seq_len) / seq_len
                       : 0;
    printf("  Usable sequences (seq_len=%d): %zu\n", seq_len, sequences);

    return rc != 0 ? 1 : 0;
}
/* data.c -- download MAESTRO v3 (piano MIDI) and emit corpus.bin as a
 * flat stream of event tokens (see tokens.h for the vocabulary).
 *
 * Pipeline:
 *   1. curl maestro-v3.0.0-midi.zip (~80 MB) if not present
 *   2. unzip it into ./maestro-v3.0.0/
 *   3. walk every .mid file, parse it, tokenize it, append to corpus.bin
 *
 * The MIDI parser handles SMF format 0/1, tempo changes, and running
 * status.  Only note-on/off and tempo events are kept; everything else
 * (CC, pitch bend, sysex, etc.) is dropped.
 *
 * Requires (apt): curl unzip
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "tokens.h"

#define MAESTRO_URL "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
#define MAESTRO_ZIP "maestro-v3.0.0-midi.zip"
#define MAESTRO_DIR "maestro-v3.0.0"
#define CORPUS_PATH "corpus.bin"

/* ---------------------------------------------------------------------- */
/*  Event record produced by the MIDI parser.                             */
/* ---------------------------------------------------------------------- */

typedef enum { EV_NOTE_OFF = 0, EV_NOTE_ON = 1, EV_TEMPO = 2 } EvKind;

typedef struct {
    uint64_t tick;     /* absolute ticks from start of file */
    uint8_t  kind;
    uint8_t  pitch;    /* for note events */
    uint32_t value;    /* velocity for notes, us-per-quarter for tempo */
} Ev;

static int ev_cmp(const void* a, const void* b) {
    const Ev* x = (const Ev*)a;
    const Ev* y = (const Ev*)b;
    if (x->tick != y->tick) return (x->tick < y->tick) ? -1 : 1;
    /* tempo first at ties so it's in effect when sibling notes are processed */
    if (x->kind != y->kind) return (x->kind == EV_TEMPO) ? -1 : 1;
    return 0;
}

/* ---------------------------------------------------------------------- */
/*  SMF parser: read whole file, collect a sorted Ev[] across all tracks. */
/*  Returns event count, or -1 on parse error.                            */
/* ---------------------------------------------------------------------- */

static long parse_smf(const char* path, Ev** out_events, int* out_division)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 14) { fclose(f); return -1; }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    fclose(f);

    if (memcmp(buf, "MThd", 4) != 0) { free(buf); return -1; }
    uint32_t hlen     = ((uint32_t)buf[4]<<24)|((uint32_t)buf[5]<<16)|((uint32_t)buf[6]<<8)|buf[7];
    int      ntrks    = (buf[10]<<8)|buf[11];
    int      division = (buf[12]<<8)|buf[13];
    if (division & 0x8000) { free(buf); return -1; }  /* SMPTE timing: skip */

    unsigned char* p   = buf + 8 + hlen;
    unsigned char* end = buf + sz;

    size_t cap = 4096, n = 0;
    Ev*    ev  = (Ev*)malloc(cap * sizeof(Ev));

    for (int t = 0; t < ntrks && p + 8 <= end; t++) {
        if (memcmp(p, "MTrk", 4) != 0) break;
        uint32_t tlen = ((uint32_t)p[4]<<24)|((uint32_t)p[5]<<16)|((uint32_t)p[6]<<8)|p[7];
        unsigned char* q    = p + 8;
        unsigned char* qend = q + tlen;
        if (qend > end) qend = end;
        p = qend;

        uint64_t tick    = 0;
        uint8_t  running = 0;

        while (q < qend) {
            /* delta-time VLQ */
            uint32_t d = 0;
            while (q < qend) {
                uint8_t b = *q++;
                d = (d << 7) | (b & 0x7F);
                if (!(b & 0x80)) break;
            }
            tick += d;
            if (q >= qend) break;

            uint8_t status = *q;
            if (status >= 0x80) { running = status; q++; }
            else                status = running;

            if (status == 0xFF) {
                /* meta event */
                if (q >= qend) break;
                uint8_t  mtype = *q++;
                uint32_t mlen  = 0;
                while (q < qend) {
                    uint8_t b = *q++;
                    mlen = (mlen << 7) | (b & 0x7F);
                    if (!(b & 0x80)) break;
                }
                if (mtype == 0x51 && mlen == 3 && q + 3 <= qend) {
                    uint32_t tempo = ((uint32_t)q[0]<<16)|((uint32_t)q[1]<<8)|q[2];
                    if (n == cap) { cap *= 2; ev = (Ev*)realloc(ev, cap * sizeof(Ev)); }
                    ev[n++] = (Ev){tick, EV_TEMPO, 0, tempo};
                }
                q += mlen;
            } else if (status == 0xF0 || status == 0xF7) {
                /* sysex: skip */
                uint32_t slen = 0;
                while (q < qend) {
                    uint8_t b = *q++;
                    slen = (slen << 7) | (b & 0x7F);
                    if (!(b & 0x80)) break;
                }
                q += slen;
            } else {
                /* channel voice */
                uint8_t hi    = status & 0xF0;
                int     nbyte = (hi == 0xC0 || hi == 0xD0) ? 1 : 2;
                if (q + nbyte > qend) break;
                uint8_t d1 = q[0];
                uint8_t d2 = (nbyte == 2) ? q[1] : 0;
                q += nbyte;
                if (hi == 0x90 || hi == 0x80) {
                    if (n == cap) { cap *= 2; ev = (Ev*)realloc(ev, cap * sizeof(Ev)); }
                    EvKind k = (hi == 0x90 && d2 > 0) ? EV_NOTE_ON : EV_NOTE_OFF;
                    ev[n++] = (Ev){tick, (uint8_t)k, d1, d2};
                }
            }
        }
    }

    free(buf);
    qsort(ev, n, sizeof(Ev), ev_cmp);
    *out_events   = ev;
    *out_division = division;
    return (long)n;
}

/* ---------------------------------------------------------------------- */
/*  Tokenizer: walk sorted events, emit event tokens to `out`.            */
/* ---------------------------------------------------------------------- */

static size_t tokenize_and_emit(FILE* out, const Ev* ev, size_t n, int division)
{
    if (n == 0) return 0;
    uint32_t tempo        = 500000;   /* default 120 BPM */
    double   accum_ms     = 0.0;      /* unflushed gap (incl. sub-grid residue) */
    uint64_t last_tick    = 0;
    int      last_vel_bin = -1;
    size_t   emitted      = 0;

    for (size_t i = 0; i < n; i++) {
        uint64_t dt = ev[i].tick - last_tick;
        last_tick  = ev[i].tick;
        accum_ms  += (double)dt * (double)tempo / (double)division / 1000.0;

        /* Flush whole 10ms bins, leave any sub-10ms residue for next time. */
        int delta_ms = (int)accum_ms;
        accum_ms -= delta_ms;
        while (delta_ms >= TIME_BIN_MS) {
            int bin = delta_ms / TIME_BIN_MS;
            if (bin > NUM_TIME_BINS) bin = NUM_TIME_BINS;
            fputc((unsigned char)(TOK_TIME_BASE + bin - 1), out);
            emitted++;
            delta_ms -= bin * TIME_BIN_MS;
        }
        accum_ms += delta_ms;

        if (ev[i].kind == EV_TEMPO) {
            tempo = ev[i].value;
            continue;
        }

        int pitch = ev[i].pitch;
        if (pitch < MIN_PITCH || pitch > MAX_PITCH) continue;
        int p_idx = pitch - MIN_PITCH;

        if (ev[i].kind == EV_NOTE_ON) {
            int vel_bin = (int)(ev[i].value / VEL_BIN_SIZE);
            if (vel_bin >= NUM_VEL_BINS) vel_bin = NUM_VEL_BINS - 1;
            if (vel_bin != last_vel_bin) {
                fputc((unsigned char)(TOK_VEL_BASE + vel_bin), out);
                emitted++;
                last_vel_bin = vel_bin;
            }
            fputc((unsigned char)(TOK_NOTE_ON_BASE + p_idx), out);
            emitted++;
        } else {
            fputc((unsigned char)(TOK_NOTE_OFF_BASE + p_idx), out);
            emitted++;
        }
    }
    return emitted;
}

/* ---------------------------------------------------------------------- */

int main(void)
{
    if (access(MAESTRO_DIR, F_OK) != 0) {
        if (access(MAESTRO_ZIP, F_OK) != 0) {
            if (system("curl -L --fail --retry 3 -o " MAESTRO_ZIP " " MAESTRO_URL) != 0) {
                fprintf(stderr, "download failed\n");
                return 1;
            }
        }
        if (system("unzip -q " MAESTRO_ZIP) != 0) {
            fprintf(stderr, "unzip failed (is the `unzip` package installed?)\n");
            return 1;
        }
    }

    FILE* out = fopen(CORPUS_PATH, "wb");
    if (!out) { perror(CORPUS_PATH); return 1; }

    FILE* find = popen("find " MAESTRO_DIR " \\( -name '*.mid' -o -name '*.midi' \\) | sort", "r");
    if (!find) { fclose(out); return 1; }

    char   path[2048];
    size_t files = 0, tokens = 0;
    while (fgets(path, sizeof(path), find)) {
        size_t L = strlen(path);
        if (L > 0 && path[L-1] == '\n') path[--L] = '\0';

        Ev*  ev = NULL;
        int  division = 0;
        long n = parse_smf(path, &ev, &division);
        if (n < 0) {
            fprintf(stderr, "  skip (parse error): %s\n", path);
            free(ev);
            continue;
        }
        tokens += tokenize_and_emit(out, ev, (size_t)n, division);
        free(ev);
        files++;
        if (files % 50 == 0)
            fprintf(stderr, "  %zu files, %.2f M tokens\n", files, tokens / 1e6);
    }
    pclose(find);
    fclose(out);

    fprintf(stderr, "Wrote %zu tokens from %zu MIDI files to %s\n",
            tokens, files, CORPUS_PATH);

    /* Clean up the downloaded archive and extracted dataset; keep corpus.bin. */
    (void)system("rm -rf " MAESTRO_ZIP " " MAESTRO_DIR);
    return 0;
}

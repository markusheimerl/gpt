/*
 * server.c – single-file HTTP chat server wrapping GPT inference
 *
 * Build:
 *   clang -O3 -march=native -ffast-math -Wall \
 *         server.c -lopenblas -lm -lpthread -o server.out
 *
 * Usage:
 *   ./server.out [model.bin] [port]          (port default: 8080)
 *
 * One inference runs at a time; extra users queue up (they see a spinner
 * while they wait, then tokens stream in as soon as it's their turn).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <stdbool.h>
#include <cblas.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <stdatomic.h>

#define SPM_ENCODE "sentencepiece/build/src/spm_encode"
#define SPM_DECODE "sentencepiece/build/src/spm_decode"
#define SPM_MODEL  "spm_model.model"

#define SEQ_LEN     512
#define TEMPERATURE 0.7f
#define MAX_NEW     400
#define MAX_TURNS   64
#define MAX_TEXT    8192

/* ═══════════════════════════════════════════════════════════════════
   Tokenisation (same as infer.c, but using different tmp file names
   to be explicit – inference is serialised anyway so no races)
 ═══════════════════════════════════════════════════════════════════ */

static int spm_encode(const char *text, unsigned short *out, int max_tokens) {
    FILE *f = fopen("/tmp/srv_spm_in.txt", "w");
    if (!f) return 0;
    fprintf(f, "%s\n", text);
    fclose(f);
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "%s --model=%s --output_format=id < /tmp/srv_spm_in.txt",
             SPM_ENCODE, SPM_MODEL);
    FILE *p = popen(cmd, "r");
    if (!p) return 0;
    int n = 0;
    char line[65536];
    while (fgets(line, sizeof(line), p) && n < max_tokens) {
        char *ptr = line, *ep;
        while (*ptr) {
            while (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r') ptr++;
            if (!*ptr) break;
            long id = strtol(ptr, &ep, 10);
            if (ep == ptr) break;
            ptr = ep;
            if (n < max_tokens) out[n++] = (unsigned short)(id & 0xFFFF);
        }
    }
    pclose(p);
    return n;
}

static void spm_decode_ids(const unsigned short *ids, int n, char *buf, int bufsz) {
    buf[0] = '\0';
    if (n <= 0) return;
    FILE *f = fopen("/tmp/srv_spm_ids.txt", "w");
    if (!f) return;
    for (int i = 0; i < n; i++) fprintf(f, i ? " %hu" : "%hu", ids[i]);
    fprintf(f, "\n");
    fclose(f);
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "%s --model=%s --input_format=id < /tmp/srv_spm_ids.txt",
             SPM_DECODE, SPM_MODEL);
    FILE *p = popen(cmd, "r");
    if (!p) return;
    if (fgets(buf, bufsz, p)) {
        int l = (int)strlen(buf);
        if (l > 0 && buf[l-1] == '\n') buf[l-1] = '\0';
    }
    pclose(p);
}

/* ═══════════════════════════════════════════════════════════════════
   HTTP chunked-transfer helpers (forward-declared for stream_push)
 ═══════════════════════════════════════════════════════════════════ */

static void write_chunk(int fd, const char *data, size_t len) {
    if (len == 0) return;
    char hdr[32];
    int hl = snprintf(hdr, sizeof(hdr), "%zx\r\n", len);
    (void)write(fd, hdr, hl);
    (void)write(fd, data, len);
    (void)write(fd, "\r\n", 2);
}

static void end_chunked(int fd) {
    (void)write(fd, "0\r\n\r\n", 5);
}

/* ═══════════════════════════════════════════════════════════════════
   Streaming token printer  (out_fd >= 0 → HTTP chunks; < 0 → stdout)
 ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    unsigned short anchor;
    unsigned short gen[2048];
    int            n_gen;
    char           prev[65536];
    int            out_fd;
} StreamState;

static void stream_init(StreamState *s, unsigned short anchor_tok, int out_fd) {
    s->anchor = anchor_tok;
    s->n_gen  = 0;
    s->out_fd = out_fd;
    s->prev[0] = '\0';
    spm_decode_ids(&anchor_tok, 1, s->prev, sizeof(s->prev));
}

static void stream_push(StreamState *s, unsigned short tok) {
    if (s->n_gen >= (int)(sizeof(s->gen)/sizeof(s->gen[0]))) return;
    s->gen[s->n_gen++] = tok;

    unsigned short tmp[2049];
    tmp[0] = s->anchor;
    memcpy(tmp + 1, s->gen, s->n_gen * sizeof(unsigned short));

    char curr[65536];
    spm_decode_ids(tmp, s->n_gen + 1, curr, sizeof(curr));

    size_t prev_len = strlen(s->prev);
    if (strlen(curr) > prev_len) {
        const char *new_text = curr + prev_len;
        size_t      new_len  = strlen(new_text);
        if (s->out_fd >= 0)
            write_chunk(s->out_fd, new_text, new_len);
        else {
            fputs(new_text, stdout);
            fflush(stdout);
        }
    }
    strcpy(s->prev, curr);
}

/* ═══════════════════════════════════════════════════════════════════
   Model structs (identical to infer.c)
 ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *W1, *W2, *h, *s, *out;
    int input_dim, hidden_dim, output_dim;
} MLP;

typedef struct {
    float *W_q, *W_k, *W_v, *W_o;
    float *q, *k, *v, *z, *out;
    float *scores, *probs;
    float *K_cache, *V_cache;
    int seq_len, d_model, num_heads, head_dim;
    float scale;
    bool is_causal, use_rope;
} Attention;

typedef struct {
    Attention **attn;
    MLP       **mlp;
    float *norm1, *norm2;
    int num_layers, d_model;
} Transformer;

typedef struct {
    float       *token_embedding;
    Transformer *transformer;
    float       *x, *final_norm, *logits;
    int seq_len, d_model, hidden_dim, num_layers, vocab_size;
} GPT;

/* ─── MLP ─────────────────────────────────────────────────────────── */

static MLP *mlp_alloc(int in, int hid, int out_d) {
    MLP *m = malloc(sizeof(MLP));
    m->input_dim = in; m->hidden_dim = hid; m->output_dim = out_d;
    m->W1 = malloc((size_t)in  * hid * sizeof(float));
    m->W2 = malloc((size_t)hid * out_d * sizeof(float));
    m->h  = malloc(hid * sizeof(float));
    m->s  = malloc(hid * sizeof(float));
    m->out = malloc(out_d * sizeof(float));
    return m;
}

static void mlp_free(MLP *m) {
    free(m->W1); free(m->W2); free(m->h); free(m->s); free(m->out); free(m);
}

static void mlp_forward(MLP *m, float *x) {
    cblas_sgemv(CblasRowMajor, CblasTrans,
                m->input_dim, m->hidden_dim,
                1.0f, m->W1, m->hidden_dim, x, 1, 0.0f, m->h, 1);
    for (int i = 0; i < m->hidden_dim; i++) {
        float h = m->h[i];
        m->s[i] = h / (1.0f + expf(-h));
    }
    cblas_sgemv(CblasRowMajor, CblasTrans,
                m->hidden_dim, m->output_dim,
                1.0f, m->W2, m->output_dim, m->s, 1, 0.0f, m->out, 1);
}

static MLP *mlp_deserialize(FILE *f) {
    int in, hid, out_d;
    fread(&in,    sizeof(int), 1, f);
    fread(&hid,   sizeof(int), 1, f);
    fread(&out_d, sizeof(int), 1, f);
    MLP *m = mlp_alloc(in, hid, out_d);
    fread(m->W1, sizeof(float), (size_t)in  * hid,   f);
    fread(m->W2, sizeof(float), (size_t)hid * out_d, f);
    return m;
}

/* ─── Attention ───────────────────────────────────────────────────── */

static Attention *attn_alloc(int seq_len, int d_model, int num_heads,
                              bool is_causal, bool use_rope) {
    Attention *a = malloc(sizeof(Attention));
    a->seq_len = seq_len; a->d_model = d_model;
    a->num_heads = num_heads; a->head_dim = d_model / num_heads;
    a->scale = 1.0f / sqrtf((float)a->head_dim);
    a->is_causal = is_causal; a->use_rope = use_rope;
    size_t w = (size_t)d_model * d_model;
    a->W_q = malloc(w * sizeof(float)); a->W_k = malloc(w * sizeof(float));
    a->W_v = malloc(w * sizeof(float)); a->W_o = malloc(w * sizeof(float));
    a->q   = malloc(d_model * sizeof(float)); a->k = malloc(d_model * sizeof(float));
    a->v   = malloc(d_model * sizeof(float)); a->z = malloc(d_model * sizeof(float));
    a->out = malloc(d_model * sizeof(float));
    a->scores  = malloc(seq_len * sizeof(float));
    a->probs   = malloc(seq_len * sizeof(float));
    a->K_cache = malloc((size_t)seq_len * d_model * sizeof(float));
    a->V_cache = malloc((size_t)seq_len * d_model * sizeof(float));
    return a;
}

static void attn_free(Attention *a) {
    free(a->W_q); free(a->W_k); free(a->W_v); free(a->W_o);
    free(a->q); free(a->k); free(a->v); free(a->z); free(a->out);
    free(a->scores); free(a->probs);
    free(a->K_cache); free(a->V_cache);
    free(a);
}

static void rope_apply(float *q, float *k, int pos, int d_model) {
    for (int dp = 0; dp < d_model / 2; dp++) {
        int d = dp * 2;
        float theta = powf(10000.0f, -((float)d / (float)d_model));
        float angle = (float)pos * theta;
        float ca = cosf(angle), sa = sinf(angle);
        float q0 = q[d], q1 = q[d+1];
        q[d]   = q0*ca - q1*sa; q[d+1] = q0*sa + q1*ca;
        float k0 = k[d], k1 = k[d+1];
        k[d]   = k0*ca - k1*sa; k[d+1] = k0*sa + k1*ca;
    }
}

static void attn_forward(Attention *a, float *x, int pos) {
    int d = a->d_model, hd = a->head_dim, nh = a->num_heads;
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, a->W_q, d, x, 1, 0.0f, a->q, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, a->W_k, d, x, 1, 0.0f, a->k, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, a->W_v, d, x, 1, 0.0f, a->v, 1);
    if (a->use_rope) rope_apply(a->q, a->k, pos, d);
    memcpy(a->K_cache + pos * d, a->k, d * sizeof(float));
    memcpy(a->V_cache + pos * d, a->v, d * sizeof(float));
    memset(a->z, 0, d * sizeof(float));
    for (int h = 0; h < nh; h++) {
        float *q_h = a->q + h * hd;
        float *z_h = a->z + h * hd;
        float max_s = -1e30f;
        for (int j = 0; j <= pos; j++) {
            float sc = cblas_sdot(hd, q_h, 1, a->K_cache + j*d + h*hd, 1) * a->scale;
            a->scores[j] = sc;
            if (sc > max_s) max_s = sc;
        }
        float sum = 0.0f;
        for (int j = 0; j <= pos; j++) {
            a->probs[j] = expf(a->scores[j] - max_s);
            sum += a->probs[j];
        }
        float inv = 1.0f / sum;
        for (int j = 0; j <= pos; j++)
            cblas_saxpy(hd, a->probs[j] * inv,
                        a->V_cache + j*d + h*hd, 1, z_h, 1);
    }
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, a->W_o, d, a->z, 1, 0.0f, a->out, 1);
}

static Attention *attn_deserialize(FILE *f, int seq_len) {
    int d_model; bool is_causal, use_rope;
    fread(&d_model,   sizeof(int),  1, f);
    fread(&is_causal, sizeof(bool), 1, f);
    fread(&use_rope,  sizeof(bool), 1, f);
    Attention *a = attn_alloc(seq_len, d_model, 8, is_causal, use_rope);
    size_t w = (size_t)d_model * d_model;
    fread(a->W_q, sizeof(float), w, f);
    fread(a->W_k, sizeof(float), w, f);
    fread(a->W_v, sizeof(float), w, f);
    fread(a->W_o, sizeof(float), w, f);
    return a;
}

/* ─── Transformer ─────────────────────────────────────────────────── */

static void rmsnorm(float *out, const float *in, int d) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += in[i] * in[i];
    float scale = 1.0f / sqrtf(ss / d + 1e-6f);
    for (int i = 0; i < d; i++) out[i] = in[i] * scale;
}

static Transformer *transformer_alloc(int d_model, int num_layers) {
    Transformer *t = malloc(sizeof(Transformer));
    t->d_model = d_model; t->num_layers = num_layers;
    t->attn  = malloc(num_layers * sizeof(Attention*));
    t->mlp   = malloc(num_layers * sizeof(MLP*));
    t->norm1 = malloc(d_model * sizeof(float));
    t->norm2 = malloc(d_model * sizeof(float));
    return t;
}

static void transformer_free(Transformer *t) {
    for (int i = 0; i < t->num_layers; i++) {
        attn_free(t->attn[i]);
        mlp_free(t->mlp[i]);
    }
    free(t->attn); free(t->mlp);
    free(t->norm1); free(t->norm2);
    free(t);
}

static void transformer_forward(Transformer *t, float *x, int pos) {
    int d = t->d_model;
    for (int l = 0; l < t->num_layers; l++) {
        rmsnorm(t->norm1, x, d);
        attn_forward(t->attn[l], t->norm1, pos);
        for (int i = 0; i < d; i++) x[i] += t->attn[l]->out[i];
        rmsnorm(t->norm2, x, d);
        mlp_forward(t->mlp[l], t->norm2);
        for (int i = 0; i < d; i++) x[i] += t->mlp[l]->out[i];
    }
}

static Transformer *transformer_deserialize(FILE *f, int seq_len) {
    int d_model, hidden_dim, num_layers; bool is_causal, use_rope;
    fread(&d_model,    sizeof(int),  1, f);
    fread(&hidden_dim, sizeof(int),  1, f);
    fread(&num_layers, sizeof(int),  1, f);
    fread(&is_causal,  sizeof(bool), 1, f);
    fread(&use_rope,   sizeof(bool), 1, f);
    (void)hidden_dim; (void)is_causal; (void)use_rope;
    Transformer *t = transformer_alloc(d_model, num_layers);
    for (int i = 0; i < num_layers; i++) {
        t->attn[i] = attn_deserialize(f, seq_len);
        t->mlp[i]  = mlp_deserialize(f);
    }
    return t;
}

/* ─── GPT ─────────────────────────────────────────────────────────── */

static void gpt_forward(GPT *gpt, unsigned short token, int pos) {
    int d = gpt->d_model;
    memcpy(gpt->x, gpt->token_embedding + (size_t)token * d, d * sizeof(float));
    transformer_forward(gpt->transformer, gpt->x, pos);
    rmsnorm(gpt->final_norm, gpt->x, d);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                gpt->vocab_size, d,
                1.0f, gpt->token_embedding, d,
                gpt->final_norm, 1,
                0.0f, gpt->logits, 1);
}

static GPT *gpt_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }
    int d_model, hidden_dim, num_layers, vocab_size;
    fread(&d_model,    sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    fread(&vocab_size, sizeof(int), 1, f);
    GPT *gpt = malloc(sizeof(GPT));
    gpt->seq_len = SEQ_LEN; gpt->d_model = d_model;
    gpt->hidden_dim = hidden_dim; gpt->num_layers = num_layers;
    gpt->vocab_size = vocab_size;
    gpt->token_embedding = malloc((size_t)vocab_size * d_model * sizeof(float));
    gpt->x          = malloc(d_model * sizeof(float));
    gpt->final_norm = malloc(d_model * sizeof(float));
    gpt->logits     = malloc((size_t)vocab_size * sizeof(float));
    fread(gpt->token_embedding, sizeof(float), (size_t)vocab_size * d_model, f);
    gpt->transformer = transformer_deserialize(f, SEQ_LEN);
    fclose(f);
    fprintf(stderr, "Loaded: d_model=%d hidden=%d layers=%d vocab=%d\n",
            d_model, hidden_dim, num_layers, vocab_size);
    return gpt;
}

static void gpt_free(GPT *gpt) {
    if (!gpt) return;
    free(gpt->token_embedding);
    free(gpt->x); free(gpt->final_norm); free(gpt->logits);
    transformer_free(gpt->transformer);
    free(gpt);
}

/* ─── Sampling ────────────────────────────────────────────────────── */

static unsigned short sample_token(float *logits, int vocab_size, float temperature) {
    float max_l = -1e30f;
    for (int v = 0; v < vocab_size; v++) {
        logits[v] /= temperature;
        if (logits[v] > max_l) max_l = logits[v];
    }
    float sum = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        logits[v] = expf(logits[v] - max_l);
        sum += logits[v];
    }
    float r = (float)rand() / (float)RAND_MAX, cumsum = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        cumsum += logits[v] / sum;
        if (r <= cumsum) return (unsigned short)v;
    }
    return (unsigned short)(vocab_size - 1);
}

/* ─── Prefill + Generate ──────────────────────────────────────────── */

static int prefill(GPT *gpt, const char *text, unsigned short *ctx, int pos) {
    unsigned short buf[512];
    int room = gpt->seq_len - pos - 2 - 256;
    if (room <= 0) return pos;
    int n = spm_encode(text, buf, room < 512 ? room : 512);
    for (int i = 0; i < n && pos < gpt->seq_len - 1; i++) {
        pos++;
        ctx[pos] = buf[i];
        gpt_forward(gpt, buf[i], pos);
    }
    return pos;
}

static int generate(GPT *gpt, unsigned short *ctx, int pos,
                    float temperature, int max_new,
                    unsigned short bos_tok, unsigned short user_tok,
                    int out_fd) {
    StreamState ss;
    stream_init(&ss, ctx[pos], out_fd);
    for (int i = 0; i < max_new && pos < gpt->seq_len - 1; i++) {
        unsigned short next = sample_token(gpt->logits, gpt->vocab_size, temperature);
        if (next == bos_tok || next == user_tok) break;
        stream_push(&ss, next);
        pos++;
        ctx[pos] = next;
        gpt_forward(gpt, next, pos);
    }
    if (out_fd < 0) printf("\n");
    return pos;
}

static unsigned short find_special(const char *text) {
    unsigned short buf[4];
    int n = spm_encode(text, buf, 4);
    return (n > 0) ? buf[0] : 0xFFFF;
}

/* ═══════════════════════════════════════════════════════════════════
   Global inference state (one model, one mutex = one chat at a time)
 ═══════════════════════════════════════════════════════════════════ */

static GPT            *g_gpt      = NULL;
static unsigned short  g_bos_tok  = 0xFFFF;
static unsigned short  g_user_tok = 0xFFFF;
static pthread_mutex_t g_infer_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_int      g_waiting  = 0;   /* threads blocked on infer_mu */

/* ═══════════════════════════════════════════════════════════════════
   Minimal JSON history parser
   Expects: [{"role":"user","text":"..."},{"role":"assistant","text":"..."},...]
 ═══════════════════════════════════════════════════════════════════ */

typedef struct { char role[16]; char text[MAX_TEXT]; } Turn;

/* Read a JSON string value starting AFTER the opening quote.
   Returns pointer past the closing quote, or NULL on error. */
static const char *json_read_string(const char *p, char *out, int outsz) {
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p+1)) {
            p++;
            char c;
            switch (*p) {
                case 'n':  c = '\n'; break;
                case 't':  c = '\t'; break;
                case 'r':  c = '\r'; break;
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                default:   c = *p;   break;
            }
            if (i < outsz - 1) out[i++] = c;
        } else {
            if (i < outsz - 1) out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    return (*p == '"') ? p + 1 : NULL;
}

static int parse_history(const char *json, Turn *turns, int max_turns) {
    int n = 0;
    const char *p = json;
    while (*p && n < max_turns) {
        while (*p && *p != '{') p++;
        if (!*p) break;
        p++;
        Turn *t = &turns[n];
        t->role[0] = t->text[0] = '\0';
        /* parse up to 2 key-value pairs per object */
        for (int field = 0; field < 2; field++) {
            while (*p && *p != '"' && *p != '}') p++;
            if (!*p || *p == '}') break;
            p++; /* skip opening " of key */
            char key[32]; int ki = 0;
            while (*p && *p != '"' && ki < 31) key[ki++] = *p++;
            key[ki] = '\0';
            if (*p == '"') p++;
            while (*p && *p != ':') p++;
            if (!*p) break;
            p++;
            while (*p && *p != '"') p++;
            if (!*p) break;
            p++; /* skip opening " of value */
            if (strcmp(key, "role") == 0)
                p = json_read_string(p, t->role, sizeof(t->role));
            else if (strcmp(key, "text") == 0)
                p = json_read_string(p, t->text, sizeof(t->text));
            else {
                /* skip unknown value */
                char tmp[32]; p = json_read_string(p, tmp, sizeof(tmp));
            }
            if (!p) break;
            while (*p && *p != ',' && *p != '}') p++;
            if (*p == ',') p++;
        }
        n++;
        while (*p && *p != '}') p++;
        if (*p == '}') p++;
    }
    return n;
}

/* Build the full prompt string from conversation history. */
static void build_prompt(char *buf, int bufsz, const Turn *turns, int n) {
    int off = 0;
    for (int i = 0; i < n; i++) {
        int rem = bufsz - off;
        if (rem <= 0) break;
        if (strcmp(turns[i].role, "user") == 0) {
            if (i == 0)
                off += snprintf(buf+off, rem, "<|bos|><|user|>\n%s\n<|assistant|>\n", turns[i].text);
            else
                off += snprintf(buf+off, rem, "<|user|>\n%s\n<|assistant|>\n", turns[i].text);
        } else {
            /* assistant turn: just emit the text; next user turn will add the marker */
            off += snprintf(buf+off, rem, "%s\n", turns[i].text);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
   HTTP utilities
 ═══════════════════════════════════════════════════════════════════ */

/* Send a complete HTTP response with a fixed body. */
static void http_send(int fd, int status, const char *ctype,
                      const char *body, int body_len) {
    char hdr[512];
    int hl = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d OK\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, ctype, body_len);
    (void)write(fd, hdr, hl);
    if (body_len > 0) (void)write(fd, body, body_len);
}

/* Read until \r\n\r\n. Returns header length (incl. blank line) or -1. */
static int read_headers(int fd, char *buf, int bufsz) {
    int n = 0;
    while (n < bufsz - 1) {
        ssize_t r = read(fd, buf + n, 1);
        if (r <= 0) return -1;
        n++;
        if (n >= 4 &&
            buf[n-4]=='\r' && buf[n-3]=='\n' &&
            buf[n-2]=='\r' && buf[n-1]=='\n')
            break;
    }
    buf[n] = '\0';
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
   Embedded HTML page
 ═══════════════════════════════════════════════════════════════════ */

static const char HTML_PAGE[] =
"<!DOCTYPE html>\n"
"<html lang='en'><head>\n"
"<meta charset='utf-8'>\n"
"<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
"<title>GPT Chat</title>\n"
"<style>\n"
"*{box-sizing:border-box;margin:0;padding:0}\n"
"body{font-family:monospace;background:#111;color:#ccc;display:flex;flex-direction:column;height:100vh;padding:16px;gap:12px}\n"
"h2{color:#7cf;font-size:1.1em;font-weight:normal}\n"
"#chat{flex:1;overflow-y:auto;border:1px solid #333;border-radius:6px;padding:12px;background:#0d0d0d;display:flex;flex-direction:column;gap:8px}\n"
".bubble{max-width:85%;padding:8px 12px;border-radius:6px;white-space:pre-wrap;word-break:break-word;line-height:1.5}\n"
".user{background:#1a3a5c;color:#9cf;align-self:flex-end}\n"
".assistant{background:#1a2a1a;color:#9f9;align-self:flex-start}\n"
".waiting{color:#fa0;font-style:italic}\n"
"#status{font-size:0.8em;color:#666;min-height:1em}\n"
"#input-row{display:flex;gap:8px}\n"
"#msg{flex:1;padding:10px;background:#1a1a1a;border:1px solid #444;color:#ccc;border-radius:6px;font-family:monospace;font-size:14px;resize:none;height:52px}\n"
"#msg:focus{outline:none;border-color:#7cf}\n"
"button{padding:10px 20px;background:#1a4a2a;border:1px solid #4a8a4a;color:#9f9;cursor:pointer;border-radius:6px;font-family:monospace;font-size:14px}\n"
"button:hover:not(:disabled){background:#2a6a3a}\n"
"button:disabled{opacity:0.4;cursor:not-allowed}\n"
"</style></head><body>\n"
"<h2>&#x25B6; GPT Chat</h2>\n"
"<div id='chat'></div>\n"
"<div id='status'></div>\n"
"<div id='input-row'>\n"
"  <textarea id='msg' placeholder='Type a message... (Enter to send, Shift+Enter for newline)'></textarea>\n"
"  <button id='send' onclick='sendMsg()'>Send</button>\n"
"</div>\n"
"<script>\n"
"var chatHistory = [];\n"
"var busy = false;\n"
"\n"
"function bubble(role, text) {\n"
"  var chat = document.getElementById('chat');\n"
"  var d = document.createElement('div');\n"
"  d.className = 'bubble ' + role;\n"
"  d.textContent = text;\n"
"  chat.appendChild(d);\n"
"  chat.scrollTop = chat.scrollHeight;\n"
"  return d;\n"
"}\n"
"\n"
"function setStatus(s) { document.getElementById('status').textContent = s; }\n"
"\n"
"async function sendMsg() {\n"
"  if (busy) return;\n"
"  var inp = document.getElementById('msg');\n"
"  var text = inp.value.trim();\n"
"  if (!text) return;\n"
"  inp.value = '';\n"
"  busy = true;\n"
"  document.getElementById('send').disabled = true;\n"
"  setStatus('');\n"
"\n"
"  chatHistory.push({role:'user', text:text});\n"
"  bubble('user', text);\n"
"\n"
"  var aDiv = bubble('assistant', '');\n"
"  var assistantText = '';\n"
"\n"
"  try {\n"
"    var resp = await fetch('/chat', {\n"
"      method: 'POST',\n"
"      headers: {'Content-Type': 'application/json'},\n"
"      body: JSON.stringify(chatHistory)\n"
"    });\n"
"    if (!resp.ok) { aDiv.textContent = '[error ' + resp.status + ']'; throw new Error(); }\n"
"\n"
"    var qpos = parseInt(resp.headers.get('X-Queue-Position') || '0');\n"
"    if (qpos > 0) setStatus('Waiting in queue (position ' + qpos + ')...');\n"
"\n"
"    var reader = resp.body.getReader();\n"
"    var decoder = new TextDecoder();\n"
"    var started = false;\n"
"    while (true) {\n"
"      var res = await reader.read();\n"
"      if (res.done) break;\n"
"      var chunk = decoder.decode(res.value, {stream:true});\n"
"      if (!started) { setStatus(''); started = true; }\n"
"      assistantText += chunk;\n"
"      aDiv.textContent = assistantText;\n"
"      document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;\n"
"    }\n"
"    chatHistory.push({role:'assistant', text:assistantText});\n"
"  } catch(e) {\n"
"    if (!assistantText) aDiv.textContent = '[connection error]';\n"
"  }\n"
"\n"
"  setStatus('');\n"
"  busy = false;\n"
"  document.getElementById('send').disabled = false;\n"
"  document.getElementById('msg').focus();\n"
"}\n"
"\n"
"document.getElementById('msg').addEventListener('keydown', function(e) {\n"
"  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }\n"
"});\n"
"</script>\n"
"</body></html>\n";

/* ═══════════════════════════════════════════════════════════════════
   Connection handler (one thread per connection)
 ═══════════════════════════════════════════════════════════════════ */

static void *handle_conn(void *arg) {
    int fd = (int)(intptr_t)arg;

    /* Disable Nagle so each chunk flushes immediately */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    char hdrbuf[8192];
    if (read_headers(fd, hdrbuf, sizeof(hdrbuf)) < 0) {
        close(fd);
        return NULL;
    }

    /* Parse first line: METHOD PATH HTTP/x.x */
    char method[8], path[256];
    if (sscanf(hdrbuf, "%7s %255s", method, path) != 2) {
        close(fd);
        return NULL;
    }

    /* ── GET / → serve chat page ── */
    if (strcmp(method, "GET") == 0 &&
        (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0)) {
        http_send(fd, 200, "text/html; charset=utf-8",
                  HTML_PAGE, (int)strlen(HTML_PAGE));
        close(fd);
        return NULL;
    }

    /* ── GET /status → queue depth ── */
    if (strcmp(method, "GET") == 0 && strcmp(path, "/status") == 0) {
        char body[64];
        int n = snprintf(body, sizeof(body), "%d\n", atomic_load(&g_waiting));
        http_send(fd, 200, "text/plain", body, n);
        close(fd);
        return NULL;
    }

    /* ── POST /chat → run inference ── */
    if (strcmp(method, "POST") == 0 && strcmp(path, "/chat") == 0) {

        /* Parse Content-Length */
        int clen = 0;
        const char *cl = strcasestr(hdrbuf, "Content-Length:");
        if (cl) clen = atoi(cl + 15);
        if (clen <= 0 || clen > 1024*1024) {
            http_send(fd, 400, "text/plain", "bad request\n", 12);
            close(fd);
            return NULL;
        }

        /* Read body */
        char *body = malloc(clen + 1);
        if (!body) { close(fd); return NULL; }
        int got = 0;
        while (got < clen) {
            ssize_t r = read(fd, body + got, clen - got);
            if (r <= 0) break;
            got += (int)r;
        }
        body[got] = '\0';

        /* Parse JSON history */
        Turn *turns = malloc(MAX_TURNS * sizeof(Turn));
        int   nturn = 0;
        if (turns) nturn = parse_history(body, turns, MAX_TURNS);
        free(body);

        if (nturn == 0 || strcmp(turns[nturn-1].role, "user") != 0) {
            http_send(fd, 400, "text/plain", "need at least one user turn\n", 28);
            free(turns);
            close(fd);
            return NULL;
        }

        /* Build full prompt */
        char *prompt = malloc(32768);
        if (!prompt) { free(turns); close(fd); return NULL; }
        build_prompt(prompt, 32768, turns, nturn);
        free(turns);

        /* Note our queue position (for the header) and wait for the lock */
        int qpos = atomic_fetch_add(&g_waiting, 1);
        pthread_mutex_lock(&g_infer_mu);
        atomic_fetch_sub(&g_waiting, 1);

        /* Send streaming response headers (includes queue position) */
        {
            char hdr[512];
            int hl = snprintf(hdr, sizeof(hdr),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/plain; charset=utf-8\r\n"
                "Transfer-Encoding: chunked\r\n"
                "Cache-Control: no-cache\r\n"
                "X-Accel-Buffering: no\r\n"
                "X-Queue-Position: %d\r\n"
                "Connection: close\r\n"
                "\r\n",
                qpos);
            (void)write(fd, hdr, hl);
        }

        /* Run inference */
        unsigned short *ctx = calloc(SEQ_LEN, sizeof(unsigned short));
        if (ctx) {
            int pos = prefill(g_gpt, prompt, ctx, -1);
            generate(g_gpt, ctx, pos, TEMPERATURE, MAX_NEW,
                     g_bos_tok, g_user_tok, fd);
            free(ctx);
        }
        free(prompt);

        pthread_mutex_unlock(&g_infer_mu);

        end_chunked(fd);
        close(fd);
        return NULL;
    }

    /* ── Anything else → 404 ── */
    http_send(fd, 404, "text/plain", "not found\n", 10);
    close(fd);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════
   main
 ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    srand((unsigned)time(NULL));
    signal(SIGPIPE, SIG_IGN);

    const char *model_path = NULL;
    int port = 8080;

    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], ".bin"))
            model_path = argv[i];
        else
            port = atoi(argv[i]);
    }

    /* Find latest trim model if not specified */
    if (!model_path) {
        FILE *fp = popen("ls -t *_gpt_trim.bin 2>/dev/null | head -n1", "r");
        static char auto_path[256];
        if (fp && fgets(auto_path, sizeof(auto_path), fp)) {
            auto_path[strcspn(auto_path, "\n")] = '\0';
            if (auto_path[0]) model_path = auto_path;
        }
        if (fp) pclose(fp);
    }
    if (!model_path) {
        fprintf(stderr, "Usage: %s [model.bin] [port]\n", argv[0]);
        return 1;
    }

    g_gpt = gpt_load(model_path);
    if (!g_gpt) return 1;

    g_bos_tok  = find_special("<|bos|>");
    g_user_tok = find_special("<|user|>");
    fprintf(stderr, "Special tokens: bos=%hu user=%hu\n", g_bos_tok, g_user_tok);

    /* Create server socket */
    int srv = socket(AF_INET6, SOCK_STREAM, 0);
    if (srv < 0) {
        /* Fall back to IPv4 if IPv6 is unavailable */
        srv = socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) { perror("socket"); return 1; }
        int opt = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr4 = {0};
        addr4.sin_family      = AF_INET;
        addr4.sin_addr.s_addr = INADDR_ANY;
        addr4.sin_port        = htons((uint16_t)port);
        if (bind(srv, (struct sockaddr*)&addr4, sizeof(addr4)) < 0) {
            perror("bind"); return 1;
        }
    } else {
        int opt = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        int off = 0;
        setsockopt(srv, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));
        struct sockaddr_in6 addr6 = {0};
        addr6.sin6_family = AF_INET6;
        addr6.sin6_addr   = in6addr_any;
        addr6.sin6_port   = htons((uint16_t)port);
        if (bind(srv, (struct sockaddr*)&addr6, sizeof(addr6)) < 0) {
            perror("bind"); return 1;
        }
    }

    if (listen(srv, 32) < 0) { perror("listen"); return 1; }

    fprintf(stderr, "Listening on http://0.0.0.0:%d/\n", port);

    /* Accept loop — each connection gets its own detached thread */
    while (1) {
        struct sockaddr_storage peer;
        socklen_t plen = sizeof(peer);
        int fd = accept(srv, (struct sockaddr*)&peer, &plen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }
        pthread_t tid;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_create(&tid, &attr, handle_conn, (void*)(intptr_t)fd);
        pthread_attr_destroy(&attr);
    }

    gpt_free(g_gpt);
    return 0;
}

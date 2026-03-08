#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <stdbool.h>
#include <cblas.h>

#define SPM_ENCODE "sentencepiece/build/src/spm_encode"
#define SPM_DECODE "sentencepiece/build/src/spm_decode"
#define SPM_MODEL  "spm_model.model"

// ============================================================================
// Tokenisation via spm_encode / spm_decode (BPE)
// ============================================================================

/* Encode text → token ids.  Returns number of tokens written into out[]. */
static int spm_encode(const char* text, unsigned short* out, int max_tokens) {
    FILE* f = fopen("/tmp/infer_in.txt", "w");
    if (!f) return 0;
    fprintf(f, "%s\n", text);
    fclose(f);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "%s --model=%s --output_format=id < /tmp/infer_in.txt",
             SPM_ENCODE, SPM_MODEL);
    FILE* p = popen(cmd, "r");
    if (!p) return 0;

    int n = 0;
    char line[65536];
    while (fgets(line, sizeof(line), p) && n < max_tokens) {
        char* ptr = line, *ep;
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

/* Decode a sequence of token ids to text (returned in caller-supplied buf). */
static void spm_decode_ids(const unsigned short* ids, int n, char* buf, int bufsz) {
    buf[0] = '\0';
    if (n <= 0) return;

    FILE* f = fopen("/tmp/infer_ids.txt", "w");
    if (!f) return;
    for (int i = 0; i < n; i++) fprintf(f, i ? " %hu" : "%hu", ids[i]);
    fprintf(f, "\n");
    fclose(f);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "%s --model=%s --input_format=id < /tmp/infer_ids.txt",
             SPM_DECODE, SPM_MODEL);
    FILE* p = popen(cmd, "r");
    if (!p) return;
    if (fgets(buf, bufsz, p)) {
        int l = (int)strlen(buf);
        if (l > 0 && buf[l-1] == '\n') buf[l-1] = '\0';
    }
    pclose(p);
}

/*
 * Streaming token printer.
 * We decode [anchor_id, ...generated_so_far, new_id] and print only the
 * suffix that wasn't there before, using the same trick as train.c.
 */
typedef struct {
    unsigned short anchor;          /* last prompt token — decoder boundary */
    unsigned short gen[2048];       /* generated tokens so far              */
    int            n_gen;
    char           prev[65536];     /* text decoded last iteration           */
    bool           first;
} StreamState;

static void stream_init(StreamState* s, unsigned short anchor_tok) {
    s->anchor = anchor_tok;
    s->n_gen  = 0;
    s->first  = true;
    s->prev[0] = '\0';
    /* Prime prev with the anchor decoded alone. */
    spm_decode_ids(&anchor_tok, 1, s->prev, sizeof(s->prev));
}

static void stream_push(StreamState* s, unsigned short tok) {
    if (s->n_gen >= (int)(sizeof(s->gen)/sizeof(s->gen[0]))) return;
    s->gen[s->n_gen++] = tok;

    /* Build id list: anchor + all generated */
    unsigned short tmp[2049];
    tmp[0] = s->anchor;
    memcpy(tmp + 1, s->gen, s->n_gen * sizeof(unsigned short));

    char curr[65536];
    spm_decode_ids(tmp, s->n_gen + 1, curr, sizeof(curr));

    /* Print only the new suffix */
    size_t prev_len = strlen(s->prev);
    if (strlen(curr) > prev_len)
        fputs(curr + prev_len, stdout);
    fflush(stdout);

    strcpy(s->prev, curr);
    s->first = false;
}

// ============================================================================
// Model structures
// ============================================================================

typedef struct {
    float* W1; float* W2;
    float* h; float* s; float* out;
    int input_dim; int hidden_dim; int output_dim;
} MLP;

typedef struct {
    float* W_q; float* W_k; float* W_v; float* W_o;
    float* q; float* k; float* v; float* z; float* out;
    float* scores; float* probs;
    float* K_cache;   /* [seq_len x d_model] */
    float* V_cache;   /* [seq_len x d_model] */
    int seq_len; int d_model; int num_heads; int head_dim;
    float scale; bool is_causal; bool use_rope;
} Attention;

typedef struct {
    Attention** attn;
    MLP**       mlp;
    float* norm1; float* norm2;
    int num_layers; int d_model;
} Transformer;

typedef struct {
    float*       token_embedding; /* [vocab_size x d_model] */
    Transformer* transformer;
    float*       x;
    float*       final_norm;
    float*       logits;          /* [vocab_size] */
    int seq_len; int d_model; int hidden_dim;
    int num_layers; int vocab_size;
} GPT;

// ============================================================================
// MLP
// ============================================================================

static MLP* mlp_alloc(int in, int hid, int out_d) {
    MLP* m = malloc(sizeof(MLP));
    m->input_dim = in; m->hidden_dim = hid; m->output_dim = out_d;
    m->W1 = malloc((size_t)in  * hid * sizeof(float));
    m->W2 = malloc((size_t)hid * out_d * sizeof(float));
    m->h  = malloc(hid * sizeof(float));
    m->s  = malloc(hid * sizeof(float));
    m->out = malloc(out_d * sizeof(float));
    return m;
}

static void mlp_free(MLP* m) {
    free(m->W1); free(m->W2);
    free(m->h); free(m->s); free(m->out);
    free(m);
}

static void mlp_forward(MLP* m, float* x) {
    /* h = x W1  (row-major: x is [1 x in], W1 is [in x hid]) */
    cblas_sgemv(CblasRowMajor, CblasTrans,
                m->input_dim, m->hidden_dim,
                1.0f, m->W1, m->hidden_dim, x, 1, 0.0f, m->h, 1);
    /* swish */
    for (int i = 0; i < m->hidden_dim; i++) {
        float h = m->h[i];
        m->s[i] = h / (1.0f + expf(-h));
    }
    /* out = s W2 */
    cblas_sgemv(CblasRowMajor, CblasTrans,
                m->hidden_dim, m->output_dim,
                1.0f, m->W2, m->output_dim, m->s, 1, 0.0f, m->out, 1);
}

static MLP* mlp_deserialize(FILE* f) {
    int in, hid, out_d;
    fread(&in,    sizeof(int), 1, f);
    fread(&hid,   sizeof(int), 1, f);
    fread(&out_d, sizeof(int), 1, f);
    MLP* m = mlp_alloc(in, hid, out_d);
    fread(m->W1, sizeof(float), (size_t)in  * hid,   f);
    fread(m->W2, sizeof(float), (size_t)hid * out_d, f);
    return m;
}

// ============================================================================
// Attention (with KV-cache for autoregressive decoding)
// ============================================================================

static Attention* attn_alloc(int seq_len, int d_model, int num_heads,
                               bool is_causal, bool use_rope) {
    Attention* a = malloc(sizeof(Attention));
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
    a->scores = malloc(seq_len * sizeof(float));
    a->probs  = malloc(seq_len * sizeof(float));
    a->K_cache = malloc((size_t)seq_len * d_model * sizeof(float));
    a->V_cache = malloc((size_t)seq_len * d_model * sizeof(float));
    return a;
}

static void attn_free(Attention* a) {
    free(a->W_q); free(a->W_k); free(a->W_v); free(a->W_o);
    free(a->q); free(a->k); free(a->v); free(a->z); free(a->out);
    free(a->scores); free(a->probs);
    free(a->K_cache); free(a->V_cache);
    free(a);
}

static void rope_apply(float* q, float* k, int pos, int d_model) {
    for (int dp = 0; dp < d_model / 2; dp++) {
        int d = dp * 2;
        float theta = powf(10000.0f, -((float)d / (float)d_model));
        float angle = (float)pos * theta;
        float ca = cosf(angle), sa = sinf(angle);
        float q0 = q[d], q1 = q[d+1];
        q[d]   = q0*ca - q1*sa;  q[d+1] = q0*sa + q1*ca;
        float k0 = k[d], k1 = k[d+1];
        k[d]   = k0*ca - k1*sa;  k[d+1] = k0*sa + k1*ca;
    }
}

static void attn_forward(Attention* a, float* x, int pos) {
    int d = a->d_model, hd = a->head_dim, nh = a->num_heads;

    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, a->W_q, d, x, 1, 0.0f, a->q, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, a->W_k, d, x, 1, 0.0f, a->k, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, a->W_v, d, x, 1, 0.0f, a->v, 1);

    if (a->use_rope) rope_apply(a->q, a->k, pos, d);

    memcpy(a->K_cache + pos * d, a->k, d * sizeof(float));
    memcpy(a->V_cache + pos * d, a->v, d * sizeof(float));

    memset(a->z, 0, d * sizeof(float));
    for (int h = 0; h < nh; h++) {
        float* q_h = a->q + h * hd;
        float* z_h = a->z + h * hd;
        float max_s = -1e30f;
        for (int j = 0; j <= pos; j++) {
            float s = cblas_sdot(hd, q_h, 1, a->K_cache + j*d + h*hd, 1) * a->scale;
            a->scores[j] = s;
            if (s > max_s) max_s = s;
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

static Attention* attn_deserialize(FILE* f, int seq_len) {
    int d_model; bool is_causal, use_rope;
    fread(&d_model,   sizeof(int),  1, f);
    fread(&is_causal, sizeof(bool), 1, f);
    fread(&use_rope,  sizeof(bool), 1, f);
    Attention* a = attn_alloc(seq_len, d_model, 8, is_causal, use_rope);
    size_t w = (size_t)d_model * d_model;
    fread(a->W_q, sizeof(float), w, f);
    fread(a->W_k, sizeof(float), w, f);
    fread(a->W_v, sizeof(float), w, f);
    fread(a->W_o, sizeof(float), w, f);
    return a;
}

// ============================================================================
// Transformer
// ============================================================================

static void rmsnorm(float* out, const float* in, int d) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += in[i] * in[i];
    float scale = 1.0f / sqrtf(ss / d + 1e-6f);
    for (int i = 0; i < d; i++) out[i] = in[i] * scale;
}

static Transformer* transformer_alloc(int d_model, int num_layers) {
    Transformer* t = malloc(sizeof(Transformer));
    t->d_model = d_model; t->num_layers = num_layers;
    t->attn = malloc(num_layers * sizeof(Attention*));
    t->mlp  = malloc(num_layers * sizeof(MLP*));
    t->norm1 = malloc(d_model * sizeof(float));
    t->norm2 = malloc(d_model * sizeof(float));
    return t;
}

static void transformer_free(Transformer* t) {
    for (int i = 0; i < t->num_layers; i++) {
        attn_free(t->attn[i]);
        mlp_free(t->mlp[i]);
    }
    free(t->attn); free(t->mlp);
    free(t->norm1); free(t->norm2);
    free(t);
}

static void transformer_forward(Transformer* t, float* x, int pos) {
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

static Transformer* transformer_deserialize(FILE* f, int seq_len) {
    int d_model, hidden_dim, num_layers; bool is_causal, use_rope;
    fread(&d_model,    sizeof(int),  1, f);
    fread(&hidden_dim, sizeof(int),  1, f);
    fread(&num_layers, sizeof(int),  1, f);
    fread(&is_causal,  sizeof(bool), 1, f);
    fread(&use_rope,   sizeof(bool), 1, f);
    (void)hidden_dim; (void)is_causal; (void)use_rope;
    Transformer* t = transformer_alloc(d_model, num_layers);
    for (int i = 0; i < num_layers; i++) {
        t->attn[i] = attn_deserialize(f, seq_len);
        t->mlp[i]  = mlp_deserialize(f);
    }
    return t;
}

// ============================================================================
// GPT
// ============================================================================

static void gpt_forward(GPT* gpt, unsigned short token, int pos) {
    int d = gpt->d_model;
    memcpy(gpt->x, gpt->token_embedding + (size_t)token * d, d * sizeof(float));
    transformer_forward(gpt->transformer, gpt->x, pos);
    rmsnorm(gpt->final_norm, gpt->x, d);
    /* logits = final_norm * token_embedding^T */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                gpt->vocab_size, d,
                1.0f, gpt->token_embedding, d,
                gpt->final_norm, 1,
                0.0f, gpt->logits, 1);
}

static GPT* gpt_load(const char* path, int seq_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }

    int d_model, hidden_dim, num_layers, vocab_size;
    fread(&d_model,    sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    fread(&vocab_size, sizeof(int), 1, f);

    GPT* gpt = malloc(sizeof(GPT));
    gpt->seq_len = seq_len; gpt->d_model = d_model;
    gpt->hidden_dim = hidden_dim; gpt->num_layers = num_layers;
    gpt->vocab_size = vocab_size;
    gpt->token_embedding = malloc((size_t)vocab_size * d_model * sizeof(float));
    gpt->x          = malloc(d_model * sizeof(float));
    gpt->final_norm = malloc(d_model * sizeof(float));
    gpt->logits     = malloc((size_t)vocab_size * sizeof(float));

    fread(gpt->token_embedding, sizeof(float), (size_t)vocab_size * d_model, f);
    gpt->transformer = transformer_deserialize(f, seq_len);
    fclose(f);

    fprintf(stderr, "Loaded: d_model=%d  hidden=%d  layers=%d  vocab=%d\n",
            d_model, hidden_dim, num_layers, vocab_size);
    return gpt;
}

static void gpt_free(GPT* gpt) {
    if (!gpt) return;
    free(gpt->token_embedding);
    free(gpt->x); free(gpt->final_norm); free(gpt->logits);
    transformer_free(gpt->transformer);
    free(gpt);
}

// ============================================================================
// Sampling
// ============================================================================

static unsigned short sample(float* logits, int vocab_size, float temperature) {
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

// ============================================================================
// Generation
// ============================================================================

/*
 * Prefill pos starting at *pos_inout, feeding every token in the prompt
 * one-by-one into the KV-cache.  Returns the position of the last token
 * processed (whose logits are ready in gpt->logits).
 */
static int prefill(GPT* gpt, const char* text,
                   unsigned short* ctx, int pos) {
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

/*
 * Greedy/temperature-sampled autoregressive generation.
 * Returns final context position.
 */
static int generate(GPT* gpt, unsigned short* ctx, int pos,
                    float temperature, int max_new,
                    unsigned short bos_tok, unsigned short user_tok) {
    /* Use the last prompt token as the decoder anchor for streaming. */
    StreamState ss;
    stream_init(&ss, ctx[pos]);

    for (int i = 0; i < max_new && pos < gpt->seq_len - 1; i++) {
        unsigned short next = sample(gpt->logits, gpt->vocab_size, temperature);
        /* Stop at BOS or <|user|> — signals end of assistant turn */
        if (next == bos_tok || next == user_tok) break;
        stream_push(&ss, next);
        pos++;
        ctx[pos] = next;
        gpt_forward(gpt, next, pos);
    }
    printf("\n");
    return pos;
}

// ============================================================================
// Special-token lookup via spm_encode
// ============================================================================

static unsigned short find_special(const char* text) {
    unsigned short buf[4];
    int n = spm_encode(text, buf, 4);
    return (n > 0) ? buf[0] : 0xFFFF;
}

// ============================================================================
// Globals for signal handler
// ============================================================================

static GPT*           g_gpt     = NULL;
static unsigned short* g_ctx    = NULL;
static bool            g_interactive = true;

static void on_sigint(int sig) {
    (void)sig;
    if (g_interactive) printf("\n\nExiting...\n");
    gpt_free(g_gpt);
    free(g_ctx);
    exit(0);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    signal(SIGINT, on_sigint);
    // openblas_set_num_threads(6);

    const int   SEQ_LEN     = 512;
    const float TEMPERATURE = 0.7f;
    const int   MAX_NEW     = 256;

    /* Argument parsing: optional model path, then optional prompt words */
    int arg_offset = 1;
    const char* model_path = "checkpoint_gpt_trim.bin";
    if (argc > 1 && strstr(argv[1], ".bin")) {
        model_path = argv[1];
        arg_offset = 2;
    }

    g_gpt = gpt_load(model_path, SEQ_LEN);
    if (!g_gpt) return 1;

    g_ctx = calloc(SEQ_LEN, sizeof(unsigned short));

    /* Discover special tokens at runtime via the BPE model */
    unsigned short tok_bos  = find_special("<|bos|>");
    unsigned short tok_user = find_special("<|user|>");
    fprintf(stderr, "Special tokens: bos=%hu  user=%hu\n", tok_bos, tok_user);

    // ── Non-interactive: prompt from CLI args ─────────────────────────────
    if (argc > arg_offset) {
        g_interactive = false;

        char question[4096] = {0};
        for (int i = arg_offset; i < argc; i++) {
            if (i > arg_offset) strcat(question, " ");
            strncat(question, argv[i], sizeof(question) - strlen(question) - 1);
        }
        char turn[4096];
        snprintf(turn, sizeof(turn),
                 "<|bos|><|user|>\n%s\n<|assistant|>\n", question);

        int pos = prefill(g_gpt, turn, g_ctx, -1);
        generate(g_gpt, g_ctx, pos, TEMPERATURE, MAX_NEW, tok_bos, tok_user);

        gpt_free(g_gpt); free(g_ctx);
        return 0;
    }

    // ── Interactive multi-turn chat ───────────────────────────────────────
    int  pos        = -1;
    bool first_turn = true;
    char question[4096];

    while (1) {
        printf("\n\033[1;36m?\033[0m ");
        fflush(stdout);
        if (!fgets(question, sizeof(question), stdin)) break;
        question[strcspn(question, "\n")] = '\0';
        if (!strlen(question)) continue;

        /* Reset context if we're running out of room */
        if (g_gpt->seq_len - pos < 300) {
            fprintf(stderr, "\n[Context full — resetting]\n");
            pos = -1; first_turn = true;
        }

        char turn[4096];
        if (first_turn) {
            snprintf(turn, sizeof(turn),
                     "<|bos|><|user|>\n%s\n<|assistant|>\n", question);
            first_turn = false;
        } else {
            snprintf(turn, sizeof(turn),
                     "<|user|>\n%s\n<|assistant|>\n", question);
        }

        pos = prefill(g_gpt, turn, g_ctx, pos);

        printf("\033[1;32m>\033[0m ");
        fflush(stdout);

        pos = generate(g_gpt, g_ctx, pos, TEMPERATURE, MAX_NEW, tok_bos, tok_user);
    }

    on_sigint(0);
    return 0;
}
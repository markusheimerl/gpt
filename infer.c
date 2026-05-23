#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <stdbool.h>
#include <cblas.h>

extern void openblas_set_num_threads(int);

// ============================================================================
// Model structures
// ============================================================================

typedef struct {
    float *W1, *W2;
    float *h, *s, *out;
    int input_dim, hidden_dim, output_dim;
} MLP;

typedef struct {
    float *W_q, *W_k, *W_v, *W_o;
    float *q, *k, *v, *z, *out;
    float *scores, *probs;
    float *K_cache;   /* [seq_len x d_model] */
    float *V_cache;   /* [seq_len x d_model] */
    int seq_len, d_model, num_heads, head_dim;
    float scale;
    bool is_causal, use_rope;
} Attention;

typedef struct {
    Attention** attn;
    MLP**       mlp;
    float *norm1, *norm2;
    int num_layers, d_model;
} Transformer;

typedef struct {
    float* token_embedding;   /* [vocab_size x d_model] */
    Transformer* transformer;
    float *x, *final_norm;
    float* logits;            /* [vocab_size] */
    int seq_len, d_model, hidden_dim, num_layers, vocab_size;
} GPT;

// ============================================================================
// MLP
// ============================================================================

static MLP* mlp_alloc(int in, int hid, int out_d) {
    MLP* m = malloc(sizeof(MLP));
    m->input_dim = in; m->hidden_dim = hid; m->output_dim = out_d;
    m->W1  = malloc((size_t)in  * hid   * sizeof(float));
    m->W2  = malloc((size_t)hid * out_d * sizeof(float));
    m->h   = malloc(hid   * sizeof(float));
    m->s   = malloc(hid   * sizeof(float));
    m->out = malloc(out_d * sizeof(float));
    return m;
}

static void mlp_free(MLP* m) {
    free(m->W1); free(m->W2);
    free(m->h);  free(m->s); free(m->out);
    free(m);
}

static void mlp_forward(MLP* m, float* x) {
    cblas_sgemv(CblasRowMajor, CblasTrans,
                m->input_dim, m->hidden_dim,
                1.0f, m->W1, m->hidden_dim, x, 1, 0.0f, m->h, 1);
    for (int i = 0; i < m->hidden_dim; i++) {
        float h = m->h[i];
        m->s[i] = h / (1.0f + expf(-h));   /* swish */
    }
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
    size_t w1 = (size_t)in  * hid;
    size_t w2 = (size_t)hid * out_d;
    fread(m->W1, sizeof(float), w1, f);
    fread(m->W2, sizeof(float), w2, f);
    /* skip optimizer state: t (int) + m,v for W1 and W2 */
    int t; fread(&t, sizeof(int), 1, f);
    fseek(f, (long)((2 * w1 + 2 * w2) * sizeof(float)), SEEK_CUR);
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

    memcpy(a->K_cache + (size_t)pos * d, a->k, d * sizeof(float));
    memcpy(a->V_cache + (size_t)pos * d, a->v, d * sizeof(float));

    memset(a->z, 0, d * sizeof(float));
    for (int h = 0; h < nh; h++) {
        float* q_h = a->q + h * hd;
        float* z_h = a->z + h * hd;
        float max_s = -1e30f;
        for (int j = 0; j <= pos; j++) {
            float s = cblas_sdot(hd, q_h, 1, a->K_cache + (size_t)j*d + h*hd, 1) * a->scale;
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
                        a->V_cache + (size_t)j*d + h*hd, 1, z_h, 1);
    }
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, a->W_o, d, a->z, 1, 0.0f, a->out, 1);
}

static Attention* attn_deserialize(FILE* f, int seq_len) {
    int d_model;
    bool is_causal, use_rope;
    fread(&d_model,   sizeof(int),  1, f);
    fread(&is_causal, sizeof(bool), 1, f);
    fread(&use_rope,  sizeof(bool), 1, f);
    Attention* a = attn_alloc(seq_len, d_model, 8, is_causal, use_rope);
    size_t w = (size_t)d_model * d_model;
    fread(a->W_q, sizeof(float), w, f);
    fread(a->W_k, sizeof(float), w, f);
    fread(a->W_v, sizeof(float), w, f);
    fread(a->W_o, sizeof(float), w, f);
    /* skip optimizer state: t (int) + m,v for each of W_q, W_k, W_v, W_o */
    int t; fread(&t, sizeof(int), 1, f);
    fseek(f, (long)(8 * w * sizeof(float)), SEEK_CUR);
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
    t->attn  = malloc(num_layers * sizeof(Attention*));
    t->mlp   = malloc(num_layers * sizeof(MLP*));
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
    int d_model, hidden_dim, num_layers;
    bool is_causal, use_rope;
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

static void gpt_forward(GPT* gpt, unsigned char token, int pos) {
    int d = gpt->d_model;
    memcpy(gpt->x, gpt->token_embedding + (size_t)token * d, d * sizeof(float));
    transformer_forward(gpt->transformer, gpt->x, pos);
    rmsnorm(gpt->final_norm, gpt->x, d);
    /* logits = token_embedding * final_norm   (weight-tied output) */
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
    size_t emb = (size_t)vocab_size * d_model;
    gpt->token_embedding = malloc(emb * sizeof(float));
    gpt->x          = malloc(d_model * sizeof(float));
    gpt->final_norm = malloc(d_model * sizeof(float));
    gpt->logits     = malloc((size_t)vocab_size * sizeof(float));

    fread(gpt->token_embedding, sizeof(float), emb, f);

    /* skip optimizer state: t (int) + m,v for token_embedding */
    int t; fread(&t, sizeof(int), 1, f);
    fseek(f, (long)(2 * emb * sizeof(float)), SEEK_CUR);

    gpt->transformer = transformer_deserialize(f, seq_len);
    fclose(f);

    fprintf(stderr, "Loaded: d_model=%d  hidden=%d  layers=%d  vocab=%d  seq_len=%d\n",
            d_model, hidden_dim, num_layers, vocab_size, seq_len);
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

static unsigned char sample(float* logits, int vocab_size, float temperature) {
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
        if (r <= cumsum) return (unsigned char)v;
    }
    return (unsigned char)(vocab_size - 1);
}

// ============================================================================
// Signal handler
// ============================================================================

static GPT* g_gpt = NULL;

static void on_sigint(int sig) {
    (void)sig;
    fprintf(stderr, "\n[interrupted]\n");
    gpt_free(g_gpt);
    exit(0);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    signal(SIGINT, on_sigint);

    int   seq_len     = 1024;
    float temperature = 0.7f;
    int   n_tokens    = 512;
    int   threads     = 8;
    unsigned seed     = (unsigned)time(NULL);
    const char* model_path = "checkpoint_gpt.bin";
    const char* prompt = "Once upon a time, there was a";

    for (int i = 1; i < argc; i++) {
        if      (strncmp(argv[i], "--prompt=",      9) == 0) prompt      = argv[i] + 9;
        else if (strncmp(argv[i], "--temperature=",14) == 0) temperature = (float)atof(argv[i] + 14);
        else if (strncmp(argv[i], "--tokens=",      9) == 0) n_tokens    = atoi(argv[i] + 9);
        else if (strncmp(argv[i], "--seed=",        7) == 0) seed        = (unsigned)atoi(argv[i] + 7);
        else if (strncmp(argv[i], "--seq_len=",    10) == 0) seq_len     = atoi(argv[i] + 10);
        else if (strncmp(argv[i], "--threads=",   10) == 0) threads     = atoi(argv[i] + 10);
        else if (strstr(argv[i], ".bin"))                    model_path  = argv[i];
        else {
            fprintf(stderr, "usage: %s [model.bin] [--prompt=...] [--temperature=X] [--tokens=N] [--seed=S] [--seq_len=N] [--threads=N]\n", argv[0]);
            return 1;
        }
    }
    srand(seed);
    openblas_set_num_threads(threads);

    g_gpt = gpt_load(model_path, seq_len);
    if (!g_gpt) return 1;

    int prompt_len = (int)strlen(prompt);
    if (prompt_len >= seq_len) { fprintf(stderr, "Prompt too long\n"); return 1; }
    if (n_tokens > seq_len - prompt_len) n_tokens = seq_len - prompt_len;

    fprintf(stderr, "Generating %d tokens (T=%.2f, seed=%u)\n",
            n_tokens, temperature, seed);

    /* Feed prompt */
    for (int pos = 0; pos < prompt_len; pos++) {
        gpt_forward(g_gpt, (unsigned char)prompt[pos], pos);
        fputc(prompt[pos], stdout);
    }
    fflush(stdout);

    /* Sample */
    for (int pos = prompt_len; pos < prompt_len + n_tokens; pos++) {
        unsigned char next = sample(g_gpt->logits, g_gpt->vocab_size, temperature);
        fputc((char)next, stdout);
        fflush(stdout);
        gpt_forward(g_gpt, next, pos);
    }
    fputc('\n', stdout);

    gpt_free(g_gpt);
    return 0;
}

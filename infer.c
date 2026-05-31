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
// Model structures (CPU, fp32)
// ============================================================================

typedef struct {
    float *W1, *W2;
    float *h, *s, *out;
    int input_dim, hidden_dim, output_dim;
} MLP;

typedef struct {
    float *W_in, *W_out;     // [d_model x d_model]
    float *A, *B, *C;        // [d_model x state_dim]; A pre-sigmoid
    float *D;                // [d_model]
    float *state;            // [d_model x state_dim]   per-channel running state
    float *u, *z, *out;      // workspace
    int d_model, state_dim;
} SSM;

typedef struct {
    SSM** ssm;
    MLP** mlp;
    float *norm1, *norm2;
    int num_layers, d_model;
} Transformer;

typedef struct {
    float* token_embedding;
    Transformer* transformer;
    float *x, *final_norm;
    float* logits;
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
    cblas_sgemv(CblasRowMajor, CblasTrans, m->input_dim, m->hidden_dim,
                1.0f, m->W1, m->hidden_dim, x, 1, 0.0f, m->h, 1);
    for (int i = 0; i < m->hidden_dim; i++) {
        float h = m->h[i];
        m->s[i] = h / (1.0f + expf(-h));
    }
    cblas_sgemv(CblasRowMajor, CblasTrans, m->hidden_dim, m->output_dim,
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
    int t; fread(&t, sizeof(int), 1, f);
    fseek(f, (long)((2 * w1 + 2 * w2) * sizeof(float)), SEEK_CUR);
    return m;
}

// ============================================================================
// SSM (per-token incremental)
// ============================================================================

static SSM* ssm_alloc(int d_model, int state_dim) {
    SSM* s = malloc(sizeof(SSM));
    s->d_model = d_model; s->state_dim = state_dim;
    size_t ws = (size_t)d_model * d_model;
    size_t ss = (size_t)d_model * state_dim;
    s->W_in  = malloc(ws * sizeof(float));
    s->W_out = malloc(ws * sizeof(float));
    s->A     = malloc(ss * sizeof(float));
    s->B     = malloc(ss * sizeof(float));
    s->C     = malloc(ss * sizeof(float));
    s->D     = malloc(d_model * sizeof(float));
    s->state = calloc(ss, sizeof(float));
    s->u   = malloc(d_model * sizeof(float));
    s->z   = malloc(d_model * sizeof(float));
    s->out = malloc(d_model * sizeof(float));
    return s;
}

static void ssm_free(SSM* s) {
    free(s->W_in); free(s->W_out);
    free(s->A); free(s->B); free(s->C); free(s->D);
    free(s->state);
    free(s->u); free(s->z); free(s->out);
    free(s);
}

// One token step. Mirrors the GPU forward kernel exactly.
static void ssm_forward(SSM* s, float* x) {
    int d = s->d_model, N = s->state_dim;

    cblas_sgemv(CblasRowMajor, CblasTrans, d, d,
                1.0f, s->W_in, d, x, 1, 0.0f, s->u, 1);

    for (int dd = 0; dd < d; dd++) {
        float u = s->u[dd];
        float acc = 0.0f;
        float* st = &s->state[dd * N];
        for (int n = 0; n < N; n++) {
            float a_raw = s->A[dd * N + n];
            float a = 1.0f / (1.0f + expf(-a_raw));
            float new_s = a * st[n] + s->B[dd * N + n] * u;
            st[n] = new_s;
            acc += s->C[dd * N + n] * new_s;
        }
        s->z[dd] = acc + s->D[dd] * u;
    }

    cblas_sgemv(CblasRowMajor, CblasTrans, d, d,
                1.0f, s->W_out, d, s->z, 1, 0.0f, s->out, 1);
}

static SSM* ssm_deserialize(FILE* f) {
    int d_model, state_dim;
    fread(&d_model,   sizeof(int), 1, f);
    fread(&state_dim, sizeof(int), 1, f);
    SSM* s = ssm_alloc(d_model, state_dim);

    size_t ws = (size_t)d_model * d_model;
    size_t ss = (size_t)d_model * state_dim;

    fread(s->W_in,  sizeof(float), ws,      f);
    fread(s->W_out, sizeof(float), ws,      f);
    fread(s->A,     sizeof(float), ss,      f);
    fread(s->B,     sizeof(float), ss,      f);
    fread(s->C,     sizeof(float), ss,      f);
    fread(s->D,     sizeof(float), d_model, f);

    // Skip optimizer state: t + (m,v) for each of {W_in,W_out,A,B,C,D}
    int t; fread(&t, sizeof(int), 1, f);
    long skip = (long)((2 * ws + 2 * ws + 2 * ss + 2 * ss + 2 * ss + 2 * (size_t)d_model) * sizeof(float));
    fseek(f, skip, SEEK_CUR);
    return s;
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
    t->ssm   = malloc(num_layers * sizeof(SSM*));
    t->mlp   = malloc(num_layers * sizeof(MLP*));
    t->norm1 = malloc(d_model * sizeof(float));
    t->norm2 = malloc(d_model * sizeof(float));
    return t;
}

static void transformer_free(Transformer* t) {
    for (int i = 0; i < t->num_layers; i++) {
        ssm_free(t->ssm[i]);
        mlp_free(t->mlp[i]);
    }
    free(t->ssm); free(t->mlp);
    free(t->norm1); free(t->norm2);
    free(t);
}

static void transformer_forward(Transformer* t, float* x) {
    int d = t->d_model;
    for (int l = 0; l < t->num_layers; l++) {
        rmsnorm(t->norm1, x, d);
        ssm_forward(t->ssm[l], t->norm1);
        for (int i = 0; i < d; i++) x[i] += t->ssm[l]->out[i];

        rmsnorm(t->norm2, x, d);
        mlp_forward(t->mlp[l], t->norm2);
        for (int i = 0; i < d; i++) x[i] += t->mlp[l]->out[i];
    }
}

static Transformer* transformer_deserialize(FILE* f) {
    int d_model, hidden_dim, num_layers;
    fread(&d_model,    sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    (void)hidden_dim;
    Transformer* t = transformer_alloc(d_model, num_layers);
    for (int i = 0; i < num_layers; i++) {
        t->ssm[i] = ssm_deserialize(f);
        t->mlp[i] = mlp_deserialize(f);
    }
    return t;
}

// ============================================================================
// GPT
// ============================================================================

static void gpt_forward(GPT* gpt, unsigned char token) {
    int d = gpt->d_model;
    memcpy(gpt->x, gpt->token_embedding + (size_t)token * d, d * sizeof(float));
    transformer_forward(gpt->transformer, gpt->x);
    rmsnorm(gpt->final_norm, gpt->x, d);
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

    int t; fread(&t, sizeof(int), 1, f);
    fseek(f, (long)(2 * emb * sizeof(float)), SEEK_CUR);

    gpt->transformer = transformer_deserialize(f);
    fclose(f);

    fprintf(stderr, "Loaded: d_model=%d hidden=%d layers=%d vocab=%d state_dim=%d\n",
            d_model, hidden_dim, num_layers, vocab_size,
            gpt->transformer->ssm[0]->state_dim);
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

    fprintf(stderr, "Generating %d tokens (T=%.2f, seed=%u)\n",
            n_tokens, temperature, seed);

    for (int pos = 0; pos < prompt_len; pos++) {
        gpt_forward(g_gpt, (unsigned char)prompt[pos]);
        fputc(prompt[pos], stdout);
    }
    fflush(stdout);

    for (int i = 0; i < n_tokens; i++) {
        unsigned char next = sample(g_gpt->logits, g_gpt->vocab_size, temperature);
        fputc((char)next, stdout);
        fflush(stdout);
        gpt_forward(g_gpt, next);
    }
    fputc('\n', stdout);

    gpt_free(g_gpt);
    return 0;
}

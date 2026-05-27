// cuDNN flash attention (forward + backward) wrapper.
// Compiled as C++; exposes a C ABI for attention.c.
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>

namespace fe = cudnn_frontend;

#define CHECK(expr) do { auto _e = (expr); \
    if (!_e.is_good()) { \
        fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__, _e.err_msg.c_str()); \
        exit(EXIT_FAILURE); \
    } } while (0)

enum { Q_UID = 1, K_UID, V_UID, O_UID, STATS_UID, SCALE_UID, dO_UID, dQ_UID, dK_UID, dV_UID };

static cudnnHandle_t g_handle = nullptr;
static void*  g_workspace = nullptr;
static size_t g_workspace_size = 0;
static std::map<std::tuple<int,int,int,int,int,int>, std::shared_ptr<fe::graph::Graph>> g_cache;

// QKV/O share layout: dims {B, NH, T, HS} over a [B, T, NH*HS] buffer (heads on fast axis).
static auto qkv(fe::graph::Graph& g, const char* name, int64_t uid, int B, int NH, int T, int HS) {
    return g.tensor(fe::graph::Tensor_attributes().set_name(name).set_uid(uid)
                    .set_dim   ({B, NH, T, HS})
                    .set_stride({(int64_t)T*NH*HS, HS, (int64_t)NH*HS, 1}));
}

// Build forward or backward SDPA graph
static std::shared_ptr<fe::graph::Graph> build_graph(int B, int NH, int T, int HS,
                                                     bool is_causal, bool is_backward, bool is_inference) {
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = qkv(*g, "Q", Q_UID, B, NH, T, HS);
    auto K = qkv(*g, "K", K_UID, B, NH, T, HS);
    auto V = qkv(*g, "V", V_UID, B, NH, T, HS);
    auto scale = g->tensor(fe::graph::Tensor_attributes().set_name("scale").set_uid(SCALE_UID)
                    .set_dim({1,1,1,1}).set_stride({1,1,1,1})
                    .set_is_pass_by_value(true).set_data_type(fe::DataType_t::FLOAT));

    const int64_t sB = (int64_t)T*NH*HS, sT = (int64_t)NH*HS;
    const std::vector<int64_t> out_dim    = {B, NH, T, HS};
    const std::vector<int64_t> out_stride = {sB, HS, sT, 1};
    const std::vector<int64_t> stats_dim    = {B, NH, T, 1};
    const std::vector<int64_t> stats_stride = {(int64_t)NH*T, T, 1, 1};

    if (!is_backward) {
        auto [O, stats] = g->sdpa(Q, K, V, fe::graph::SDPA_attributes().set_name("sdpa_fwd")
                                    .set_is_inference(is_inference)
                                    .set_attn_scale(scale)
                                    .set_causal_mask(is_causal));
        O->set_output(true).set_uid(O_UID).set_dim(out_dim).set_stride(out_stride);
        if (!is_inference)
            stats->set_output(true).set_uid(STATS_UID).set_data_type(fe::DataType_t::FLOAT)
                  .set_dim(stats_dim).set_stride(stats_stride);
    } else {
        auto O     = qkv(*g, "O",  O_UID,  B, NH, T, HS);
        auto dO    = qkv(*g, "dO", dO_UID, B, NH, T, HS);
        auto stats = g->tensor(fe::graph::Tensor_attributes().set_name("stats").set_uid(STATS_UID)
                        .set_dim(stats_dim).set_stride(stats_stride)
                        .set_data_type(fe::DataType_t::FLOAT));
        auto [dQ, dK, dV] = g->sdpa_backward(Q, K, V, O, dO, stats,
                                fe::graph::SDPA_backward_attributes().set_name("sdpa_bwd")
                                    .set_deterministic_algorithm(true)
                                    .set_causal_mask(is_causal)
                                    .set_attn_scale(scale));
        dQ->set_output(true).set_uid(dQ_UID).set_dim(out_dim).set_stride(out_stride);
        dK->set_output(true).set_uid(dK_UID).set_dim(out_dim).set_stride(out_stride);
        dV->set_output(true).set_uid(dV_UID).set_dim(out_dim).set_stride(out_stride);
    }

    CHECK(g->validate());
    CHECK(g->build_operation_graph(g_handle));
    CHECK(g->create_execution_plans({fe::HeurMode_t::A}));
    CHECK(g->check_support(g_handle));
    CHECK(g->build_plans(g_handle));
    return g;
}

// Cache graphs by shape/mode; grow shared workspace as needed
static std::shared_ptr<fe::graph::Graph> get_graph(int B, int NH, int T, int HS,
                                                   bool is_causal, bool is_backward, bool is_inference) {
    if (!g_handle) cudnnCreate(&g_handle);
    auto& slot = g_cache[{B, NH, T, HS, (int)is_causal, is_backward ? 2 : (int)is_inference}];
    if (!slot) slot = build_graph(B, NH, T, HS, is_causal, is_backward, is_inference);
    size_t need = slot->get_workspace_size();
    if (need > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, need);
        g_workspace_size = need;
    }
    return slot;
}

extern "C" {

// O = softmax(QKᵀ/√d) V    (optionally causal; saves softmax stats for backward)
void cudnn_attention_forward(void* Q, void* K, void* V, void* O, void* stats,
                             int B, int NH, int T, int HS, int is_causal) {
    auto graph = get_graph(B, NH, T, HS, is_causal, false, stats == nullptr);
    float scale = 1.0f / std::sqrt((float)HS);
    std::unordered_map<int64_t, void*> pack = {
        {Q_UID, Q}, {K_UID, K}, {V_UID, V}, {O_UID, O}, {SCALE_UID, &scale}
    };
    if (stats) pack[STATS_UID] = stats;
    CHECK(graph->execute(g_handle, pack, g_workspace));
}

// (dQ, dK, dV) = ∂L/∂(Q, K, V) given dO and saved softmax stats
void cudnn_attention_backward(void* Q, void* K, void* V, void* O, void* dO, void* stats,
                              void* dQ, void* dK, void* dV,
                              int B, int NH, int T, int HS, int is_causal) {
    auto graph = get_graph(B, NH, T, HS, is_causal, true, false);
    float scale = 1.0f / std::sqrt((float)HS);
    std::unordered_map<int64_t, void*> pack = {
        {Q_UID, Q},   {K_UID, K},   {V_UID, V},   {O_UID, O}, {dO_UID, dO}, {STATS_UID, stats},
        {dQ_UID, dQ}, {dK_UID, dK}, {dV_UID, dV}, {SCALE_UID, &scale}
    };
    CHECK(graph->execute(g_handle, pack, g_workspace));
}

void cudnn_attention_destroy(void) {
    g_cache.clear();
    if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; g_workspace_size = 0; }
    if (g_handle)    { cudnnDestroy(g_handle); g_handle = nullptr; }
}

} // extern "C"
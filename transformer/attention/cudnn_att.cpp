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

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t s = (call); \
    if (s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(s)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define FE_CHECK(expr) do { \
    auto _e = (expr); \
    if (!_e.is_good()) { \
        fprintf(stderr, "cuDNN-FE error %s:%d: %s\n", __FILE__, __LINE__, _e.err_msg.c_str()); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

static cudnnHandle_t g_handle = nullptr;
static void*  g_workspace      = nullptr;
static size_t g_workspace_size = 0;

enum UIDs { Q_UID = 1, K_UID, V_UID, O_UID, STATS_UID, SCALE_UID, dO_UID, dQ_UID, dK_UID, dV_UID };

using fwd_key = std::tuple<int,int,int,int,int,int>; // B, NH, T, HS, is_causal, is_inference
using bwd_key = std::tuple<int,int,int,int,int>;     // B, NH, T, HS, is_causal

static std::map<fwd_key, std::shared_ptr<fe::graph::Graph>> g_fwd_cache;
static std::map<bwd_key, std::shared_ptr<fe::graph::Graph>> g_bwd_cache;

static void ensure_handle() {
    if (!g_handle) CUDNN_CHECK(cudnnCreate(&g_handle));
}

static void ensure_workspace(size_t need) {
    if (need > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, need);
        g_workspace_size = need;
    }
}

static std::shared_ptr<fe::graph::Graph> build_fwd_graph(int B, int NH, int T, int HS,
                                                        bool is_causal, bool is_inference) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q, K, V live in separate [B, T, NH*HS] buffers (heads on fast axis).
    // View them as {B, NH, T, HS} with strides {T*NH*HS, HS, NH*HS, 1}.
    const int64_t sB = (int64_t)T * NH * HS;
    const int64_t sH = HS;
    const int64_t sT = (int64_t)NH * HS;

    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                .set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(Q_UID));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                .set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(K_UID));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                .set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(V_UID));

    auto scale = graph->tensor(fe::graph::Tensor_attributes().set_name("scale")
                .set_dim({1,1,1,1}).set_stride({1,1,1,1})
                .set_uid(SCALE_UID).set_is_pass_by_value(true)
                .set_data_type(fe::DataType_t::FLOAT));

    auto opts = fe::graph::SDPA_attributes().set_name("sdpa_fwd")
                .set_is_inference(is_inference)
                .set_attn_scale(scale)
                .set_causal_mask(is_causal);

    auto [O, stats] = graph->sdpa(Q, K, V, opts);
    O->set_output(true).set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(O_UID);
    if (!is_inference) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
              .set_dim({B, NH, T, 1}).set_stride({(int64_t)NH*T, T, 1, 1}).set_uid(STATS_UID);
    }

    FE_CHECK(graph->validate());
    FE_CHECK(graph->build_operation_graph(g_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    FE_CHECK(graph->check_support(g_handle));
    FE_CHECK(graph->build_plans(g_handle));
    ensure_workspace(graph->get_workspace_size());
    return graph;
}

static std::shared_ptr<fe::graph::Graph> build_bwd_graph(int B, int NH, int T, int HS, bool is_causal) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t sB = (int64_t)T * NH * HS;
    const int64_t sH = HS;
    const int64_t sT = (int64_t)NH * HS;

    auto Q  = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                .set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(Q_UID));
    auto K  = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                .set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(K_UID));
    auto V  = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                .set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(V_UID));
    auto O  = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                .set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(O_UID));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                .set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(dO_UID));
    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                .set_dim({B, NH, T, 1}).set_stride({(int64_t)NH*T, T, 1, 1})
                .set_uid(STATS_UID).set_data_type(fe::DataType_t::FLOAT));
    auto scale = graph->tensor(fe::graph::Tensor_attributes().set_name("scale")
                .set_dim({1,1,1,1}).set_stride({1,1,1,1})
                .set_uid(SCALE_UID).set_is_pass_by_value(true)
                .set_data_type(fe::DataType_t::FLOAT));

    auto opts = fe::graph::SDPA_backward_attributes().set_name("sdpa_bwd")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                .set_deterministic_algorithm(true)
#endif
                .set_causal_mask(is_causal)
                .set_attn_scale(scale);

    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, opts);
    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride({sB, sH, sT, 1}).set_uid(dV_UID);

    FE_CHECK(graph->validate());
    FE_CHECK(graph->build_operation_graph(g_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    FE_CHECK(graph->check_support(g_handle));
    FE_CHECK(graph->build_plans(g_handle));
    ensure_workspace(graph->get_workspace_size());
    return graph;
}

extern "C" {

void cudnn_attention_destroy(void) {
    g_fwd_cache.clear();
    g_bwd_cache.clear();
    if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; g_workspace_size = 0; }
    if (g_handle)    { cudnnDestroy(g_handle); g_handle = nullptr; }
}

void cudnn_attention_forward(void* Q, void* K, void* V, void* O, void* stats,
                             int B, int NH, int T, int HS, int is_causal) {
    ensure_handle();
    const bool is_inference = (stats == nullptr);
    fwd_key key{B, NH, T, HS, is_causal, is_inference};
    auto it = g_fwd_cache.find(key);
    std::shared_ptr<fe::graph::Graph> graph;
    if (it == g_fwd_cache.end()) {
        graph = build_fwd_graph(B, NH, T, HS, is_causal, is_inference);
        g_fwd_cache[key] = graph;
    } else {
        graph = it->second;
    }

    float scale_cpu = 1.0f / std::sqrt((float)HS);
    std::unordered_map<int64_t, void*> pack = {
        {Q_UID, Q}, {K_UID, K}, {V_UID, V}, {O_UID, O}, {SCALE_UID, &scale_cpu}
    };
    if (!is_inference) pack[STATS_UID] = stats;

    ensure_workspace(graph->get_workspace_size());
    FE_CHECK(graph->execute(g_handle, pack, g_workspace));
}

void cudnn_attention_backward(void* Q, void* K, void* V, void* O, void* dO, void* stats,
                              void* dQ, void* dK, void* dV,
                              int B, int NH, int T, int HS, int is_causal) {
    ensure_handle();
    bwd_key key{B, NH, T, HS, is_causal};
    auto it = g_bwd_cache.find(key);
    std::shared_ptr<fe::graph::Graph> graph;
    if (it == g_bwd_cache.end()) {
        graph = build_bwd_graph(B, NH, T, HS, is_causal);
        g_bwd_cache[key] = graph;
    } else {
        graph = it->second;
    }

    float scale_cpu = 1.0f / std::sqrt((float)HS);
    std::unordered_map<int64_t, void*> pack = {
        {Q_UID, Q},  {K_UID, K},  {V_UID, V},  {O_UID, O},
        {dO_UID, dO}, {STATS_UID, stats},
        {dQ_UID, dQ}, {dK_UID, dK}, {dV_UID, dV},
        {SCALE_UID, &scale_cpu}
    };

    ensure_workspace(graph->get_workspace_size());
    FE_CHECK(graph->execute(g_handle, pack, g_workspace));
}

} // extern "C"

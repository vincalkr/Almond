// src/hy4/hy4_graph.cpp - GGUF -> ggml tensor loading and the Engram projection graph.
#include "hy4_graph.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace almond;
using namespace almond::hy4;

namespace almond::hy4 {
namespace {

bool read_f32(const GgufModel & m, const std::string & name, std::vector<float> & out, std::string & err) {
    size_t which = 0;
    const TensorInfo * t = m.find(name, &which);
    if (!t) { err = "missing " + name; return false; }
    const uint64_t n = t->elements();
    out.resize(n);
    const uint8_t * p = m.shard(which).tensor_data(*t);
    if (t->type == 0) {  // F32
        std::memcpy(out.data(), p, n * sizeof(float));
    } else if (t->type == 30) {  // BF16: the high 16 bits of the float
        const uint16_t * b = reinterpret_cast<const uint16_t *>(p);
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t bits = (uint32_t) b[i] << 16;
            std::memcpy(&out[i], &bits, 4);
        }
    } else { err = name + " has type " + std::to_string(t->type) + " (expected F32 or BF16)"; return false; }
    return true;
}

}  // namespace

bool hy4_engram_load_weights(const GgufModel & m, const Hy4Geometry & G, int module,
                             EngramWeights & w, std::string & err) {
    if (module < 0 || module >= (int) G.ple_layers.size()) { err = "bad Engram module"; return false; }
    const std::string b = "blk." + std::to_string(G.ple_layers[module]) + ".";
    return read_f32(m, b + "engram_lookup_norm.weight", w.lookup_norm, err) &&
           read_f32(m, b + "engram_kv.weight", w.kv, err) &&
           read_f32(m, b + "engram_kv_b.weight", w.kv_b, err) &&
           read_f32(m, b + "engram_gate.weight", w.gate, err) &&
           read_f32(m, b + "engram_pre_gamma.weight", w.pre_gamma, err) &&
           read_f32(m, b + "engram_hc_scale.weight", w.hc_scale, err);
}

bool hy4_engram_forward_cpu(const GgufModel & m, const Hy4Geometry & G, int module, int n_tokens,
                            const float * x, const int32_t * ids, const float * valid,
                            const uint8_t * table, std::vector<float> & delta, std::string & err) {
    EngramWeights w;
    if (!hy4_engram_load_weights(m, G, module, w, err)) return false;
    const int64_t D = G.n_embd, HC = G.hc, dim = G.ple_dim;
    const int64_t n_cols = G.ple_heads * (G.ple_ngram - 1);
    const int64_t kv_cols = D * (HC + 1);

    // A test-sized context: engram_kv alone is 94 MB of F32, so let ggml own the tensor data here. The engine will
    // use the scheduler and preallocated device buffers instead.
    ggml_init_params ip = { 256u * 1024u * 1024u, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { err = "ggml_init failed"; return false; }

    auto t2 = [&](int64_t a, int64_t b_) { return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b_); };
    auto t3 = [&](int64_t a, int64_t b_, int64_t c) { return ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b_, c); };

    ggml_tensor * in_rows = t3(dim, n_cols, n_tokens);
    ggml_tensor * in_valid = t2(n_cols, n_tokens);
    ggml_tensor * in_x = t3(D, HC, n_tokens);
    ggml_tensor * wt_norm = t2(dim, n_cols);
    ggml_tensor * wt_kv = t2(dim * n_cols, kv_cols);
    ggml_tensor * wt_kvb = t2(D, HC + 1);
    ggml_tensor * wt_gate = t2(D, HC);
    ggml_tensor * wt_gamma = t2(D, 1);
    ggml_tensor * wt_scale = t2(D, HC);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    const float eps = 1.1920929e-07f;

    ggml_tensor * r0 = ggml_rms_norm(ctx, in_rows, eps);
    ggml_tensor * r1 = ggml_mul(ctx, r0, wt_norm);
    ggml_tensor * r2 = ggml_mul(ctx, r1, ggml_reshape_3d(ctx, in_valid, 1, n_cols, n_tokens));
    ggml_tensor * r = ggml_reshape_2d(ctx, r2, dim * n_cols, n_tokens);

    ggml_tensor * kv = ggml_mul_mat(ctx, wt_kv, r);
    kv = ggml_add(ctx, ggml_reshape_3d(ctx, kv, D, HC + 1, n_tokens), wt_kvb);
    ggml_tensor * key = ggml_cont(ctx, ggml_view_3d(ctx, kv, D, HC, n_tokens, kv->nb[1], kv->nb[2], 0));
    ggml_tensor * value = ggml_cont(ctx, ggml_view_3d(ctx, kv, D, 1, n_tokens, kv->nb[1], kv->nb[2], HC * kv->nb[1]));

    ggml_tensor * dot = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, in_x, eps), ggml_rms_norm(ctx, key, eps)), wt_gate);
    dot = ggml_scale(ctx, ggml_sum_rows(ctx, dot), 1.0f / sqrtf((float) D));
    ggml_tensor * mag = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx, ggml_step(ctx, ggml_neg(ctx, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, mag, sign));

    ggml_tensor * signal = ggml_mul(ctx, ggml_rms_norm(ctx, value, eps), wt_gamma);
    ggml_tensor * scale = ggml_add(ctx, ggml_repeat_4d(ctx, gate, D, HC, n_tokens, 1), wt_scale);
    ggml_tensor * d = ggml_mul(ctx, ggml_repeat_4d(ctx, signal, D, HC, n_tokens, 1), scale);

    ggml_tensor * any = ggml_step(ctx, ggml_sum_rows(ctx, ggml_reshape_2d(ctx, in_valid, n_cols, n_tokens)));
    d = ggml_mul(ctx, d, ggml_reshape_3d(ctx, any, 1, 1, n_tokens));
    ggml_tensor * out = ggml_add(ctx, in_x, d);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(ctx); err = "no CPU backend"; return false; }
    float * rows_p = (float *) ggml_get_data(in_rows);
    for (int64_t t = 0; t < n_tokens; ++t)
        for (int64_t c = 0; c < n_cols; ++c) {
            const int64_t row = ids[(size_t) (t * n_cols + c)];
            for (int64_t j = 0; j < dim; ++j)
                rows_p[(t * n_cols + c) * dim + j] = hy4_f8_e4m3_to_float(table[(uint64_t) row * (uint64_t) dim + (uint64_t) j]);
        }
    std::memcpy(ggml_get_data(in_valid), valid, (size_t) n_cols * n_tokens * sizeof(float));
    std::memcpy(ggml_get_data(in_x), x, (size_t) D * HC * n_tokens * sizeof(float));
    std::memcpy(ggml_get_data(wt_norm), w.lookup_norm.data(), w.lookup_norm.size() * sizeof(float));
    std::memcpy(ggml_get_data(wt_kv), w.kv.data(), w.kv.size() * sizeof(float));
    std::memcpy(ggml_get_data(wt_kvb), w.kv_b.data(), w.kv_b.size() * sizeof(float));
    std::memcpy(ggml_get_data(wt_gate), w.gate.data(), w.gate.size() * sizeof(float));
    std::memcpy(ggml_get_data(wt_gamma), w.pre_gamma.data(), w.pre_gamma.size() * sizeof(float));
    std::memcpy(ggml_get_data(wt_scale), w.hc_scale.data(), w.hc_scale.size() * sizeof(float));

    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_free(cpu); ggml_free(ctx); err = "graph compute failed"; return false;
    }
    delta.assign((size_t) D * HC * n_tokens, 0.0f);
    const float * o = (const float *) ggml_get_data(out);
    for (size_t i = 0; i < delta.size(); ++i) delta[i] = o[i];
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return true;
}

}  // namespace almond::hy4

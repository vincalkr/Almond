// src/hy4/hy4_forward.cpp - trunk loop + iHC head + output projection, following PR #32517's graph.
#include "hy4_forward.hpp"

#include "hy4_layer.hpp"

#include "hy4_weights.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace almond::hy4 {
namespace {

ggml_tensor * W2(ggml_context * c, const GgufModel & m, const std::string & n, int64_t a, int64_t b, std::string & err) {
    (void) c;   // the weight is a tensor over the mapped file (hy4_weights.hpp), not a copy
    return hy4_weight(m, n, a, b, 0, err);
}

// HY4_DUMP_DIR=<dir> saves each layer's streams as raw f32 ([n_embd, hc, nt], named like PR #32517's callbacks)
// so they can be compared with the reference engine's activations.
void dump(const char * name, const float * p, size_t n) {
    const char * d = std::getenv("HY4_DUMP_DIR");
    if (!d) return;
    FILE * f = std::fopen((std::string(d) + "/" + name + ".bin").c_str(), "wb");
    if (!f) return;
    std::fwrite(p, 4, n, f);
    std::fclose(f);
}

}  // namespace

bool hy4_forward_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t nt,
                     const int64_t * tokens, const int64_t * prev_ids, const int32_t * pos,
                     std::vector<float> & logits, std::string & err, Hy4KvCache * cache) {
    const int64_t D = G.n_embd, HC = G.hc, V = G.n_vocab;
    const float eps = (float) G.rms_eps;

    // One embedding feeds hc parallel residual streams (all streams start equal).
    std::vector<float> streams((size_t) D * HC * nt);
    {
        ggml_init_params ip = { 1024u * 1024u * 1024u, nullptr, false };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * emb = W2(c, m, "token_embd.weight", D, V, err);
        ggml_tensor * idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, nt);
        for (int64_t t = 0; t < nt; ++t) ((int32_t *) ggml_get_data(idx))[t] = (int32_t) tokens[t];  // ids are int64
        ggml_tensor * e = ggml_get_rows(c, emb, idx);                    // [D, nt]
        ggml_tensor * s = ggml_repeat_4d(c, ggml_reshape_3d(c, e, D, 1, nt), D, HC, nt, 1);
        ggml_set_output(s);
        ggml_cgraph * gf = ggml_new_graph_custom(c, 32, false);
        ggml_build_forward_expand(gf, s);
        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) { ggml_free(c); err = "no CPU backend"; return false; }
        bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_free(cpu);
        if (!ok) { ggml_free(c); err = "embedding failed"; return false; }
        std::memcpy(streams.data(), ggml_get_data(s), streams.size() * 4);
        ggml_free(c);
    }
    dump("hc_init", streams.data(), streams.size());

    const int64_t n_trunk = G.n_layer - G.n_nextn;
    if (cache) {
        if (cache->k.empty()) cache->k.resize((size_t) n_trunk);
        for (int64_t t = 0; t < nt; ++t)
            if (pos[t] != cache->n_past + t) { err = "positions must continue the cache"; return false; }
    }
    for (int64_t il = 0; il < n_trunk; ++il) {
        std::vector<float> next;
        if (!hy4_layer_cpu(m, G, il, nt, streams.data(), tokens, prev_ids, pos, next, err,
                           cache ? &cache->k[(size_t) il] : nullptr)) {
            err = "layer " + std::to_string(il) + ": " + err;
            return false;
        }
        streams.swap(next);
        dump(("l_out-" + std::to_string(il)).c_str(), streams.data(), streams.size());
    }

    // iHC head: reduce the streams with the output block's gates.
    ggml_init_params ip = { 1024u * 1024u * 1024u, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tx = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, nt);
    std::memcpy(ggml_get_data(tx), streams.data(), streams.size() * 4);
    ggml_tensor * fn = W2(c, m, "output_hc_fn.weight", HC * D, HC, err);
    ggml_tensor * sc = W2(c, m, "output_hc_scale.weight", 1, 1, err);
    ggml_tensor * bs = W2(c, m, "output_hc_base.weight", HC, 1, err);
    ggml_tensor * on = W2(c, m, "output_norm.weight", D, 1, err);
    ggml_tensor * head = W2(c, m, "output.weight", D, V, err);
    if (err.size()) { ggml_free(c); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(c, 128, false);
    ggml_tensor * flat = ggml_reshape_2d(c, tx, HC * D, nt);
    ggml_tensor * mixes = ggml_mul_mat(c, fn, ggml_rms_norm(c, flat, eps));   // [HC, nt]
    ggml_tensor * gates = ggml_sigmoid(c, ggml_scale_bias(c, ggml_add(c, ggml_mul(c, mixes, sc), bs), 1.0f, eps));
    ggml_tensor * reduced = nullptr;
    for (int64_t h = 0; h < HC; ++h) {
        ggml_tensor * xh = ggml_view_2d(c, tx, D, nt, tx->nb[2], (size_t) h * tx->nb[1]);
        ggml_tensor * gh = ggml_view_2d(c, gates, 1, nt, gates->nb[1], (size_t) h * gates->nb[0]);
        ggml_tensor * cur = ggml_mul(c, xh, gh);
        reduced = reduced ? ggml_add(c, reduced, cur) : cur;
    }
    ggml_tensor * y = ggml_mul_mat(c, head, ggml_mul(c, ggml_rms_norm(c, reduced, eps), on));
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(c); err = "no CPU backend"; return false; }
    bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_free(cpu);
    if (!ok) { ggml_free(c); err = "head failed"; return false; }
    logits.assign((size_t) V * nt, 0.0f);
    std::memcpy(logits.data(), ggml_get_data(y), logits.size() * 4);
    dump("result_output", logits.data(), logits.size());
    if (cache) cache->n_past += nt;
    ggml_free(c);
    return true;
}

}  // namespace almond::hy4

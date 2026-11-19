// src/hy4/hy4_layer.cpp - one trunk layer, following hy-v4-flash.cpp's loop (PR #32517).
#include "hy4_layer.hpp"

#include "hy4_attn.hpp"
#include "hy4_engram.hpp"
#include "hy4_graph.hpp"
#include "hy4_hc.hpp"
#include "hy4_moe.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace almond::hy4 {
namespace {

ggml_tensor * load(ggml_context * c, const GgufModel & m, const std::string & name, int64_t n, std::string & err) {
    size_t which = 0;
    const TensorInfo * t = m.find(name, &which);
    if (!t || t->type != 0) { err = "missing F32 " + name; return nullptr; }
    ggml_tensor * g = ggml_new_tensor_1d(c, GGML_TYPE_F32, n);
    std::memcpy(ggml_get_data(g), m.shard(which).tensor_data(*t), (size_t) n * 4);
    return g;
}

/// RMS norm over n_embd followed by the layer's learned per-channel scale.
bool norm_mul(int64_t D, int64_t nt, const float * in, const float * scale, float eps,
              std::vector<float> & out, std::string & err) {
    ggml_init_params ip = { 4u * 1024u * 1024u, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, nt);
    ggml_tensor * s = ggml_new_tensor_1d(c, GGML_TYPE_F32, D);
    std::memcpy(ggml_get_data(x), in, (size_t) D * nt * 4);
    std::memcpy(ggml_get_data(s), scale, (size_t) D * 4);
    ggml_tensor * y = ggml_mul(c, ggml_rms_norm(c, x, eps), s);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 32, false);
    ggml_build_forward_expand(gf, y);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { err = "no CPU backend"; ggml_free(c); return false; }
    bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_free(cpu);
    if (!ok) { err = "norm graph failed"; ggml_free(c); return false; }
    out.assign((size_t) D * nt, 0.0f);
    std::memcpy(out.data(), ggml_get_data(y), out.size() * 4);
    ggml_free(c);
    return true;
}

}  // namespace

bool hy4_layer_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer, int64_t nt,
                   const float * streams, const int64_t * tokens, const int64_t * prev_ids,
                   const int32_t * pos, std::vector<float> & out, std::string & err,
                   std::vector<float> * kcache) {
    const std::string b = "blk." + std::to_string(layer) + ".";
    const int64_t D = G.n_embd, HC = G.hc;
    const float eps = (float) G.rms_eps;
    out.assign((size_t) D * HC * nt, 0.0f);

    // --- Engram, before the attention pre-mix (layers 1, 11, 21, 31) -------------------------
    std::vector<float> cur_x(streams, streams + (size_t) D * HC * nt);
    const EngramLayout L = hy4_engram_layout(G);
    for (size_t e = 0; e < L.modules.size(); ++e)
        if (L.modules[e].layer == layer && tokens) {
            const int64_t cols = L.heads * (L.n_gram - 1);
            size_t which = 0;
            const TensorInfo * tt = m.find("per_layer_token_embd.weight", &which);
            if (!tt) { err = "no Engram table"; return false; }
            const uint8_t * table = m.shard(which).tensor_data(*tt);
            std::vector<float> delta;
            std::vector<int32_t> ids((size_t) nt * cols);
            std::vector<float> valid((size_t) nt * cols);
            for (int64_t t = 0; t < nt; ++t)
                hy4_engram_hash(L, e, tokens[t], prev_ids ? prev_ids + t * (L.n_gram - 1) : nullptr, ids.data() + t * cols, valid.data() + t * cols);
            if (!hy4_engram_forward_cpu(m, G, e, nt, cur_x.data(), ids.data(), valid.data(), table, delta, err))
                return false;
            cur_x.swap(delta);  // the Engram graph returns x + delta, not the delta alone
        }

    const std::string L_ = "-" + std::to_string(layer);
    auto dump = [&](const std::string & name, const std::vector<float> & v) {  // HY4_DUMP_DIR, as in hy4_forward.cpp
        const char * d = std::getenv("HY4_DUMP_DIR");
        if (!d) return;
        if (FILE * f = std::fopen((std::string(d) + "/" + name + L_ + ".bin").c_str(), "wb")) {
            std::fwrite(v.data(), 4, v.size(), f);
            std::fclose(f);
        }
    };
    for (const auto & em : L.modules) if (em.layer == layer) dump("engram_out", cur_x);

    // --- attention half ---------------------------------------------------------------------
    HcPre ahc;
    {
        ggml_init_params ip = { 4u * 1024u * 1024u, nullptr, false };
        ggml_context * c = ggml_init(ip);
        auto W = [&](const char * n, int64_t a, int64_t b_) {
            size_t w = 0; const TensorInfo * t = m.find(b + n, &w);
            if (!t || t->type != 0) { err = std::string("missing F32 ") + n; return (ggml_tensor *) nullptr; }
            ggml_tensor * g = ggml_new_tensor_2d(c, GGML_TYPE_F32, a, b_);
            std::memcpy(ggml_get_data(g), m.shard(w).tensor_data(*t), (size_t) a * b_ * 4);
            return g;
        };
        ggml_tensor * fn = W("hc_attn_fn.weight", HC * D, 2 * HC);
        ggml_tensor * sc = W("hc_attn_scale.weight", 2, 1);
        ggml_tensor * bs = W("hc_attn_base.weight", 2 * HC, 1);
        if (err.size()) { ggml_free(c); return false; }
        // the F32 arrays are contiguous in ggml order; copy them into plain vectors and rerun
        std::vector<float> fv((size_t) HC * D * 2 * HC), sv(2), bv((size_t) 2 * HC);
        std::memcpy(fv.data(), ggml_get_data(fn), fv.size() * 4);
        std::memcpy(sv.data(), ggml_get_data(sc), 8);
        std::memcpy(bv.data(), ggml_get_data(bs), bv.size() * 4);
        ggml_free(c);
        if (!hy4_hc_pre_cpu(D, HC, nt, cur_x.data(), fv.data(), sv.data(), bv.data(),
                            eps, (float) G.hc_mag, ahc, err)) return false;
    }

    std::vector<float> attn_in((size_t) D * nt);
    {   // rms_norm * attn_norm
        ggml_init_params ip = { 4u * 1024u * 1024u, nullptr, false };
        ggml_context * c = ggml_init(ip);
        size_t w = 0; const TensorInfo * t = m.find(b + "attn_norm.weight", &w);
        std::vector<float> ns((size_t) D);
        std::memcpy(ns.data(), m.shard(w).tensor_data(*t), (size_t) D * 4);
        ggml_free(c);
        if (!norm_mul(D, nt, ahc.reduced.data(), ns.data(), eps, attn_in, err)) return false;
    }
    std::vector<float> attn_out;
    if (!hy4_mla_attn_multi_cpu(m, G, layer, nt, attn_in.data(), pos, attn_out, err, kcache)) return false;
    dump("attn_out", attn_out);

    std::vector<float> after_attn((size_t) D * HC * nt);
    if (!hy4_hc_post_cpu(D, HC, nt, attn_out.data(), cur_x.data(), ahc.post.data(), after_attn, err)) return false;
    dump("hc_attn_out", after_attn);

    // --- FFN half ---------------------------------------------------------------------------
    HcPre fhc;
    {
        ggml_init_params ip = { 4u * 1024u * 1024u, nullptr, false };
        ggml_context * c = ggml_init(ip);
        auto W = [&](const char * n, int64_t a, int64_t b_) {
            size_t w = 0; const TensorInfo * t = m.find(b + n, &w);
            if (!t || t->type != 0) { err = std::string("missing F32 ") + n; return (ggml_tensor *) nullptr; }
            ggml_tensor * g = ggml_new_tensor_2d(c, GGML_TYPE_F32, a, b_);
            std::memcpy(ggml_get_data(g), m.shard(w).tensor_data(*t), (size_t) a * b_ * 4);
            return g;
        };
        ggml_tensor * fn = W("hc_ffn_fn.weight", HC * D, 2 * HC);
        ggml_tensor * sc = W("hc_ffn_scale.weight", 2, 1);
        ggml_tensor * bs = W("hc_ffn_base.weight", 2 * HC, 1);
        if (err.size()) { ggml_free(c); return false; }
        std::vector<float> fv((size_t) HC * D * 2 * HC), sv(2), bv((size_t) 2 * HC);
        std::memcpy(fv.data(), ggml_get_data(fn), fv.size() * 4);
        std::memcpy(sv.data(), ggml_get_data(sc), 8);
        std::memcpy(bv.data(), ggml_get_data(bs), bv.size() * 4);
        ggml_free(c);
        if (!hy4_hc_pre_cpu(D, HC, nt, after_attn.data(), fv.data(), sv.data(), bv.data(),
                            eps, (float) G.hc_mag, fhc, err)) return false;
    }

    std::vector<float> ffn_in((size_t) D * nt);
    {
        ggml_init_params ip = { 4u * 1024u * 1024u, nullptr, false };
        ggml_context * c = ggml_init(ip);
        size_t w = 0; const TensorInfo * t = m.find(b + "ffn_norm.weight", &w);
        std::vector<float> ns((size_t) D);
        std::memcpy(ns.data(), m.shard(w).tensor_data(*t), (size_t) D * 4);
        ggml_free(c);
        if (!norm_mul(D, nt, fhc.reduced.data(), ns.data(), eps, ffn_in, err)) return false;
    }
    std::vector<float> ffn_raw((size_t) D * nt);
    if (layer < G.n_dense) {
        if (!hy4_dense_ffn_cpu(m, G, layer, nt, ffn_in.data(), ffn_raw, err)) return false;
    } else if (!hy4_latent_moe_cpu(m, G, layer, nt, ffn_in.data(), ffn_raw, err)) return false;

    std::vector<float> ffn_post((size_t) D * nt);
    {
        ggml_init_params ip = { 4u * 1024u * 1024u, nullptr, false };
        ggml_context * c = ggml_init(ip);
        size_t w = 0; const TensorInfo * t = m.find(b + "post_ffw_norm.weight", &w);
        std::vector<float> ns((size_t) D);
        std::memcpy(ns.data(), m.shard(w).tensor_data(*t), (size_t) D * 4);
        ggml_free(c);
        if (!norm_mul(D, nt, ffn_raw.data(), ns.data(), eps, ffn_post, err)) return false;
    }
    dump("ffn_out", ffn_post);
    if (!hy4_hc_post_cpu(D, HC, nt, ffn_post.data(), after_attn.data(), fhc.post.data(), out, err)) return false;
    return true;
}

}  // namespace almond::hy4

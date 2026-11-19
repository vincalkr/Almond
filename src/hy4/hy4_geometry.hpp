// src/hy4/hy4_geometry.hpp - Hy4-flash (`hy_v4_flash`) geometry and tensor map.
//
// Every number comes from the file's `hy_v4_flash.*` keys; a missing key is an error, never a default.  The reader
// keeps full copies of the long arrays (token map, PLE offsets, per-layer head counts) - GgufFile's metadata sample
// holds only the first 64 entries, so those are read straight from the header region like DeepSeek's geometry does.
//
// The GGUF inspected on 2026-11-18 has block_count 43 (42 trunk layers + one MTP draft block), leading_dense_block_count 1,
// a 96 x 660,006,416 F8_E4M3 Engram table in shard 2, and experts at IQ2_XXS / Q4_K - see docs/hy4/GGUF_NOTES.md.
#pragma once

#include "almond/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <initializer_list>
#include <string>
#include <unistd.h>
#include <vector>

namespace almond::hy4 {

namespace ty {
constexpr uint32_t F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, IQ2_XXS = 16, I32 = 26, BF16 = 30, F8_E4M3 = 43;
}

struct Hy4Geometry {
    int64_t n_embd = 0, n_layer = 0, n_dense = 0, n_ctx = 0, n_vocab = 0, n_ff = 0;
    int64_t n_head = 0, n_head_swa = 0, n_head_kv = 0, d_key = 0, d_val = 0;
    int64_t q_lora = 0, kv_lora = 0, d_key_mla = 0, d_val_mla = 0, d_rope = 0;
    double rms_eps = 0, hc_eps = 0, hc_mag = 0;
    int64_t rope_base = 0, rope_base_swa = 0, swa = 0, hc = 0;
    int64_t n_expert = 0, n_used = 0, n_shared = 0, d_expert = 0, d_latent = 0, gating = 0, weights_norm = 0;
    double weights_scale = 0;
    int64_t idx_heads = 0, idx_dim = 0, idx_topk = 0, idx_kpool = 0, idx_tail = 0;
    int64_t n_nextn = 0, ple_ngram = 0, ple_heads = 0, ple_dim = 0, engram_pad = 0;
    std::vector<int64_t> head_count, swa_pattern, idx_types, ple_layers, ple_mult, ple_offsets, ple_sizes, token_map;
};

namespace detail {
inline size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}
}  // namespace detail

/// Full integer array out of a GGUF header (the metadata sample keeps only the first 64 entries).
inline std::string hy4_read_int_array(const std::string & path, const std::string & key, std::vector<int64_t> & out) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return "cannot open " + path;
    struct Closer { int fd; ~Closer() { ::close(fd); } } closer { fd };
    uint64_t pos = 0;
    auto get = [&](void * p, size_t n) -> bool {
        ssize_t got = ::pread(fd, p, n, (off_t) pos);
        pos += (uint64_t) n;
        return got == (ssize_t) n;
    };
    uint32_t magic = 0, version = 0;
    uint64_t n_t = 0, n_kv = 0;
    if (!get(&magic, 4) || magic != 0x46554747u || !get(&version, 4) || !get(&n_t, 8) || !get(&n_kv, 8))
        return "bad GGUF header";
    auto skip_str = [&]() { uint64_t n; if (!get(&n, 8)) return false; pos += n; return true; };
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t kl = 0;
        if (!get(&kl, 8) || kl > 4096) return "bad key";
        std::string k(kl, '\0');
        if (!get(k.data(), kl)) return "bad key";
        uint32_t t = 0;
        if (!get(&t, 4)) return "bad type";
        if (t == 8) { if (!skip_str()) return "bad string"; continue; }
        if (t != 9) { size_t s = detail::scalar_size(t); if (!s) return "bad scalar type"; pos += s; continue; }
        uint32_t et = 0;
        uint64_t cnt = 0;
        if (!get(&et, 4) || !get(&cnt, 8)) return "bad array";
        size_t s = detail::scalar_size(et);
        if (k != key) {
            if (et == 8) { for (uint64_t j = 0; j < cnt; ++j) if (!skip_str()) return "bad string array"; }
            else { if (!s) return "bad array type"; pos += s * cnt; }
            continue;
        }
        if (!s || et == 6 || et == 12) return key + ": not an integer array";
        out.resize(cnt);
        for (uint64_t j = 0; j < cnt; ++j) {
            uint8_t b[8] = {};
            if (!get(b, s)) return key + ": truncated";
            int64_t v = 0;
            switch (et) {
                case 0: case 7: v = b[0]; break;
                case 1: v = (int8_t) b[0]; break;
                case 2: { uint16_t x; std::memcpy(&x, b, 2); v = x; break; }
                case 3: { int16_t x; std::memcpy(&x, b, 2); v = x; break; }
                case 4: { uint32_t x; std::memcpy(&x, b, 4); v = x; break; }
                case 5: { int32_t x; std::memcpy(&x, b, 4); v = x; break; }
                default: std::memcpy(&v, b, 8); break;
            }
            out[j] = v;
        }
        return {};
    }
    return "missing " + key;
}

namespace detail {
inline std::string i64(const GgufFile & g, const std::string & k, int64_t & out) {
    const MetaValue * v = g.get(k);
    if (!v) return "missing " + k;
    if (!v->is_num()) return k + " is not a number";
    out = (int64_t) v->num();
    return {};
}
inline std::string f64(const GgufFile & g, const std::string & k, double & out) {
    const MetaValue * v = g.get(k);
    if (!v) return "missing " + k;
    if (!v->is_num()) return k + " is not a number";
    out = v->num();
    return {};
}
}  // namespace detail

inline std::string hy4_read_geometry(const GgufModel & model, Hy4Geometry & G) {
    const GgufFile & g = model.meta();
    const MetaValue * arch = g.get("general.architecture");
    if (!arch || arch->s != "hy_v4_flash") return "not a hy_v4_flash GGUF";
    std::string e;
    const std::string P = "hy_v4_flash.";
    struct U { const char * k; int64_t * v; };
    for (const U & u : std::initializer_list<U>{
             { "block_count", &G.n_layer }, { "leading_dense_block_count", &G.n_dense },
             { "context_length", &G.n_ctx }, { "vocab_size", &G.n_vocab },
             { "embedding_length", &G.n_embd }, { "feed_forward_length", &G.n_ff },
             { "attention.head_count_kv", &G.n_head_kv }, { "attention.key_length", &G.d_key },
             { "attention.value_length", &G.d_val }, { "attention.q_lora_rank", &G.q_lora },
             { "attention.kv_lora_rank", &G.kv_lora }, { "attention.key_length_mla", &G.d_key_mla },
             { "attention.value_length_mla", &G.d_val_mla }, { "rope.dimension_count", &G.d_rope },
             { "rope.freq_base", &G.rope_base },
             { "rope.freq_base_swa", &G.rope_base_swa }, { "attention.sliding_window", &G.swa },
             { "hyper_connection.count", &G.hc },
             { "expert_count", &G.n_expert }, { "expert_used_count", &G.n_used },
             { "expert_shared_count", &G.n_shared }, { "expert_feed_forward_length", &G.d_expert },
             { "expert_latent_length", &G.d_latent }, { "expert_gating_func", &G.gating },
             { "expert_weights_norm", &G.weights_norm }, { "attention.indexer.head_count", &G.idx_heads },
             { "attention.indexer.key_length", &G.idx_dim }, { "attention.indexer.top_k", &G.idx_topk },
             { "attention.indexer.kpool", &G.idx_kpool }, { "attention.indexer.kpool_select_tail", &G.idx_tail },
             { "nextn_predict_layers", &G.n_nextn }, { "ple.ngram_size", &G.ple_ngram },
             { "ple.heads_per_ngram", &G.ple_heads }, { "embedding_length_per_layer_input", &G.ple_dim },
             { "engram.pad_token_id", &G.engram_pad } })
        if (!(e = detail::i64(g, P + u.k, *u.v)).empty()) return e;
    if (!(e = detail::f64(g, P + "expert_weights_scale", G.weights_scale)).empty()) return e;
    if (!(e = detail::f64(g, P + "attention.layer_norm_rms_epsilon", G.rms_eps)).empty()) return e;
    if (!(e = detail::f64(g, P + "hyper_connection.epsilon", G.hc_eps)).empty()) return e;
    if (!(e = detail::f64(g, P + "hyper_connection.magnitude", G.hc_mag)).empty()) return e;

    const std::string & path = g.path();
    if (!(e = hy4_read_int_array(path, P + "attention.head_count", G.head_count)).empty()) return e;
    if (!(e = hy4_read_int_array(path, P + "attention.sliding_window_pattern", G.swa_pattern)).empty()) return e;
    if (!(e = hy4_read_int_array(path, P + "attention.indexer.types", G.idx_types)).empty()) return e;
    if (!(e = hy4_read_int_array(path, P + "ple.layers", G.ple_layers)).empty()) return e;
    if (!(e = hy4_read_int_array(path, P + "ple.layer_multipliers", G.ple_mult)).empty()) return e;
    if (!(e = hy4_read_int_array(path, P + "ple.head_offsets", G.ple_offsets)).empty()) return e;
    if (!(e = hy4_read_int_array(path, P + "ple.head_vocab_sizes", G.ple_sizes)).empty()) return e;
    if (!(e = hy4_read_int_array(path, P + "engram.token_map", G.token_map)).empty()) return e;

    if ((int64_t) G.head_count.size() != G.n_layer || (int64_t) G.swa_pattern.size() != G.n_layer ||
        (int64_t) G.idx_types.size() != G.n_layer - G.n_nextn)
        return "per-layer arrays do not match block_count";
    if ((int64_t) G.ple_offsets.size() != (int64_t) G.ple_layers.size() * G.ple_heads * (G.ple_ngram - 1) ||
        (int64_t) G.ple_sizes.size() != (int64_t) G.ple_offsets.size() ||
        (int64_t) G.ple_mult.size() != (int64_t) G.ple_layers.size() * G.ple_ngram)
        return "PLE arrays do not match their shape";
    if ((int64_t) G.token_map.size() != G.n_vocab) return "engram.token_map size";
    G.n_head = G.head_count[0];
    for (int64_t h : G.head_count) if (h != G.n_head && h != G.n_head_swa) {
        if (!G.n_head_swa) G.n_head_swa = h;
        else if (h != G.n_head && h != G.n_head_swa) return "more than two head counts";
    }
    if (!G.n_head_swa) G.n_head_swa = G.n_head;
    if (G.n_dense < 1 || G.n_expert < 1 || G.n_used < 1 || G.hc < 1 || G.d_latent < 1) return "unsupported shape";
    if (!g.get("tokenizer.ggml.bos_token_id") || !g.get("tokenizer.chat_template")) return "missing tokenizer metadata";
    return {};
}

/// Presence/type check of the tensor map.  Reports the first problem, does not load tensor data.
inline std::string hy4_check_tensors(const GgufModel & m, const Hy4Geometry & G) {
    for (int64_t l = 0; l < G.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const char * common[] = { "attn_norm", "attn_q_a", "attn_q_b", "attn_q_a_norm", "attn_kv_a_mqa",
            "attn_kv_a_norm", "attn_k_b", "attn_v_b", "attn_gate", "attn_output", "attn_sinks", "ffn_norm",
            "hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
            "post_attention_norm", "post_ffw_norm" };
        const bool draft = G.n_nextn && l >= G.n_layer - G.n_nextn;
        for (const char * n : common) {
            if (draft && !std::strncmp(n, "hc_", 3)) continue;  // the MTP block has no hyper-connections of its own
            if (!m.find(b + n + ".weight")) return "missing " + b + n + ".weight";
        }
        if (l < (int64_t) G.idx_types.size() && G.idx_types[(size_t) l]) {
            const char * idx[] = { "indexer.attn_k", "indexer.attn_q_b", "indexer.k_norm", "indexer.proj",
                "indexer_compressor_ape", "indexer_compressor_gate", "indexer_compressor_norm",
                "indexer_compressor_value" };
            for (const char * n : idx)
                if (!m.find(b + n + ".weight")) return "missing " + b + n + ".weight";
        }
        if (l == G.n_layer - G.n_nextn && G.n_nextn) {
            const char * mtp[] = { "nextn.eh_proj", "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm" };
            for (const char * n : mtp)
                if (!m.find(b + n + ".weight")) return "missing " + b + n + ".weight";
        }
        if (l < G.n_dense) {
            const char * dense[] = { "ffn_gate", "ffn_up", "ffn_down" };
            for (const char * n : dense)
                if (!m.find(b + n + ".weight")) return "missing " + b + n + ".weight";
        } else {
            const char * exps[] = { "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
                "ffn_routed_down", "ffn_routed_up", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" };
            for (const char * n : exps)
                if (!m.find(b + n + ".weight")) return "missing " + b + n + ".weight";
            if (!m.find(b + "exp_probs_b.bias")) return "missing " + b + "exp_probs_b.bias";
        }
    }
    for (int64_t l : G.ple_layers) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const char * en[] = { "engram_gate", "engram_hc_scale", "engram_kv", "engram_kv_b",
            "engram_lookup_norm", "engram_pre_gamma" };
        for (const char * n : en)
            if (!m.find(b + n + ".weight")) return "missing " + b + n + ".weight";
    }
    if (!m.find("token_embd.weight") || !m.find("output.weight") || !m.find("output_norm.weight") ||
        !m.find("output_hc_base.weight") || !m.find("output_hc_fn.weight") || !m.find("output_hc_scale.weight") ||
        !m.find("per_layer_token_embd.weight")) return "missing output / embedding / Engram table";
    return {};
}

}  // namespace almond::hy4

// src/hy4/hy4_forward.hpp - the full single-pass Hy4-flash forward (no persistent KV cache), ending in logits.
#pragma once

#include "hy4_geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace almond::hy4 {

/// The keys of every trunk layer for the positions already run (see hy4_mla_attn_multi_cpu).
struct Hy4KvCache {
    std::vector<std::vector<float>> k;   // [trunk layer] -> F32 rows of kv_lora + rope_dim
    int64_t n_past = 0;
};

/// `tokens` [nt] and `prev_ids` [nt * (ngram-1)] feed the Engram layers. `logits` is ggml-order [vocab, nt].
/// With `cache`, the tokens continue the cached sequence: `pos` must be n_past .. n_past+nt-1, and the cache grows.
/// Without it the pass sees only these nt tokens, as before.
bool hy4_forward_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t nt,
                     const int64_t * tokens, const int64_t * prev_ids, const int32_t * pos,
                     std::vector<float> & logits, std::string & err, Hy4KvCache * cache = nullptr);

}  // namespace almond::hy4

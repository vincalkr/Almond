// src/hy4/hy4_layer.hpp - one full Hy4-flash trunk layer (iHC, Engram, gated MLA attention, LatentMoE).
#pragma once

#include "hy4_geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace almond::hy4 {

/// Runs trunk layer `layer` for `nt` tokens. `streams` is ggml-order [n_embd, hc, nt] on input and output.
/// `tokens` and `prev_ids` ([nt * (n_gram-1)] raw ids, newest first, -1 before the sequence) feed the Engram hash
/// when the layer owns one; pass nullptr otherwise.
bool hy4_layer_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer, int64_t nt,
                   const float * streams, const int64_t * tokens, const int64_t * prev_ids,
                   const int32_t * pos, std::vector<float> & out, std::string & err,
                   std::vector<float> * kcache = nullptr);  // this layer's keys, see hy4_mla_attn_multi_cpu

}  // namespace almond::hy4

// src/hy4/hy4_moe.hpp - the LatentMoE layer of Hy4-flash as a ggml graph.
#pragma once

#include "hy4_geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace almond::hy4 {

/// Router at full width, experts at latent width. `x` and `out` are ggml-order [n_embd, n_tokens].
bool hy4_latent_moe_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer, int64_t nt,
                        const float * x, std::vector<float> & out, std::string & err);

/// The dense SwiGLU FFN of layer 0 (hidden -> 13824 -> hidden). x and out are ggml-order [n_embd, n_tokens].
bool hy4_dense_ffn_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer, int64_t nt,
                       const float * x, std::vector<float> & out, std::string & err);

}  // namespace almond::hy4

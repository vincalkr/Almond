// src/hy4/hy4_attn.hpp - the gated MLA attention of a Hy4-flash sparse layer, as a ggml graph.
#pragma once

#include "hy4_geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace almond::hy4 {

/// One-token (single-key) MLA attention: with one key the softmax is 1 and RoPE cancels in q·k, so this isolates the
/// weight layout, the norms, the latent absorption and the output projection for verification. `x` and `out` are
/// ggml-order [n_embd, n_tokens] with n_tokens == 1 in this first block.
bool hy4_mla_attn_one_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer,
                          const float * x, std::vector<float> & out, std::string & err);

/// Multi-token MLA attention with NORM RoPE, a causal mask and the layer's attention sinks, using ggml's flash
/// attention. `x` is ggml-order [n_embd, n_tokens] and `out` the same shape.
/// `kcache`, when given, holds this layer's earlier keys as F32 rows of kv_lora + rope_dim (normed latent, then the
/// roped part) for positions 0 .. n_past-1; the new tokens must sit at positions n_past .. n_past+nt-1, attend to the
/// cached keys too, and their keys are appended. Sliding-window layers see only the last `sliding_window` positions.
/// Sparse layers attend to every key: that is what PR #32517's k-pool indexer selects while the context fits in its
/// top-k (hy4_max_exact_ctx); beyond that this is not the model's attention.
bool hy4_mla_attn_multi_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer, int64_t nt,
                            const float * x, const int32_t * pos, std::vector<float> & out, std::string & err,
                            std::vector<float> * kcache = nullptr);

/// The longest context for which the sparse layers' full attention equals the indexer's selection.
inline int64_t hy4_max_exact_ctx(const Hy4Geometry & G) {
    return G.idx_topk > 0 && G.idx_kpool > 0 ? G.idx_topk - G.idx_kpool * G.idx_tail : (int64_t) 1 << 40;
}

}  // namespace almond::hy4

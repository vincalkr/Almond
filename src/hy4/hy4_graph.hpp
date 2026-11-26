// src/hy4/hy4_graph.hpp - the Engram projection as a ggml graph (CPU reference for the engine path).
#pragma once

#include "hy4_engram.hpp"

#include <string>
#include <vector>

namespace almond::hy4 {

/// Loads the six Engram weight tensors of one module from the GGUF into contiguous F32 arrays in GGUF order
/// (ne0 fastest). Only F32 and BF16 occur in these tensors in the IQ2-AH build.
struct EngramWeights {
    std::vector<float> lookup_norm;  ///< [dim, n_cols]
    std::vector<float> kv;           ///< [dim*n_cols, n_embd*(hc+1)]
    std::vector<float> kv_b;         ///< [n_embd, hc+1]
    std::vector<float> gate;         ///< [n_embd, hc]
    std::vector<float> pre_gamma;    ///< [n_embd]
    std::vector<float> hc_scale;     ///< [n_embd, hc]
};

bool hy4_engram_load_weights(const GgufModel & m, const Hy4Geometry & G, int module,
                             EngramWeights & w, std::string & err);

/// Runs PR #32517's build_engram on the CPU. `x` is [n_embd, hc, n_tokens], `ids` are the row ids from the hash
/// ([n_tokens, n_cols]), `valid` the per-column masks. Returns the updated streams x + delta [n_embd, hc, n_tokens]
/// (the output parameter is still called `delta`).
bool hy4_engram_forward_cpu(const GgufModel & m, const Hy4Geometry & G, int module, int n_tokens,
                            const float * x, const int32_t * ids, const float * valid,
                            const uint8_t * table, std::vector<float> & delta, std::string & err);

}  // namespace almond::hy4

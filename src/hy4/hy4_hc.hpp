// src/hy4/hy4_hc.hpp - identity Hyper-Connections (iHC) pre/post mixing, as ggml graphs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace almond::hy4 {

struct HcPre {
    std::vector<float> reduced;  ///< [n_embd, n_tokens]
    std::vector<float> post;     ///< [hc, n_tokens]
    std::vector<float> mixes;    ///< [2*hc, n_tokens], for verification
    std::vector<float> pre_g;    ///< [hc, n_tokens], for verification
};

/// The pre-mix: reduce the hc streams with the learned sigmoid gates, and produce the post gates.
bool hy4_hc_pre_cpu(int64_t D, int64_t HC, int64_t nt, const float * x, const float * fn,
                    const float * scale, const float * base, float eps, float mag,
                    HcPre & out, std::string & err);

/// The post-mix: out[:, h, t] = residual[:, h, t] + x[:, t] * post[h, t].
bool hy4_hc_post_cpu(int64_t D, int64_t HC, int64_t nt, const float * x, const float * residual,
                     const float * post, std::vector<float> & out, std::string & err);

}  // namespace almond::hy4

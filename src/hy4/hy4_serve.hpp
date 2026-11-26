// src/hy4/hy4_serve.hpp - the serve protocol of the other paths (INFO / READY, GEN -> RESUME / PP / T / DONE, STOP,
// CONVCLEAR, QUIT) over Hy4Engine, so serve/server.py drives it unchanged (tools/hy4/serve/serve_hy4.py).
#pragma once

#include "hy4_engine.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace almond::hy4 {

struct Hy4Sampling {
    float temp = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    float rep = 1.0f, freq = 0.0f, pres = 0.0f;
    int last_n = 64;
    uint64_t seed = 0;
};

/// One token from `logits` [V] under `s`; `hist` (the tokens so far) feeds the penalties.
int64_t hy4_sample(const float * logits, int64_t V, const Hy4Sampling & s, const std::vector<int64_t> & hist,
                   std::mt19937_64 & rng);

/// Reads commands from stdin until it closes or QUIT. `eos` ends a turn.
int hy4_serve(Hy4Engine & eng, const std::vector<int64_t> & eos, const Hy4Sampling & defaults, int mtp_n = 1,
              int64_t image_pad = 120120);   // <｜hy_image▁pad｜>

}  // namespace almond::hy4

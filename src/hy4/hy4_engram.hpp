// src/hy4/hy4_engram.hpp - the Engram (PLE) n-gram hash and the FP8 row reader.
//
// The algorithm is the one in llama.cpp PR #32517 (src/models/hy-v4-flash.cpp, engram_hash): for each of the four
// Engram modules a token's compressed id and its two previous ids are combined with the module's three multipliers;
// for each n-gram order >= 2 and each of the eight heads the rolling hash picks one row of the shared table.
// Every number (multipliers, bucket counts, head offsets, token map) comes from the GGUF metadata - see hy4_geometry.
#pragma once

#include "hy4_geometry.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace almond::hy4 {

struct EngramModule {
    int64_t layer = 0;                 ///< 0-based trunk layer that owns it
    std::vector<uint64_t> mult;        ///< [n_gram]
    std::vector<uint32_t> prime;       ///< [(n_gram-1) * heads]
    std::vector<uint32_t> offset;      ///< [(n_gram-1) * heads], rebased into the one table
};

struct EngramLayout {
    int64_t n_gram = 0, heads = 0, dim = 0, pad = 0;
    std::vector<EngramModule> modules;
    std::vector<int64_t> token_map;
};

inline EngramLayout hy4_engram_layout(const Hy4Geometry & G) {
    EngramLayout L;
    L.n_gram = G.ple_ngram;
    L.heads = G.ple_heads;
    L.dim = G.ple_dim;
    L.pad = G.engram_pad;
    L.token_map = G.token_map;
    const int64_t cols = G.ple_heads * (G.ple_ngram - 1);
    L.modules.resize(G.ple_layers.size());
    for (size_t e = 0; e < L.modules.size(); ++e) {
        auto & m = L.modules[e];
        m.layer = G.ple_layers[e];
        m.mult.assign(G.ple_mult.begin() + (ptrdiff_t) (e * G.ple_ngram),
                      G.ple_mult.begin() + (ptrdiff_t) ((e + 1) * G.ple_ngram));
        m.prime.assign(G.ple_sizes.begin() + (ptrdiff_t) (e * cols),
                       G.ple_sizes.begin() + (ptrdiff_t) ((e + 1) * cols));
        m.offset.assign(G.ple_offsets.begin() + (ptrdiff_t) (e * cols),
                        G.ple_offsets.begin() + (ptrdiff_t) ((e + 1) * cols));
    }
    return L;
}

/// The compressed id of a token; -1 when the token cannot be hashed (out of range, image position).
inline int64_t hy4_engram_cid(const EngramLayout & L, int64_t token) {
    if (token < 0 || token >= (int64_t) L.token_map.size()) return -1;
    return L.token_map[(size_t) token];
}

/// Fill `ids` with [n_cols] row ids for one token of module `e`. `prev` holds the previous n_gram-1 raw token ids,
/// newest first (prev[0] is the token just before); -1 or an unmapped id is replaced by the pad id, as PR #32517 does. `valid` is 0 for an image/absent token.
inline void hy4_engram_hash(const EngramLayout & L, int64_t e, int64_t token, const int64_t * prev,
                            int32_t * ids, float * valid) {
    const EngramModule & m = L.modules[(size_t) e];
    const int64_t n_cols = L.heads * (L.n_gram - 1);
    const int64_t pad = L.pad < (int64_t) L.token_map.size() ? L.token_map[(size_t) L.pad] : L.pad;
    int64_t c = hy4_engram_cid(L, token);
    const bool dead = c < 0;
    uint64_t toks[8];  // n_gram is 3 in the file; the reference derives the multiplier count from it
    if ((int64_t) m.mult.size() > 8) return;
    toks[0] = dead ? (uint64_t) pad : (uint64_t) c;
    for (int64_t s = 1; s < L.n_gram; ++s) {
        c = prev ? hy4_engram_cid(L, prev[(size_t) (s - 1)]) : -1;
        toks[s] = c < 0 ? (uint64_t) pad : (uint64_t) c;
    }
    uint64_t rolling = toks[0] * m.mult[0];
    for (int64_t o = 1; o < L.n_gram; ++o) {
        rolling ^= toks[(size_t) o] * m.mult[(size_t) o];
        for (int64_t h = 0; h < L.heads; ++h) {
            const int64_t col = (o - 1) * L.heads + h;
            ids[col] = (int32_t) (rolling % m.prime[(size_t) col] + m.offset[(size_t) col]);
            valid[col] = dead ? 0.0f : 1.0f;
        }
    }
    (void) n_cols;
}

/// Decode one FP8 E4M3 byte (type 43) to float. E4M3FN: 4 exponent, 3 mantissa, no infinity; 0x7F is NaN.
inline float hy4_f8_e4m3_to_float(uint8_t b) {
    const uint32_t sign = (uint32_t) (b & 0x80) << 24;
    const uint32_t exp = (b >> 3) & 0x0F;
    const uint32_t mant = b & 0x07;
    uint32_t bits;
    if (exp == 0) {
        if (!mant) bits = sign;                                     // +/- 0
        else {                                                      // subnormal: 2^-6 * mant/8
            float v = (float) mant * (1.0f / 512.0f);
            std::memcpy(&bits, &v, 4);
            return (b & 0x80) ? -v : v;
        }
    } else if (exp == 15 && mant == 7) {
        bits = sign | 0x7FC00000u;                                  // NaN
    } else {
        bits = sign | ((exp + 120u) << 23) | (mant << 20);          // bias 7 -> IEEE bias 127
    }
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

}  // namespace almond::hy4

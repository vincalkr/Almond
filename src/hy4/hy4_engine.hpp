// src/hy4/hy4_engine.hpp - the Hy4-flash engine on the cards: weights uploaded once, one ggml graph per step over the
// HIP backends and the CPU (ggml_backend_sched), the KV cache in VRAM.
//
// Placement: the trunk is cut into contiguous layer ranges, one per card. Each card holds its layers' attention,
// shared expert, norms, iHC and Engram weights and their KV rows; a card's first layers also keep their routed experts
// in VRAM while they fit, and the experts of the remaining layers are read in place from the mapped GGUF and run on
// the CPU (ggml's mul_mat_id touches only the 16 experts a token picks). The Engram table (FP8, 63 GB) stays in the
// mapped second shard; its rows are gathered and decoded on the host for each step.
#pragma once

#include "hy4_geometry.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace almond::hy4 {

struct Hy4EngineOptions {
    std::vector<int> gpus{0, 1};
    int64_t n_ctx = 4096;          ///< KV rows per layer (rounded up to 256)
    int64_t n_ubatch = 512;        ///< tokens per graph during the prompt
    int threads = 0;               ///< CPU threads for the experts in RAM; 0 = hardware threads / 2
    int64_t reserve_mib = 2048;    ///< VRAM left free per card for the compute buffers
    int expert_layers_gpu = -1;    ///< routed-expert layers in VRAM; -1 = as many as fit
    std::string engram = "auto";  ///< the Engram table: ram, ssd (O_DIRECT rows), or auto (RAM when it fits)
    bool dense_q8 = false;
    bool dense_f16 = false;        ///< the BF16 matrices stored as F16 on the cards (--dense-f16): the same values
    bool mtp = false;
    bool fused_hc = false;
    bool fused_proj = false;       ///< with --dense-q8: the attention input's four projections as one product (--fused-proj)         ///< iHC through the vendored ggml's fused DeepSeek-V4 ops (--fused-hc)
    int mtp_n = 1;                 ///< guesses per step (--mtp-n)              ///< load the MTP block (layer 42) and draft one token ahead (--mtp)         ///< the 2-D BF16 matrices quantized to Q8_0 on load (--dense-q8)
    bool verbose = true;
};

class Hy4Engine {
public:
    Hy4Engine();
    ~Hy4Engine();
    Hy4Engine(const Hy4Engine &) = delete;
    Hy4Engine & operator=(const Hy4Engine &) = delete;

    /// Opens the GGUF (any shard), uploads the weights and makes the KV cache.
    bool load(const std::string & gguf, const Hy4EngineOptions & o, std::string & err);

    /// Runs `nt` tokens after the ones already in the cache and returns the last token's logits [n_vocab].
    /// `rows` (optional, [nt]): an image embedding row [n_embd] for a position, or nullptr for a token; a position with
    /// a row should carry a negative id (an image key: the cache compares it, the Engram treats it as no token).
    /// `yx` (with rows, [2 nt]): each image row's (line, column) in its picture, -1 for a token; a picture starts at
    /// (0, 0). The rows then take Tencent's 2-D positions (interleaved M-RoPE) and the text after it its offset.
    bool decode(const int64_t * tokens, int64_t nt, std::vector<float> & logits, std::string & err,
                const float * const * rows = nullptr, const int32_t * yx = nullptr);

    /// As decode, but `logits` gets every token's row: [nt * n_vocab] (perplexity).
    bool decode_all(const int64_t * tokens, int64_t nt, std::vector<float> & logits, std::string & err);

    /// --mtp: the block's greedy guess for the token after `token`, where `token` is the next one at n_past (it
    /// is not added to the cache; decode it, with the guess after it, to check the guess).
    bool draft(int64_t token, int64_t & out, std::string & err);
    /// Up to n guesses in a row, each block step fed the previous one's state and guess.
    bool draft_chain(int64_t token, int n, std::vector<int64_t> & out, std::string & err);
    bool has_mtp() const;

    /// The conversation in the cache to a file (tokens, the caches' used rows, the MTP states) and back; `bytes` is
    /// the file's size. restore() refuses a file of another model geometry or context layout.
    bool save(const std::string & path, int64_t & tokens, uint64_t & bytes, std::string & err);
    bool restore(const std::string & path, int64_t & tokens, uint64_t & bytes, std::string & err);

    /// Drops the cache back to `n` tokens (0 = empty); the next decode continues from there.
    void truncate(int64_t n);

    int64_t n_past() const;
    int64_t n_ctx() const;
    const Hy4Geometry & geometry() const;
    const std::vector<int64_t> & history() const;   ///< the tokens in the cache, in order

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace almond::hy4

// test_expert_cache_chunks - the segmented ExpertCache's shrink / grow (the DeepSeek cache loan) on this GPU, small:
// slots of 0.5-3 MiB in chunks of ~8 MiB, every slot filled with its own pattern; then CYCLES times: shrink to a
// part, scribble the freed VRAM with a foreign allocation (what the prompt batch does with it), grow, refill the
// slots that came back, and compare EVERY slot with its pattern.  A HIP build's chunked cache must pass; the VMM
// segments it replaced did not (2026-11-15).  Usage: test_expert_cache_chunks [CYCLES] [SLOTS]
#include "almond/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using almond::core::ExpertCache;

static std::vector<uint8_t> pattern(int64_t slot, int64_t bytes, int gen) {
    std::vector<uint8_t> v((size_t) bytes);
    uint32_t x = (uint32_t) (slot * 2654435761u) ^ (uint32_t) (gen * 40503u) ^ 0x9e3779b9u;
    for (auto& c : v) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; c = (uint8_t) x; }
    return v;
}

int main(int argc, char** argv) {
    const int cycles = argc > 1 ? std::atoi(argv[1]) : 30;
    const int n = argc > 2 ? std::atoi(argv[2]) : 48;
    std::vector<int64_t> sizes((size_t) n);
    for (int i = 0; i < n; ++i) sizes[(size_t) i] = (int64_t) (512 + (i * 977) % 2560) * 1024;   // 0.5-3 MiB
    ExpertCache c;
    c.set_segment_bytes(8 << 20);
    std::string err;
    if (!c.open_sized(sizes, 4, n, err)) { std::printf("open: %s\n", err.c_str()); return 1; }
    std::printf("open: %d slots, %.1f MiB, segmented %d\n", n, c.full_bytes() / 1048576.0, (int) c.segmented());
    std::vector<int> gen((size_t) n, 0);
    auto fill = [&](int s) -> bool {
        const auto p = pattern(s, sizes[(size_t) s], gen[(size_t) s]);
        return c.fill_slot_blocking(s, p.data(), err, sizes[(size_t) s]);
    };
    auto verify_all = [&](const char* when) -> int {
        int bad = 0;
        for (int s = 0; s < c.slots(); ++s) {
            const auto p = pattern(s, sizes[(size_t) s], gen[(size_t) s]);
            if (!c.verify_slot(s, p.data(), err, sizes[(size_t) s])) { ++bad; if (bad <= 4) std::printf("  %s: slot %d BAD\n", when, s); }
        }
        return bad;
    };
    for (int s = 0; s < n; ++s) if (!fill(s)) { std::printf("fill: %s\n", err.c_str()); return 1; }
    int fails = verify_all("after the fill") > 0;
    for (int k = 0; k < cycles; ++k) {
        const int64_t keep = c.full_bytes() * (1 + k % 3) / 4;
        const int64_t live0 = c.slots();
        if (!c.shrink(keep, err)) { std::printf("shrink: %s\n", err.c_str()); return 1; }
        const int64_t live1 = c.slots();
        fails += verify_all("after the shrink") > 0;
        void* foreign = nullptr;   // the batch's buffers in the freed VRAM, written over
        const size_t fb = (size_t) (c.full_bytes() - c.bytes());
        if (cudaMalloc(&foreign, fb) == cudaSuccess) { cudaMemset(foreign, 0xA5, fb); cudaDeviceSynchronize(); cudaFree(foreign); }
        if (!c.grow(c.full_bytes(), err)) { std::printf("grow: %s\n", err.c_str()); return 1; }
        for (int64_t s = live1; s < c.slots(); ++s) { ++gen[(size_t) s]; if (!fill((int) s)) { std::printf("refill: %s\n", err.c_str()); return 1; } }
        const int bad = verify_all("after the grow");
        fails += bad > 0;
        if (k < 3 || bad) std::printf("cycle %d: %lld -> %lld -> %lld slots live, %d bad\n", k, (long long) live0, (long long) live1,
                                      (long long) c.slots(), bad);
    }
    std::printf("%s: %d cycles, %d checks failed\n", fails ? "FAIL" : "PASS", cycles, fails);
    return fails ? 1 : 0;
}

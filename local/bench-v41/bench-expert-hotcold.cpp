// bench-expert-hotcold <shard1.gguf> - one decode token's grouped expert call (6 experts, 1 entry each) on real DeepSeek
// blobs, phase 1 (gate/up + swiglu + quantize) and phase 2 (down), with the weights hot (the same 6 experts every
// call: 50-60 MB, inside the 96 MB Infinity Cache) and cold (48 experts rotated, ~430 MB, read from VRAM).
#include "ds41_moe.hpp"
#include "almond/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <tuple>
#include <vector>
using namespace almond;
using namespace almond::v41;
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); } }

int main(int argc, char** argv) {
    Ds41FileBlobs src(argv[1]);
    std::string err;
    if (!src.open(err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom& g = src.geom();
    const int64_t D = g.n_embd, FF = g.n_ff;
    const int K = 6, NE = 48, ITERS = 300;
    std::printf("n_embd %lld n_ff %lld layers %lld experts %lld per_expert %d\n", (long long) D, (long long) FF,
                (long long) g.n_layers, (long long) g.n_experts, (int) g.per_expert());
    // one layer per (gate, up, down) triple
    // per format class: the first NE (layer, expert) of that class
    std::map<std::tuple<int, int, int>, std::vector<std::pair<int64_t, int64_t>>> pick;
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (int64_t e = 0; e < g.n_experts; ++e) {
            const int c = g.cls(l, e);
            if (c < 0) continue;
            auto& v = pick[std::make_tuple(g.cls_gu[c], g.cls_gu[c], g.cls_d[c])];
            if ((int) v.size() < NE) v.emplace_back(l, e);
        }
    // activations: one token
    std::mt19937 rng(3); std::normal_distribution<float> nd(0, 0.3f);
    std::vector<float> x(D); for (auto& v : x) v = nd(rng);
    float* d_xf; void* d_xq; float* d_out; void* d_scr;
    ck(cudaMalloc((void**) &d_xf, D * 4), "m"); ck(cudaMalloc(&d_xq, D / 32 * 36), "m");
    ck(cudaMalloc((void**) &d_out, K * D * 4), "m"); ck(cudaMalloc(&d_scr, kernels::native_expert_scratch_bytes(K, FF)), "m");
    ck(cudaMemcpy(d_xf, x.data(), D * 4, cudaMemcpyHostToDevice), "c");
    kernels::quantize_q8_1_rows(d_xf, 1, D, d_xq, nullptr);
    int32_t start[K + 1], ent_dst[K], ent_tok[K], ng = K;
    for (int i = 0; i <= K; ++i) start[i] = i;
    for (int i = 0; i < K; ++i) { ent_dst[i] = i; ent_tok[i] = 0; }
    int32_t *d_start, *d_ng, *d_dst, *d_tok;
    ck(cudaMalloc((void**) &d_start, sizeof start), "m"); ck(cudaMalloc((void**) &d_ng, 4), "m");
    ck(cudaMalloc((void**) &d_dst, sizeof ent_dst), "m"); ck(cudaMalloc((void**) &d_tok, sizeof ent_tok), "m");
    ck(cudaMemcpy(d_start, start, sizeof start, cudaMemcpyHostToDevice), "c"); ck(cudaMemcpy(d_ng, &ng, 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(d_dst, ent_dst, sizeof ent_dst, cudaMemcpyHostToDevice), "c"); ck(cudaMemcpy(d_tok, ent_tok, sizeof ent_tok, cudaMemcpyHostToDevice), "c");
    std::printf("%-26s %-5s %10s %10s %8s %8s %6s\n", "gate/up/down (layer)", "phase", "hot us", "cold us", "hot GB/s", "cold GB/s", "c/h");
    for (auto& [t, le] : pick) {
        const auto [gt, ut, dt] = t;
        if ((int) le.size() < NE) { std::printf("%d/%d/%d: only %zu experts, skipped\n", gt, ut, dt, le.size()); continue; }
        const long long l = le[0].first;
        const kernels::NativeExpertLayout L = kernels::native_expert_layout3(gt, ut, dt, D, FF);
        if (!kernels::native_expert_supported3(gt, ut, dt, D, FF)) { std::printf("unsupported %d/%d/%d\n", gt, ut, dt); continue; }
        std::vector<uint8_t> blob(src.blob_bytes());
        std::vector<void*> d_blobs(NE);
        for (int e = 0; e < NE; ++e) {
            src.read(le[e].first, le[e].second, blob.data());
            ck(cudaMalloc(&d_blobs[e], L.bytes), "m");
            ck(cudaMemcpy(d_blobs[e], blob.data(), L.bytes, cudaMemcpyHostToDevice), "c");
        }
        // NE/K pointer sets, back to back: set s = experts s*K .. s*K+K-1
        const int NS = NE / K;
        std::vector<unsigned long long> ptrs(NE);
        for (int e = 0; e < NE; ++e) ptrs[e] = (unsigned long long) (uintptr_t) d_blobs[e];
        unsigned long long* d_ptrs;
        ck(cudaMalloc((void**) &d_ptrs, NE * 8), "m"); ck(cudaMemcpy(d_ptrs, ptrs.data(), NE * 8, cudaMemcpyHostToDevice), "c");
        const double gu_bytes = (double) K * (L.down_off), d_bytes = (double) K * (L.bytes - L.down_off);
        for (int phase = 1; phase <= 2; ++phase) {
            kernels::native_expert_set_mode(-1, phase);
            float res[2];
            static const bool rev = std::getenv("REV") != nullptr;
            for (int k = 0; k <= 1; ++k) {
                const int cold = rev ? 1 - k : k;
                auto call = [&](int i) {
                    const unsigned long long* gp = d_ptrs + (cold ? (i % NS) * K : 0);
                    kernels::native_expert_grouped(L, gp, d_start, d_ng, d_dst, d_tok, K, K, d_xq, d_scr, d_out, nullptr, K);
                };
                for (int i = 0; i < 20; ++i) call(i);
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                ck(cudaDeviceSynchronize(), "s");
                cudaEventRecord(a, nullptr);
                for (int i = 0; i < ITERS; ++i) call(i);
                cudaEventRecord(b, nullptr); cudaEventSynchronize(b);
                float ms; cudaEventElapsedTime(&ms, a, b);
                res[cold] = ms * 1000.f / ITERS;
            }
            const double by = phase == 1 ? gu_bytes : d_bytes;
            std::printf("%3d/%3d/%3d (L%-3lld)          %-5s %10.1f %10.1f %8.0f %8.0f %6.2f\n", gt, ut, dt, (long long) l,
                        phase == 1 ? "gu" : "down", res[0], res[1], by / res[0] / 1e3, by / res[1] / 1e3, res[1] / res[0]);
        }
        for (auto p : d_blobs) cudaFree(p);
        cudaFree(d_ptrs);
    }
    return 0;
}

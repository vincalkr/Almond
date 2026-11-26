// bench_hy4_q8 [device] - the Q8_0 x F32 matrix-vector products the decode trunk pays, on the real dense shapes.
//
// Why this exists: the decode kernel that costs most is the dense Q8_0 mat-vec (rocprofv3, 2026-11-21: 114 ms of the
// 417 ms of kernel time in a 19-token window at 256K).  Our own row kernel (tools/v41/ggml/mmvq-rows.patch,
// ALMOND_MMVQ_ROWS=1) is worth +10,1% of decode where a step carries one token and nothing at the two to four tokens
// of an MTP verify batch (docs/hy4/GGUF_NOTES.md, "The same for Q8_0 rows").  Measuring that in the engine costs a
// model load and four minutes a run, and the wall clock of a whole round jitters ~2%; here the same products run on
// their own, so a variant is read in seconds and the number is the kernel's, not the machine's.
//
// The kernels are chosen by the environment ggml reads once per process, so each variant is a separate process:
// see local/hy4-q8-bench.sh for the loop that prints the table.
//
//   bench_hy4_q8 [device] [iters]        # device 1 is the card with room while the engine holds card 0
//
// One line per (shape, tokens a step, variant): the time, the weight bytes a call moves and their rate, and a hash of
// the result so variants that compute the same numbers are visible (the hash of the whole output, FNV-1a).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The dense products of Hy4-flash as the engine builds them: src0 is the weight [K, N] in Q8_0, src1 the activations
// [K, tokens].  From the GGUF with --dense-q8 (the file's BF16 dense matrices loaded as Q8_0).
struct Shape { const char * what; int K, N; };
static const Shape kShapes[] = {
    { "ffn_up / ffn_gate", 3072, 13824 },
    { "ffn_down",         13824,  3072 },
    { "attn_gate",         3072,  8192 },
    { "attn_output",       8192,  3072 },
    { "attn_q_b",          1024,  8192 },
    { "attn_q_a / kv_a",   3072,  1024 },
};

int main(int argc, char ** argv) {
    const int device  = argc > 1 ? std::atoi(argv[1]) : 1;
    const int iters   = argc > 2 ? std::atoi(argv[2]) : 40;
    const int nrep    = argc > 3 ? std::atoi(argv[3]) : 8;
    const char * knob = std::getenv("ALMOND_MMVQ_ROWS");
    const char * wpr  = std::getenv("ALMOND_MMVQ_WPR");
    std::printf("# device %d  rows-knob=%s wpr=%s  iters=%d  reps=%d\n", device,
                knob && knob[0] == '1' ? "on" : "off", wpr ? wpr : "-", iters, nrep);

    ggml_backend_t be = ggml_backend_cuda_init(device);
    if (!be) { std::fprintf(stderr, "ggml_backend_cuda_init(%d) failed\n", device); return 1; }

    for (const Shape & S : kShapes) {
        for (int nt = 1; nt <= 4; ++nt) {
            ggml_init_params p{ 8u << 20, nullptr, /* no_alloc */ true };
            ggml_context * ctx = ggml_init(p);
            // nrep distinct weights, one graph: per call the engine reads a layer's weights once and never reuses
            // them, so one weight replayed n times sits in the 48 MB of L2 and measures the cache, not the HBM.
            std::vector<ggml_tensor *> w(nrep);
            for (int r = 0; r < nrep; ++r) w[r] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, S.K, S.N);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  S.K, nt);
            std::vector<ggml_tensor *> y(nrep);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            for (int r = 0; r < nrep; ++r) { y[r] = ggml_mul_mat(ctx, w[r], x); ggml_build_forward_expand(gf, y[r]); }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            if (!buf) { std::fprintf(stderr, "%s: no room on device %d\n", S.what, device); return 1; }

            std::vector<uint8_t> wb;
            for (int r = 0; r < nrep; ++r) {
                wb.resize(ggml_nbytes(w[r]));
                for (size_t i = 0; i < wb.size(); ++i) wb[i] = (uint8_t) ((i * 1103515245u + 12345u + 7919u * r) >> 13);
                ggml_backend_tensor_set(w[r], wb.data(), 0, ggml_nbytes(w[r]));
            }
            std::vector<float> xf((size_t) S.K * nt);
            for (size_t i = 0; i < xf.size(); ++i) xf[i] = 0.02f * (float) ((int) ((i * 23) % 89) - 44);
            ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));

            for (int i = 0; i < 2; ++i) ggml_backend_graph_compute(be, gf);      // warm the allocator and the caches
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, gf);
            const double us = std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - t0).count() / (iters * nrep);

            std::vector<float> out((size_t) S.N * nt);
            ggml_backend_tensor_get(y[nrep - 1], out.data(), 0, ggml_nbytes(y[nrep - 1]));
            uint64_t h = 1469598103934665603ull;
            for (float v : out) { const uint8_t * b = (const uint8_t *) &v;
                for (size_t i = 0; i < 4; ++i) { h ^= b[i]; h *= 1099511628211ull; } }

            const double wbytes = (double) ggml_nbytes(w[0]);
            std::printf("%-18s K=%-6d N=%-6d nt=%d  %8.1f us  %7.1f GB/s (pesi)  hash %016llx\n",
                        S.what, S.K, S.N, nt, us, wbytes / (us * 1e-6) / 1e9, (unsigned long long) h);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(be);
    return 0;
}

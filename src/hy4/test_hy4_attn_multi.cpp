// test_hy4_attn_multi <gguf> <out.bin> - three-token gated MLA attention with NORM RoPE and causal mask.
#include "hy4_attn.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace almond;
using namespace almond::hy4;

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_hy4_attn_multi <gguf> <out.bin>\n"); return 2; }
    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(argv[1])); }
    catch (const std::exception & e) { std::fprintf(stderr, "open: %s\n", e.what()); return 1; }
    Hy4Geometry G;
    std::string err;
    if (!(err = hy4_read_geometry(*mp, G)).empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int64_t D = G.n_embd, nt = 3;
    const int32_t pos[3] = { 0, 1, 2 };
    std::vector<float> x((size_t) D * nt), out;
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.01f * (float) ((int64_t) ((i * 11) % 97) - 48);
    if (!hy4_mla_attn_multi_cpu(*mp, G, 0, nt, x.data(), pos, out, err)) {
        std::fprintf(stderr, "%s\n", err.c_str()); return 1;
    }
    FILE * f = std::fopen(argv[2], "wb");
    std::fwrite(x.data(), 4, x.size(), f);
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    std::printf("hy4 attn multi: D=%lld nt=%lld out %g %g %g\n", (long long) D, (long long) nt, out[0], out[1], out[2]);
    return 0;
}

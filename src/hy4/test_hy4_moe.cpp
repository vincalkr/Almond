// test_hy4_moe <gguf> <out.bin> - one-token LatentMoE on layer 1's real weights.
#include "hy4_moe.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
using namespace almond;
using namespace almond::hy4;
int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_hy4_moe <gguf> <out.bin>\n"); return 2; }
    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(argv[1])); }
    catch (const std::exception & e) { std::fprintf(stderr, "open: %s\n", e.what()); return 1; }
    Hy4Geometry G; std::string err;
    if (!(err = hy4_read_geometry(*mp, G)).empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<float> x((size_t) G.n_embd), out;
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.01f * (float) ((int64_t) ((i * 23) % 89) - 44);
    if (!hy4_latent_moe_cpu(*mp, G, 1, 1, x.data(), out, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    FILE * f = std::fopen(argv[2], "wb");
    std::fwrite(x.data(), 4, x.size(), f);
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    std::printf("hy4 moe: D=%lld out %g %g %g\n", (long long) G.n_embd, out[0], out[1], out[2]);
    return 0;
}

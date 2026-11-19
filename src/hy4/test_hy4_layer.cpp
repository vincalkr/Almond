// test_hy4_layer <gguf> <out.bin> - one full layer (layer 1: Engram + attention + MoE) on real weights.
#include "hy4_layer.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
using namespace almond;
using namespace almond::hy4;
int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_hy4_layer <gguf> <out.bin>\n"); return 2; }
    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(argv[1])); }
    catch (const std::exception & e) { std::fprintf(stderr, "open: %s\n", e.what()); return 1; }
    Hy4Geometry G; std::string err;
    if (!(err = hy4_read_geometry(*mp, G)).empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int64_t D = G.n_embd, HC = G.hc, nt = 2;
    std::vector<float> streams((size_t) D * HC * nt);
    for (size_t i = 0; i < streams.size(); ++i) streams[i] = 0.01f * (float) ((int64_t) ((i * 17) % 101) - 50);
    const int64_t tokens[2] = { 10, 20 }, prev[4] = { -1, -1, 10, -1 };
    const int32_t pos[2] = { 0, 1 };
    std::vector<float> out;
    if (!hy4_layer_cpu(*mp, G, 1, nt, streams.data(), tokens, prev, pos, out, err)) {
        std::fprintf(stderr, "%s\n", err.c_str()); return 1;
    }
    FILE * f = std::fopen(argv[2], "wb");
    std::fwrite(streams.data(), 4, streams.size(), f);
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    std::printf("hy4 layer: D=%lld HC=%lld nt=%lld out %g %g %g\n", (long long) D, (long long) HC, (long long) nt, out[0], out[1], out[2]);
    return 0;
}

// test_hy4_forward <gguf> <out.bin> - full 42-layer single pass over two tokens, dump logits.
#include "hy4_forward.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
using namespace almond;
using namespace almond::hy4;
int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_hy4_forward <gguf> <out.bin>\n"); return 2; }
    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(argv[1])); }
    catch (const std::exception & e) { std::fprintf(stderr, "open: %s\n", e.what()); return 1; }
    Hy4Geometry G; std::string err;
    if (!(err = hy4_read_geometry(*mp, G)).empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int64_t nt = 2;
    const int64_t tokens[2] = { 10, 20 }, prev[4] = { -1, -1, 10, -1 };
    const int32_t pos[2] = { 0, 1 };
    std::vector<float> logits;
    if (!hy4_forward_cpu(*mp, G, nt, tokens, prev, pos, logits, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    FILE * f = std::fopen(argv[2], "wb");
    std::fwrite(logits.data(), 4, logits.size(), f);
    std::fclose(f);
    std::printf("hy4 forward: layers=%lld logits=%zu first=%g %g\n", (long long) G.n_layer, logits.size(), logits[0], logits[1]);
    return 0;
}

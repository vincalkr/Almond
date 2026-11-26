// test_hy4_attn <gguf> <out.bin> - run the one-token MLA attention of a sparse layer (layer 0) and dump for the
// numpy reference in tools/hy4/hy4_attn_ref.py.
#include "hy4_attn.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace almond;
using namespace almond::hy4;

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_hy4_attn <gguf> <out.bin>\n"); return 2; }
    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(argv[1])); }
    catch (const std::exception & e) { std::fprintf(stderr, "open: %s\n", e.what()); return 1; }
    Hy4Geometry G;
    std::string err;
    if (!(err = hy4_read_geometry(*mp, G)).empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<float> x((size_t) G.n_embd), out;
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.01f * (float) ((int64_t) ((i * 11) % 97) - 48);
    if (!hy4_mla_attn_one_cpu(*mp, G, 0, x.data(), out, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    FILE * f = std::fopen(argv[2], "wb");
    std::fwrite(x.data(), 4, x.size(), f);
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    std::printf("hy4 attn: D=%lld out %g %g %g\n", (long long) G.n_embd, out[0], out[1], out[2]);
    return 0;
}

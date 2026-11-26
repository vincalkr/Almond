// test_hy4_graph <gguf> <out.bin> - run the Engram projection on four fixed tokens and dump inputs + output so that
// tools/hy4/hy4_engram_ref.py can recompute it independently with numpy.
#include "hy4_graph.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace almond;
using namespace almond::hy4;

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_hy4_graph <hy4.gguf> <out.bin>\n"); return 2; }
    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(argv[1])); }
    catch (const std::exception & e) { std::fprintf(stderr, "open: %s\n", e.what()); return 1; }
    Hy4Geometry G;
    std::string err;
    if (!(err = hy4_read_geometry(*mp, G)).empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const EngramLayout L = hy4_engram_layout(G);
    const int64_t D = G.n_embd, HC = G.hc, nt = 4, cols = L.heads * (L.n_gram - 1);
    const int64_t tokens[4] = { 10, 20, 30, 40 };
    const int64_t prev[2] = { 5, 6 };

    std::vector<float> x((size_t) D * HC * nt);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.01f * (float) ((int64_t) ((i * 37) % 101) - 50);
    std::vector<int32_t> ids((size_t) nt * cols);
    std::vector<float> valid((size_t) nt * cols);
    for (int64_t t = 0; t < nt; ++t)
        hy4_engram_hash(L, 0, tokens[t], prev, ids.data() + t * cols, valid.data() + t * cols);

    size_t which = 0;
    const TensorInfo * table = mp->find("per_layer_token_embd.weight", &which);
    if (!table) { std::fprintf(stderr, "no table\n"); return 1; }
    const uint8_t * tbl = mp->shard(which).tensor_data(*table);

    std::vector<float> delta;
    if (!hy4_engram_forward_cpu(*mp, G, 0, nt, x.data(), ids.data(), valid.data(), tbl, delta, err)) {
        std::fprintf(stderr, "forward: %s\n", err.c_str()); return 1;
    }

    FILE * f = std::fopen(argv[2], "wb");
    if (!f) { std::perror("fopen"); return 1; }
    const int64_t hdr[5] = { D, HC, nt, cols, G.ple_dim };
    std::fwrite(hdr, sizeof hdr, 1, f);
    std::fwrite(x.data(), sizeof(float), x.size(), f);
    std::fwrite(ids.data(), sizeof(int32_t), ids.size(), f);
    std::fwrite(valid.data(), sizeof(float), valid.size(), f);
    std::fwrite(delta.data(), sizeof(float), delta.size(), f);
    std::fclose(f);
    std::printf("hy4 graph: wrote %s (%lld floats of delta, first %g %g %g)\n", argv[2],
                (long long) delta.size(), delta[0], delta[1], delta[2]);
    return 0;
}

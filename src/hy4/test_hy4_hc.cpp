// test_hy4_hc <gguf> <out.bin> - run iHC pre/post on real layer-0 weights and dump inputs + outputs for the numpy
// reference in tools/hy4/hy4_hc_ref.py.
#include "hy4_hc.hpp"
#include "hy4_geometry.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace almond;
using namespace almond::hy4;

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_hy4_hc <gguf> <out.bin>\n"); return 2; }
    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(argv[1])); }
    catch (const std::exception & e) { std::fprintf(stderr, "open: %s\n", e.what()); return 1; }
    Hy4Geometry G;
    std::string err;
    if (!(err = hy4_read_geometry(*mp, G)).empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    const int64_t D = G.n_embd, HC = G.hc, nt = 3;
    auto load = [&](const std::string & name, std::vector<float> & v) {
        size_t which = 0;
        const TensorInfo * t = mp->find(name, &which);
        if (!t || t->type != 0) { err = "missing F32 " + name; return false; }
        v.resize(t->elements());
        std::memcpy(v.data(), mp->shard(which).tensor_data(*t), v.size() * 4);
        return true;
    };
    std::vector<float> fn, scale, base;
    if (!load("blk.0.hc_attn_fn.weight", fn) || !load("blk.0.hc_attn_scale.weight", scale) ||
        !load("blk.0.hc_attn_base.weight", base)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    std::vector<float> x((size_t) D * HC * nt), x2((size_t) D * nt), res((size_t) D * HC * nt);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.02f * (float) ((int64_t) ((i * 17) % 101) - 50);
    for (size_t i = 0; i < x2.size(); ++i) x2[i] = 0.01f * (float) ((int64_t) ((i * 29) % 97) - 48);
    for (size_t i = 0; i < res.size(); ++i) res[i] = 0.03f * (float) ((int64_t) ((i * 13) % 89) - 44);

    HcPre pre;
    if (!hy4_hc_pre_cpu(D, HC, nt, x.data(), fn.data(), scale.data(), base.data(),
                        (float) G.hc_eps, (float) G.hc_mag, pre, err)) {
        std::fprintf(stderr, "pre: %s\n", err.c_str()); return 1;
    }
    std::vector<float> post_out;
    if (!hy4_hc_post_cpu(D, HC, nt, x2.data(), res.data(), pre.post.data(), post_out, err)) {
        std::fprintf(stderr, "post: %s\n", err.c_str()); return 1;
    }
    FILE * f = std::fopen(argv[2], "wb");
    if (!f) { std::perror("fopen"); return 1; }
    const double hdr[2] = { G.hc_eps, G.hc_mag };
    const int64_t dims[3] = { D, HC, nt };
    std::fwrite(dims, sizeof dims, 1, f); std::fwrite(hdr, sizeof hdr, 1, f);
    auto wr = [&](const std::vector<float> & v) { std::fwrite(v.data(), 4, v.size(), f); };
    wr(x); wr(fn); wr(scale); wr(base); wr(pre.reduced); wr(pre.post); wr(pre.mixes); wr(pre.pre_g); wr(x2); wr(res); wr(post_out);
    std::fclose(f);
    std::printf("hy4 hc: reduced %g %g, post %g %g, out %g\n", pre.reduced[0], pre.reduced[1], pre.post[0],
                pre.post[1], post_out[0]);
    return 0;
}

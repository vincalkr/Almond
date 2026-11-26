// src/hy4/hy4_hc.cpp - iHC pre/post, ported from llama.cpp PR #32517 (build_hc_pre / build_hc_post).
#include "hy4_hc.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>

namespace almond::hy4 {
namespace {

ggml_tensor * tensor(ggml_context * c, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 0) {
    return ne2 ? ggml_new_tensor_3d(c, GGML_TYPE_F32, ne0, ne1, ne2) : ggml_new_tensor_2d(c, GGML_TYPE_F32, ne0, ne1);
}

bool run(ggml_context * c, ggml_cgraph * gf, ggml_tensor * out, std::string & err) {
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { err = "no CPU backend"; return false; }
    bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_free(cpu);
    if (!ok) err = "graph compute failed";
    return ok;
}

}  // namespace

bool hy4_hc_pre_cpu(int64_t D, int64_t HC, int64_t nt, const float * x, const float * fn,
                    const float * scale, const float * base, float eps, float mag,
                    HcPre & out, std::string & err) {
    ggml_init_params ip = { 4u * 1024u * 1024u, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tx = tensor(c, D, HC, nt);
    ggml_tensor * tfn = tensor(c, HC * D, 2 * HC);
    ggml_tensor * tscale = tensor(c, 2);
    ggml_tensor * tbase = tensor(c, 2 * HC);
    std::memcpy(ggml_get_data(tx), x, (size_t) D * HC * nt * 4);
    std::memcpy(ggml_get_data(tfn), fn, (size_t) HC * D * 2 * HC * 4);
    std::memcpy(ggml_get_data(tscale), scale, 8);
    std::memcpy(ggml_get_data(tbase), base, (size_t) 2 * HC * 4);

    ggml_cgraph * gf = ggml_new_graph_custom(c, 128, false);
    ggml_tensor * flat = ggml_reshape_2d(c, tx, HC * D, nt);
    ggml_tensor * norm = ggml_rms_norm(c, flat, eps);
    ggml_tensor * mixes = ggml_mul_mat(c, tfn, norm);             // [2*HC, nt]
    ggml_tensor * mg = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
    ggml_tensor * pg = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);

    ggml_tensor * s0 = ggml_view_1d(c, tscale, 1, 0);
    ggml_tensor * s1 = ggml_view_1d(c, tscale, 1, 4);
    ggml_tensor * b0 = ggml_view_1d(c, tbase, HC, 0);
    ggml_tensor * b1 = ggml_view_1d(c, tbase, HC, (size_t) HC * 4);

    ggml_tensor * pre = ggml_sigmoid(c, ggml_scale_bias(c, ggml_add(c, ggml_mul(c, mg, s0), b0), 1.0f, eps));
    ggml_tensor * post = ggml_sigmoid(c, ggml_scale_bias(c, ggml_add(c, ggml_mul(c, pg, s1), b1), 1.0f, eps));
    post = ggml_scale_bias(c, ggml_scale(c, post, mag), 1.0f, eps);

    ggml_tensor * red = nullptr;
    for (int64_t h = 0; h < HC; ++h) {
        ggml_tensor * xh = ggml_view_2d(c, tx, D, nt, tx->nb[2], (size_t) h * tx->nb[1]);
        ggml_tensor * ph = ggml_view_2d(c, pre, 1, nt, pre->nb[1], (size_t) h * pre->nb[0]);
        ggml_tensor * cur = ggml_mul(c, xh, ph);
        red = red ? ggml_add(c, red, cur) : cur;
    }
    for (ggml_tensor * t : { mixes, pre, post }) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }  // set_output alone does not put a tensor into the graph
    if (!run(c, gf, red, err)) { ggml_free(c); return false; }
    out.reduced.assign((size_t) D * nt, 0.0f);
    out.post.assign((size_t) HC * nt, 0.0f);
    std::memcpy(out.reduced.data(), ggml_get_data(red), out.reduced.size() * 4);
    std::memcpy(out.post.data(), ggml_get_data(post), out.post.size() * 4);
    out.mixes.assign((size_t) 2 * HC * nt, 0.0f);
    out.pre_g.assign((size_t) HC * nt, 0.0f);
    std::memcpy(out.mixes.data(), ggml_get_data(mixes), out.mixes.size() * 4);
    std::memcpy(out.pre_g.data(), ggml_get_data(pre), out.pre_g.size() * 4);
    ggml_free(c);
    return true;
}

bool hy4_hc_post_cpu(int64_t D, int64_t HC, int64_t nt, const float * x, const float * residual,
                     const float * post, std::vector<float> & out, std::string & err) {
    ggml_init_params ip = { 4u * 1024u * 1024u, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tx = tensor(c, D, nt), * tres = tensor(c, D, HC, nt), * tpost = tensor(c, HC, nt);
    std::memcpy(ggml_get_data(tx), x, (size_t) D * nt * 4);
    std::memcpy(ggml_get_data(tres), residual, (size_t) D * HC * nt * 4);
    std::memcpy(ggml_get_data(tpost), post, (size_t) HC * nt * 4);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 128, false);
    ggml_tensor * result = nullptr;
    for (int64_t h = 0; h < HC; ++h) {
        ggml_tensor * rh = ggml_view_2d(c, tres, D, nt, tres->nb[2], (size_t) h * tres->nb[1]);
        ggml_tensor * ph = ggml_view_2d(c, tpost, 1, nt, tpost->nb[1], (size_t) h * tpost->nb[0]);
        ggml_tensor * cur = ggml_add(c, rh, ggml_mul(c, tx, ph));
        cur = ggml_reshape_3d(c, cur, D, 1, nt);
        result = result ? ggml_concat(c, result, cur, 1) : cur;
    }
    if (!run(c, gf, result, err)) { ggml_free(c); return false; }
    out.assign((size_t) D * HC * nt, 0.0f);
    std::memcpy(out.data(), ggml_get_data(result), out.size() * 4);
    ggml_free(c);
    return true;
}

}  // namespace almond::hy4

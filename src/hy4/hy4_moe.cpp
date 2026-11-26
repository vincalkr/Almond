// src/hy4/hy4_moe.cpp - LatentMoE, ported from PR #32517 (build_latent_moe + llama.cpp's build_moe_ffn).
#include "hy4_moe.hpp"

#include "hy4_weights.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>

namespace almond::hy4 {
namespace {

ggml_tensor * load(ggml_context * c, const GgufModel & m, const std::string & name,
                   int64_t ne0, int64_t ne1, int64_t ne2, std::string & err) {
    (void) c;   // the weight is a tensor over the mapped file (hy4_weights.hpp), not a copy
    return hy4_weight(m, name, ne0, ne1, ne2, err);
}

}  // namespace

bool hy4_latent_moe_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer, int64_t nt,
                        const float * x, std::vector<float> & out, std::string & err) {
    const std::string b = "blk." + std::to_string(layer) + ".";
    const int64_t D = G.n_embd, NE = G.n_expert, USED = G.n_used;
    const int64_t LAT = G.d_latent, FF = G.d_expert;
    const float scale = (float) G.weights_scale;

    // The stacked expert weights: 512 x 1536 x 1536 at IQ2_XXS / Q4_K, about 95 MB in total.
    ggml_init_params ip = { 1536u * 1024u * 1024u, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tx = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, nt);
    std::memcpy(ggml_get_data(tx), x, (size_t) D * nt * 4);
    auto W = [&](const char * n, int64_t a, int64_t b_, int64_t d) { return load(c, m, b + n, a, b_, d, err); };
    ggml_tensor * router = W("ffn_gate_inp.weight", D, NE, 0);
    ggml_tensor * bias = W("exp_probs_b.bias", NE, 1, 0);
    ggml_tensor * rdown = W("ffn_routed_down.weight", D, LAT, 0);
    ggml_tensor * rup = W("ffn_routed_up.weight", LAT, D, 0);
    ggml_tensor * gate_e = W("ffn_gate_exps.weight", FF, LAT, NE);
    ggml_tensor * up_e = W("ffn_up_exps.weight", FF, LAT, NE);
    ggml_tensor * down_e = W("ffn_down_exps.weight", LAT, FF, NE);
    ggml_tensor * sgate = W("ffn_gate_shexp.weight", D, FF, 0);
    ggml_tensor * sup = W("ffn_up_shexp.weight", D, FF, 0);
    ggml_tensor * sdown = W("ffn_down_shexp.weight", FF, D, 0);
    if (err.size()) { ggml_free(c); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(c, 512, false);

    ggml_tensor * logits = ggml_mul_mat(c, router, tx);                       // [NE, nt]
    ggml_tensor * probs = ggml_sigmoid(c, logits);                            // gating func 2
    ggml_tensor * selection = ggml_add(c, probs, bias);                       // [NE, nt]
    ggml_tensor * ids = ggml_argsort_top_k(c, selection, USED);               // [USED, nt] i32

    ggml_tensor * weights = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, NE, nt), ids); // [1, USED, nt]
    ggml_tensor * wsum = ggml_sum_rows(c, ggml_reshape_2d(c, weights, USED, nt));        // [1, nt]
    wsum = ggml_clamp(c, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(c, ggml_reshape_2d(c, weights, USED, nt), wsum);
    weights = ggml_scale(c, weights, scale);
    weights = ggml_reshape_3d(c, weights, 1, USED, nt);

    ggml_tensor * routed = ggml_mul_mat(c, rdown, tx);                        // [LAT, nt]
    routed = ggml_repeat_4d(c, ggml_reshape_3d(c, routed, LAT, 1, nt), LAT, USED, nt, 1);

    ggml_tensor * g = ggml_mul_mat_id(c, gate_e, routed, ids);                // [FF, USED, nt]
    ggml_tensor * u = ggml_mul_mat_id(c, up_e, routed, ids);
    ggml_tensor * h = ggml_mul(c, ggml_silu(c, g), u);
    ggml_tensor * d = ggml_mul_mat_id(c, down_e, h, ids);                     // [LAT, USED, nt]
    d = ggml_mul(c, d, weights);
    ggml_tensor * summed = nullptr;
    for (int64_t i = 0; i < USED; ++i) {
        ggml_tensor * slot = ggml_view_3d(c, d, LAT, 1, nt, d->nb[1], d->nb[2], (size_t) i * d->nb[1]);
        summed = summed ? ggml_add(c, summed, slot) : slot;
    }
    ggml_tensor * sum2 = ggml_reshape_2d(c, summed, LAT, nt);
    ggml_tensor * moe = ggml_mul_mat(c, rup, sum2);        // [D, nt]

    ggml_tensor * sg = ggml_mul_mat(c, sgate, tx);
    ggml_tensor * su = ggml_mul_mat(c, sup, tx);
    ggml_tensor * sh = ggml_mul(c, ggml_silu(c, sg), su);
    ggml_tensor * shexp = ggml_mul_mat(c, sdown, sh);

    ggml_tensor * out_t = ggml_add(c, moe, shexp);
    ggml_set_output(out_t);
    ggml_build_forward_expand(gf, out_t);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(c); err = "no CPU backend"; return false; }
    bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_free(cpu);
    if (!ok) { ggml_free(c); err = "graph compute failed"; return false; }
    out.assign((size_t) D * nt, 0.0f);
    std::memcpy(out.data(), ggml_get_data(out_t), out.size() * 4);
    ggml_free(c);
    return true;
}

}  // namespace almond::hy4

namespace almond::hy4 {

bool hy4_dense_ffn_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer, int64_t nt,
                       const float * x, std::vector<float> & out, std::string & err) {
    const std::string b = "blk." + std::to_string(layer) + ".";
    const int64_t D = G.n_embd, FF = G.n_ff;
    ggml_init_params ip = { 256u * 1024u * 1024u, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tx = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, nt);
    std::memcpy(ggml_get_data(tx), x, (size_t) D * nt * 4);
    auto W = [&](const char * n, int64_t a, int64_t b_) { return load(c, m, b + n, a, b_, 0, err); };
    ggml_tensor * wg = W("ffn_gate.weight", D, FF), * wu = W("ffn_up.weight", D, FF);
    ggml_tensor * wd = W("ffn_down.weight", FF, D);
    if (err.size()) { ggml_free(c); return false; }
    ggml_cgraph * gf = ggml_new_graph_custom(c, 64, false);
    ggml_tensor * g = ggml_mul_mat(c, wg, tx), * u = ggml_mul_mat(c, wu, tx);
    ggml_tensor * y = ggml_mul_mat(c, wd, ggml_mul(c, ggml_silu(c, g), u));
    ggml_set_output(y); ggml_build_forward_expand(gf, y);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(c); err = "no CPU backend"; return false; }
    bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_free(cpu);
    if (!ok) { ggml_free(c); err = "dense FFN failed"; return false; }
    out.assign((size_t) D * nt, 0.0f);
    std::memcpy(out.data(), ggml_get_data(y), out.size() * 4);
    ggml_free(c);
    return true;
}

}  // namespace almond::hy4

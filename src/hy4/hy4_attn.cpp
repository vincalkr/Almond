// src/hy4/hy4_attn.cpp - gated MLA attention, ported from PR #32517 (build_attention), single-key case.
#include "hy4_attn.hpp"

#include "hy4_weights.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>

namespace almond::hy4 {
namespace {

/// The weight as a tensor over the mapped file (hy4_weights.hpp); `c` is unused and kept for the call sites.
ggml_tensor * load(ggml_context * c, const GgufModel & m, const std::string & name, int64_t ne0, int64_t ne1,
                   int64_t ne2, std::string & err) {
    (void) c;
    return hy4_weight(m, name, ne0, ne1, ne2, err);
}

}  // namespace

bool hy4_mla_attn_one_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer,
                          const float * x, std::vector<float> & out, std::string & err) {
    const std::string b = "blk." + std::to_string(layer) + ".";
    const int64_t D = G.n_embd, RQ = G.q_lora, KR = G.kv_lora, R = G.d_rope;
    const int64_t heads = G.head_count[(size_t) layer];
    const int64_t nope = G.d_key_mla - R;          // 192
    const int64_t VD = G.d_val_mla;                // 256
    const int64_t kdim = G.d_key_mla;              // 256 = 192 + 64 (latent q head)

    // A context large enough for the weight tensors themselves: the gate and output are Q8_0 (32 MB together).
    ggml_init_params ip = { 96u * 1024u * 1024u, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tx = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 1);
    std::memcpy(ggml_get_data(tx), x, (size_t) D * 4);

    ggml_tensor * wqa = load(c, m, b + "attn_q_a.weight", D, RQ, 0, err);
    ggml_tensor * qan = load(c, m, b + "attn_q_a_norm.weight", RQ, 1, 0, err);
    ggml_tensor * wqb = load(c, m, b + "attn_q_b.weight", RQ, heads * kdim, 0, err);
    ggml_tensor * wkv = load(c, m, b + "attn_kv_a_mqa.weight", D, KR + R, 0, err);
    ggml_tensor * kvn = load(c, m, b + "attn_kv_a_norm.weight", KR, 1, 0, err);
    ggml_tensor * wkb = load(c, m, b + "attn_k_b.weight", nope, KR, heads, err);
    ggml_tensor * wvb = load(c, m, b + "attn_v_b.weight", KR, VD, heads, err);
    ggml_tensor * gate = load(c, m, b + "attn_gate.weight", D, heads * VD, 0, err);
    ggml_tensor * wo = load(c, m, b + "attn_output.weight", heads * VD, D, 0, err);
    ggml_tensor * post = load(c, m, b + "post_attention_norm.weight", D, 1, 0, err);
    if (err.size()) { ggml_free(c); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(c, 256, false);
    const float norm_eps = (float) G.rms_eps;

    ggml_tensor * qr = ggml_mul_mat(c, wqa, tx);
    qr = ggml_rms_norm(c, qr, norm_eps);
    qr = ggml_mul(c, qr, qan);
    ggml_tensor * q = ggml_mul_mat(c, wqb, qr);                    // [kdim, heads, 1]

    // Split the head into nope/rope parts, absorb the nope part with wk_b into the latent KV space.
    ggml_tensor * q_nope = ggml_view_3d(c, q, nope, heads, 1, (size_t) kdim * ggml_type_size(GGML_TYPE_F32),
                                        (size_t) kdim * heads * 4, 0);
    ggml_tensor * q_pe = ggml_view_3d(c, q, R, heads, 1, (size_t) kdim * 4,
                                      (size_t) kdim * heads * 4, (size_t) nope * 4);
    q_nope = ggml_permute(c, q_nope, 0, 2, 1, 3);
    q_nope = ggml_mul_mat(c, wkb, q_nope);                         // [KR, heads, 1]
    q_nope = ggml_permute(c, q_nope, 0, 2, 1, 3);                  // [KR, heads, 1] after permute back
    ggml_tensor * Q = ggml_concat(c, q_nope, q_pe, 0);            // [KR + R, heads, 1]

    ggml_tensor * kv = ggml_mul_mat(c, wkv, tx);                  // [KR + R, 1]
    ggml_tensor * kv_lat = ggml_view_2d(c, kv, KR, 1, (size_t) (KR + R) * 4, 0);
    ggml_tensor * k_pe = ggml_view_2d(c, kv, R, 1, (size_t) (KR + R) * 4, (size_t) KR * 4);
    kv_lat = ggml_rms_norm(c, kv_lat, norm_eps);
    kv_lat = ggml_mul(c, kv_lat, kvn);
    (void) k_pe;  // with one key, q_pe·k_pe is rotation-invariant; the full block will apply RoPE to both

    // One key: softmax is 1, so the attention output is simply the latent value, broadcast to every head.
    ggml_tensor * A = kv_lat;                                     // [KR, 1]
    A = ggml_reshape_3d(c, A, KR, 1, 1);
    A = ggml_repeat_4d(c, A, KR, 1, heads, 1);                    // [KR, 1, heads] for the per-head wv_b
    A = ggml_mul_mat(c, wvb, A);                                  // [VD, 1, heads]
    A = ggml_permute(c, A, 0, 2, 1, 3);                           // [VD, heads, 1]
    A = ggml_cont(c, A);
    A = ggml_reshape_2d(c, A, heads * VD, 1);

    ggml_tensor * g = ggml_sigmoid(c, ggml_mul_mat(c, gate, tx));
    A = ggml_mul(c, A, g);
    ggml_tensor * y = ggml_mul_mat(c, wo, A);
    y = ggml_rms_norm(c, y, norm_eps);
    y = ggml_mul(c, y, post);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(c); err = "no CPU backend"; return false; }
    bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_free(cpu);
    if (!ok) { ggml_free(c); err = "graph compute failed"; return false; }
    out.assign((size_t) D, 0.0f);
    std::memcpy(out.data(), ggml_get_data(y), (size_t) D * 4);
    ggml_free(c);
    (void) Q;
    return true;
}


bool hy4_mla_attn_multi_cpu(const GgufModel & m, const Hy4Geometry & G, int64_t layer, int64_t nt,
                            const float * x, const int32_t * pos, std::vector<float> & out, std::string & err,
                            std::vector<float> * kcache) {
    const std::string b = "blk." + std::to_string(layer) + ".";
    const int64_t D = G.n_embd, RQ = G.q_lora, KR = G.kv_lora, R = G.d_rope;
    const int64_t H = G.head_count[(size_t) layer];
    const int64_t qhead = G.d_key_mla;
    const int64_t khead = KR + R;
    const int64_t nope = qhead - R;
    const int64_t VD = G.d_val_mla;
    const float base = G.swa_pattern[(size_t) layer] ? (float) G.rope_base_swa : (float) G.rope_base;

    ggml_init_params ip = { 256u * 1024u * 1024u, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tx = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, nt);
    ggml_tensor * tpos = ggml_new_tensor_1d(c, GGML_TYPE_I32, nt);
    std::memcpy(ggml_get_data(tx), x, (size_t) D * nt * 4);
    std::memcpy(ggml_get_data(tpos), pos, (size_t) nt * 4);
    auto W = [&](const char * n, int64_t a, int64_t b_, int64_t d) { return load(c, m, b + n, a, b_, d, err); };
    ggml_tensor * wqa = W("attn_q_a.weight", D, RQ, 0), * qan = W("attn_q_a_norm.weight", RQ, 1, 0);
    ggml_tensor * wqb = W("attn_q_b.weight", RQ, H * qhead, 0);
    ggml_tensor * wkv = W("attn_kv_a_mqa.weight", D, KR + R, 0), * kvn = W("attn_kv_a_norm.weight", KR, 1, 0);
    ggml_tensor * wkb = W("attn_k_b.weight", nope, KR, H), * wvb = W("attn_v_b.weight", KR, VD, H);
    ggml_tensor * gate = W("attn_gate.weight", D, H * VD, 0), * wo = W("attn_output.weight", H * VD, D, 0);
    ggml_tensor * sinks = W("attn_sinks.weight", H, 1, 0), * post = W("post_attention_norm.weight", D, 1, 0);
    if (err.size()) { ggml_free(c); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(c, 512, false);
    const float eps = (float) G.rms_eps;

    ggml_tensor * qr0 = ggml_mul_mat(c, wqa, tx);
    ggml_tensor * qr = ggml_rms_norm(c, qr0, eps);
    qr = ggml_mul(c, qr, qan);
    ggml_tensor * q = ggml_mul_mat(c, wqb, qr);                             // [kdim, H, nt]
    ggml_tensor * qn = ggml_view_3d(c, q, nope, H, nt, (size_t) qhead * 4, (size_t) qhead * H * 4, 0);
    ggml_tensor * qp = ggml_view_3d(c, q, R, H, nt, (size_t) qhead * 4, (size_t) qhead * H * 4, (size_t) nope * 4);
    // PR: hy_v4_flash uses NORM RoPE (consecutive pairs), base 1e7 on sparse layers.
    qp = ggml_rope_ext(c, qp, tpos, nullptr, R, 0, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    qp = ggml_permute(c, qp, 0, 2, 1, 3);                                   // [R, nt, H]
    qn = ggml_permute(c, qn, 0, 2, 1, 3);                                   // [nope, nt, H]
    qn = ggml_mul_mat(c, wkb, qn);                                          // [KR, nt, H]
    ggml_tensor * Q = ggml_concat(c, qn, qp, 0);                            // [khead, nt, H]
    Q = ggml_reshape_4d(c, ggml_cont(c, Q), khead, nt, H, 1);

    ggml_tensor * kv = ggml_mul_mat(c, wkv, tx);                            // [KR + R, nt]
    ggml_tensor * lat = ggml_view_2d(c, kv, KR, nt, (size_t) (KR + R) * 4, 0);
    ggml_tensor * kp = ggml_cont(c, ggml_view_2d(c, kv, R, nt, (size_t) (KR + R) * 4, (size_t) KR * 4));
    lat = ggml_cont(c, ggml_mul(c, ggml_rms_norm(c, lat, eps), kvn));
    kp = ggml_rope_ext(c, ggml_reshape_3d(c, kp, R, 1, nt), tpos, nullptr, R, 0, 0, base,
                       1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    kp = ggml_cont(c, ggml_permute(c, kp, 0, 2, 1, 3));                     // [R, nt, 1]
    ggml_tensor * Knew = ggml_concat(c, ggml_reshape_4d(c, lat, KR, nt, 1, 1),
                                     ggml_reshape_4d(c, kp, R, nt, 1, 1), 0); // [khead, nt, 1, 1]
    ggml_set_output(Knew);
    ggml_build_forward_expand(gf, Knew);
    const int64_t n_past = kcache ? (int64_t) (kcache->size() / (size_t) khead) : 0;
    const int64_t n_kv = n_past + nt;
    ggml_tensor * K = Knew;
    if (n_past > 0) {
        ggml_tensor * Kc = ggml_new_tensor_4d(c, GGML_TYPE_F32, khead, n_past, 1, 1);
        std::memcpy(ggml_get_data(Kc), kcache->data(), (size_t) khead * n_past * 4);
        K = ggml_concat(c, Kc, Knew, 1);                                    // [khead, n_kv, 1, 1]
    }
    // V is the normed latent, the first KR values of each key row.
    ggml_tensor * V = ggml_view_4d(c, K, KR, n_kv, 1, 1, K->nb[1], K->nb[2], K->nb[3], 0);
    V = ggml_cast(c, ggml_cont(c, V), GGML_TYPE_F16);
    K = ggml_cast(c, K, GGML_TYPE_F16);

    // Causal over all n_kv keys; a sliding-window layer also hides keys `swa` or more positions back
    // (llama.cpp's LLAMA_SWA_TYPE_STANDARD, which the PR uses).
    const bool swa = G.swa_pattern[(size_t) layer] != 0 && G.swa > 0;
    ggml_tensor * mask_f = ggml_new_tensor_4d(c, GGML_TYPE_F32, n_kv, nt, 1, 1);
    float * mp = (float *) ggml_get_data(mask_f);
    for (int64_t t = 0; t < nt; ++t) {
        const int64_t pt = n_past + t;
        for (int64_t s = 0; s < n_kv; ++s)
            mp[t * n_kv + s] = s <= pt && (!swa || pt - s < G.swa) ? 0.0f : -INFINITY;
    }

    ggml_tensor * mask = ggml_cast(c, mask_f, GGML_TYPE_F16);
    ggml_tensor * attn = ggml_flash_attn_ext(c, Q, K, V, mask, 1.0f / sqrtf((float) qhead), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(attn, sinks);
    ggml_prec_set_acc(attn, GGML_PREC_F32);

    // wv_b: [KR, VD, H] applied per head to the latent result [KR, H, nt].
    ggml_tensor * A = ggml_reshape_3d(c, ggml_cont(c, attn), KR, H, nt);
    A = ggml_mul_mat(c, wvb, ggml_permute(c, A, 0, 2, 1, 3));              // [VD, nt, H]
    A = ggml_permute(c, A, 0, 2, 1, 3);                                    // [VD, H, nt]
    A = ggml_reshape_2d(c, ggml_cont(c, A), H * VD, nt);

    ggml_tensor * g = ggml_sigmoid(c, ggml_mul_mat(c, gate, tx));
    A = ggml_mul(c, A, g);
    ggml_tensor * y = ggml_mul_mat(c, wo, A);
    y = ggml_mul(c, ggml_rms_norm(c, y, eps), post);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(c); err = "no CPU backend"; return false; }
    bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_free(cpu);
    if (!ok) { ggml_free(c); err = "graph compute failed"; return false; }
    out.assign((size_t) D * nt, 0.0f);
    std::memcpy(out.data(), ggml_get_data(y), out.size() * 4);
    if (kcache) {
        const float * kn = (const float *) ggml_get_data(Knew);
        kcache->insert(kcache->end(), kn, kn + (size_t) khead * nt);
    }
    ggml_free(c);
    return true;
}

}  // namespace almond::hy4

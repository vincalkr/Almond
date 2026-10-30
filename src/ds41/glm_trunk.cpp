// src/ds41/glm_trunk.cpp - see glm_trunk.hpp.  Function and line references are to llama.cpp's
// src/models/glm5-next.cpp (build_hc_pre / build_hc_post / build_kda_layer / build_kpool_select / build_dsa_layer),
// src/llama-graph.cpp (build_moe_ffn, build_attn_mha) and src/llama-memory-hybrid-idx.cpp (set_input_kpool).
#include "glm_trunk.hpp"

#include "almond/artifact/gguf_reader.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <functional>
#include <map>
#include <tuple>
#include <vector>

using namespace almond;

namespace {

constexpr float NEG_INF = -INFINITY;

// the KV-key prefix: llama.cpp writes "glm5-next", project-maya's packs "glm5next"; tensor names are the same
const MetaValue * key(const GgufFile & f, const std::string & arch, const char * k) {
    return f.get(arch + "." + k);
}
double num(const GgufFile & f, const std::string & arch, const char * k, double dflt) {
    const MetaValue * v = key(f, arch, k);
    if (!v) return dflt;
    if (v->type == MetaType::ARRAY) return v->items.empty() ? dflt : v->items[0].num();
    return v->num();
}
std::vector<double> arr(const GgufFile & f, const std::string & arch, const char * k, size_t n, double dflt) {
    std::vector<double> out(n, dflt);
    const MetaValue * v = key(f, arch, k);
    if (!v) return out;
    if (v->type != MetaType::ARRAY) { std::fill(out.begin(), out.end(), v->num()); return out; }
    for (size_t i = 0; i < n && i < v->items.size(); ++i) {
        const MetaValue & it = v->items[i];
        // the GGUF reader keeps integers unsigned: an I32 array of 0 / 1 reads back as is
        out[i] = it.num();
    }
    return out;
}

}  // namespace

struct GlmTrunk::Impl {
    TrunkInfo info;
    TrunkConfig cfg;
    std::string err;
    std::unique_ptr<GgufModel> model;
    ggml_backend_t backend = nullptr;
    bool own_backend = false;

    // ---- geometry (glm5-next.cpp load_arch_hparams)
    std::string arch;
    int64_t D = 0, NL = 0, HC = 4, NH = 0, HD = 0, DCONV = 0, DIN = 0, QL = 0, KVL = 0, DKM = 0, DVM = 0;
    int64_t IH = 0, ID = 0, ITOP = 0, KP = 0, NE = 0, NU = 0, NDENSE = 0, NV = 0, SINK = 20;
    float eps_rms = 1e-5f, eps_norm = 1e-6f, hc_eps = 1e-6f, gate_lb = -INFINITY, w_scale = 1.0f;
    bool w_norm = true, sel_tail = false;
    std::vector<uint8_t> recr, idx_full;
    std::vector<float> clamp_exp, clamp_shexp;

    // ---- weights (non-routed, on the backend) + the token embedding on the mmap
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor *> w;
    const uint8_t * embd = nullptr;
    uint32_t embd_type = 0;
    size_t embd_row = 0;

    // ---- persistent state
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    int64_t NC = 1, CTX = 0;
    ggml_tensor * x_state = nullptr;   // [D, HC, NC]
    ggml_tensor * hres = nullptr;      // [D, HC, NC] the FFN half's residual (hc_post of the attention)
    ggml_tensor * post_f = nullptr;    // [HC, NC]
    ggml_tensor * comb_f = nullptr;    // [HC, HC, NC]
    ggml_tensor * fout = nullptr;      // [D, NC] shared expert (or the dense FFN) of the current layer
    ggml_tensor * routed = nullptr;    // [D, NC] the routed experts' sum, written by the tier on the card (routed_dev)
    ggml_tensor * sel_buf = nullptr;   // [CTX * NC] the last full indexer's selection mask, [n_kv, n] contiguous
    struct Layer {
        ggml_tensor * conv = nullptr;  // KDA: [3 * (DCONV-1) * DIN] the q | k | v conv windows
        ggml_tensor * ssm = nullptr;   // KDA: [HD, HD, NH] the delta-rule state
        ggml_tensor * mla = nullptr;   // MLA: [KVL, 1, CTX] F16, the latent per cell
        ggml_tensor * idx = nullptr;   // MLA full indexer: [3 * ID, CTX] F16, key | gate | pooled per cell
    };
    std::vector<Layer> ly;

    // ---- the pass
    int pos0 = 0, n = 0;
    std::vector<float> host_emb, host_fn, host_wts, host_x, host_routed;
    // the cells the pass's attention reads: pos0 + n, rounded up to a multiple of kKvStep for a one-token pass so
    // its cached graphs are rebuilt once every kKvStep tokens (the extra cells are masked: never selected)
    static constexpr int64_t kKvStep = 256;
    int64_t n_kv = 0;
    std::vector<int32_t> host_ids;
    // k-pool inputs of the pass (set_input_kpool for one sequence: cell = position, pools of KP consecutive cells)
    int64_t n_pool = 0, n_new = 0, n_top = 0, n_sel = 0;
    std::vector<int32_t> pool_cells, pool_idxs, tail_idxs, new_pool_idxs;
    std::vector<int64_t> new_pool_rep, k_idxs;
    std::vector<float> pool_mask, sel_mask;

    ggml_gallocr_t galloc = nullptr;

    ~Impl() {
        cache.clear();
        if (galloc) ggml_gallocr_free(galloc);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (sctx) ggml_free(sctx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (own_backend && backend) ggml_backend_free(backend);
    }

    ggml_tensor * W(const std::string & name) const {
        const auto it = w.find(name);
        return it == w.end() ? nullptr : it->second;
    }
    ggml_tensor * BL(int il, const char * s) const { return W("blk." + std::to_string(il) + "." + s); }

    // ---------------------------------------------------------------- graph helpers
    // A graph and the host buffers its inputs are read from at every run (Impl members: their contents change per
    // pass, the objects do not).  A one-token pass's graphs are kept (`cache`, each with its own allocator) and run
    // again; a longer pass builds its graphs per call on the shared allocator.
    struct G {
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t ga = nullptr;
        std::vector<std::pair<ggml_tensor *, std::function<const void *()>>> inputs;
        std::vector<ggml_tensor *> out;
        ~G() { if (ga) ggml_gallocr_free(ga); if (ctx) ggml_free(ctx); }
    };
    enum Kind { K_BEGIN, K_AR, K_FIN, K_LOGITS };
    std::map<std::tuple<int, int, int64_t>, std::unique_ptr<G>> cache;
    // the graph for (kind, il): the cached one for a one-token pass (built by `build` the first time), else a new one
    G * graph(int kind, int il, int64_t kv, std::unique_ptr<G> & tmp, const std::function<void(G &)> & build) {
        if (n == 1) {
            std::unique_ptr<G> & slot = cache[{ kind, il, kv }];
            if (!slot) {
                slot = std::make_unique<G>();
                open(*slot);
                build(*slot);
            }
            return slot.get();
        }
        tmp = std::make_unique<G>();
        open(*tmp);
        build(*tmp);
        return tmp.get();
    }
    void open(G & g) {
        ggml_init_params ip = { ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), nullptr, true };
        g.ctx = ggml_init(ip);
        g.gf = ggml_new_graph_custom(g.ctx, 16384, false);
    }
    template <class T> ggml_tensor * input(G & g, ggml_type t, std::initializer_list<int64_t> ne, const std::vector<T> & v) {
        int64_t d[4] = { 1, 1, 1, 1 };
        int k = 0;
        for (int64_t x : ne) d[k++] = x;
        ggml_tensor * x = ggml_new_tensor(g.ctx, t, k, d);
        ggml_set_input(x);
        const std::vector<T> * src = &v;
        g.inputs.push_back({ x, [src]() { return (const void *) src->data(); } });
        return x;
    }
    static double pnow() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    int prof_tag = 3;
    const bool hc_composite = std::getenv("ALMOND_GLM_HC_COMPOSITE") != nullptr;
    double pw[4] = {}, pc[4] = {};
    int64_t pn[4] = {}, pcalls = 0;
    int pnodes[4] = {};
    bool run(G & g, bool kept) {
        if (kept) {   // allocated once, in its own buffer: nothing else runs there
            if (!g.ga) {
                g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                if (!ggml_gallocr_alloc_graph(g.ga, g.gf)) { err = "glm: graph allocation failed"; return false; }
            }
        } else {
            if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(galloc, g.gf)) { err = "glm: graph allocation failed"; return false; }
        }
        // ALMOND_GLM_PROF=1 (diagnostic): the time waiting for earlier work, then this graph's, per tag
        static const bool prof = std::getenv("ALMOND_GLM_PROF") != nullptr;
        double t0 = 0, t1 = 0;
        if (prof) { t0 = pnow(); ggml_backend_synchronize(backend); t1 = pnow(); }
        for (auto & in : g.inputs) ggml_backend_tensor_set(in.first, in.second(), 0, ggml_nbytes(in.first));
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) { err = "glm: graph compute failed"; return false; }
        if (prof) {
            ggml_backend_synchronize(backend);
            const double t2 = pnow();
            static int dumped[4] = {};
            if (std::getenv("ALMOND_GLM_PROF_OPS") && !dumped[prof_tag]++) {
                std::fprintf(stderr, "glmops tag %d:", prof_tag);
                for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
                    ggml_tensor * t = ggml_graph_node(g.gf, i);
                    std::fprintf(stderr, " %s%s", ggml_op_desc(t), t->op == GGML_OP_MUL_MAT ? (std::string("/") + ggml_type_name(t->src[0]->type)).c_str() : "");
                }
                std::fprintf(stderr, "\n");
            }
            pw[prof_tag] += t1 - t0; pc[prof_tag] += t2 - t1; ++pn[prof_tag]; pnodes[prof_tag] = ggml_graph_n_nodes(g.gf);
            if (++pcalls % (45 * 2 * 64) == 0)
                for (int i = 0; i < 4; ++i)
                    std::fprintf(stderr, "glmprof %s: %lld calls, wait %.3f ms, graph %.3f ms per call (%d nodes)\n",
                                 i == 0 ? "kda" : i == 1 ? "mla" : i == 2 ? "finish" : "other", (long long) pn[i],
                                 pw[i] / std::max<int64_t>(pn[i], 1), pc[i] / std::max<int64_t>(pn[i], 1), pnodes[i]);
        }
        prof_tag = 3;
        return true;
    }

    ggml_tensor * rms(ggml_context * c, ggml_tensor * x, ggml_tensor * wt) const {
        x = ggml_rms_norm(c, x, eps_rms);
        return wt ? ggml_mul(c, x, wt) : x;
    }
    // views of the persistent state's first n columns
    ggml_tensor * cols3(ggml_context * c, ggml_tensor * t) const {   // [D, HC, n]
        return ggml_view_3d(c, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], 0);
    }
    ggml_tensor * cols2(ggml_context * c, ggml_tensor * t) const {   // [ne0, n] (or [HC, HC, n] for comb)
        // by the tensor, not ggml_n_dims: with one column (--prefill-chunk 1) comb_f [HC, HC, 1] counts as 2-D
        if (t == comb_f) return ggml_view_3d(c, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], 0);
        return ggml_view_2d(c, t, t->ne[0], n, t->nb[1], 0);
    }

    // build_hc_pre (weights) + the mixes: {the pre-mixed stream [D, n], post [HC, n], comb [HC, HC, n]}
    struct HcPre { ggml_tensor * cur, * post, * comb; };
    HcPre hc_pre(ggml_context * c, ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base) const {
        const int64_t nt = x->ne[2];
        ggml_tensor * flat = ggml_reshape_2d(c, ggml_cont(c, x), HC * D, nt);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, ggml_rms_norm(c, flat, eps_rms));   // [(2+HC)*HC, nt]
        if (!hc_composite) {
            // the fused DeepSeek-V4 HC ops (llama.cpp's fused_dsv4_hc_* path of glm5-next): the Sinkhorn alone is ~130
            // tiny kernels per call as stock ops.  ALMOND_GLM_HC_COMPOSITE=1: the stock-op composite below.
            ggml_tensor * pre, * post;
#if defined(ALMOND_DSV41_HC_MIX)
            ggml_tensor * mix = ggml_dsv41_hc_mixes(c, ggml_cont(c, ggml_view_2d(c, mixes, 2 * HC, nt, mixes->nb[1], 0)),
                                                    ggml_view_1d(c, scale, 2, 0), ggml_view_1d(c, base, 2 * HC, 0), hc_eps);
            pre = ggml_view_2d(c, mix, HC, nt, mix->nb[1], 0);
            post = ggml_view_2d(c, mix, HC, nt, mix->nb[1], (size_t) nt * mix->nb[1]);
#else
            auto w1 = [&](ggml_tensor * t, int64_t ne0, int64_t i0) { return ggml_view_1d(c, t, ne0, ggml_row_size(t->type, i0)); };
            auto w2 = [&](int64_t ne0, int64_t i0) { return ggml_view_2d(c, mixes, ne0, nt, mixes->nb[1], ggml_row_size(mixes->type, i0)); };
            pre = ggml_scale_bias(c, ggml_sigmoid(c, ggml_add(c, ggml_mul(c, w2(HC, 0), w1(scale, 1, 0)), w1(base, HC, 0))), 1.0f, hc_eps);
            post = ggml_scale(c, ggml_sigmoid(c, ggml_add(c, ggml_mul(c, w2(HC, HC), w1(scale, 1, 1)), w1(base, HC, HC))), 2.0f);
#endif
            ggml_tensor * comb = ggml_dsv4_hc_comb(c, mixes, scale, base, hc_eps, (int32_t) SINK);
            return { ggml_dsv4_hc_pre(c, x, ggml_cont(c, pre)), post, comb };
        }
        auto v1 = [&](ggml_tensor * t, int64_t ne0, int64_t i0) { return ggml_view_1d(c, t, ne0, ggml_row_size(t->type, i0)); };
        auto v2 = [&](int64_t ne0, int64_t i0) { return ggml_view_2d(c, mixes, ne0, nt, mixes->nb[1], ggml_row_size(mixes->type, i0)); };
        ggml_tensor * pre = ggml_add(c, ggml_mul(c, v2(HC, 0), v1(scale, 1, 0)), v1(base, HC, 0));
        pre = ggml_scale_bias(c, ggml_sigmoid(c, pre), 1.0f, hc_eps);
        ggml_tensor * post = ggml_add(c, ggml_mul(c, v2(HC, HC), v1(scale, 1, 1)), v1(base, HC, HC));
        post = ggml_scale(c, ggml_sigmoid(c, post), 2.0f);
        ggml_tensor * comb = ggml_add(c, ggml_mul(c, v2(HC * HC, 2 * HC), v1(scale, 1, 2)), v1(base, HC * HC, 2 * HC));
        comb = ggml_reshape_3d(c, ggml_cont(c, comb), HC, HC, nt);
        // build_hc_sinkhorn
        comb = ggml_soft_max(c, comb);
        ggml_tensor * eps = ggml_fill(c, ggml_new_tensor_1d(c, GGML_TYPE_F32, 1), hc_eps);
        comb = ggml_add(c, comb, eps);
        auto norm_cols = [&]() {
            ggml_tensor * t = ggml_cont(c, ggml_permute(c, comb, 1, 0, 2, 3));
            ggml_tensor * s = ggml_add(c, ggml_sum_rows(c, t), eps);
            comb = ggml_div(c, comb, ggml_permute(c, s, 1, 0, 2, 3));
        };
        auto norm_rows = [&]() { comb = ggml_div(c, comb, ggml_add(c, ggml_sum_rows(c, comb), eps)); };
        norm_cols();
        for (int64_t i = 1; i < SINK; ++i) { norm_rows(); norm_cols(); }
        // the weighted sum of the streams
        ggml_tensor * res = nullptr;
        for (int64_t ih = 0; ih < HC; ++ih) {
            ggml_tensor * xh = ggml_view_2d(c, x, D, nt, x->nb[2], ih * x->nb[1]);
            ggml_tensor * wh = ggml_view_2d(c, pre, 1, nt, pre->nb[1], ih * pre->nb[0]);
            ggml_tensor * t = ggml_mul(c, xh, wh);
            res = res ? ggml_add(c, res, t) : t;
        }
        return { res, post, comb };
    }
    // build_hc_post: [D, HC, n]
    ggml_tensor * hc_post(ggml_context * c, ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb) const {
        if (!hc_composite) return ggml_dsv4_hc_post(c, x, residual, ggml_cont(c, post), ggml_cont(c, comb));
        const int64_t nt = x->ne[1];
        ggml_tensor * out = nullptr;
        for (int64_t dst = 0; dst < HC; ++dst) {
            ggml_tensor * pd = ggml_view_2d(c, post, 1, nt, post->nb[1], dst * post->nb[0]);
            ggml_tensor * cur = ggml_mul(c, x, pd);
            for (int64_t src = 0; src < HC; ++src) {
                ggml_tensor * rs = ggml_view_2d(c, residual, D, nt, residual->nb[2], src * residual->nb[1]);
                ggml_tensor * cs = ggml_view_2d(c, comb, 1, nt, comb->nb[2], dst * comb->nb[0] + src * comb->nb[1]);
                cur = ggml_add(c, cur, ggml_mul(c, rs, cs));
            }
            cur = ggml_reshape_3d(c, cur, D, 1, nt);
            out = out ? ggml_concat(c, out, cur, 1) : cur;
        }
        return out;
    }
    // build_ffn (SiLU, parallel gate) with llama.cpp's swiglu clamp
    ggml_tensor * ffn(ggml_context * c, ggml_tensor * x, ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down, float limit) const {
        ggml_tensor * u = ggml_mul_mat(c, up, x);
        ggml_tensor * g = ggml_mul_mat(c, gate, x);
        ggml_tensor * h = limit > 1e-6f ? ggml_swiglu_clamp(c, g, u, limit) : ggml_mul(c, ggml_silu(c, g), u);
        return ggml_mul_mat(c, down, h);
    }

    // ---------------------------------------------------------------- KDA (build_kda_layer + glm5_conv1d)
    ggml_tensor * kda(G & g, ggml_tensor * cur, int il) {
        ggml_context * c = g.ctx;
        const Layer & L = ly[(size_t) il];
        const int64_t csz = (DCONV - 1) * DIN;
        auto conv1d = [&](int qkv, ggml_tensor * pw, ggml_tensor * cw) {
            ggml_tensor * st = ggml_view_3d(c, L.conv, DCONV - 1, DIN, 1, (DCONV - 1) * 4, csz * 4, (size_t) qkv * csz * 4);
            ggml_tensor * xp = ggml_reshape_3d(c, ggml_mul_mat(c, pw, cur), DIN, n, 1);
            ggml_tensor * cx = ggml_concat(c, st, ggml_transpose(c, xp), 0);           // [DCONV-1+n, DIN, 1]
            // the window after this pass: its last DCONV-1 columns
            ggml_tensor * last = ggml_view_3d(c, cx, DCONV - 1, DIN, 1, cx->nb[1], cx->nb[2], (size_t) n * cx->nb[0]);
            ggml_build_forward_expand(g.gf, ggml_cpy(c, last, st));
            ggml_tensor * y = ggml_ssm_conv(c, cx, ggml_reshape_2d(c, cw, DCONV, DIN));
            y = ggml_silu(c, ggml_reshape_2d(c, y, DIN, n));
            return ggml_reshape_4d(c, y, HD, NH, n, 1);
        };
        ggml_tensor * q = conv1d(0, BL(il, "attn_q.weight"), BL(il, "ssm_conv1d_q.weight"));
        ggml_tensor * k = conv1d(1, BL(il, "attn_k.weight"), BL(il, "ssm_conv1d_k.weight"));
        ggml_tensor * v = conv1d(2, BL(il, "attn_v.weight"), BL(il, "ssm_conv1d_v.weight"));
        // the decay gate (ssm_a holds -exp(A_log))
        ggml_tensor * g1 = ggml_add(c, ggml_mul_mat(c, BL(il, "ssm_f_b.weight"), ggml_mul_mat(c, BL(il, "ssm_f_a.weight"), cur)),
                                    BL(il, "ssm_dt.bias"));
        ggml_tensor * A = ggml_reshape_3d(c, BL(il, "ssm_a"), 1, NH, 1);
        if (gate_lb > -INFINITY) {
            g1 = ggml_mul(c, ggml_reshape_3d(c, g1, HD, NH, n), A);
            g1 = ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, g1, -1.0f)), gate_lb);
        } else {
            g1 = ggml_mul(c, ggml_reshape_3d(c, ggml_softplus(c, g1), HD, NH, n), A);
        }
        g1 = ggml_reshape_4d(c, g1, HD, NH, n, 1);
        ggml_tensor * beta = ggml_sigmoid(c, ggml_reshape_4d(c, ggml_mul_mat(c, BL(il, "ssm_beta.weight"), cur), 1, NH, n, 1));
        ggml_tensor * s = ggml_reshape_4d(c, L.ssm, HD, HD, NH, 1);
        auto l2 = [&](ggml_tensor * t) { return ggml_scale(c, ggml_rms_norm(c, t, 1e-6f / (float) HD), 1.0f / std::sqrt((float) HD)); };
        q = l2(q);
        k = l2(k);
        ggml_tensor * gd = ggml_gated_delta_net(c, q, k, v, g1, beta, s, 1);
        const int64_t na = HD * NH * n;
        ggml_tensor * out = ggml_view_4d(c, gd, HD, NH, n, 1, ggml_row_size(gd->type, HD), ggml_row_size(gd->type, HD * NH),
                                         ggml_row_size(gd->type, na), 0);
        ggml_tensor * ns = ggml_view_1d(c, gd, HD * HD * NH, ggml_row_size(gd->type, na));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, ns, ggml_reshape_1d(c, L.ssm, HD * HD * NH)));
        out = ggml_cont(c, out);
        // the output gate: RMSNorm(o) * sigmoid(g2)
        ggml_tensor * g2 = ggml_reshape_3d(c, ggml_mul_mat(c, BL(il, "ssm_g_b.weight"), ggml_mul_mat(c, BL(il, "ssm_g_a.weight"), cur)), HD, NH, n);
        ggml_tensor * o = rms(c, ggml_reshape_3d(c, out, HD, NH, n), BL(il, "ssm_norm.weight"));
        ggml_tensor * gated = ggml_cont_2d(c, ggml_mul(c, o, ggml_sigmoid(c, g2)), DIN, n);
        return ggml_mul_mat(c, BL(il, "attn_output.weight"), gated);
    }

    // ---------------------------------------------------------------- the k-pool indexer (build_kpool_select)
    // the selection mask [n_kv, n] (0 = attended, -inf = not)
    ggml_tensor * kpool_select(G & g, ggml_tensor * cur, ggml_tensor * qr, ggml_tensor * kid, int il) {
        ggml_context * c = g.ctx;
        const Layer & L = ly[(size_t) il];
        ggml_tensor * iq = ggml_reshape_3d(c, ggml_mul_mat(c, BL(il, "indexer.attn_q_b.weight"), qr), ID, IH, n);
        ggml_tensor * ik = ggml_mul_mat(c, BL(il, "indexer.attn_k.weight"), cur);
        ik = ggml_add(c, ggml_mul(c, ggml_norm(c, ik, eps_norm), BL(il, "indexer.k_norm.weight")), BL(il, "indexer.k_norm.bias"));
        ggml_tensor * ig = ggml_mul_mat(c, BL(il, "indexer_compressor_gate.weight"), cur);
        ggml_tensor * pz = ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, ID, n), 0.0f);
        ggml_tensor * packed = ggml_concat(c, ggml_concat(c, ik, ig, 0), pz, 0);   // [3*ID, n]
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, L.idx, packed, kid));
        ggml_tensor * key_gate = ggml_view_2d(c, L.idx, 2 * ID, CTX, L.idx->nb[1], 0);
        ggml_tensor * pooledv = ggml_view_2d(c, L.idx, ID, CTX, L.idx->nb[1], ggml_row_size(L.idx->type, 2 * ID));
        // pool the entries this pass completes (a dummy one when it completes none)
        ggml_tensor * rows = ggml_get_rows(c, key_gate, input(g, GGML_TYPE_I32, { KP * n_new }, new_pool_idxs));
        rows = ggml_reshape_3d(c, rows, 2 * ID, KP, n_new);
        ggml_tensor * pk = ggml_view_3d(c, rows, ID, KP, n_new, rows->nb[1], rows->nb[2], 0);
        ggml_tensor * pg = ggml_view_3d(c, rows, ID, KP, n_new, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, ID));
        ggml_tensor * lg = ggml_add(c, pg, BL(il, "indexer_compressor_ape.weight"));
        lg = ggml_cont(c, ggml_permute(c, lg, 1, 0, 2, 3));                                          // [KP, ID, n_new]
        ggml_tensor * pr = ggml_reshape_3d(c, ggml_soft_max(c, ggml_reshape_2d(c, lg, KP, ID * n_new)), KP, ID, n_new);
        pk = ggml_cont(c, ggml_permute(c, pk, 1, 0, 2, 3));
        ggml_tensor * pn = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_mul(c, pr, pk)), ID, n_new);
        ggml_tensor * scat = ggml_set_rows(c, pooledv, pn, input(g, GGML_TYPE_I64, { n_new }, new_pool_rep));
        ggml_build_forward_expand(g.gf, scat);
        ggml_tensor * pooled = ggml_get_rows(c, pooledv, input(g, GGML_TYPE_I32, { n_pool }, pool_cells));
        pooled = ggml_reshape_3d(c, pooled, ID, 1, n_pool);
        ggml_tensor * wts = ggml_scale(c, ggml_mul_mat(c, BL(il, "indexer.proj.weight"), cur), 1.0f / std::sqrt((float) (ID * IH)));
        ggml_tensor * qp = ggml_permute(c, iq, 0, 2, 1, 3);
        ggml_tensor * kp = ggml_permute(c, pooled, 0, 2, 1, 3);
        ggml_tensor * kq = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, kp, qp), 2, 1, 0, 3));         // [IH, n, n_pool]
        ggml_tensor * score = ggml_sum_rows(c, ggml_mul(c, ggml_relu(c, kq), wts));                   // [1, n, n_pool]
        score = ggml_cont(c, ggml_permute(c, score, 2, 1, 0, 3));                                     // [n_pool, n, 1]
        score = ggml_add(c, ggml_reshape_2d(c, score, n_pool, n), input(g, GGML_TYPE_F32, { n_pool, n }, pool_mask));
        ggml_tensor * top = ggml_top_k(c, score, (int) n_top);                                        // [n_top, n]
        ggml_tensor * ss = ggml_get_rows(c, ggml_reshape_3d(c, score, 1, n_pool, n), top);
        ggml_tensor * order = ggml_argsort(c, ggml_reshape_2d(c, ss, n_top, n), GGML_SORT_ORDER_DESC);
        top = ggml_get_rows(c, ggml_reshape_3d(c, ggml_cast(c, top, GGML_TYPE_F32), 1, n_top, n), order);
        top = ggml_cast(c, ggml_cont(c, ggml_reshape_2d(c, top, n_top, n)), GGML_TYPE_I32);
        ggml_tensor * pidx = input(g, GGML_TYPE_I32, { KP, n_pool }, pool_idxs);
        ggml_tensor * si = ggml_reshape_2d(c, ggml_get_rows(c, pidx, ggml_reshape_1d(c, top, n_top * n)), KP * n_top, n);
        if (sel_tail) si = ggml_concat(c, si, input(g, GGML_TYPE_I32, { KP - 1, n }, tail_idxs), 0);
        // the mask: every slot -inf, then the live slots' cells to 0 (dead slots write their own dump rows past n_kv)
        ggml_tensor * all = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, n_kv + n_sel, n), NEG_INF);
        ggml_tensor * zeros = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, n_sel, n), 0.0f);
        ggml_tensor * live = ggml_reshape_2d(c, ggml_exp(c, input(g, GGML_TYPE_F32, { n_sel, n }, sel_mask)), n_sel, n);
        ggml_tensor * dump = ggml_arange(c, (float) n_kv, (float) (n_kv + n_sel), 1.0f);
        ggml_tensor * idf = ggml_cast(c, si, GGML_TYPE_F32);
        idf = ggml_add(c, ggml_mul(c, ggml_sub(c, idf, dump), live), dump);
        ggml_tensor * sel = ggml_set_rows(c, all, zeros, ggml_reshape_3d(c, ggml_cast(c, idf, GGML_TYPE_I32), n_sel, n, 1));
        sel = ggml_cont(c, ggml_view_2d(c, sel, n_kv, n, sel->nb[2], 0));
        return sel;
    }

    // ---------------------------------------------------------------- nope MLA (build_dsa_layer + build_attn_mha)
    ggml_tensor * dsa(G & g, ggml_tensor * cur, int il) {
        ggml_context * c = g.ctx;
        const Layer & L = ly[(size_t) il];
        ggml_tensor * qr = rms(c, ggml_mul_mat(c, BL(il, "attn_q_a.weight"), cur), BL(il, "attn_q_a_norm.weight"));
        ggml_tensor * q = ggml_reshape_3d(c, ggml_mul_mat(c, BL(il, "attn_q_b.weight"), qr), DKM, NH, n);
        ggml_tensor * kv = rms(c, ggml_mul_mat(c, BL(il, "attn_kv_a_mqa.weight"), cur), BL(il, "attn_kv_a_norm.weight"));
        // the latent cache rows of this pass (by position: an input, so a kept graph serves every position)
        ggml_tensor * kid = input(g, GGML_TYPE_I64, { n }, k_idxs);
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, ggml_reshape_2d(c, L.mla, KVL, CTX), kv, kid));
        // absorb wk_b: the query in latent space [KVL, NH, n]
        ggml_tensor * qa = ggml_mul_mat(c, BL(il, "attn_k_b.weight"), ggml_permute(c, q, 0, 2, 1, 3));   // [KVL, n, NH]
        // the mask: this layer's own indexer, or the last full indexer's (sel_buf)
        ggml_tensor * selv = ggml_view_2d(c, sel_buf, n_kv, n, n_kv * 4, 0);
        ggml_tensor * mask;
        if (idx_full[(size_t) il]) {
            ggml_tensor * sel = kpool_select(g, cur, qr, kid, il);
            ggml_build_forward_expand(g.gf, ggml_cpy(c, sel, selv));
            mask = sel;
        } else {
            mask = selv;
        }
        ggml_tensor * k = ggml_view_2d(c, L.mla, KVL, n_kv, L.mla->nb[2], 0);                      // [KVL, n_kv]
        ggml_tensor * kq = ggml_mul_mat(c, k, qa);                                                     // [n_kv, n, NH]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(c, kq, mask, 1.0f / std::sqrt((float) DKM), 0.0f);
        ggml_tensor * vt = ggml_cont(c, ggml_transpose(c, ggml_cast(c, k, GGML_TYPE_F32)));            // [n_kv, KVL]
        ggml_tensor * kqv = ggml_mul_mat(c, vt, kq);                                                   // [KVL, n, NH]
        kqv = ggml_mul_mat(c, BL(il, "attn_v_b.weight"), kqv);                                         // [DVM, n, NH]
        ggml_tensor * o = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), DVM * NH, n);
        return ggml_mul_mat(c, BL(il, "attn_output.weight"), o);
    }

    // ---------------------------------------------------------------- the pass's k-pool inputs
    void kpool_inputs() {
        const int64_t complete = (pos0 + n) / KP;        // pools whose cells all exist after this pass
        // padded by one past the cells read (llama.cpp's always-present dummy); the padding pools are masked
        n_pool = std::max<int64_t>(n_kv / KP, 1) + 1;
        n_top = std::min<int64_t>(n_pool, ITOP / KP);
        n_sel = KP * n_top + (sel_tail ? KP - 1 : 0);
        const int32_t sentinel = (int32_t) n_kv;
        pool_cells.assign((size_t) n_pool, 0);
        pool_idxs.assign((size_t) (KP * n_pool), sentinel);
        for (int64_t j = 0; j < complete; ++j) {
            pool_cells[(size_t) j] = (int32_t) (j * KP + KP - 1);
            for (int64_t k = 0; k < KP; ++k) pool_idxs[(size_t) (j * KP + k)] = (int32_t) (j * KP + k);
        }
        for (int64_t j = complete; j < n_pool; ++j) pool_cells[(size_t) j] = pos0;   // a cell of this pass: finite rows
        // the pools this pass completes: their last cell lies in [pos0, pos0 + n)
        new_pool_idxs.clear();
        new_pool_rep.clear();
        for (int64_t j = pos0 / KP; j < complete; ++j) {
            const int64_t last = j * KP + KP - 1;
            if (last < pos0) continue;
            for (int64_t k = 0; k < KP; ++k) new_pool_idxs.push_back((int32_t) (j * KP + k));
            new_pool_rep.push_back(last);
        }
        if (new_pool_rep.empty()) {   // one dummy entry: a cell no complete pool's rep uses
            int64_t pad = 0;
            while (pad < complete * KP && (pad % KP) == KP - 1) ++pad;
            for (int64_t k = 0; k < KP; ++k) new_pool_idxs.push_back((int32_t) pad);
            new_pool_rep.push_back(pad);
        }
        n_new = (int64_t) new_pool_rep.size();
        pool_mask.assign((size_t) (n_pool * n), NEG_INF);
        sel_mask.assign((size_t) (n_sel * n), NEG_INF);
        tail_idxs.assign((size_t) ((KP - 1) * n), sentinel);
        k_idxs.resize((size_t) n);
        for (int i = 0; i < n; ++i) {
            const int64_t p = pos0 + i;
            k_idxs[(size_t) i] = p;
            const int64_t nv = (p + 1) / KP;             // pools ending at or before p
            for (int64_t j = 0; j < nv; ++j) pool_mask[(size_t) (i * n_pool + j)] = 0.0f;
            const int64_t nvc = std::min(nv, n_top);
            for (int64_t s = 0; s < nvc * KP; ++s) sel_mask[(size_t) (i * n_sel + s)] = 0.0f;
            const int64_t n_tail = (p + 1) % KP;
            for (int64_t k = 0; k < KP - 1; ++k) {
                const bool real = k < n_tail;
                tail_idxs[(size_t) (i * (KP - 1) + k)] = real ? (int32_t) (p - k) : sentinel;
                if (sel_tail) sel_mask[(size_t) (i * n_sel + n_top * KP + k)] = real ? 0.0f : NEG_INF;
            }
        }
    }
};

GlmTrunk::GlmTrunk() : p_(new Impl) {}
GlmTrunk::~GlmTrunk() = default;

const TrunkInfo & GlmTrunk::info() const { return p_->info; }
ggml_backend_t GlmTrunk::backend() const { return p_->backend; }
const std::string & GlmTrunk::last_error() const { return p_->err; }

bool GlmTrunk::init(const std::string & path, const TrunkConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    try {
        im.model = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception & e) { err = e.what(); return false; }
    const GgufFile & f = im.model->meta();
    const MetaValue * a = f.get("general.architecture");
    im.arch = a ? a->s : "";
    if (im.arch != "glm5-next" && im.arch != "glm5next") { err = "glm: not a GLM-5.3 GGUF (" + im.arch + ")"; return false; }
    const std::string & A = im.arch;
    const int64_t nblk = (int64_t) num(f, A, "block_count", 0);
    const int64_t nextn = (int64_t) num(f, A, "nextn_predict_layers", 0);
    im.NL = nblk - nextn;
    im.D = (int64_t) num(f, A, "embedding_length", 0);
    im.NH = (int64_t) num(f, A, "attention.head_count", 0);
    im.HD = (int64_t) num(f, A, "kda.head_dim", 0);
    im.DCONV = (int64_t) num(f, A, "ssm.conv_kernel", 0);
    im.DIN = im.HD * im.NH;
    im.gate_lb = (float) num(f, A, "kda.gate_lower_bound", -INFINITY);
    im.QL = (int64_t) num(f, A, "attention.q_lora_rank", 0);
    im.KVL = (int64_t) num(f, A, "attention.kv_lora_rank", 0);
    im.DKM = (int64_t) num(f, A, "attention.key_length_mla", 0);
    im.DVM = (int64_t) num(f, A, "attention.value_length_mla", 0);
    im.IH = (int64_t) num(f, A, "attention.indexer.head_count", 0);
    im.ID = (int64_t) num(f, A, "attention.indexer.key_length", 0);
    im.ITOP = (int64_t) num(f, A, "attention.indexer.top_k", 0);
    im.KP = (int64_t) num(f, A, "attention.indexer.kpool", 0);
    im.sel_tail = num(f, A, "attention.indexer.kpool_select_tail", 0) != 0;
    im.HC = (int64_t) num(f, A, "hyper_connection.count", 4);
    im.SINK = (int64_t) num(f, A, "hyper_connection.sinkhorn_iterations", 20);
    im.hc_eps = (float) num(f, A, "hyper_connection.epsilon", 1e-6);
    im.eps_rms = (float) num(f, A, "attention.layer_norm_rms_epsilon", 1e-5);
    im.eps_norm = (float) num(f, A, "attention.layer_norm_epsilon", 1e-6);
    im.NE = (int64_t) num(f, A, "expert_count", 0);
    im.NU = (int64_t) num(f, A, "expert_used_count", 0);
    im.NDENSE = (int64_t) num(f, A, "leading_dense_block_count", 0);
    im.w_scale = (float) num(f, A, "expert_weights_scale", 1.0);
    im.w_norm = num(f, A, "expert_weights_norm", 1) != 0;
    if (num(f, A, "expert_group_count", 1) > 1) { err = "glm: expert groups are not supported"; return false; }
    if (num(f, A, "expert_gating_func", 2) != 2) { err = "glm: only the sigmoid router is supported"; return false; }
    const std::vector<double> hkv = arr(f, A, "attention.head_count_kv", (size_t) nblk, 0);
    const std::vector<double> itype = arr(f, A, "attention.indexer.types", (size_t) nblk, 1);
    const std::vector<double> ce = arr(f, A, "swiglu_clamp_exp", (size_t) nblk, 0);
    const std::vector<double> cs = key(f, A, "swiglu_clamp_shexp") ? arr(f, A, "swiglu_clamp_shexp", (size_t) nblk, 0) : ce;
    im.recr.resize((size_t) im.NL);
    im.idx_full.resize((size_t) im.NL);
    im.clamp_exp.resize((size_t) im.NL);
    im.clamp_shexp.resize((size_t) im.NL);
    for (int64_t l = 0; l < im.NL; ++l) {
        im.recr[(size_t) l] = hkv[(size_t) l] == 0;
        im.idx_full[(size_t) l] = itype[(size_t) l] != 0;
        im.clamp_exp[(size_t) l] = (float) ce[(size_t) l];
        im.clamp_shexp[(size_t) l] = (float) cs[(size_t) l];
    }
    if (im.HC != 4 || im.D <= 0 || im.NL <= 0 || im.KP < 2) { err = "glm: unexpected geometry"; return false; }

    // ---- the backend
    im.backend = cfg.backend;
    if (!im.backend) { im.backend = ggml_backend_cpu_init(); im.own_backend = true; ggml_backend_cpu_set_n_threads(im.backend, cfg.n_threads); }

    // ---- weights: every non-routed tensor of the trunk's layers + the head; the embedding stays on the mmap
    std::vector<std::pair<const TensorInfo *, size_t>> want;
    for (size_t s = 0; s < im.model->size(); ++s)
        for (const TensorInfo & t : im.model->shard(s).tensors()) {
            const std::string & nm = t.name;
            if (nm.find("_exps.") != std::string::npos) continue;
            if (nm == "token_embd.weight") continue;
            if (nm.rfind("blk.", 0) == 0 && std::atoll(nm.c_str() + 4) >= im.NL) continue;   // the MTP block
            want.push_back({ &t, s });
        }
    ggml_init_params wp = { ggml_tensor_overhead() * (want.size() + 8), nullptr, true };
    im.wctx = ggml_init(wp);
    for (auto & [t, s] : want) {
        int64_t ne[4] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < t->shape.size() && d < 4; ++d) ne[d] = (int64_t) t->shape[d];
        ggml_tensor * x = ggml_new_tensor(im.wctx, (ggml_type) t->type, (int) std::max<size_t>(1, t->shape.size()), ne);
        ggml_set_name(x, t->name.c_str());
        im.w[t->name] = x;
    }
    im.wbuf = ggml_backend_alloc_ctx_tensors(im.wctx, im.backend);
    if (!im.wbuf) { err = "glm: no memory for the trunk's weights"; return false; }
    size_t wbytes = 0;
    for (auto & [t, s] : want) {
        ggml_tensor * x = im.w[t->name];
        ggml_backend_tensor_set(x, im.model->shard(s).tensor_data(*t), 0, ggml_nbytes(x));
        wbytes += ggml_nbytes(x);
    }
    size_t es = 0;
    const TensorInfo * te = im.model->find("token_embd.weight", &es);
    if (!te) { err = "glm: no token_embd.weight"; return false; }
    im.embd = im.model->shard(es).tensor_data(*te);
    im.embd_type = te->type;
    im.embd_row = ggml_row_size((ggml_type) te->type, im.D);
    im.NV = (int64_t) te->shape[1];
    if (!ggml_get_type_traits((ggml_type) im.embd_type)->to_float) { err = "glm: token_embd type has no dequantizer"; return false; }
    std::fprintf(stderr, "glm: %s, %lld layers (%d KDA, %d MLA), %.2f GB of trunk weights on the backend\n", im.arch.c_str(),
                 (long long) im.NL, (int) std::count(im.recr.begin(), im.recr.end(), 1),
                 (int) std::count(im.recr.begin(), im.recr.end(), 0), wbytes / 1e9);

    // ---- state
    im.NC = std::max(1, cfg.max_pass);
    im.CTX = cfg.ctx_max;
    im.ly.resize((size_t) im.NL);
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (16 + 4 * im.NL), nullptr, true };
    im.sctx = ggml_init(sp);
    im.x_state = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, im.NC);
    im.hres = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, im.NC);
    im.post_f = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.HC, im.NC);
    im.comb_f = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.HC, im.HC, im.NC);
    im.fout = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, im.NC);
    im.routed = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, im.NC);
    im.sel_buf = ggml_new_tensor_1d(im.sctx, GGML_TYPE_F32, im.CTX * im.NC);
    for (int64_t l = 0; l < im.NL; ++l) {
        Impl::Layer & L = im.ly[(size_t) l];
        if (im.recr[(size_t) l]) {
            L.conv = ggml_new_tensor_1d(im.sctx, GGML_TYPE_F32, 3 * (im.DCONV - 1) * im.DIN);
            L.ssm = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.HD, im.HD, im.NH);
        } else {
            L.mla = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F16, im.KVL, 1, im.CTX);
            if (im.idx_full[(size_t) l]) L.idx = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F16, 3 * im.ID, im.CTX);
        }
    }
    im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
    if (!im.sbuf) { err = "glm: no memory for the state (lower --ctx)"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);
    std::fprintf(stderr, "glm: state %.2f GB (ctx %lld, %lld pass columns)\n", ggml_backend_buffer_get_size(im.sbuf) / 1e9,
                 (long long) im.CTX, (long long) im.NC);

    // ---- info
    TrunkInfo & i = im.info;
    i.arch = "glm53";
    i.n_embd = im.D;
    i.n_layer = im.NL;
    i.n_used = im.NU;
    i.vocab = im.NV;
    const MetaValue * eos = f.get("tokenizer.ggml.eos_token_id");
    i.eos = eos ? (int64_t) eos->u : -1;
    i.enc_cut_layer = -1;
    i.n_swa = 0;
    return true;
}

bool GlmTrunk::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC) { im.err = "glm: a pass of " + std::to_string(n) + " tokens (max " + std::to_string(im.NC) + ")"; return false; }
    if (pos0 + n > im.CTX) { im.err = "glm: past the context (" + std::to_string(im.CTX) + ")"; return false; }
    im.pos0 = pos0;
    im.n = n;
    // the embeddings, dequantized on the host, repeated into the HC streams
    const auto * tr = ggml_get_type_traits((ggml_type) im.embd_type);
    im.host_emb.resize((size_t) (im.D * n));
    for (int i = 0; i < n; ++i) {
        if (tids[i] < 0 || tids[i] >= im.NV) { im.err = "glm: token id out of range"; return false; }
        tr->to_float(im.embd + (size_t) tids[i] * im.embd_row, im.host_emb.data() + (size_t) i * im.D, im.D);
    }
    im.n_kv = n == 1 ? std::min<int64_t>(im.CTX, (pos0 + 1 + Impl::kKvStep - 1) / Impl::kKvStep * Impl::kKvStep) : pos0 + n;
    im.kpool_inputs();
    if (std::getenv("ALMOND_GLM_PROF_OPS")) std::fprintf(stderr, "glm begin pos0 %d n %d n_kv %lld n_new %lld\n", pos0, n, (long long) im.n_kv, (long long) im.n_new);
    std::unique_ptr<Impl::G> tmp;
    Impl::G * g = im.graph(Impl::K_BEGIN, 0, 0, tmp, [&](Impl::G & g) {
        ggml_context * c = g.ctx;
        ggml_tensor * e = im.input(g, GGML_TYPE_F32, { im.D, 1, im.n }, im.host_emb);
        ggml_tensor * x = ggml_repeat_4d(c, e, im.D, im.HC, im.n, 1);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, x, im.cols3(c, im.x_state)));
    });
    return im.run(*g, !tmp);
}

bool GlmTrunk::attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    Impl & im = *p_;
    if (fn_dev) *fn_dev = nullptr;
    const int n = im.n;
    const bool dense = il < im.NDENSE;
    static const bool dbg = std::getenv("ALMOND_GLM_DEBUG") != nullptr;
    std::unique_ptr<Impl::G> tmp;
    // out: fn, [sel, w], [attn_norm, mixer, x1 when debugging]
    Impl::G * g = im.graph(Impl::K_AR, il, im.recr[(size_t) il] ? 0 : im.n_kv, tmp, [&](Impl::G & g) {
        ggml_context * c = g.ctx;
        ggml_tensor * x = im.cols3(c, im.x_state);
        Impl::HcPre ah = im.hc_pre(c, x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"));
        ggml_tensor * an = im.rms(c, ah.cur, im.BL(il, "attn_norm.weight"));
        ggml_tensor * cur = im.recr[(size_t) il] ? im.kda(g, an, il) : im.dsa(g, an, il);
        ggml_tensor * x1 = im.hc_post(c, cur, x, ah.post, ah.comb);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, x1, im.cols3(c, im.hres)));
        Impl::HcPre fh = im.hc_pre(c, x1, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_cont(c, fh.post), im.cols2(c, im.post_f)));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_cont(c, fh.comb), im.cols2(c, im.comb_f)));
        ggml_tensor * fn = im.rms(c, fh.cur, im.BL(il, "ffn_norm.weight"));
        ggml_set_output(fn);
        ggml_build_forward_expand(g.gf, fn);
        g.out = { fn, nullptr, nullptr };
        if (dense) {
            ggml_tensor * o = im.ffn(c, fn, im.BL(il, "ffn_up.weight"), im.BL(il, "ffn_gate.weight"), im.BL(il, "ffn_down.weight"),
                                     im.clamp_shexp[(size_t) il]);
            ggml_build_forward_expand(g.gf, ggml_cpy(c, o, im.cols2(c, im.fout)));
        } else {
            ggml_tensor * o = im.ffn(c, fn, im.BL(il, "ffn_up_shexp.weight"), im.BL(il, "ffn_gate_shexp.weight"),
                                     im.BL(il, "ffn_down_shexp.weight"), im.clamp_shexp[(size_t) il]);
            ggml_build_forward_expand(g.gf, ggml_cpy(c, o, im.cols2(c, im.fout)));
            // build_moe_ffn's router: sigmoid, the selection bias for picking only, weights from the unbiased probs
            ggml_tensor * probs = ggml_sigmoid(c, ggml_mul_mat(c, im.BL(il, "ffn_gate_inp.weight"), fn));      // [NE, n]
            ggml_tensor * sp = ggml_add(c, probs, im.BL(il, "exp_probs_b.bias"));
            // argsort_top_k is a view (the first NU of each sorted row): contiguous before it is read back
            ggml_tensor * sel = ggml_argsort_top_k(c, sp, (int) im.NU);                                          // [NU, n]
            ggml_tensor * w = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, im.NE, n), sel), im.NU, n);
            if (im.w_norm) w = ggml_div(c, w, ggml_sum_rows(c, w));
            w = ggml_scale(c, w, im.w_scale);
            sel = ggml_cont(c, sel);
            ggml_set_output(sel);
            ggml_set_output(w);
            ggml_build_forward_expand(g.gf, sel);
            ggml_build_forward_expand(g.gf, w);
            g.out[1] = sel;
            g.out[2] = w;
        }
        if (dbg) {
            for (ggml_tensor * t : { an, cur, x1 }) { ggml_set_output(t); ggml_build_forward_expand(g.gf, t); g.out.push_back(t); }
        }
    });
    im.prof_tag = im.recr[(size_t) il] ? 0 : 1;
    if (!im.run(*g, !tmp)) return false;
    ggml_tensor * fn = g->out[0];
    if (dbg) {
        auto sum = [](ggml_tensor * t) {
            std::vector<float> v((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
            double s = 0;
            for (float f : v) s += f;
            return s;
        };
        std::fprintf(stderr, "glmdbg L%d attn_norm %.6f mixer_out %.6f hc_attn_post %.6f ffn_norm %.6f\n", il, sum(g->out[3]),
                     sum(g->out[4]), sum(g->out[5]), sum(fn));
    }
    im.host_fn.resize((size_t) (im.D * n));
    ggml_backend_tensor_get(fn, im.host_fn.data(), 0, im.host_fn.size() * 4);
    if (dense) {
        for (int64_t i = 0; i < im.NU * n; ++i) { ids[i] = -1; wts[i] = 0.0f; }
    } else {
        ggml_backend_tensor_get(g->out[1], ids, 0, (size_t) (im.NU * n) * 4);
        ggml_backend_tensor_get(g->out[2], wts, 0, (size_t) (im.NU * n) * 4);
    }
    *fn_host = im.host_fn.data();
    if (dbg && !dense) {
        double si = 0, sw = 0;
        for (int64_t i = 0; i < im.NU * n; ++i) { si += ids[i]; sw += wts[i]; }
        std::fprintf(stderr, "glmdbg L%d topk %.0f weights %.6f\n", il, si, sw);
    }
    return true;
}

bool GlmTrunk::finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    const int n = im.n;
    const bool routed = il >= im.NDENSE;
    // routed_sum null: the tier already wrote the sum into `routed` on the card (routed_dev)
    const bool on_dev = routed && !routed_sum;
    if (routed && !on_dev) im.host_routed.assign(routed_sum, routed_sum + (size_t) (im.D * n));
    static const bool dbg = std::getenv("ALMOND_GLM_DEBUG") != nullptr;
    std::unique_ptr<Impl::G> tmp;
    Impl::G * g = im.graph(Impl::K_FIN, il, on_dev ? 1 : 0, tmp, [&](Impl::G & g) {
        ggml_context * c = g.ctx;
        ggml_tensor * out = im.cols2(c, im.fout);
        if (on_dev) out = ggml_add(c, out, im.cols2(c, im.routed));
        else if (routed) out = ggml_add(c, out, im.input(g, GGML_TYPE_F32, { im.D, n }, im.host_routed));
        ggml_tensor * x = im.hc_post(c, out, im.cols3(c, im.hres), im.cols2(c, im.post_f), im.cols2(c, im.comb_f));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, x, im.cols3(c, im.x_state)));
        if (dbg) { ggml_set_output(x); ggml_set_output(out); g.out = { x, out }; }
    });
    im.prof_tag = 2;
    if (!im.run(*g, !tmp)) return false;
    if (dbg) {
        auto sum = [](ggml_tensor * t) {
            std::vector<float> v((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
            double s = 0;
            for (float f : v) s += f;
            return s;
        };
        if (routed && !on_dev) { double s = 0; for (float f : im.host_routed) s += f; std::fprintf(stderr, "glmdbg L%d moe_out %.6f\n", il, s); }
        std::fprintf(stderr, "glmdbg L%d ffn_out %.6f l_out %.6f\n", il, sum(g->out[1]), sum(g->out[0]));
    }
    return true;
}

bool GlmTrunk::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    std::unique_ptr<Impl::G> tmp;
    auto build = [&](Impl::G & g) {
        ggml_context * c = g.ctx;
        ggml_tensor * xs = ggml_view_3d(c, im.x_state, im.D, im.HC, nrows, im.x_state->nb[1], im.x_state->nb[2],
                                        (size_t) row0 * im.x_state->nb[2]);
        // build_hc_mean
        ggml_tensor * acc = ggml_view_2d(c, xs, im.D, nrows, xs->nb[2], 0);
        for (int64_t s = 1; s < im.HC; ++s) acc = ggml_add(c, acc, ggml_view_2d(c, xs, im.D, nrows, xs->nb[2], s * xs->nb[1]));
        ggml_tensor * h = im.rms(c, ggml_scale(c, acc, 1.0f / (float) im.HC), im.W("output_norm.weight"));
        ggml_tensor * lg = ggml_mul_mat(c, im.W("output.weight"), h);
        ggml_set_output(lg);
        ggml_build_forward_expand(g.gf, lg);
        g.out = { lg };
    };
    Impl::G * g;
    if (im.n == 1 && row0 == 0 && nrows == 1) {
        g = im.graph(Impl::K_LOGITS, 0, 0, tmp, build);
    } else {
        tmp = std::make_unique<Impl::G>();
        im.open(*tmp);
        build(*tmp);
        g = tmp.get();
    }
    if (!im.run(*g, !tmp)) return false;
    ggml_backend_tensor_get(g->out[0], out, 0, (size_t) nrows * (size_t) im.NV * 4);
    return true;
}

// ---- the next layer's prediction: none yet (the driver then runs without the prefetch, as with --pf-b 0)
bool GlmTrunk::predict(int, int *, int) { return false; }
bool GlmTrunk::predict_multi(int, int *) { return false; }
bool GlmTrunk::finish_predict(int il, const float * routed_sum, int, int * top_ids, int n_top) {
    for (int i = 0; i < n_top; ++i) top_ids[i] = -1;
    return finish(il, routed_sum);
}
bool GlmTrunk::finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead, int * ahead_ids) {
    for (int i = 0; i < ahead * 16; ++i) ahead_ids[i] = -1;
    return finish_predict(il, routed_sum, next_il, top_ids, n_top);
}

// ---- not in this first version: the driver checks and takes the chunked prompt / no conversation reuse
bool GlmTrunk::begin_batch(const int *, int, int) { p_->err = "glm: no layer-major prompt batch yet (--prompt-batch 0)"; return false; }
bool GlmTrunk::attn_router_span(int, int, int) { p_->err = "glm: no layer-major prompt batch yet"; return false; }
bool GlmTrunk::attn_router_span_kv(int, int, int) { p_->err = "glm: no layer-major prompt batch yet"; return false; }
const int32_t * GlmTrunk::batch_ids() const { return nullptr; }
const float * GlmTrunk::batch_wts() const { return nullptr; }
const float * GlmTrunk::batch_fn() const { return nullptr; }
int GlmTrunk::batch_size() const { return 0; }
bool GlmTrunk::upload_routed(const float *, int) { return false; }
float * GlmTrunk::routed_dev() { return p_->routed ? (float *) p_->routed->data : nullptr; }
bool GlmTrunk::finish_span(int, int, int) { return false; }
bool GlmTrunk::set_columns(int nc) { return nc <= p_->NC; }
int GlmTrunk::columns() const { return (int) p_->NC; }
void GlmTrunk::release_pass() {}
void GlmTrunk::set_oom_hook(std::function<bool()>) {}
bool GlmTrunk::snapshot(int) { return false; }
bool GlmTrunk::restore() { return false; }
int GlmTrunk::snapshot_pos() const { return -1; }
bool GlmTrunk::park(int64_t) { return false; }
bool GlmTrunk::unpark() { return false; }
void GlmTrunk::drop_park() {}
bool GlmTrunk::parked() const { return false; }
void GlmTrunk::rearm(int) {}
std::vector<MoeTrunk::StateRegion> GlmTrunk::state_regions(int64_t) const { return {}; }
void GlmTrunk::pool_regions(bool, std::vector<PoolRegion> & s, std::vector<PoolRegion> & r) const { s.clear(); r.clear(); }
void GlmTrunk::conv_tensors(std::vector<StreamTensor> & s, std::vector<ggml_tensor *> & r) const { s.clear(); r.clear(); }
bool GlmTrunk::read_state(std::vector<float> & x, std::vector<float> & pre) {
    Impl & im = *p_;
    x.resize((size_t) (im.D * im.HC * im.n));
    ggml_backend_tensor_get(im.x_state, x.data(), 0, x.size() * 4);
    pre.clear();
    return true;
}
bool GlmTrunk::read_state_cols(int c0, int n, std::vector<float> & x) {
    Impl & im = *p_;
    x.resize((size_t) (im.D * im.HC * n));
    ggml_backend_tensor_get(im.x_state, x.data(), (size_t) c0 * im.x_state->nb[2], x.size() * 4);
    return true;
}
ggml_tensor * GlmTrunk::weight(const std::string & name) const { return p_->W(name); }

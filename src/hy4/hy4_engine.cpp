// src/hy4/hy4_engine.cpp - see hy4_engine.hpp. The graph follows PR #32517 (llama.cpp hy_v4_flash) and the CPU
// blocks in this folder that were checked against it (docs/hy4/GGUF_NOTES.md).
#include "hy4_engine.hpp"

#include "hy4_attn.hpp"
#include "hy4_engram.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>

#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace almond::hy4 {

namespace {

constexpr int64_t KV_PAD = 256;      // the HIP flash attention wants n_kv in multiples of 256
constexpr size_t GRAPH_NODES = 24576;

int64_t pad_to(int64_t v, int64_t p) { return (v + p - 1) / p * p; }

float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

struct Hy4Engine::Impl {
    std::unique_ptr<GgufModel> model;
    Hy4Geometry G;
    EngramLayout L;
    Hy4EngineOptions o;
    int64_t n_trunk = 0, n_ctx = 0, n_past = 0;
    std::vector<int64_t> hist;

    std::vector<ggml_backend_t> gpu;
    ggml_backend_t cpu = nullptr;
    std::vector<ggml_backend_t> all;    // gpus then cpu
    ggml_backend_sched_t sched = nullptr;

    // weights: one no_alloc context per device (index ng = CPU); the CPU tensors point into the mapping
    std::vector<ggml_context *> wctx;
    std::vector<ggml_backend_buffer_t> wbuf;
    std::map<std::string, ggml_tensor *> w;
    std::vector<int> dev;       // [layer] card of the layer's dense weights and KV
    std::vector<int> edev;      // [layer] card of its routed experts, or ng for the CPU
    std::vector<ggml_tensor *> kv;   // [layer] F16 [kv_lora + rope, rows]: n_ctx rows, a ring of r_swa on SWA layers
    std::vector<ggml_tensor *> itok;  // [layer] F32 [2 * idx_dim, r_tok]: the indexer's value | gate, a ring
    std::vector<ggml_tensor *> ipool; // [layer] F32 [idx_dim, n_ctx / kpool]: the pooled keys
    int64_t r_swa = 0, r_tok = 0;
    // --fused-proj: per layer, the row offsets of attn_q_a / attn_kv_a_mqa / attn_gate / indexer.attn_k (-1 when the
    // layer has none) inside blk.L.attn_in_cat, the four Q8_0 matrices that read the attention input stacked as one
    struct ProjCat { int64_t qa = -1, kv = -1, gate = -1, ik = -1; };
    std::map<int64_t, ProjCat> proj_cat;
    int64_t n_alloc = 0;              // layers with weights and caches: the trunk, plus the MTP block with --mtp
    int64_t draft_off = 0;            // a chained draft: its position past n_past
    std::vector<float> draft_h;       // and the state it starts from (the previous draft's), empty = the trunk's
    // HY4_ROUTE_STATS=<file>: how often each layer routes to each expert, written when the engine ends (a diagnostic
    // for an expert cache)
    const char * route_stats = std::getenv("HY4_ROUTE_STATS");
    std::vector<std::pair<int64_t, ggml_tensor *>> route_ids;
    std::vector<std::vector<uint64_t>> route_count;
    std::vector<float> hhist;         // --mtp: the trunk's final state (output-normed) of every position in the cache

    bool is_swa(int64_t il) const { return G.swa_pattern[(size_t) il] != 0 && G.swa > 0; }
    bool has_indexer(int64_t il) const { return W("blk." + std::to_string(il) + ".indexer.attn_k.weight") != nullptr; }
    int64_t n_top_max() const { return G.idx_topk / G.idx_kpool - G.idx_tail; }
    std::vector<std::pair<const uint8_t *, size_t>> cpu_ranges;   // the mapped weights the CPU reads

    // the per-step graph
    std::vector<uint8_t> meta;

    ~Impl() {
        if (route_stats && !route_count.empty()) {
            if (std::FILE * f = std::fopen(route_stats, "w")) {
                for (size_t l = 0; l < route_count.size(); ++l) {
                    std::fprintf(f, "%zu", l);
                    for (uint64_t v : route_count[l]) std::fprintf(f, " %llu", (unsigned long long) v);
                    std::fprintf(f, "\n");
                }
                std::fclose(f);
            }
        }
        if (eng_fd >= 0) ::close(eng_fd);
        if (eng_ram) ::munmap(eng_ram - (eng_off & 4095ull), eng_ram_bytes);
        if (sched) ggml_backend_sched_free(sched);
        for (auto b : wbuf) if (b) ggml_backend_buffer_free(b);
        for (auto c : wctx) if (c) ggml_free(c);
        for (auto b : gpu) ggml_backend_free(b);
        if (cpu) ggml_backend_free(cpu);
    }

    int ng() const { return (int) gpu.size(); }

    ggml_tensor * W(const std::string & n) const {
        auto it = w.find(n);
        return it == w.end() ? nullptr : it->second;
    }

    static bool as_f32(const TensorInfo & t) {
        if (t.type != ty::BF16) return false;
        if (t.shape.size() == 1) return true;
        return t.name.find("engram_") != std::string::npos && t.name.find("engram_kv.weight") == std::string::npos;
    }

    /// The type a weight is kept in: BF16 vectors and the small Engram tensors become F32 (ggml's element-wise ops
    /// want F32); with --dense-q8 the 2-D BF16 matrices become Q8_0 (the batched per-head attn_k_b / attn_v_b stay).
    enum ggml_type kept_type(const TensorInfo & t) const {
        if (as_f32(t)) return GGML_TYPE_F32;
        if (o.dense_q8 && t.type == ty::BF16 && t.shape.size() == 2 && t.shape[0] % 32 == 0 &&
            t.name != "token_embd.weight" && t.name.find("attn_k_b") == std::string::npos &&
            t.name.find("attn_v_b") == std::string::npos)
            return GGML_TYPE_Q8_0;
        if (o.dense_f16 && t.type == ty::BF16 && t.shape.size() >= 2 && t.name != "token_embd.weight")
            return GGML_TYPE_F16;
        return (enum ggml_type) t.type;
    }
    uint64_t kept_bytes(const TensorInfo & t) const {
        const auto ty_ = kept_type(t);
        if (ty_ == (enum ggml_type) t.type) return tensor_payload_bytes(t);
        uint64_t b = ggml_row_size(ty_, (int64_t) t.shape[0]);
        for (size_t i = 1; i < t.shape.size(); ++i) b *= t.shape[i];
        return b;
    }

    // ---- placement ------------------------------------------------------------------------------------------------
    int layer_of(const std::string & n) const {
        if (n.rfind("blk.", 0) != 0) return -1;
        return std::atoi(n.c_str() + 4);
    }

    bool place(std::string & err) {
        const int NG = ng();
        // bytes per layer from the file
        std::vector<uint64_t> dense((size_t) n_trunk, 0), exps((size_t) n_trunk, 0);
        uint64_t head = 0;
        for (size_t s = 0; s < model->size(); ++s)
            for (const auto & t : model->shard(s).tensors()) {
                const int l = layer_of(t.name);
                const uint64_t b = kept_bytes(t);
                if (l >= 0 && l < n_trunk) {
                    if (t.name.find("_exps") != std::string::npos) exps[(size_t) l] += b; else dense[(size_t) l] += b;
                } else if (t.name.rfind("output", 0) == 0) head += b;
            }
        // the biggest a layer's caches get: a full layer's KV and indexer caches (the SWA layers keep a ring)
        const uint64_t kv_layer = (uint64_t) (G.kv_lora + G.d_rope) * 2 * (uint64_t) n_ctx +
                                  (uint64_t) G.idx_dim * 4 * ((uint64_t) n_ctx / 4 + 2 * (uint64_t) (o.n_ubatch + 8));
        std::vector<int64_t> budget((size_t) NG);
        for (int g = 0; g < NG; ++g) {
            size_t fr = 0, tot = 0;
            ggml_backend_cuda_get_device_memory(o.gpus[(size_t) g], &fr, &tot);
            budget[(size_t) g] = (int64_t) fr - o.reserve_mib * 1048576;
            if (o.verbose)
                std::fprintf(stderr, "hy4: card %d: %.1f GiB free\n", o.gpus[(size_t) g], fr / 1073741824.0);
        }
        budget[(size_t) NG - 1] -= (int64_t) head;
        // dense layers split between the cards in proportion to what each can hold of the whole trunk
        uint64_t dense_all = 0;
        for (int64_t l = 0; l < n_trunk; ++l) dense_all += dense[(size_t) l] + kv_layer;
        dev.assign((size_t) n_alloc, NG - 1);
        edev.assign((size_t) n_alloc, NG);
        if (n_alloc > n_trunk) {         // the MTP block's dense weights and caches on the last card, its experts in RAM
            uint64_t d = 0;
            for (size_t s = 0; s < model->size(); ++s)
                for (const auto & t : model->shard(s).tensors())
                    if (layer_of(t.name) == n_trunk && t.name.find("_exps") == std::string::npos) d += kept_bytes(t);
            budget[(size_t) NG - 1] -= (int64_t) (d + kv_layer);
        }
        // The dense part of each layer goes to the cards in contiguous ranges (one card-to-card step per token); every
        // cut between the ranges is tried, and on each card its layers' experts go into what is left, in layer order.
        // The cut that keeps the most expert layers in VRAM wins (a card that cannot take the next layer's experts
        // still takes dense layers, which leaves the other card room for experts).
        int64_t l = n_trunk;
        {
            const int expert_cap = o.expert_layers_gpu < 0 ? (int) n_trunk : o.expert_layers_gpu;
            std::vector<int> best_dev, best_edev;
            int best = -1;
            std::vector<int64_t> cuts;
            if (NG == 1) cuts.push_back(n_trunk);
            else for (int64_t a = 0; a <= n_trunk; ++a) cuts.push_back(a);
            for (int64_t a : cuts) {
                std::vector<int> dv((size_t) n_alloc, NG - 1), ev((size_t) n_alloc, NG);
                std::vector<int64_t> b(budget);
                bool ok = true;
                for (int64_t k = 0; k < n_trunk; ++k) {
                    const int g = NG == 1 || k < a ? 0 : 1;
                    dv[(size_t) k] = g;
                    b[(size_t) g] -= (int64_t) (dense[(size_t) k] + kv_layer);
                }
                for (int g = 0; g < NG; ++g) ok = ok && b[(size_t) g] >= 0;
                if (!ok) continue;
                int n_exp = 0;
                for (int64_t k = 0; k < n_trunk && n_exp < expert_cap; ++k) {
                    const int g = dv[(size_t) k];
                    if ((int64_t) exps[(size_t) k] <= b[(size_t) g]) {
                        ev[(size_t) k] = g;
                        b[(size_t) g] -= (int64_t) exps[(size_t) k];
                        ++n_exp;
                    }
                }
                // a layer whose experts are in RAM costs more on an earlier card than on the last one (measured
                // 2026-11-19, 64K: 13.6-14.7 tok/s with one to three such layers on card 0, 16.0-17.1 without), so a
                // cut that leaves one on an earlier card is not taken
                for (int64_t k = 0; k < n_trunk; ++k) ok = ok && (dv[(size_t) k] == NG - 1 || ev[(size_t) k] < NG);
                if (ok && n_exp > best) { best = n_exp; best_dev = dv; best_edev = ev; }
            }
            if (best < 0) l = 0;
            else { dev = best_dev; edev = best_edev; }
        }
        if (l < n_trunk) {
            uint64_t ds = 0, es = 0;
            for (int64_t k = 0; k < n_trunk; ++k) { ds += dense[(size_t) k]; es += exps[(size_t) k]; }
            err = "the dense weights do not fit in VRAM (stopped at layer " + std::to_string(l) + "; dense " +
                  std::to_string(ds >> 20) + " MiB, experts " + std::to_string(es >> 20) + " MiB, head " +
                  std::to_string(head >> 20) + " MiB)";
            return false;
        }
        (void) dense_all;
        if (o.verbose) {
            int on_gpu = 0;
            for (int64_t k = 0; k < n_trunk; ++k) on_gpu += edev[(size_t) k] < NG;
            std::fprintf(stderr, "hy4: layers per card:");
            for (int g = 0; g < NG; ++g)
                std::fprintf(stderr, " %d", (int) std::count(dev.begin(), dev.end(), g));
            std::fprintf(stderr, "; routed experts in VRAM for %d of %lld layers, the rest on the CPU\n", on_gpu,
                         (long long) n_trunk);
            std::string map;   // per layer: the card of its dense part, upper case when its experts are there too
            for (int64_t k = 0; k < n_trunk; ++k) map += (char) ((edev[(size_t) k] < NG ? 'A' : 'a') + dev[(size_t) k]);
            std::fprintf(stderr, "hy4: placement %s\n", map.c_str());
        }
        return true;
    }

    // ---- weights --------------------------------------------------------------------------------------------------
    bool upload(std::string & err) {
        const int NG = ng();
        wctx.assign((size_t) NG + 1, nullptr);
        for (auto & c : wctx) {
            ggml_init_params ip = { 4096 * ggml_tensor_overhead(), nullptr, true };
            c = ggml_init(ip);
        }
        struct Item { const TensorInfo * t; size_t shard; ggml_tensor * g; int d; };
        std::vector<Item> items;
        static const char * PROJ_PARTS[4] = { "attn_q_a.weight", "attn_kv_a_mqa.weight", "attn_gate.weight",
                                              "indexer.attn_k.weight" };
        std::map<int64_t, std::array<std::pair<const TensorInfo *, size_t>, 4>> cat_parts;
        for (size_t s = 0; s < model->size(); ++s)
            for (const auto & t : model->shard(s).tensors()) {
                const std::string & n = t.name;
                if (n == "per_layer_token_embd.weight") continue;            // the Engram table: host gathers
                const int l = layer_of(n);
                if (l >= n_alloc) continue;                                 // the MTP block without --mtp
                int d;
                if (n == "token_embd.weight") d = NG;
                else if (l < 0) d = dev[(size_t) n_trunk - 1];
                else d = n.find("_exps") != std::string::npos ? edev[(size_t) l] : dev[(size_t) l];
                const auto type = d == NG ? (enum ggml_type) t.type : kept_type(t);
                if (o.fused_proj && l >= 0 && d < NG && type == GGML_TYPE_Q8_0 && t.shape.size() == 2 &&
                    (int64_t) t.shape[0] == G.n_embd) {
                    const std::string b = "blk." + std::to_string(l) + ".";
                    int k = -1;
                    for (int j = 0; j < 4; ++j) if (n == b + PROJ_PARTS[j]) k = j;
                    if (k >= 0) { cat_parts[l][k] = { &t, s }; continue; }
                }
                int64_t ne[4] = { 1, 1, 1, 1 };
                for (size_t i = 0; i < t.shape.size() && i < 4; ++i) ne[i] = (int64_t) t.shape[i];
                ggml_tensor * g = ggml_new_tensor(wctx[(size_t) d], type, (int) std::max<size_t>(1, t.shape.size()), ne);
                ggml_set_name(g, n.c_str());
                w[n] = g;
                items.push_back({ &t, s, g, d });
            }
        // --fused-proj: the stacked matrix of each layer whose parts were all kept as Q8_0
        std::map<int64_t, ggml_tensor *> cat_t;
        for (auto & [l, parts] : cat_parts) {
            ProjCat pc;
            int64_t rows = 0;
            int64_t * offs[4] = { &pc.qa, &pc.kv, &pc.gate, &pc.ik };
            for (int j = 0; j < 4; ++j)
                if (parts[(size_t) j].first) { *offs[j] = rows; rows += (int64_t) parts[(size_t) j].first->shape[1]; }
            ggml_tensor * g = ggml_new_tensor_2d(wctx[(size_t) dev[(size_t) l]], GGML_TYPE_Q8_0, G.n_embd, rows);
            const std::string name = "blk." + std::to_string(l) + ".attn_in_cat";
            ggml_set_name(g, name.c_str());
            w[name] = g;
            cat_t[l] = g;
            proj_cat[l] = pc;
        }
        wbuf.assign((size_t) NG + 1, nullptr);
        for (int d = 0; d < NG; ++d) {
            wbuf[(size_t) d] = ggml_backend_alloc_ctx_tensors_from_buft(wctx[(size_t) d],
                                                                          ggml_backend_get_default_buffer_type(gpu[(size_t) d]));
            if (!wbuf[(size_t) d]) { err = "cannot allocate the weights on card " + std::to_string(o.gpus[(size_t) d]); return false; }
            ggml_backend_buffer_set_usage(wbuf[(size_t) d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        // CPU tensors: a buffer over each tensor's bytes in the mapping (zero copy)
        uint64_t up = 0;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> conv;
        for (auto & it : items) {
            const uint8_t * src = model->shard(it.shard).tensor_data(*it.t);
            const size_t bytes = ggml_nbytes(it.g);
            if (it.d == NG) {
                if (it.g->type != (enum ggml_type) it.t->type) { err = it.t->name + ": CPU tensor needs conversion"; return false; }
                ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(const_cast<uint8_t *>(src), bytes);
                ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                ggml_backend_tensor_alloc(b, it.g, const_cast<uint8_t *>(src));
                cpu_ranges.emplace_back(src, bytes);
                wbuf.push_back(b);
                continue;
            }
            const std::string & tn = it.t->name;
            if (o.fused_hc && it.t->type == ty::F32 && tn.rfind("blk.", 0) == 0 && tn.find("hc_") != std::string::npos &&
                tn.size() > 12 && tn.compare(tn.size() - 12, 12, "_base.weight") == 0) {
                // --fused-hc: Hy4 adds eps inside the gates' sigmoid; folded into the base here, so the fused op (eps 0)
                // gives the same pre gate, and the post gate lacks only Hy4's +eps after it
                const uint64_t n = it.t->elements();
                conv.assign(reinterpret_cast<const float *>(src), reinterpret_cast<const float *>(src) + n);
                for (auto & v : conv) v += (float) G.rms_eps;
                ggml_backend_tensor_set(it.g, conv.data(), 0, bytes);
            } else if (it.g->type == GGML_TYPE_F32 && it.t->type == ty::BF16) {
                const uint64_t n = it.t->elements();
                conv.resize(n);
                const uint16_t * h = reinterpret_cast<const uint16_t *>(src);
                for (uint64_t i = 0; i < n; ++i) conv[i] = bf16_to_f32(h[i]);
                ggml_backend_tensor_set(it.g, conv.data(), 0, bytes);
            } else if (it.g->type == GGML_TYPE_F16 && it.t->type == ty::BF16) {   // --dense-f16: exact unless out of range
                const uint64_t n = it.t->elements();
                const uint16_t * h = reinterpret_cast<const uint16_t *>(src);
                std::vector<ggml_fp16_t> f((size_t) n);
                uint64_t clipped = 0;
                for (uint64_t i = 0; i < n; ++i) {
                    float v = bf16_to_f32(h[i]);
                    if (std::fabs(v) > 65504.0f) { v = v > 0 ? 65504.0f : -65504.0f; ++clipped; }
                    f[i] = ggml_fp32_to_fp16(v);
                }
                if (clipped) std::fprintf(stderr, "hy4: %s: %llu values past the F16 range clipped\n", it.t->name.c_str(),
                                          (unsigned long long) clipped);
                ggml_backend_tensor_set(it.g, f.data(), 0, bytes);
            } else if (it.g->type != (enum ggml_type) it.t->type) {     // BF16 -> Q8_0, rows split over threads
                const int64_t ne0 = it.g->ne[0], nrows = ggml_nrows(it.g);
                conv.resize((size_t) (ne0 * nrows));
                const uint16_t * h = reinterpret_cast<const uint16_t *>(src);
                std::vector<uint8_t> q(bytes);
                const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
                std::vector<std::thread> th;
                for (int k = 0; k < nth; ++k)
                    th.emplace_back([&, k] {
                        const int64_t r0 = nrows * k / nth, r1 = nrows * (k + 1) / nth;
                        for (int64_t i = r0 * ne0; i < r1 * ne0; ++i) conv[(size_t) i] = bf16_to_f32(h[i]);
                        ggml_quantize_chunk(it.g->type, conv.data(), q.data(), r0 * ne0, r1 - r0, ne0, nullptr);  // start in elements
                    });
                for (auto & t : th) t.join();
                ggml_backend_tensor_set(it.g, q.data(), 0, bytes);
            } else {
                ggml_backend_tensor_set(it.g, src, 0, bytes);
            }
            up += bytes;
            // the file's copy of a weight now in VRAM is not needed in RAM: give its pages back, so they do not
            // compete with the experts the CPU reads (with the Engram table in RAM they pushed those out to disk)
            const uintptr_t a = (uintptr_t) src & ~(uintptr_t) 4095;
            ::madvise((void *) a, it.t->elements() ? tensor_payload_bytes(*it.t) + ((uintptr_t) src - a) : 0, MADV_PAGEOUT);
        }
        for (auto & [l, parts] : cat_parts) {
            ggml_tensor * g = cat_t[l];
            const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, G.n_embd);
            size_t at = 0;
            for (int j = 0; j < 4; ++j) {
                const TensorInfo * t = parts[(size_t) j].first;
                if (!t) continue;
                const uint8_t * src = model->shard(parts[(size_t) j].second).tensor_data(*t);
                const int64_t nr = (int64_t) t->shape[1];
                if (t->type == ty::Q8_0) {
                    ggml_backend_tensor_set(g, src, at, rb * (size_t) nr);
                } else {                                   // BF16 -> Q8_0
                    const uint16_t * h = reinterpret_cast<const uint16_t *>(src);
                    conv.resize((size_t) (G.n_embd * nr));
                    for (size_t i = 0; i < conv.size(); ++i) conv[i] = bf16_to_f32(h[i]);
                    std::vector<uint8_t> q(rb * (size_t) nr);
                    ggml_quantize_chunk(GGML_TYPE_Q8_0, conv.data(), q.data(), 0, nr, G.n_embd, nullptr);
                    ggml_backend_tensor_set(g, q.data(), at, q.size());
                }
                at += rb * (size_t) nr;
                up += rb * (size_t) nr;
            }
        }
        if (o.verbose)
            std::fprintf(stderr, "hy4: %.1f GiB of weights uploaded in %.1f s\n", up / 1073741824.0,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        // KV cache
        kv.assign((size_t) n_alloc, nullptr);
        itok.assign((size_t) n_alloc, nullptr);
        ipool.assign((size_t) n_alloc, nullptr);
        r_swa = pad_to(o.n_ubatch + G.swa, KV_PAD);      // a ubatch never overwrites a row its own tokens still see
        r_tok = o.n_ubatch + 2 * G.idx_kpool;            // the current group's members, up to kpool-1 back
        for (int d = 0; d < NG; ++d) {
            ggml_init_params ip = { (size_t) (3 * n_alloc + 1) * ggml_tensor_overhead(), nullptr, true };
            ggml_context * c = ggml_init(ip);
            int cnt = 0;
            for (int64_t l = 0; l < n_alloc; ++l)
                if (dev[(size_t) l] == d) {
                    kv[(size_t) l] = ggml_new_tensor_2d(c, GGML_TYPE_F16, G.kv_lora + G.d_rope, is_swa(l) ? r_swa : n_ctx);
                    ggml_set_name(kv[(size_t) l], ("kv-" + std::to_string(l)).c_str());
                    if (has_indexer(l)) {
                        itok[(size_t) l] = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * G.idx_dim, r_tok);
                        ipool[(size_t) l] = ggml_new_tensor_2d(c, GGML_TYPE_F32, G.idx_dim, n_ctx / G.idx_kpool);
                        ggml_set_name(itok[(size_t) l], ("itok-" + std::to_string(l)).c_str());
                        ggml_set_name(ipool[(size_t) l], ("ipool-" + std::to_string(l)).c_str());
                    }
                    ++cnt;
                }
            wctx.push_back(c);
            if (!cnt) continue;
            ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, ggml_backend_get_default_buffer_type(gpu[(size_t) d]));
            if (!b) { err = "cannot allocate the KV cache"; return false; }
            ggml_backend_buffer_clear(b, 0);
            wbuf.push_back(b);
        }
        return true;
    }

    // ---- graph ----------------------------------------------------------------------------------------------------
    /// What one step looks like, from where it starts in the sequence and how many tokens it has.
    struct Shape {
        int64_t first = 0, nt = 0, n_kv = 0;
        bool index = false;                  // the sparse layers select through the indexer (some token past top-k)
        bool img = false;                    // some positions take an image row instead of a token embedding
        bool mrope = false;                  // some rotation in the step has h or w apart from t
        int64_t n_pool = 0, n_top = 0, n_sel = 0, n_new = 0, g_new = 0;   // g_new: the first group completed here
    };

    Shape shape(int64_t first, int64_t nt) const {
        Shape sh;
        sh.first = first; sh.nt = nt;
        sh.n_kv = std::min(n_ctx, pad_to(first + nt, KV_PAD));
        const int64_t kp = G.idx_kpool, last = first + nt - 1;
        sh.g_new = (first + kp - 1) / kp;                       // groups whose last member is in [first, last]
        const int64_t g_end = (last + 1) / kp;
        sh.n_new = std::max<int64_t>(0, g_end - sh.g_new);
        // a step that completes no group pools the last complete one again (the same rows, the same result): the
        // graph keeps one shape from token to token, so the HIP graphs it was captured into are reused, not rebuilt
        if (sh.n_new == 0 && g_end > 0) { sh.g_new = g_end - 1; sh.n_new = 1; }
        sh.n_pool = std::max<int64_t>(1, last / kp);            // the pools the last token can see
        sh.index = last / kp > n_top_max() && !std::getenv("HY4_FULL_ATTENTION");   // (a diagnostic: full attention)
        sh.n_top = std::min(sh.n_pool, n_top_max());
        sh.n_sel = kp * sh.n_top + (G.idx_tail ? kp : 0);
        return sh;
    }

    struct Inputs {
        ggml_tensor * tok = nullptr, * pos = nullptr, * kidx = nullptr, * mask = nullptr, * mask_swa = nullptr;
        ggml_tensor * kidx_swa = nullptr, * kidx_tok = nullptr;
        ggml_tensor * new_cells = nullptr, * new_rep = nullptr, * new_pos = nullptr;
        ggml_tensor * pool_mask = nullptr, * pool_idx = nullptr, * tail = nullptr, * live = nullptr, * dump = nullptr;
        ggml_tensor * mask_base = nullptr, * zeros = nullptr;
        ggml_tensor * hprev = nullptr, * h = nullptr;
        ggml_tensor * emb_over = nullptr, * emb_sel = nullptr, * eng_valid = nullptr;   // image positions
        ggml_tensor * pos4 = nullptr, * new_pos4 = nullptr;   // M-RoPE positions (t, h, w, t) when the step needs them   // --mtp: the state before the first token; the states out
        std::vector<ggml_tensor *> eng;      // [module] F32 [dim, cols, nt]
        ggml_tensor * logits = nullptr;
        Shape sh;
    };

    ggml_tensor * input(ggml_context * c, enum ggml_type t, int64_t a, int64_t b_, const char * name, int64_t c3 = 1) {
        ggml_tensor * x = ggml_new_tensor_3d(c, t, a, b_, c3);
        ggml_set_input(x);
        ggml_set_name(x, name);
        return x;
    }

    /// The rotation of x [rope_dim, heads, n]: consecutive pairs (Tencent's rotate_half over 0::2 / 1::2). With `pos4`
    /// ([4 n]: t, h, w, then t again) it is the interleaved M-RoPE of Tencent's config.json (mrope_section [12, 10, 10]:
    /// pair i takes h when i % 3 == 1 and w when i % 3 == 2 below pair 30, t otherwise). ggml's IMROPE pairs (i, i + 32),
    /// so the pairs are moved to that order and back around it. A step of text only has t = h = w, which is the plain
    /// rotation, and takes it.
    ggml_tensor * rope(ggml_context * c, ggml_tensor * x, ggml_tensor * pos, ggml_tensor * pos4, float base) {
        const int64_t R = G.d_rope;
        if (!pos4) return ggml_rope_ext(c, x, pos, nullptr, (int) R, 0, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        const int64_t H = x->ne[1], n = x->ne[2];
        ggml_tensor * v = ggml_reshape_4d(c, ggml_cont(c, x), 2, R / 2, H, n);
        v = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3)), R, H, n);       // (2j + r) -> (j + 32 r)
        int sections[GGML_MROPE_SECTIONS] = { 12, 10, 10, 0 };
        v = ggml_rope_multi(c, v, pos4, nullptr, (int) R, sections, GGML_ROPE_TYPE_IMROPE, 0, base, 1.0f, 0.0f, 1.0f,
                            0.0f, 0.0f);
        v = ggml_reshape_4d(c, v, R / 2, 2, H, n);
        return ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3)), R, H, n);    // back to (2j + r)
    }

    /// The trailing rope_dim values of each indexer head rotated, the front left as it is.
    ggml_tensor * idx_rope(ggml_context * c, ggml_tensor * x, ggml_tensor * pos, ggml_tensor * pos4, float base) {
        const int64_t R = G.d_rope, nope = G.idx_dim - R;
        ggml_tensor * front = ggml_view_3d(c, x, nope, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
        ggml_tensor * back = ggml_view_3d(c, x, R, x->ne[1], x->ne[2], x->nb[1], x->nb[2], ggml_row_size(x->type, nope));
        back = rope(c, ggml_cont(c, back), pos, pos4, base);
        return ggml_concat(c, ggml_cont(c, front), back, 0);
    }

    /// PR #32517's build_kpool_select: stores this ubatch's indexer rows, pools the groups it completed, and (when
    /// the context is past top-k) returns the sparse layer's mask [n_kv, nt] F16; otherwise nullptr (causal mask).
    ggml_tensor * indexer(ggml_context * c, ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * qr, int64_t il,
                          float base, const Inputs & in, ggml_tensor * ik_raw = nullptr) {
        const Shape & sh = in.sh;
        const std::string b = "blk." + std::to_string(il) + ".";
        const int64_t NI = G.idx_dim, NH = G.idx_heads, kp = G.idx_kpool, nt = sh.nt;
        const float eps = (float) G.rms_eps;
        ggml_tensor * ik = ggml_mul(c, ggml_rms_norm(c, ik_raw ? ik_raw : ggml_mul_mat(c, W(b + "indexer.attn_k.weight"), x), eps),
                                    W(b + "indexer.k_norm.weight"));
        ggml_tensor * iv = ggml_mul_mat(c, W(b + "indexer_compressor_value.weight"), ik);
        ggml_tensor * ig = ggml_mul_mat(c, W(b + "indexer_compressor_gate.weight"), ik);
        ggml_tensor * tokc = ggml_set_rows(c, itok[(size_t) il], ggml_concat(c, iv, ig, 0), in.kidx_tok);
        ggml_build_forward_expand(gf, tokc);
        ggml_tensor * poolc = ipool[(size_t) il];
        if (sh.n_new > 0) {
            ggml_tensor * rows = ggml_get_rows(c, tokc, in.new_cells);                // [2NI, kp * n_new]
            rows = ggml_reshape_3d(c, rows, 2 * NI, kp, sh.n_new);
            ggml_tensor * pk = ggml_view_3d(c, rows, NI, kp, sh.n_new, rows->nb[1], rows->nb[2], 0);
            ggml_tensor * pg = ggml_view_3d(c, rows, NI, kp, sh.n_new, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, NI));
            ggml_tensor * lg = ggml_add(c, pg, W(b + "indexer_compressor_ape.weight"));      // ape [NI, kp]
            lg = ggml_cont(c, ggml_permute(c, lg, 1, 0, 2, 3));                       // [kp, NI, n_new]
            ggml_tensor * pr = ggml_reshape_3d(c, ggml_soft_max(c, ggml_reshape_2d(c, lg, kp, NI * sh.n_new)), kp, NI, sh.n_new);
            pk = ggml_cont(c, ggml_permute(c, pk, 1, 0, 2, 3));
            ggml_tensor * pooled = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_mul(c, pr, pk)), NI, sh.n_new);
            pooled = ggml_mul(c, ggml_rms_norm(c, pooled, eps), W(b + "indexer_compressor_norm.weight"));
            pooled = ggml_reshape_2d(c, idx_rope(c, ggml_reshape_3d(c, pooled, NI, 1, sh.n_new), in.new_pos, in.new_pos4, base), NI, sh.n_new);
            poolc = ggml_set_rows(c, poolc, pooled, in.new_rep);
            ggml_build_forward_expand(gf, poolc);
        }
        if (!sh.index) return nullptr;
        ggml_tensor * iq = ggml_reshape_3d(c, ggml_mul_mat(c, W(b + "indexer.attn_q_b.weight"), qr), NI, NH, nt);
        iq = idx_rope(c, iq, in.pos, in.pos4, base);
        ggml_tensor * pooled = ggml_view_3d(c, poolc, NI, sh.n_pool, 1, poolc->nb[1], poolc->nb[1] * sh.n_pool, 0);
        ggml_tensor * wts = ggml_scale(c, ggml_mul_mat(c, W(b + "indexer.proj.weight"), x), 1.0f / sqrtf((float) (NI * NH)));
        ggml_tensor * qp = ggml_permute(c, iq, 0, 2, 1, 3);                          // [NI, nt, NH]
        ggml_tensor * kq = ggml_mul_mat(c, pooled, qp);                               // [n_pool, nt, NH]
        kq = ggml_relu(c, ggml_cont(c, ggml_permute(c, kq, 2, 1, 0, 3)));             // [NH, nt, n_pool]
        kq = ggml_mul(c, kq, ggml_reshape_3d(c, wts, NH, nt, 1));
        ggml_tensor * score = ggml_sum_rows(c, kq);                                   // [1, nt, n_pool]
        score = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, score, 2, 1, 0, 3)), sh.n_pool, nt);
        score = ggml_add(c, score, in.pool_mask);
        ggml_tensor * top = ggml_argsort_top_k(c, score, (int) sh.n_top);            // [n_top, nt], best first
        ggml_tensor * sel = ggml_get_rows(c, in.pool_idx, ggml_reshape_1d(c, ggml_cont(c, top), sh.n_top * nt));
        sel = ggml_reshape_2d(c, sel, kp * sh.n_top, nt);
        if (G.idx_tail) sel = ggml_concat(c, sel, in.tail, 0);                        // [n_sel, nt]
        // a slot that is not real (a pool the token cannot see yet, a tail cell before the start) writes its own dump
        // row past n_kv: idx = dump + live * (idx - dump)
        ggml_tensor * f = ggml_cast(c, sel, GGML_TYPE_F32);
        f = ggml_add(c, ggml_mul(c, ggml_sub(c, f, in.dump), in.live), in.dump);
        ggml_tensor * idx = ggml_cast(c, f, GGML_TYPE_I32);
        ggml_tensor * m = ggml_set_rows(c, ggml_dup(c, in.mask_base), in.zeros, ggml_reshape_3d(c, idx, sh.n_sel, nt, 1));
        m = ggml_cont(c, ggml_view_2d(c, m, sh.n_kv, nt, m->nb[2], 0));
        return ggml_cast(c, m, GGML_TYPE_F16);
    }

    ggml_tensor * hc_pre(ggml_context * c, ggml_tensor * x, const std::string & p, int64_t nt, ggml_tensor ** post) {
        const int64_t D = G.n_embd, HC = G.hc;
        const float eps = (float) G.rms_eps;
        ggml_tensor * fn = W(p + "_fn.weight"), * sc = W(p + "_scale.weight"), * bs = W(p + "_base.weight");
        ggml_tensor * flat = ggml_reshape_2d(c, x, HC * D, nt);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, ggml_rms_norm(c, flat, eps));    // [2HC, nt]
        if (o.fused_hc && G.hc_mag == 2.0) {
            // --fused-hc: DeepSeek-V4's fused ops (vendored ggml); Hy4's eps inside the sigmoid is in the base (upload)
            ggml_tensor * g = ggml_dsv41_hc_mixes(c, ggml_cont(c, mixes), sc, bs, 0.0f);         // [HC, 2 nt]
            ggml_tensor * pre = ggml_view_2d(c, g, HC, nt, g->nb[1], 0);
            *post = ggml_view_2d(c, g, HC, nt, g->nb[1], (size_t) nt * g->nb[1]);
            return ggml_dsv4_hc_pre(c, x, pre);
        }
        ggml_tensor * mg = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * pg = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        ggml_tensor * s0 = ggml_view_1d(c, sc, 1, 0), * s1 = ggml_view_1d(c, sc, 1, 4);
        ggml_tensor * b0 = ggml_view_1d(c, bs, HC, 0), * b1 = ggml_view_1d(c, bs, HC, (size_t) HC * 4);
        ggml_tensor * pre = ggml_sigmoid(c, ggml_scale_bias(c, ggml_add(c, ggml_mul(c, mg, s0), b0), 1.0f, eps));
        ggml_tensor * po = ggml_sigmoid(c, ggml_scale_bias(c, ggml_add(c, ggml_mul(c, pg, s1), b1), 1.0f, eps));
        *post = ggml_scale_bias(c, ggml_scale(c, po, (float) G.hc_mag), 1.0f, eps);
        // sum_h x[:, h, :] * pre[h, :]
        ggml_tensor * pw = ggml_reshape_3d(c, ggml_cont(c, pre), 1, HC, nt);
        ggml_tensor * prod = ggml_mul(c, x, pw);                                      // [D, HC, nt]
        ggml_tensor * red = ggml_sum_rows(c, ggml_cont(c, ggml_permute(c, prod, 1, 0, 2, 3)));   // [1, D, nt]
        return ggml_reshape_2d(c, red, D, nt);
    }

    ggml_tensor * hc_post(ggml_context * c, ggml_tensor * y, ggml_tensor * residual, ggml_tensor * post, int64_t nt) {
        const int64_t D = G.n_embd, HC = G.hc;
        if (o.fused_hc) return ggml_dsv4_hc_post(c, y, residual, post, nullptr);   // residual + y * post, one kernel
        // residual[:, h, :] + y * post[h, :]
        ggml_tensor * yb = ggml_reshape_3d(c, y, D, 1, nt);
        ggml_tensor * pw = ggml_reshape_3d(c, ggml_cont(c, post), 1, HC, nt);
        ggml_tensor * add = ggml_mul(c, ggml_repeat_4d(c, yb, D, HC, nt, 1), pw);
        return ggml_add(c, residual, add);
    }

    ggml_tensor * norm(ggml_context * c, ggml_tensor * x, const std::string & n) {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) G.rms_eps), W(n));
    }

    ggml_tensor * engram(ggml_context * c, ggml_tensor * x, ggml_tensor * rows, const std::string & b, int64_t nt,
                         ggml_tensor * in_valid = nullptr) {
        const int64_t D = G.n_embd, HC = G.hc, cols = L.heads * (L.n_gram - 1), dim = L.dim;
        const float eps = 1.1920929e-07f;
        ggml_tensor * r = ggml_mul(c, ggml_rms_norm(c, rows, eps), W(b + "engram_lookup_norm.weight"));
        r = ggml_reshape_2d(c, r, dim * cols, nt);
        ggml_tensor * kvt = ggml_mul_mat(c, W(b + "engram_kv.weight"), r);
        kvt = ggml_add(c, ggml_reshape_3d(c, kvt, D, HC + 1, nt), W(b + "engram_kv_b.weight"));
        ggml_tensor * key = ggml_cont(c, ggml_view_3d(c, kvt, D, HC, nt, kvt->nb[1], kvt->nb[2], 0));
        ggml_tensor * value = ggml_cont(c, ggml_view_3d(c, kvt, D, 1, nt, kvt->nb[1], kvt->nb[2], HC * kvt->nb[1]));
        ggml_tensor * dot = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, x, eps), ggml_rms_norm(c, key, eps)),
                                     W(b + "engram_gate.weight"));
        dot = ggml_scale(c, ggml_sum_rows(c, dot), 1.0f / sqrtf((float) D));
        ggml_tensor * mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, dot), 1e-6f, INFINITY));
        ggml_tensor * sign = ggml_scale_bias(c, ggml_step(c, ggml_neg(c, dot)), -2.0f, 1.0f);
        ggml_tensor * gate = ggml_sigmoid(c, ggml_mul(c, mag, sign));
        ggml_tensor * signal = ggml_mul(c, ggml_rms_norm(c, value, eps), W(b + "engram_pre_gamma.weight"));
        ggml_tensor * scale = ggml_add(c, ggml_repeat_4d(c, gate, D, HC, nt, 1), W(b + "engram_hc_scale.weight"));
        ggml_tensor * delta = ggml_mul(c, ggml_repeat_4d(c, signal, D, HC, nt, 1), scale);
        if (in_valid) delta = ggml_mul(c, delta, in_valid);   // an image position takes no Engram delta (the PR's mask)
        return ggml_add(c, x, delta);
    }

    ggml_tensor * attention(ggml_context * c, ggml_cgraph * gf, ggml_tensor * x, int64_t il, int64_t nt,
                            const Inputs & in) {
        const std::string b = "blk." + std::to_string(il) + ".";
        const int64_t KR = G.kv_lora, R = G.d_rope, H = G.head_count[(size_t) il];
        const int64_t qhead = G.d_key_mla, khead = KR + R, nope = qhead - R, VD = G.d_val_mla;
        const float eps = (float) G.rms_eps;
        const bool swa = is_swa(il);
        const float base = swa ? (float) G.rope_base_swa : (float) G.rope_base;
        const int64_t n_kv = swa ? r_swa : in.sh.n_kv;

        // --fused-proj: one product for the four projections of x, then views of its rows
        ggml_tensor * cat = nullptr, * v_qa = nullptr, * v_kv = nullptr, * v_gate = nullptr, * v_ik = nullptr;
        if (auto it = proj_cat.find(il); it != proj_cat.end()) {
            cat = ggml_mul_mat(c, W(b + "attn_in_cat"), x);                          // [rows, nt]
            auto rows = [&](int64_t off, int64_t n) {
                return ggml_view_2d(c, cat, n, nt, cat->nb[1], (size_t) off * ggml_element_size(cat));
            };
            v_qa = rows(it->second.qa, G.q_lora);
            v_kv = rows(it->second.kv, KR + R);
            v_gate = rows(it->second.gate, H * VD);
            if (it->second.ik >= 0) v_ik = rows(it->second.ik, G.idx_dim);
        }
        auto contig = [&](ggml_tensor * v) { return ggml_is_contiguous(v) ? v : ggml_cont(c, v); };   // one token: as is
        if (v_ik) v_ik = contig(v_ik);
        ggml_tensor * qr = norm(c, v_qa ? contig(v_qa) : ggml_mul_mat(c, W(b + "attn_q_a.weight"), x),
                                b + "attn_q_a_norm.weight");
        ggml_tensor * q = ggml_reshape_3d(c, ggml_mul_mat(c, W(b + "attn_q_b.weight"), qr), qhead, H, nt);
        ggml_tensor * qn = ggml_view_3d(c, q, nope, H, nt, q->nb[1], q->nb[2], 0);
        ggml_tensor * qp = ggml_view_3d(c, q, R, H, nt, q->nb[1], q->nb[2], (size_t) nope * ggml_element_size(q));
        qp = rope(c, qp, in.pos, in.pos4, base);
        qp = ggml_permute(c, qp, 0, 2, 1, 3);                                       // [R, nt, H]
        qn = ggml_permute(c, qn, 0, 2, 1, 3);                                       // [nope, nt, H]
        ggml_tensor * wkb = ggml_reshape_3d(c, W(b + "attn_k_b.weight"), nope, KR, H);
        qn = ggml_mul_mat(c, wkb, qn);                                              // [KR, nt, H]
        ggml_tensor * Q = ggml_concat(c, qn, qp, 0);                                // [khead, nt, H]

        ggml_tensor * kvc = v_kv ? v_kv : ggml_mul_mat(c, W(b + "attn_kv_a_mqa.weight"), x);   // [KR + R, nt]
        ggml_tensor * lat = ggml_view_2d(c, kvc, KR, nt, kvc->nb[1], 0);
        ggml_tensor * kp = ggml_view_2d(c, kvc, R, nt, kvc->nb[1], (size_t) KR * ggml_element_size(kvc));
        lat = norm(c, ggml_cont(c, lat), b + "attn_kv_a_norm.weight");
        kp = rope(c, ggml_reshape_3d(c, ggml_cont(c, kp), R, 1, nt), in.pos, in.pos4, base);
        ggml_tensor * Knew = ggml_concat(c, lat, ggml_reshape_2d(c, kp, R, nt), 0);    // [khead, nt]
        ggml_tensor * cache = ggml_set_rows(c, kv[(size_t) il], Knew, swa ? in.kidx_swa : in.kidx);
        ggml_build_forward_expand(gf, cache);
        ggml_tensor * K = ggml_view_3d(c, cache, khead, n_kv, 1, cache->nb[1], cache->nb[1] * n_kv, 0);
        ggml_tensor * V = ggml_view_3d(c, cache, KR, n_kv, 1, cache->nb[1], cache->nb[1] * n_kv, 0);

        ggml_tensor * mask = swa ? in.mask_swa : in.mask;
        if (!swa && has_indexer(il)) {
            ggml_tensor * sel = indexer(c, gf, x, qr, il, base, in, v_ik);
            if (sel) mask = sel;
        }
        ggml_tensor * attn = ggml_flash_attn_ext(c, Q, K, V, mask, 1.0f / sqrtf((float) qhead), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(attn, W(b + "attn_sinks.weight"));
        ggml_prec_set_acc(attn, GGML_PREC_F32);
        ggml_tensor * A = ggml_reshape_3d(c, attn, KR, H, nt);
        ggml_tensor * wvb = ggml_reshape_3d(c, W(b + "attn_v_b.weight"), KR, VD, H);
        A = ggml_mul_mat(c, wvb, ggml_permute(c, A, 0, 2, 1, 3));                   // [VD, nt, H]
        A = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, A, 0, 2, 1, 3)), H * VD, nt);
        A = ggml_mul(c, A, ggml_sigmoid(c, v_gate ? contig(v_gate) : ggml_mul_mat(c, W(b + "attn_gate.weight"), x)));
        ggml_tensor * y = ggml_mul_mat(c, W(b + "attn_output.weight"), A);
        return norm(c, y, b + "post_attention_norm.weight");
    }

    ggml_tensor * ffn(ggml_context * c, ggml_cgraph * gf, ggml_tensor * x, int64_t il, int64_t nt) {
        const std::string b = "blk." + std::to_string(il) + ".";
        if (il < G.n_dense) {
            ggml_tensor * g = ggml_mul_mat(c, W(b + "ffn_gate.weight"), x), * u = ggml_mul_mat(c, W(b + "ffn_up.weight"), x);
            return ggml_mul_mat(c, W(b + "ffn_down.weight"), ggml_swiglu_split(c, g, u));
        }
        const int64_t NE = G.n_expert, USED = G.n_used, LAT = G.d_latent;
        ggml_tensor * logits = ggml_mul_mat(c, W(b + "ffn_gate_inp.weight"), x);   // [NE, nt]
        // the routing in llama.cpp's build_moe_ffn order (sigmoid, reshape, bias, argsort/view, get_rows, then
        // reshape, sum_rows, clamp, div, reshape, scale) and expanded here, before the experts, so its nodes stay
        // together: ggml-cuda then runs it as one fused top-k MoE kernel
        ggml_tensor * probs = ggml_sigmoid(c, logits);
        ggml_tensor * probs3 = ggml_reshape_3d(c, probs, 1, NE, nt);
        ggml_tensor * sel = ggml_add(c, probs, W(b + "exp_probs_b.bias"));
        ggml_tensor * ids = ggml_argsort_top_k(c, sel, (int) USED);                // [USED, nt]
        if (route_stats) { ggml_set_output(ids); route_ids.emplace_back(il, ids); }
        ggml_tensor * wts = ggml_get_rows(c, probs3, ids);
        ggml_tensor * w2 = ggml_reshape_2d(c, wts, USED, nt);
        ggml_tensor * wsum = ggml_clamp(c, ggml_sum_rows(c, w2), 6.103515625e-5f, INFINITY);
        wts = ggml_reshape_3d(c, ggml_div(c, w2, wsum), 1, USED, nt);
        wts = ggml_scale(c, wts, (float) G.weights_scale);
        ggml_build_forward_expand(gf, wts);
        ggml_tensor * routed = ggml_mul_mat(c, W(b + "ffn_routed_down.weight"), x);  // [LAT, nt]
        routed = ggml_reshape_3d(c, routed, LAT, 1, nt);
        ggml_tensor * g = ggml_mul_mat_id(c, W(b + "ffn_gate_exps.weight"), routed, ids);
        ggml_tensor * u = ggml_mul_mat_id(c, W(b + "ffn_up_exps.weight"), routed, ids);
        ggml_tensor * d = ggml_mul_mat_id(c, W(b + "ffn_down_exps.weight"), ggml_swiglu_split(c, g, u), ids);
        d = ggml_mul(c, d, wts);                                                    // [LAT, USED, nt]
        ggml_tensor * summed = ggml_sum_rows(c, ggml_cont(c, ggml_permute(c, d, 1, 0, 2, 3)));   // [1, LAT, nt]
        ggml_tensor * moe = ggml_mul_mat(c, W(b + "ffn_routed_up.weight"), ggml_reshape_2d(c, summed, LAT, nt));
        ggml_tensor * sg = ggml_mul_mat(c, W(b + "ffn_gate_shexp.weight"), x);
        ggml_tensor * su = ggml_mul_mat(c, W(b + "ffn_up_shexp.weight"), x);
        ggml_tensor * sh = ggml_mul_mat(c, W(b + "ffn_down_shexp.weight"), ggml_swiglu_split(c, sg, su));
        return ggml_add(c, moe, sh);
    }

    /// PR #32517's graph_mtp: the next token's embedding and the trunk's state before it -> eh_proj -> an ordinary
    /// sparse layer with a plain residual -> shared_head_norm. `emb` and `hs` are [D, nt].
    ggml_tensor * mtp_layer(ggml_context * c, ggml_cgraph * gf, ggml_tensor * emb, ggml_tensor * hs, int64_t nt,
                            const Inputs & in, bool cache_only = false) {
        const int64_t il = n_trunk;
        const std::string b = "blk." + std::to_string(il) + ".";
        ggml_tensor * cur = ggml_concat(c, norm(c, emb, b + "nextn.enorm.weight"), norm(c, hs, b + "nextn.hnorm.weight"), 0);
        cur = ggml_mul_mat(c, W(b + "nextn.eh_proj.weight"), cur);
        ggml_tensor * res = cur;
        cur = attention(c, gf, norm(c, cur, b + "attn_norm.weight"), il, nt, in);   // expands the cache writes
        if (cache_only) return nullptr;      // the trunk's step: only the block's caches are wanted, not its output
        cur = ggml_add(c, cur, res);
        res = cur;
        cur = norm(c, ffn(c, gf, norm(c, cur, b + "ffn_norm.weight"), il, nt), b + "post_ffw_norm.weight");
        cur = ggml_add(c, cur, res);
        return norm(c, cur, b + "nextn.shared_head_norm.weight");
    }

    /// The draft: the MTP block alone on (state before, token) at the next position -> logits [V, 1].
    ggml_cgraph * build_draft(ggml_context * c, const Shape & sh, Inputs & in) {
        ggml_cgraph * gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
        make_inputs(c, sh, in);
        ggml_tensor * emb = ggml_get_rows(c, W("token_embd.weight"), in.tok);
        ggml_tensor * hm = mtp_layer(c, gf, emb, in.hprev, sh.nt, in);
        ggml_set_output(hm);
        in.h = hm;              // a chained draft feeds the block's own state back in
        ggml_tensor * y = ggml_mul_mat(c, W("output.weight"), hm);
        ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        in.logits = y;
        return gf;
    }

    ggml_cgraph * build(ggml_context * c, const Shape & sh, Inputs & in, bool all_logits = false) {
        const int64_t D = G.n_embd, HC = G.hc, nt = sh.nt;
        const float eps = (float) G.rms_eps;
        ggml_cgraph * gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
        make_inputs(c, sh, in);
        return build_trunk(c, gf, sh, in, all_logits, D, HC, nt, eps);
    }

    void make_inputs(ggml_context * c, const Shape & sh, Inputs & in) {
        const int64_t nt = sh.nt, kp = G.idx_kpool;
        in.sh = sh;
        if (n_alloc > n_trunk) in.hprev = input(c, GGML_TYPE_F32, G.n_embd, 1, "inp_hprev");
        if (sh.mrope) {
            in.pos4 = input(c, GGML_TYPE_I32, 4 * nt, 1, "inp_pos4");
            if (sh.n_new > 0) in.new_pos4 = input(c, GGML_TYPE_I32, 4 * sh.n_new, 1, "inp_new_pos4");
        }
        if (sh.img) {
            in.emb_over = input(c, GGML_TYPE_F32, G.n_embd, nt, "inp_emb_over");
            in.emb_sel = input(c, GGML_TYPE_F32, 1, nt, "inp_emb_sel");
            in.eng_valid = input(c, GGML_TYPE_F32, 1, 1, "inp_eng_valid", nt);
        }
        in.tok = input(c, GGML_TYPE_I32, nt, 1, "inp_tok");
        in.pos = input(c, GGML_TYPE_I32, nt, 1, "inp_pos");
        in.kidx = input(c, GGML_TYPE_I64, nt, 1, "inp_kidx");
        in.kidx_swa = input(c, GGML_TYPE_I64, nt, 1, "inp_kidx_swa");
        in.kidx_tok = input(c, GGML_TYPE_I64, nt, 1, "inp_kidx_tok");
        in.mask = input(c, GGML_TYPE_F16, sh.n_kv, nt, "inp_mask");
        in.mask_swa = input(c, GGML_TYPE_F16, r_swa, nt, "inp_mask_swa");
        if (sh.n_new > 0) {
            in.new_cells = input(c, GGML_TYPE_I32, kp * sh.n_new, 1, "inp_new_cells");
            in.new_rep = input(c, GGML_TYPE_I64, sh.n_new, 1, "inp_new_rep");
            in.new_pos = input(c, GGML_TYPE_I32, sh.n_new, 1, "inp_new_pos");
        }
        if (sh.index) {
            in.pool_mask = input(c, GGML_TYPE_F32, sh.n_pool, nt, "inp_pool_mask");
            in.pool_idx = input(c, GGML_TYPE_I32, kp, sh.n_pool, "inp_pool_idx");
            in.tail = input(c, GGML_TYPE_I32, kp, nt, "inp_tail");
            in.live = input(c, GGML_TYPE_F32, sh.n_sel, nt, "inp_live");
            in.dump = input(c, GGML_TYPE_F32, sh.n_sel, 1, "inp_dump");
            in.mask_base = input(c, GGML_TYPE_F32, 1, sh.n_kv + sh.n_sel, "inp_mask_base", nt);
            in.zeros = input(c, GGML_TYPE_F32, 1, sh.n_sel, "inp_zeros", nt);
        }
        in.eng.clear();
        for (size_t e = 0; e < L.modules.size(); ++e) {
            ggml_tensor * t = ggml_new_tensor_3d(c, GGML_TYPE_F32, L.dim, L.heads * (L.n_gram - 1), nt);
            ggml_set_input(t);
            ggml_set_name(t, ("inp_engram-" + std::to_string(e)).c_str());
            in.eng.push_back(t);
        }
    }

    ggml_cgraph * build_trunk(ggml_context * c, ggml_cgraph * gf, const Shape & sh, Inputs & in, bool all_logits,
                              int64_t D, int64_t HC, int64_t nt, float eps) {
        (void) sh;
        route_ids.clear();
        ggml_tensor * emb = ggml_get_rows(c, W("token_embd.weight"), in.tok);        // [D, nt]
        if (in.emb_over) emb = ggml_add(c, emb, ggml_mul(c, ggml_sub(c, in.emb_over, emb), in.emb_sel));   // image rows
        ggml_tensor * x = ggml_repeat_4d(c, ggml_reshape_3d(c, emb, D, 1, nt), D, HC, nt, 1);
        for (int64_t il = 0; il < n_trunk; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            for (size_t e = 0; e < L.modules.size(); ++e)
                if (L.modules[e].layer == il) x = engram(c, x, in.eng[e], b, nt, in.eng_valid);
            ggml_tensor * post = nullptr;
            ggml_tensor * cur = hc_pre(c, x, b + "hc_attn", nt, &post);
            cur = norm(c, cur, b + "attn_norm.weight");
            cur = attention(c, gf, cur, il, nt, in);
            x = hc_post(c, cur, x, post, nt);
            cur = hc_pre(c, x, b + "hc_ffn", nt, &post);
            cur = norm(c, cur, b + "ffn_norm.weight");
            cur = ffn(c, gf, cur, il, nt);
            cur = norm(c, cur, b + "post_ffw_norm.weight");
            x = hc_post(c, cur, x, post, nt);
        }
        // iHC head, output norm, vocabulary projection: the last token only, or every token
        const bool mtp = n_alloc > n_trunk;
        const int64_t no = all_logits || mtp ? nt : 1;
        ggml_tensor * xl = no == nt ? x : ggml_cont(c, ggml_view_3d(c, x, D, HC, 1, x->nb[1], x->nb[2],
                                                                    (size_t) (nt - 1) * x->nb[2]));
        ggml_tensor * flat = ggml_reshape_2d(c, xl, HC * D, no);
        ggml_tensor * mixes = ggml_mul_mat(c, W("output_hc_fn.weight"), ggml_rms_norm(c, flat, eps));   // [HC, no]
        ggml_tensor * gates = ggml_sigmoid(c, ggml_scale_bias(c, ggml_add(c, ggml_mul(c, mixes, W("output_hc_scale.weight")),
                                                                         W("output_hc_base.weight")), 1.0f, eps));
        ggml_tensor * prod = ggml_mul(c, xl, ggml_reshape_3d(c, gates, 1, HC, no));
        ggml_tensor * red = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_cont(c, ggml_permute(c, prod, 1, 0, 2, 3))), D, no);
        ggml_tensor * hn = norm(c, red, "output_norm.weight");                     // [D, no]
        ggml_tensor * hl = hn;
        if (no == nt && !all_logits) hl = ggml_cont(c, ggml_view_2d(c, hn, D, 1, hn->nb[1], (size_t) (nt - 1) * hn->nb[1]));
        ggml_tensor * y = ggml_mul_mat(c, W("output.weight"), hl);
        ggml_set_output(y);
        ggml_set_name(y, "logits");
        ggml_build_forward_expand(gf, y);
        in.logits = y;
        if (mtp) {
            // the MTP block's own cache follows the trunk: (state at p-1, token at p) for every token of the step
            in.h = hn;
            ggml_set_output(hn);
            ggml_build_forward_expand(gf, hn);
            ggml_tensor * hs = nt > 1 ? ggml_concat(c, in.hprev, ggml_view_2d(c, hn, D, nt - 1, hn->nb[1], 0), 1) : in.hprev;
            mtp_layer(c, gf, emb, hs, nt, in, true);
        }
        return gf;
    }

    // ---- host inputs ----------------------------------------------------------------------------------------------
    // ---- Engram rows from the SSD ---------------------------------------------------------------------------------
    // The table (63 GB FP8) is read with O_DIRECT, one 4 KiB-aligned block per row (two when a row straddles a block),
    // never through the mapping or the page cache, so it does not push the experts in RAM out of memory.
    int eng_fd = -1;
    bool eng_direct = false;
    uint64_t eng_off = 0;     // the table's first byte in its file
    uint8_t * eng_ram = nullptr;   // the whole table in RAM (--engram ram, or auto when it fits)
    size_t eng_ram_bytes = 0;

    static uint64_t mem_available() {
        std::FILE * f = std::fopen("/proc/meminfo", "r");
        if (!f) return 0;
        char key[64];
        unsigned long long kb = 0, got = 0;
        while (std::fscanf(f, "%63s %llu kB", key, &kb) == 2)
            if (!std::strcmp(key, "MemAvailable:")) { got = kb; break; }
        std::fclose(f);
        return (uint64_t) got * 1024;
    }

    /// Brings the mapped weights the CPU reads into RAM now (16 threads touching one byte a page), so the first
    /// tokens do not wait on page faults to the disk.
    void warm_cpu_weights() {
        uint64_t total = 0;
        for (auto & r : cpu_ranges) {
            total += r.second;
            const uintptr_t a = (uintptr_t) r.first & ~(uintptr_t) 4095;
            ::madvise((void *) a, r.second + ((uintptr_t) r.first - a), MADV_WILLNEED);
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0};
        std::atomic<uint64_t> sink{0};
        std::vector<std::thread> th;
        for (int k = 0; k < 16; ++k)
            th.emplace_back([&] {
                uint64_t acc = 0;
                for (size_t i; (i = next.fetch_add(1)) < cpu_ranges.size();) {
                    const volatile uint8_t * p = cpu_ranges[i].first;
                    for (size_t o = 0; o < cpu_ranges[i].second; o += 4096) acc += p[o];
                }
                sink += acc;
            });
        for (auto & t : th) t.join();
        if (o.verbose)
            std::fprintf(stderr, "hy4: %.1f GiB of CPU weights brought into RAM in %.1f s\n", total / 1073741824.0,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    /// The table read whole into anonymous memory with parallel O_DIRECT reads of 64 MiB.
    bool load_engram_ram(const std::string & path, uint64_t bytes, std::string & err) {
        const uint64_t base = eng_off & ~4095ull, span = ((eng_off + bytes + 4095) & ~4095ull) - base;
        void * p = ::mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) { err = "cannot allocate the Engram table in RAM"; return false; }
        int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        const bool direct = fd >= 0;
        if (!direct) fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { ::munmap(p, span); err = "cannot open " + path; return false; }
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t piece = 64ull << 20;
        std::atomic<uint64_t> next{0};
        std::atomic<bool> bad{false};
        std::vector<std::thread> th;
        for (int k = 0; k < 16; ++k)
            th.emplace_back([&] {
                for (uint64_t at; (at = next.fetch_add(piece)) < span && !bad;) {
                    const uint64_t n = std::min(piece, span - at);
                    uint64_t done = 0;
                    while (done < n) {
                        const ssize_t r = ::pread(fd, (uint8_t *) p + at + done, n - done, (off_t) (base + at + done));
                        if (r <= 0) break;
                        done += (uint64_t) r;
                    }
                    if (done < n && base + at + done < base + span - 4096) { bad = true; return; }
                }
            });
        for (auto & t : th) t.join();
        if (!direct) ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
        if (bad) { ::munmap(p, span); err = "reading the Engram table failed"; return false; }
        eng_ram = (uint8_t *) p + (eng_off - base);
        eng_ram_bytes = span;
        if (o.verbose)
            std::fprintf(stderr, "hy4: Engram table in RAM: %.1f GiB read in %.1f s\n", bytes / 1073741824.0,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return true;
    }

    bool open_engram(std::string & err) {
        size_t which = 0;
        const TensorInfo * tt = model->find("per_layer_token_embd.weight", &which);
        if (!tt) { err = "no Engram table"; return false; }
        const GgufFile & f = model->shard(which);
        eng_off = f.data_start() + tt->offset;
        const uint64_t bytes = tt->elements();   // FP8 E4M3: one byte each (ggml has no size for type 43)
        // auto: the table goes to RAM when it fits next to the experts read from RAM, with 8 GiB to spare
        bool ram = o.engram == "ram";
        if (o.engram == "auto") {
            uint64_t cpu_exps = 0, gpu_w = 0;
            for (size_t s = 0; s < model->size(); ++s)
                for (const auto & t : model->shard(s).tensors()) {
                    const int l = layer_of(t.name);
                    if (l < 0 || l >= n_trunk) continue;
                    const bool e = t.name.find("_exps") != std::string::npos;
                    if (e && edev[(size_t) l] == ng()) cpu_exps += tensor_payload_bytes(t);
                    else gpu_w += tensor_payload_bytes(t);
                }
            // The file pages of the weights copied to VRAM stayed resident (about 45 GiB on 2026-11-18, MADV_PAGEOUT
            // did not free them), and with the table in RAM next to them the experts read from RAM were paged in from
            // the disk at every token (a 128-token run did not finish in 15 minutes on 122 GiB). So the table goes to
            // RAM only when it fits beside those pages too.
            const uint64_t avail = mem_available();
            ram = avail > bytes + cpu_exps + gpu_w + (12ull << 30);
            if (o.verbose)
                std::fprintf(stderr, "hy4: --engram-place auto: %.1f GiB available, table %.1f GiB + experts in RAM %.1f GiB "
                             "+ weights in VRAM %.1f GiB -> %s\n", avail / 1073741824.0, bytes / 1073741824.0,
                             cpu_exps / 1073741824.0, gpu_w / 1073741824.0,
                             ram ? "RAM" : "SSD");
        }
        if (ram) {
            if (load_engram_ram(f.path(), bytes, err)) return true;
            if (o.engram == "ram") return false;
            std::fprintf(stderr, "hy4: %s; reading it from the SSD instead\n", err.c_str());
            err.clear();
        }
        eng_fd = ::open(f.path().c_str(), O_RDONLY | O_DIRECT);
        eng_direct = eng_fd >= 0;
        if (!eng_direct) {
            eng_fd = ::open(f.path().c_str(), O_RDONLY);
            if (eng_fd < 0) { err = "cannot open " + f.path(); return false; }
            ::posix_fadvise(eng_fd, 0, 0, POSIX_FADV_RANDOM);
        }
        if (o.verbose)
            std::fprintf(stderr, "hy4: Engram table read from %s (%s)\n", f.path().c_str(),
                         eng_direct ? "O_DIRECT" : "buffered pread");
        return true;
    }

    /// Reads row `id` (dim bytes) into `out` as floats.
    bool read_row(int64_t id, int64_t dim, float * out, uint8_t * blk) {
        if (eng_ram) {
            const uint8_t * r = eng_ram + (uint64_t) id * (uint64_t) dim;
            for (int64_t j = 0; j < dim; ++j) out[j] = hy4_f8_e4m3_to_float(r[j]);
            return true;
        }
        const uint64_t at = eng_off + (uint64_t) id * (uint64_t) dim;
        uint64_t base = at, len = (uint64_t) dim;
        if (eng_direct) {
            base = at & ~4095ull;
            len = ((at + (uint64_t) dim + 4095) & ~4095ull) - base;
        }
        if (::pread(eng_fd, blk, len, (off_t) base) != (ssize_t) len) return false;
        const uint8_t * r = blk + (at - base);
        for (int64_t j = 0; j < dim; ++j) out[j] = hy4_f8_e4m3_to_float(r[j]);
        return true;
    }

    bool fill_engram(const int64_t * toks, int64_t nt, int64_t first, std::vector<std::vector<float>> & rows,
                     std::string & err) {
        const int64_t cols = L.heads * (L.n_gram - 1), ng1 = L.n_gram - 1, dim = L.dim;
        const int64_t NM = (int64_t) L.modules.size();
        rows.assign((size_t) NM, std::vector<float>((size_t) (dim * cols * nt)));
        // every (token, module, column) row id first, then the reads in parallel
        std::vector<int32_t> ids((size_t) (nt * NM * cols));
        std::vector<float> valid((size_t) cols);
        std::vector<int64_t> prev((size_t) ng1);
        for (int64_t i = 0; i < nt; ++i) {
            const int64_t t = first + i;
            for (int64_t s = 0; s < ng1; ++s) {
                const int64_t p = t - 1 - s;
                prev[(size_t) s] = p < 0 ? -1 : p >= first ? toks[p - first] : hist[(size_t) p];
            }
            for (int64_t e = 0; e < NM; ++e)
                hy4_engram_hash(L, e, toks[i], prev.data(), ids.data() + (size_t) ((i * NM + e) * cols), valid.data());
        }
        const int64_t n = nt * NM * cols;
        const int nth = eng_ram ? (int) std::max<int64_t>(1, std::min<int64_t>(n / 4096, 16))
                                : (int) std::max<int64_t>(1, std::min<int64_t>(n / 8, 32));
        std::vector<std::thread> th;
        std::vector<int> bad((size_t) nth, 0);
        for (int k = 0; k < nth; ++k)
            th.emplace_back([&, k] {
                alignas(4096) static thread_local uint8_t blk[8192];
                for (int64_t q = n * k / nth; q < n * (k + 1) / nth; ++q) {
                    const int64_t i = q / (NM * cols), e = (q / cols) % NM, col = q % cols;
                    float * dst = rows[(size_t) e].data() + (size_t) ((i * cols + col) * dim);
                    if (!read_row(ids[(size_t) q], dim, dst, blk)) { bad[(size_t) k] = 1; return; }
                }
            });
        for (auto & t : th) t.join();
        for (int b : bad) if (b) { err = "Engram row read failed"; return false; }
        return true;
    }

    /// An input the graph does not read (the causal mask once the indexer selects) has no buffer: skipped.
    static void set_in(ggml_tensor * t, const void * data, size_t off, size_t bytes) {
        if (t && t->buffer) ggml_backend_tensor_set(t, data, off, bytes);
    }

    const float * const * img_rows = nullptr;   // decode(): per token of the call, an image row or nullptr
    const int32_t * img_yx = nullptr;            // decode(): per token, the row's (line, column) in its picture, or -1
    int64_t img_at = 0;                           // the call's offset of this step

    // Tencent's get_rope_index: text goes on at the next position; a picture starting at p0 puts its row (y, x) at
    // t = p0, h = p0 + y, w = p0 + x, and the text after it starts past the largest of those
    std::vector<int32_t> rpos;   // [3 per cached position] t, h, w
    int32_t next_pos = 0, img_p0 = 0;

    void recompute_next_pos() {
        next_pos = 0;
        for (int32_t v : rpos) next_pos = std::max(next_pos, v + 1);
    }

    /// Positions of `nt` tokens from `first`, without changing the state unless `commit`.
    void positions(int64_t nt, const int32_t * yx, std::vector<int32_t> & t3, bool commit) {
        int32_t np = next_pos, p0 = img_p0;
        t3.resize((size_t) (3 * nt));
        for (int64_t i = 0; i < nt; ++i) {
            const int32_t y = yx ? yx[2 * i] : -1, x = yx ? yx[2 * i + 1] : -1;
            if (y < 0) {
                t3[3 * i] = t3[3 * i + 1] = t3[3 * i + 2] = np;
                ++np;
            } else {
                if (y == 0 && x == 0) p0 = np;
                t3[3 * i] = p0; t3[3 * i + 1] = p0 + y; t3[3 * i + 2] = p0 + x;
                np = std::max(np, std::max(p0 + y, p0 + x) + 1);
            }
        }
        if (commit) { next_pos = np; img_p0 = p0; }
    }

    bool step(const int64_t * toks, int64_t nt, std::vector<float> & logits, std::string & err, bool all = false,
              bool draft = false) {
        const int64_t first = n_past + (draft ? draft_off : 0), last = first + nt - 1, kp = G.idx_kpool;
        Shape sh = shape(first, nt);
        if (!draft && img_rows)
            for (int64_t i = 0; i < nt; ++i) sh.img = sh.img || img_rows[img_at + i];
        // this step's rotation positions; a draft sits past the trunk's next position
        std::vector<int32_t> t3;
        {
            const int32_t save = next_pos;
            if (draft) next_pos += (int32_t) draft_off;
            positions(nt, draft || !img_yx ? nullptr : img_yx + 2 * img_at, t3, false);
            next_pos = save;
        }
        auto pos3 = [&](int64_t q, int k) -> int32_t {         // position plane k of sequence index q
            return q < first ? rpos[(size_t) (3 * q + k)] : t3[(size_t) (3 * (q - first) + k)];
        };
        for (int64_t i = 0; i < nt; ++i) sh.mrope = sh.mrope || t3[3 * i + 1] != t3[3 * i] || t3[3 * i + 2] != t3[3 * i];
        for (int64_t k = 0; k < sh.n_new; ++k) {
            const int64_t q = (sh.g_new + k) * kp;
            sh.mrope = sh.mrope || pos3(q, 1) != pos3(q, 0) || pos3(q, 2) != pos3(q, 0);
        }
        const int64_t n_kv = sh.n_kv;
        // the Engram rows come from the SSD while the graph is built and allocated
        std::vector<std::vector<float>> rows;
        std::string eng_err;
        std::thread reader([&] {
            if (!draft && !fill_engram(toks, nt, first, rows, eng_err) && eng_err.empty()) eng_err = "Engram";
        });
        ggml_init_params ip = { meta.size(), meta.data(), true };
        ggml_context * c = ggml_init(ip);
        Inputs in;
        ggml_cgraph * gf = draft ? build_draft(c, sh, in) : build(c, sh, in, all);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { reader.join(); ggml_free(c); err = "graph allocation failed"; return false; }

        std::vector<int32_t> tok((size_t) nt), pos((size_t) nt);
        std::vector<int64_t> kidx((size_t) nt), kswa((size_t) nt), ktok((size_t) nt);
        for (int64_t i = 0; i < nt; ++i) {
            const int64_t p = first + i;
            tok[(size_t) i] = toks[i] >= 0 ? (int32_t) toks[i] : 0; pos[(size_t) i] = t3[(size_t) (3 * i)];
            kidx[(size_t) i] = p; kswa[(size_t) i] = p % r_swa; ktok[(size_t) i] = p % r_tok;
        }
        set_in(in.tok, tok.data(), 0, tok.size() * 4);
        set_in(in.pos, pos.data(), 0, pos.size() * 4);
        if (in.pos4) {          // t, h, w, then t again (the plane IMROPE gives the pairs past the sections)
            std::vector<int32_t> p4((size_t) (4 * nt));
            for (int64_t i = 0; i < nt; ++i)
                for (int k = 0; k < 4; ++k) p4[(size_t) (k * nt + i)] = t3[(size_t) (3 * i + (k == 3 ? 0 : k))];
            set_in(in.pos4, p4.data(), 0, p4.size() * 4);
        }
        set_in(in.kidx, kidx.data(), 0, kidx.size() * 8);
        set_in(in.kidx_swa, kswa.data(), 0, kswa.size() * 8);
        set_in(in.kidx_tok, ktok.data(), 0, ktok.size() * 8);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        {   // full layers: causal over the cache rows; SWA layers: the ring, row r holds the last position = r mod R
            std::vector<ggml_fp16_t> m((size_t) (n_kv * nt)), ms((size_t) (r_swa * nt));
            for (int64_t i = 0; i < nt; ++i) {
                const int64_t pt = first + i;
                for (int64_t q = 0; q < n_kv; ++q) m[(size_t) (i * n_kv + q)] = q <= pt ? zero : ninf;
                for (int64_t r = 0; r < r_swa; ++r) {
                    const int64_t q = last - ((last - r) % r_swa + r_swa) % r_swa;     // the position in row r
                    ms[(size_t) (i * r_swa + r)] = q >= 0 && q <= pt && pt - q < G.swa ? zero : ninf;
                }
            }
            set_in(in.mask, m.data(), 0, m.size() * 2);
            set_in(in.mask_swa, ms.data(), 0, ms.size() * 2);
        }
        if (sh.n_new > 0) {     // the groups this ubatch completed: their members' ring rows, their index, first position
            std::vector<int32_t> cells((size_t) (kp * sh.n_new)), npos((size_t) sh.n_new);
            std::vector<int64_t> rep((size_t) sh.n_new);
            for (int64_t k = 0; k < sh.n_new; ++k) {
                const int64_t g = sh.g_new + k;
                for (int64_t j = 0; j < kp; ++j) cells[(size_t) (k * kp + j)] = (int32_t) ((g * kp + j) % r_tok);
                rep[(size_t) k] = g;
                npos[(size_t) k] = pos3(g * kp, 0);     // the group's first member's position
            }
            set_in(in.new_cells, cells.data(), 0, cells.size() * 4);
            set_in(in.new_rep, rep.data(), 0, rep.size() * 8);
            set_in(in.new_pos, npos.data(), 0, npos.size() * 4);
            if (in.new_pos4) {
                std::vector<int32_t> p4((size_t) (4 * sh.n_new));
                for (int64_t k = 0; k < sh.n_new; ++k)
                    for (int j = 0; j < 4; ++j) p4[(size_t) (j * sh.n_new + k)] = pos3((sh.g_new + k) * kp, j == 3 ? 0 : j);
                set_in(in.new_pos4, p4.data(), 0, p4.size() * 4);
            }
        }
        if (sh.index) {         // PR #32517's set_input_kpool for one sequence at positions 0..: pool g = 4g..4g+3
            const int64_t NP = sh.n_pool, NT = sh.n_top, NS = sh.n_sel;
            std::vector<float> pm((size_t) (NP * nt)), live((size_t) (NS * nt)), dump((size_t) NS);
            std::vector<int32_t> pidx((size_t) (kp * NP)), tail((size_t) (kp * nt), 0);
            for (int64_t g = 0; g < NP; ++g) for (int64_t j = 0; j < kp; ++j) pidx[(size_t) (g * kp + j)] = (int32_t) (g * kp + j);
            for (int64_t j = 0; j < NS; ++j) dump[(size_t) j] = (float) (n_kv + j);
            for (int64_t i = 0; i < nt; ++i) {
                const int64_t pt = first + i, nv = pt / kp;            // pools that end before pt (tail is full width)
                for (int64_t g = 0; g < NP; ++g) pm[(size_t) (i * NP + g)] = g < nv ? 0.0f : -INFINITY;
                for (int64_t r = 0; r < NT; ++r)
                    for (int64_t j = 0; j < kp; ++j) live[(size_t) (i * NS + r * kp + j)] = r < nv ? 1.0f : 0.0f;
                if (G.idx_tail) {
                    const int64_t n_tail = pt % kp + 1;
                    for (int64_t k = 0; k < kp; ++k) {
                        const bool real = k < n_tail && pt - k >= 0;
                        tail[(size_t) (i * kp + k)] = real ? (int32_t) (pt - k) : 0;
                        live[(size_t) (i * NS + NT * kp + k)] = real ? 1.0f : 0.0f;
                    }
                }
            }
            std::vector<float> base((size_t) ((n_kv + NS) * nt), -INFINITY), zeros((size_t) (NS * nt), 0.0f);
            set_in(in.pool_mask, pm.data(), 0, pm.size() * 4);
            set_in(in.pool_idx, pidx.data(), 0, pidx.size() * 4);
            set_in(in.tail, tail.data(), 0, tail.size() * 4);
            set_in(in.live, live.data(), 0, live.size() * 4);
            set_in(in.dump, dump.data(), 0, dump.size() * 4);
            set_in(in.mask_base, base.data(), 0, base.size() * 4);
            set_in(in.zeros, zeros.data(), 0, zeros.size() * 4);
        }
        if (sh.img) {           // image rows replace the token embedding; no Engram delta there
            std::vector<float> over((size_t) (G.n_embd * nt), 0.0f), sel((size_t) nt, 0.0f), val((size_t) nt, 1.0f);
            for (int64_t i = 0; i < nt; ++i)
                if (const float * r = img_rows[img_at + i]) {
                    std::copy(r, r + G.n_embd, over.begin() + i * G.n_embd);
                    sel[(size_t) i] = 1.0f;
                    val[(size_t) i] = 0.0f;
                }
            set_in(in.emb_over, over.data(), 0, over.size() * 4);
            set_in(in.emb_sel, sel.data(), 0, sel.size() * 4);
            set_in(in.eng_valid, val.data(), 0, val.size() * 4);
        }
        if (in.hprev) {         // the trunk's state at first-1 (zeros before the start)
            std::vector<float> hp((size_t) G.n_embd, 0.0f);
            if (draft && !draft_h.empty()) hp = draft_h;
            else if (first > 0 && (int64_t) hhist.size() >= first * G.n_embd)
                std::copy(hhist.begin() + (first - 1) * G.n_embd, hhist.begin() + first * G.n_embd, hp.begin());
            set_in(in.hprev, hp.data(), 0, hp.size() * 4);
        }
        reader.join();
        if (!eng_err.empty()) { ggml_free(c); err = eng_err; return false; }
        for (size_t e = 0; e < in.eng.size() && e < rows.size(); ++e) set_in(in.eng[e], rows[e].data(), 0, rows[e].size() * 4);

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); err = "graph compute failed"; return false; }
        if (std::getenv("HY4_SPLITS"))
            std::fprintf(stderr, "hy4: %lld tokens: %d nodes, %d splits, %d copies\n", (long long) nt, ggml_graph_n_nodes(gf),
                         ggml_backend_sched_get_n_splits(sched), ggml_backend_sched_get_n_copies(sched));
        logits.resize((size_t) G.n_vocab * (all ? nt : 1));
        ggml_backend_tensor_get(in.logits, logits.data(), 0, logits.size() * 4);
        if (route_stats && !draft) {
            if (route_count.empty()) route_count.assign((size_t) n_alloc, std::vector<uint64_t>((size_t) G.n_expert, 0));
            std::vector<int32_t> v;
            for (auto & [l, t] : route_ids) {
                if (!t->buffer) continue;
                v.resize((size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
                for (int32_t e : v) if (e >= 0 && e < G.n_expert) ++route_count[(size_t) l][(size_t) e];
            }
        }
        if (draft) {
            draft_h.resize((size_t) G.n_embd);
            ggml_backend_tensor_get(in.h, draft_h.data(), 0, draft_h.size() * 4);
            ggml_free(c);
            return true;
        }
        if (in.h) {
            hhist.resize((size_t) (first * G.n_embd));
            hhist.resize((size_t) ((first + nt) * G.n_embd));
            ggml_backend_tensor_get(in.h, hhist.data() + first * G.n_embd, 0, (size_t) (nt * G.n_embd) * 4);
        }
        ggml_free(c);
        rpos.resize((size_t) (3 * first));
        rpos.insert(rpos.end(), t3.begin(), t3.end());
        positions(nt, !img_yx ? nullptr : img_yx + 2 * img_at, t3, true);
        n_past += nt;
        hist.insert(hist.end(), toks, toks + nt);
        return true;
    }
};

Hy4Engine::Hy4Engine() : p_(std::make_unique<Impl>()) {}
Hy4Engine::~Hy4Engine() = default;

bool Hy4Engine::load(const std::string & gguf, const Hy4EngineOptions & o, std::string & err) {
    Impl & P = *p_;
    P.o = o;
    try { P.model = std::make_unique<GgufModel>(GgufModel::open(gguf)); }
    catch (const std::exception & e) { err = e.what(); return false; }
    if (!(err = hy4_read_geometry(*P.model, P.G)).empty()) return false;
    P.L = hy4_engram_layout(P.G);
    P.n_trunk = P.G.n_layer - P.G.n_nextn;
    P.n_alloc = P.n_trunk + (o.mtp && P.G.n_nextn > 0 ? 1 : 0);
    P.n_ctx = pad_to(std::max<int64_t>(o.n_ctx, KV_PAD), KV_PAD);
    for (int g : o.gpus) {
        ggml_backend_t b = ggml_backend_cuda_init(g);
        if (!b) { err = "cannot start the HIP backend on card " + std::to_string(g); return false; }
        P.gpu.push_back(b);
    }
    if (P.gpu.empty()) { err = "no card given"; return false; }
    P.cpu = ggml_backend_cpu_init();
    const int th = o.threads > 0 ? o.threads : std::max(1u, std::thread::hardware_concurrency() / 2);
    ggml_backend_cpu_set_n_threads(P.cpu, th);
    if (!P.place(err) || !P.upload(err) || !P.open_engram(err)) return false;
    P.warm_cpu_weights();
    P.all = P.gpu;
    P.all.push_back(P.cpu);
    P.sched = ggml_backend_sched_new(P.all.data(), nullptr, (int) P.all.size(), GRAPH_NODES, false, true);
    P.meta.resize(ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false));
    // reserve the compute buffers with the largest graph
    {
        ggml_init_params ip = { P.meta.size(), P.meta.data(), true };
        ggml_context * c = ggml_init(ip);
        Impl::Inputs in;
        const int64_t nt = std::min<int64_t>(o.n_ubatch, P.n_ctx);
        ggml_cgraph * gf = P.build(c, P.shape(P.n_ctx - nt, nt), in);
        const bool ok = ggml_backend_sched_reserve(P.sched, gf);
        ggml_free(c);
        if (!ok) { err = "cannot reserve the compute buffers"; return false; }
    }
    if (o.verbose) {
        std::fprintf(stderr, "hy4: compute buffers:");
        for (auto * b : P.all)
            std::fprintf(stderr, " %s %.0f MiB", ggml_backend_name(b), ggml_backend_sched_get_buffer_size(P.sched, b) / 1048576.0);
        for (size_t g = 0; g < o.gpus.size(); ++g) {
            size_t fr = 0, tot = 0;
            ggml_backend_cuda_get_device_memory(o.gpus[g], &fr, &tot);
            std::fprintf(stderr, "; card %d %.2f GiB free", o.gpus[g], fr / 1073741824.0);
        }
        std::fprintf(stderr, "\n");
    }
    return true;
}

bool Hy4Engine::decode(const int64_t * tokens, int64_t nt, std::vector<float> & logits, std::string & err,
                       const float * const * rows, const int32_t * yx) {
    Impl & P = *p_;
    P.img_rows = rows;
    P.img_yx = rows ? yx : nullptr;
    struct Reset { Impl & p; ~Reset() { p.img_rows = nullptr; p.img_yx = nullptr; p.img_at = 0; } } reset{ P };
    if (P.n_past + nt > P.n_ctx) { err = "context full (" + std::to_string(P.n_ctx) + " tokens)"; return false; }
    for (int64_t done = 0; done < nt;) {
        const int64_t n = std::min<int64_t>(P.o.n_ubatch, nt - done);
        P.img_at = done;
        if (!P.step(tokens + done, n, logits, err)) return false;
        done += n;
    }
    return true;
}

bool Hy4Engine::decode_all(const int64_t * tokens, int64_t nt, std::vector<float> & logits, std::string & err) {
    Impl & P = *p_;
    if (P.n_past + nt > P.n_ctx) { err = "context full (" + std::to_string(P.n_ctx) + " tokens)"; return false; }
    logits.clear();
    std::vector<float> part;
    for (int64_t done = 0; done < nt;) {
        const int64_t n = std::min<int64_t>(P.o.n_ubatch, nt - done);
        if (!P.step(tokens + done, n, part, err, true)) return false;
        logits.insert(logits.end(), part.begin(), part.end());
        done += n;
    }
    return true;
}

void Hy4Engine::truncate(int64_t n) {
    Impl & P = *p_;
    n = std::max<int64_t>(0, std::min(n, P.n_past));
    P.n_past = n;
    P.hist.resize((size_t) n);
    if (!P.hhist.empty()) P.hhist.resize((size_t) (n * P.G.n_embd));
    P.rpos.resize((size_t) (3 * n));
    P.recompute_next_pos();
    P.img_p0 = 0;
}

namespace {

struct SessHead {
    char magic[8] = { 'H', 'Y', '4', 'S', 'E', 'S', 'S', '2' };   // 2: the rotation positions follow the tokens
    int64_t n_past = 0, n_embd = 0, n_alloc = 0, r_swa = 0, r_tok = 0, kpool = 0, n_hh = 0;
};

}  // namespace

bool Hy4Engine::save(const std::string & path, int64_t & tokens, uint64_t & bytes, std::string & err) {
    Impl & P = *p_;
    SessHead h;
    h.n_past = P.n_past; h.n_embd = P.G.n_embd; h.n_alloc = P.n_alloc; h.r_swa = P.r_swa; h.r_tok = P.r_tok;
    h.kpool = P.G.idx_kpool; h.n_hh = (int64_t) std::min(P.hhist.size(), (size_t) (P.n_past * P.G.n_embd));
    const std::string tmp = path + ".tmp";
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "cannot write " + tmp; return false; }
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 &&
              std::fwrite(P.hist.data(), 8, (size_t) h.n_past, f) == (size_t) h.n_past &&
              std::fwrite(P.rpos.data(), 4, (size_t) (3 * h.n_past), f) == (size_t) (3 * h.n_past) &&
              std::fwrite(P.hhist.data(), 4, (size_t) h.n_hh, f) == (size_t) h.n_hh;
    // ring layers: the whole ring (a sliding layer's ring holds only its window)
    std::vector<uint8_t> buf;
    for (size_t l = 0; ok && l < P.kv.size(); ++l) {
        ggml_tensor * parts[3] = { P.kv[l], P.itok[l], P.ipool[l] };
        for (int k = 0; k < 3 && ok; ++k) {
            ggml_tensor * t = parts[k];
            if (!t) continue;
            size_t n = ggml_nbytes(t);
            if (k == 0 && !P.is_swa((int64_t) l)) n = std::min(n, (size_t) t->nb[1] * (size_t) h.n_past);
            if (k == 2) n = std::min(n, (size_t) t->nb[1] * (size_t) (h.n_past / h.kpool));
            buf.resize(n);
            if (n) ggml_backend_tensor_get(t, buf.data(), 0, n);
            const uint64_t n64 = n;
            ok = std::fwrite(&n64, 8, 1, f) == 1 && std::fwrite(buf.data(), 1, n, f) == n;
        }
    }
    ok = std::fflush(f) == 0 && ok;
    bytes = (uint64_t) std::ftell(f);
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); err = "writing " + path + " failed"; return false; }
    tokens = h.n_past;
    return true;
}

bool Hy4Engine::restore(const std::string & path, int64_t & tokens, uint64_t & bytes, std::string & err) {
    Impl & P = *p_;
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot read " + path; return false; }
    SessHead h, want;
    bool ok = std::fread(&h, sizeof h, 1, f) == 1;
    if (!ok || std::memcmp(h.magic, want.magic, 8) || h.n_embd != P.G.n_embd || h.n_alloc != P.n_alloc ||
        h.r_swa != P.r_swa || h.r_tok != P.r_tok || h.kpool != P.G.idx_kpool || h.n_past < 0 || h.n_past > P.n_ctx ||
        h.n_hh != (P.n_alloc > P.n_trunk ? h.n_past * h.n_embd : 0)) {
        std::fclose(f);
        err = "not a session of this model and layout";
        return false;
    }
    std::vector<int64_t> hist((size_t) h.n_past);
    std::vector<int32_t> rp((size_t) (3 * h.n_past));
    std::vector<float> hh((size_t) h.n_hh);
    ok = std::fread(hist.data(), 8, hist.size(), f) == hist.size() && std::fread(rp.data(), 4, rp.size(), f) == rp.size() &&
         std::fread(hh.data(), 4, hh.size(), f) == hh.size();
    std::vector<uint8_t> buf;
    for (size_t l = 0; ok && l < P.kv.size(); ++l) {
        ggml_tensor * parts[3] = { P.kv[l], P.itok[l], P.ipool[l] };
        for (int k = 0; k < 3 && ok; ++k) {
            ggml_tensor * t = parts[k];
            if (!t) continue;
            uint64_t n = 0;
            ok = std::fread(&n, 8, 1, f) == 1 && n <= ggml_nbytes(t);
            if (!ok) break;
            buf.resize((size_t) n);
            ok = std::fread(buf.data(), 1, (size_t) n, f) == n;
            if (ok && n) ggml_backend_tensor_set(t, buf.data(), 0, (size_t) n);
        }
    }
    bytes = (uint64_t) std::ftell(f);
    std::fclose(f);
    if (!ok) { P.n_past = 0; P.hist.clear(); P.hhist.clear(); P.rpos.clear(); P.next_pos = 0; err = "the session file is cut short"; return false; }
    P.n_past = h.n_past;
    P.hist = std::move(hist);
    P.rpos = std::move(rp);
    P.recompute_next_pos();
    P.img_p0 = 0;
    P.hhist = std::move(hh);
    tokens = h.n_past;
    return true;
}

bool Hy4Engine::has_mtp() const { return p_->n_alloc > p_->n_trunk; }

bool Hy4Engine::draft(int64_t token, int64_t & out, std::string & err) {
    std::vector<int64_t> v;
    if (!draft_chain(token, 1, v, err)) return false;
    out = v[0];
    return true;
}

bool Hy4Engine::draft_chain(int64_t token, int n, std::vector<int64_t> & out, std::string & err) {
    Impl & P = *p_;
    if (!has_mtp()) { err = "no MTP block (--mtp)"; return false; }
    out.clear();
    std::vector<float> lg;
    P.draft_h.clear();
    int64_t t = token;
    for (int k = 0; k < n && P.n_past + k + 2 <= P.n_ctx; ++k) {
        P.draft_off = k;
        const bool ok = P.step(&t, 1, lg, err, false, true);
        if (!ok) { P.draft_off = 0; P.draft_h.clear(); return false; }
        t = (int64_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        out.push_back(t);
    }
    P.draft_off = 0;
    P.draft_h.clear();
    if (out.empty()) { err = "context full"; return false; }
    return true;
}

int64_t Hy4Engine::n_past() const { return p_->n_past; }
int64_t Hy4Engine::n_ctx() const { return p_->n_ctx; }
const Hy4Geometry & Hy4Engine::geometry() const { return p_->G; }
const std::vector<int64_t> & Hy4Engine::history() const { return p_->hist; }

}  // namespace almond::hy4

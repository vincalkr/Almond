// src/qwen35/qwen35.cpp - the dense Qwen3.8-27B path of almond-engine (GGUF architecture `qwen35`): 64 layers, three
// Gated DeltaNet layers to one gated full-attention layer, a dense SwiGLU FFN, and the GGUF's own nextn (MTP) block.
// It runs on ggml's graphs (the vendored third_party/llama.cpp/ggml, HIP backend), one card or a layer split over
// two, and speaks the serve protocol of the other paths (GEN / T / PP / DONE / STOP), so serve/server.py drives it
// unchanged.  almond-engine runs it when its arguments name --qwen35 (src/engine_main.cpp).
//
//   almond-engine --qwen35 -m <27B.gguf> [--gpus 0[,1]] [--max-context N] [--kv f16|q8_0] [--ubatch N]
//                 [--mtp N] [--spec-min-p P] [--dflash <DFlash2 GGUF> [--dflash-n N]] [--spec auto]
//                 [--serve | --bench N | --ppl ids.i32 --ppl-ctx N]
//
// The recurrent state: each DeltaNet layer keeps its conv window (the last W columns of the conv input, W = KSNAP+2)
// and KSNAP snapshots of its SSM state (ggml_gated_delta_net with K = KSNAP: slot s = the state s tokens before the
// end of the window).  A verify window of n tokens of which a are kept leaves the state in slot n-a and the conv
// state at rows W-(n+3)+a.. (W = KSNAP+3); the next graph reads them through get_rows with those indices as inputs,
// so a rejected draft costs no copy and the graph stays the same (each graph shape keeps its own scheduler and runs
// again with new inputs).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <deque>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "qwen35: %s\n", m.c_str());
    std::printf("ERR %s\n", m.c_str());
    std::fflush(stdout);
    std::exit(1);
}

constexpr int KSNAP = 8;          // SSM snapshots per DeltaNet layer: a verify window of at most KSNAP tokens
constexpr int CONV_W = KSNAP + 3; // conv columns kept (3 of state + KSNAP of rollback)
constexpr int GRAPH_NODES = 16384;

struct Hp {
    int n_layer = 0, n_nextn = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int ssm_conv = 4, ssm_state = 128, ssm_groups = 16, ssm_vheads = 48, ssm_inner = 6144, full_every = 4;
    int sections[4] = {0, 0, 0, 0};
    float eps = 1e-6f, rope_base = 1e7f;
    int n_vocab = 0, n_ctx_train = 0;
    bool recurrent(int il) const { return il < n_layer && (il + 1) % full_every != 0; }
    int conv_ch() const { return ssm_inner + 2 * ssm_groups * ssm_state; }
    int64_t ssm_d() const { return (int64_t) ssm_state * ssm_state * ssm_vheads; }
    int kv_dim() const { return head_dim * n_head_kv; }
};

struct Layer {
    ggml_tensor *attn_norm = nullptr, *post_norm = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *wqkv = nullptr, *wgate = nullptr, *conv1d = nullptr, *dt = nullptr, *ssm_a = nullptr, *beta = nullptr,
                *alpha = nullptr, *ssm_norm = nullptr, *ssm_out = nullptr;
    ggml_tensor *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
    // nextn
    ggml_tensor *eh_proj = nullptr, *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    // state
    ggml_tensor *k_cache = nullptr, *v_cache = nullptr;   // attention: [kv_dim, n_ctx]
    ggml_tensor *conv = nullptr, *ssm = nullptr;          // DeltaNet: [C, CONV_W] (a row per time step), [D, KSNAP]
    int dev = 0;
};

struct Opts {
    std::string model;
    std::vector<int> gpus = {0};
    int split = -1;                 // first layer of the second card (-1: by free VRAM)
    int64_t max_ctx = 32768;
    ggml_type kv_type = GGML_TYPE_Q8_0;   // q8_0 by default (f16 with --kv f16)
    int ubatch = 1024;
    int mtp = 3;                    // MTP drafts per round (0: off)
    float spec_min_p = 0.0f;        // stop drafting when the draft's top probability is below this
    std::string dflash;             // --dflash <DFlash2 GGUF>: block drafts in place of the MTP block
    int dflash_n = 5;               // drafts per block (block = 1 + this, at most the drafter's block size - 1):
                                    // 5 measured best over code / technical / free prose (7 only on code)
    bool spec_auto = false;         // --spec auto (with --dflash and --mtp): per round the drafter whose recent
                                    // tokens per millisecond are higher; the other one is tried every few rounds
    bool serve = false;
    int bench = 0;
    std::string ppl_file;
    int ppl_ctx = 2048;
    std::string prompt_ids;         // --bench: the prompt's ids (comma list), default a fixed short one
    std::vector<int32_t> eos = {248044, 248046};
};

// The DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2): five non-causal layers over a block [last token, MASK x n]
// whose K/V context is the trunk's residual stream at five of its layers (fc + norm, injected at every position the
// trunk reads), two-tap dynamic convolutions around attention and FFN, the trunk's embeddings and head, and a
// selector that scores top-16 candidates per block position with pairwise transitions (walked on the CPU).
//
// DSpine (tencent/Qwen3.8-27B-DSpine-b16) is the same frame without the convolutions and the selector: causal inside
// the block, an 8192-token window, and after every layer a "code bridge" (each row reads a unit code from its hidden
// state, the block's first row takes its token's codebook row, and row t gets a gated message from row t-1's code);
// its candidates are scored by redoing the last layer's bridge once per exact predecessor candidate.
struct DfLayer {
    ggml_tensor *attn_norm, *wq, *wk, *wv, *wo, *q_norm, *k_norm;
    ggml_tensor *attn_conv_base = nullptr, *attn_conv_proj = nullptr;
    ggml_tensor *ffn_norm, *ffn_gate, *ffn_up, *ffn_down;
    ggml_tensor *ffn_conv_base = nullptr, *ffn_conv_proj = nullptr;
    ggml_tensor *sp_read = nullptr, *sp_write = nullptr;   // DSpine: [n_embd -> code], [rank -> n_embd]
    ggml_tensor *k_ring = nullptr, *v_ring = nullptr;      // [kv_dim, ring] f16
};
struct Drafter {
    gguf_context* gguf = nullptr;
    ggml_context* meta = nullptr;
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, block = 8, conv_k = 2, conv_g = 16,
        rank = 256, top_k = 16, swa = 2048;
    int ring = 4096;          // the KV ring's cells, by position: at least the window + a block + a verify window
    bool dspine = false;      // DSpine (code bridge, causal block) rather than DFlash2 (convolutions, selector)
    bool causal = false;
    int sp_code = 0, sp_rank = 0;
    ggml_tensor *sp_codebook = nullptr, *sp_gate = nullptr, *sp_gate_b = nullptr, *sp_pred_gate = nullptr,
                *sp_prev = nullptr;
    float eps = 1e-6f, rope_base = 1e7f;
    int32_t mask_id = 248070;
    std::vector<int> target_layers;
    ggml_tensor *fc = nullptr, *enc_norm = nullptr, *out_norm = nullptr, *sel_hidden = nullptr, *sel_prev = nullptr,
                *sel_next = nullptr;
    std::vector<DfLayer> L;
    std::vector<int64_t> ring_pos;   // the position each ring cell holds (-1: none)
    int kv_dim() const { return head_dim * n_head_kv; }
};

struct Model {
    Hp hp;
    std::unique_ptr<Drafter> df;
    Opts o;
    gguf_context* gguf = nullptr;
    ggml_context* meta = nullptr;
    std::vector<ggml_backend_t> gpu;
    ggml_backend_t cpu = nullptr;
    std::vector<ggml_context*> wctx;         // per device (+ CPU last)
    std::vector<ggml_backend_buffer_t> bufs;
    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *output = nullptr;
    std::vector<Layer> L;                     // n_layer + n_nextn
    // persistent tensors on the last card
    ggml_tensor *hbuf = nullptr;              // [E, hb_cols+1]: the trunk's h_nextn rows of its last window in
                                              // columns 0.., and in the last column (pend_col()) the h of the last
                                              // token whose MTP pair is not made yet
    ggml_tensor *mtph = nullptr;              // [E, 1]: the MTP block's own h_nextn of its last row
    int hb_cols = 0;
    int pend_col() const { return hb_cols; }
    std::vector<ggml_backend_t> bk;           // the cards, then the CPU: each graph's scheduler runs on these
    // recurrent state position (same in every DeltaNet layer)
    int conv_off = CONV_W - 3, ssm_slot = 0;
};

// ---------------------------------------------------------------------------------------------------------------- load
uint32_t kv_u32(gguf_context* g, const std::string& k, bool req = true, uint32_t def = 0) {
    int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) {
        if (req) die("the GGUF has no " + k);
        return def;
    }
    return gguf_get_val_u32(g, id);
}
float kv_f32(gguf_context* g, const std::string& k) {
    int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) die("the GGUF has no " + k);
    return gguf_get_val_f32(g, id);
}

ggml_tensor* need(Model& m, const std::string& name, bool req = true) {
    ggml_tensor* t = ggml_get_tensor(m.meta, name.c_str());
    if (!t && req) die("the GGUF has no tensor " + name);
    return t;
}

size_t free_vram(int dev) {
    size_t fr = 0, tot = 0;
    ggml_backend_cuda_get_device_memory(dev, &fr, &tot);
    return fr;
}

void load(Model& m) {
    const auto t0 = Clock::now();
    gguf_init_params gp = {true, &m.meta};
    m.gguf = gguf_init_from_file(m.o.model.c_str(), gp);
    if (!m.gguf) die("cannot read " + m.o.model);
    {
        int64_t id = gguf_find_key(m.gguf, "general.architecture");
        std::string arch = id >= 0 ? gguf_get_val_str(m.gguf, id) : "";
        if (arch != "qwen35") die(m.o.model + " is '" + arch + "', not qwen35 (a dense Qwen3.5/3.8 model)");
    }
    Hp& h = m.hp;
    const int blocks = (int) kv_u32(m.gguf, "qwen35.block_count");
    h.n_nextn = (int) kv_u32(m.gguf, "qwen35.nextn_predict_layers", false, 0);
    h.n_layer = blocks - h.n_nextn;
    h.n_embd = (int) kv_u32(m.gguf, "qwen35.embedding_length");
    h.n_ff = (int) kv_u32(m.gguf, "qwen35.feed_forward_length");
    h.n_head = (int) kv_u32(m.gguf, "qwen35.attention.head_count");
    h.n_head_kv = (int) kv_u32(m.gguf, "qwen35.attention.head_count_kv");
    h.head_dim = (int) kv_u32(m.gguf, "qwen35.attention.key_length");
    h.n_rot = (int) kv_u32(m.gguf, "qwen35.rope.dimension_count");
    h.ssm_conv = (int) kv_u32(m.gguf, "qwen35.ssm.conv_kernel");
    h.ssm_state = (int) kv_u32(m.gguf, "qwen35.ssm.state_size");
    h.ssm_groups = (int) kv_u32(m.gguf, "qwen35.ssm.group_count");
    h.ssm_vheads = (int) kv_u32(m.gguf, "qwen35.ssm.time_step_rank");
    h.ssm_inner = (int) kv_u32(m.gguf, "qwen35.ssm.inner_size");
    h.full_every = (int) kv_u32(m.gguf, "qwen35.full_attention_interval", false, 4);
    h.n_ctx_train = (int) kv_u32(m.gguf, "qwen35.context_length");
    h.eps = kv_f32(m.gguf, "qwen35.attention.layer_norm_rms_epsilon");
    h.rope_base = kv_f32(m.gguf, "qwen35.rope.freq_base");
    {
        int64_t id = gguf_find_key(m.gguf, "qwen35.rope.dimension_sections");
        if (id < 0) die("the GGUF has no qwen35.rope.dimension_sections");
        const int n = (int) gguf_get_arr_n(m.gguf, id);
        const int32_t* s = (const int32_t*) gguf_get_arr_data(m.gguf, id);
        for (int i = 0; i < 4 && i < n; ++i) h.sections[i] = s[i];
    }
    if (h.ssm_conv != 4) die("conv kernel " + std::to_string(h.ssm_conv) + " (this path is written for 4)");
    if (h.ssm_inner != h.ssm_state * h.ssm_vheads) die("ssm.inner_size is not state_size x time_step_rank");
    m.tok_embd = need(m, "token_embd.weight");
    h.n_vocab = (int) m.tok_embd->ne[1];
    if (m.o.mtp > 0 && h.n_nextn < 1) {
        std::fprintf(stderr, "qwen35: the GGUF has no nextn (MTP) block: MTP is off\n");
        m.o.mtp = 0;
    }
    m.o.mtp = std::min(m.o.mtp, KSNAP - 1);

    // ---- backends
    for (int g : m.o.gpus) {
        ggml_backend_t b = ggml_backend_cuda_init(g);
        if (!b) die("cannot start the HIP backend on card " + std::to_string(g));
        m.gpu.push_back(b);
    }
    m.cpu = ggml_backend_cpu_init();
    const int ng = (int) m.gpu.size();

    // ---- which card each layer is on
    m.L.resize(h.n_layer + h.n_nextn);
    int split = h.n_layer;
    if (ng == 2) {
        if (m.o.split > 0) split = m.o.split;
        else {
            // weight bytes per layer from the file, KV and state per layer: split by free memory
            const double f0 = (double) free_vram(m.o.gpus[0]), f1 = (double) free_vram(m.o.gpus[1]);
            split = (int) std::lround(h.n_layer * f0 / (f0 + f1));
            split -= 2;   // card 0 also holds the compute buffers of the larger half
        }
        split = std::clamp(split, 1, h.n_layer - 1);
    }
    for (int il = 0; il < (int) m.L.size(); ++il) m.L[il].dev = (ng == 2 && il >= split) ? 1 : 0;
    const int last = ng - 1;

    // ---- weight contexts: one per card, one for the CPU (the token table: its get_rows runs there)
    m.wctx.resize(ng + 1);
    for (auto& c : m.wctx) {
        ggml_init_params ip = {ggml_tensor_overhead() * 2048, nullptr, true};
        c = ggml_init(ip);
    }
    auto place = [&](const std::string& name, int dev, bool req = true) -> ggml_tensor* {
        ggml_tensor* t = need(m, name, req);
        if (!t) return nullptr;
        ggml_tensor* d = ggml_dup_tensor(m.wctx[dev < 0 ? ng : dev], t);
        ggml_set_name(d, name.c_str());
        return d;
    };
    m.tok_embd = place("token_embd.weight", -1);
    m.out_norm = place("output_norm.weight", last);
    m.output = place("output.weight", last, false);
    if (!m.output) die("the GGUF has no output.weight (tied embeddings are not supported here)");
    for (int il = 0; il < (int) m.L.size(); ++il) {
        Layer& l = m.L[il];
        const int d = l.dev;
        auto P = [&](const char* suf, bool req = true) {
            return place("blk." + std::to_string(il) + "." + suf, d, req);
        };
        l.attn_norm = P("attn_norm.weight");
        l.post_norm = P("post_attention_norm.weight");
        if (il >= h.n_layer || !h.recurrent(il)) {
            l.wq = P("attn_q.weight");
            l.wk = P("attn_k.weight");
            l.wv = P("attn_v.weight");
            l.wo = P("attn_output.weight");
            l.q_norm = P("attn_q_norm.weight");
            l.k_norm = P("attn_k_norm.weight");
        } else {
            l.wqkv = P("attn_qkv.weight");
            l.wgate = P("attn_gate.weight");
            l.conv1d = P("ssm_conv1d.weight");
            l.dt = P("ssm_dt.bias");
            l.ssm_a = P("ssm_a");
            l.beta = P("ssm_beta.weight");
            l.alpha = P("ssm_alpha.weight");
            l.ssm_norm = P("ssm_norm.weight");
            l.ssm_out = P("ssm_out.weight");
        }
        l.ffn_gate = P("ffn_gate.weight");
        l.ffn_up = P("ffn_up.weight");
        l.ffn_down = P("ffn_down.weight");
        if (il >= h.n_layer) {
            l.eh_proj = P("nextn.eh_proj.weight");
            l.enorm = P("nextn.enorm.weight");
            l.hnorm = P("nextn.hnorm.weight");
            l.head_norm = P("nextn.shared_head_norm.weight", false);
            if (P("nextn.embed_tokens.weight", false) || P("nextn.shared_head_head.weight", false))
                die("this MTP block has its own embeddings or head, which this path does not read yet");
        }
    }
    // ---- allocate and read
    for (int d = 0; d <= ng; ++d) {
        ggml_backend_buffer_type_t bt = d < ng ? ggml_backend_get_default_buffer_type(m.gpu[d])
                                               : ggml_backend_cuda_host_buffer_type();
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(m.wctx[d], bt);
        if (!b) die("cannot allocate the weights of " + std::string(d < ng ? "card " + std::to_string(m.o.gpus[d]) : "the host"));
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        m.bufs.push_back(b);
    }
    const int fd = ::open(m.o.model.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open " + m.o.model);
    const size_t data_off = gguf_get_data_offset(m.gguf);
    std::vector<uint8_t> stage(64u << 20);
    size_t total = 0;
    for (int d = 0; d <= ng; ++d)
        for (ggml_tensor* t = ggml_get_first_tensor(m.wctx[d]); t; t = ggml_get_next_tensor(m.wctx[d], t)) {
            const int64_t id = gguf_find_tensor(m.gguf, ggml_get_name(t));
            const size_t off = data_off + gguf_get_tensor_offset(m.gguf, id), n = ggml_nbytes(t);
            for (size_t done = 0; done < n;) {
                const size_t c = std::min(stage.size(), n - done);
                ssize_t r = ::pread(fd, stage.data(), c, (off_t) (off + done));
                if (r != (ssize_t) c) die("short read of " + std::string(ggml_get_name(t)));
                ggml_backend_tensor_set(t, stage.data(), done, c);
                done += c;
            }
            total += n;
        }
    ::close(fd);

    // ---- state: KV caches, DeltaNet windows, the MTP rows
    m.hb_cols = std::max(m.o.ubatch, KSNAP) + 1;
    std::vector<ggml_context*> sctx(ng);
    for (auto& c : sctx) {
        ggml_init_params ip = {ggml_tensor_overhead() * 1024, nullptr, true};
        c = ggml_init(ip);
    }
    const int64_t ctx_n = m.o.max_ctx;
    for (int il = 0; il < (int) m.L.size(); ++il) {
        Layer& l = m.L[il];
        ggml_context* c = sctx[l.dev];
        if (l.wq) {
            l.k_cache = ggml_new_tensor_2d(c, m.o.kv_type, h.kv_dim(), ctx_n);
            l.v_cache = ggml_new_tensor_2d(c, m.o.kv_type, h.kv_dim(), ctx_n);
        } else {
            l.conv = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.conv_ch(), CONV_W);
            l.ssm = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.ssm_d(), KSNAP);
        }
    }
    m.hbuf = ggml_new_tensor_2d(sctx[last], GGML_TYPE_F32, h.n_embd, m.hb_cols + 1);
    m.mtph = ggml_new_tensor_2d(sctx[last], GGML_TYPE_F32, h.n_embd, 1);
    size_t state_bytes = 0;
    for (int d = 0; d < ng; ++d) {
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(sctx[d], m.gpu[d]);
        if (!b) die("cannot allocate the KV cache and state on card " + std::to_string(m.o.gpus[d]) +
                    " (lower --max-context or use --kv q8_0)");
        ggml_backend_buffer_clear(b, 0);
        state_bytes += ggml_backend_buffer_get_size(b);
        m.bufs.push_back(b);
    }

    m.bk = m.gpu;
    m.bk.push_back(m.cpu);
    std::fprintf(stderr,
                 "qwen35: %d layers (+%d nextn), embd %d, ff %d, heads %d/%d, vocab %d; weights %.2f GiB read in %.1f s; "
                 "KV %s x %lld + state %.2f GiB; %s\n",
                 h.n_layer, h.n_nextn, h.n_embd, h.n_ff, h.n_head, h.n_head_kv, h.n_vocab, total / 1073741824.0,
                 ms_since(t0) / 1000, ggml_type_name(m.o.kv_type), (long long) ctx_n, state_bytes / 1073741824.0,
                 ng == 2 ? ("layers 0-" + std::to_string(split - 1) + " on card " + std::to_string(m.o.gpus[0]) +
                            ", the rest on card " + std::to_string(m.o.gpus[1])).c_str()
                         : ("card " + std::to_string(m.o.gpus[0])).c_str());
}

void load_drafter(Model& m) {
    auto d = std::make_unique<Drafter>();
    gguf_init_params gp = {true, &d->meta};
    d->gguf = gguf_init_from_file(m.o.dflash.c_str(), gp);
    if (!d->gguf) die("cannot read " + m.o.dflash);
    {
        int64_t id = gguf_find_key(d->gguf, "general.architecture");
        std::string arch = id >= 0 ? gguf_get_val_str(d->gguf, id) : "";
        if (arch != "dflash") die(m.o.dflash + " is '" + arch + "', not a DFlash drafter");
    }
    auto u = [&](const char* k, bool req = true, uint32_t def = 0) { return kv_u32(d->gguf, k, req, def); };
    d->n_layer = (int) u("dflash.block_count");
    d->n_embd = (int) u("dflash.embedding_length");
    d->n_head = (int) u("dflash.attention.head_count");
    d->n_head_kv = (int) u("dflash.attention.head_count_kv");
    d->head_dim = (int) u("dflash.attention.key_length");
    d->block = (int) u("dflash.block_size");
    d->conv_k = (int) u("dflash.conv_kernel_size", false, 2);
    d->conv_g = (int) u("dflash.conv_group_size", false, 16);
    d->rank = (int) u("dflash.selector_rank", false, 0);
    d->top_k = (int) u("dflash.selector_top_k");
    d->swa = (int) u("dflash.attention.sliding_window", false, 2048);
    d->eps = kv_f32(d->gguf, "dflash.attention.layer_norm_rms_epsilon");
    d->rope_base = kv_f32(d->gguf, "dflash.rope.freq_base");
    {
        int64_t id = gguf_find_key(d->gguf, "tokenizer.ggml.mask_token_id");
        if (id >= 0) d->mask_id = (int32_t) gguf_get_val_u32(d->gguf, id);
        id = gguf_find_key(d->gguf, "dflash.target_layers");
        if (id < 0) die("the drafter has no dflash.target_layers");
        const int n = (int) gguf_get_arr_n(d->gguf, id);
        const int32_t* v = (const int32_t*) gguf_get_arr_data(d->gguf, id);
        d->target_layers.assign(v, v + n);
    }
    d->dspine = ggml_get_tensor(d->meta, "dspine_codebook.weight") != nullptr;
    {
        const int64_t id = gguf_find_key(d->gguf, "dflash.attention.causal");
        d->causal = id >= 0 && gguf_get_val_bool(d->gguf, id);
    }
    if (d->n_embd != m.hp.n_embd) die("the drafter's width is not the model's");
    if (d->top_k < 2 || d->top_k > 64) die("the drafter's top-k is outside 2..64");
    if (!d->dspine && d->conv_k != 2) die("a DFlash2 drafter with conv kernel 2 is expected");
    d->ring = 4096;
    while (d->ring < d->swa + d->block + KSNAP) d->ring *= 2;
    for (int t : d->target_layers)
        if (t < 0 || t >= m.hp.n_layer) die("the drafter reads a layer the model does not have");
    m.o.dflash_n = std::clamp(m.o.dflash_n, 1, std::min(d->block - 1, KSNAP - 1));
    const int dev = (int) m.gpu.size() - 1;
    ggml_init_params ip = {ggml_tensor_overhead() * 256, nullptr, true};
    ggml_context* wc = ggml_init(ip);
    auto P = [&](const std::string& name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(d->meta, name.c_str());
        if (!t) die("the drafter has no tensor " + name);
        ggml_tensor* x = ggml_dup_tensor(wc, t);
        ggml_set_name(x, name.c_str());
        return x;
    };
    d->fc = P("fc.weight");
    d->enc_norm = P("enc.output_norm.weight");
    d->out_norm = P("output_norm.weight");
    if (d->dspine) {
        d->sp_codebook = P("dspine_codebook.weight");
        d->sp_gate = P("dspine_gate.weight");
        d->sp_gate_b = P("dspine_gate.bias");
        d->sp_pred_gate = P("dspine_pred_gate.weight");
        d->sp_prev = P("dspine_prev.weight");
        d->sp_code = (int) d->sp_codebook->ne[0];
        d->sp_rank = (int) d->sp_gate->ne[1];
    } else {
        d->sel_hidden = P("selector_hidden.weight");
        d->sel_prev = P("selector_predecessor.weight");
        d->sel_next = P("selector_successor.weight");
    }
    d->L.resize(d->n_layer);
    for (int il = 0; il < d->n_layer; ++il) {
        DfLayer& l = d->L[il];
        auto B = [&](const char* suf) { return P("blk." + std::to_string(il) + "." + suf); };
        l.attn_norm = B("attn_norm.weight");
        l.wq = B("attn_q.weight");
        l.wk = B("attn_k.weight");
        l.wv = B("attn_v.weight");
        l.wo = B("attn_output.weight");
        l.q_norm = B("attn_q_norm.weight");
        l.k_norm = B("attn_k_norm.weight");
        if (!d->dspine) {
            l.attn_conv_base = B("attn_conv_base");
            l.attn_conv_proj = B("attn_conv_proj.weight");
            l.ffn_conv_base = B("ffn_conv_base");
            l.ffn_conv_proj = B("ffn_conv_proj.weight");
        } else {
            l.sp_read = B("dspine_read.weight");
            l.sp_write = B("dspine_write.weight");
        }
        l.ffn_norm = B("ffn_norm.weight");
        l.ffn_gate = B("ffn_gate.weight");
        l.ffn_up = B("ffn_up.weight");
        l.ffn_down = B("ffn_down.weight");
    }
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(wc, ggml_backend_get_default_buffer_type(m.gpu[dev]));
    if (!b) die("cannot allocate the drafter's weights");
    ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.bufs.push_back(b);
    const int fd = ::open(m.o.dflash.c_str(), O_RDONLY);
    const size_t off0 = gguf_get_data_offset(d->gguf);
    std::vector<uint8_t> stage;
    size_t total = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(wc); t; t = ggml_get_next_tensor(wc, t)) {
        const size_t off = off0 + gguf_get_tensor_offset(d->gguf, gguf_find_tensor(d->gguf, ggml_get_name(t)));
        stage.resize(ggml_nbytes(t));
        if (::pread(fd, stage.data(), stage.size(), (off_t) off) != (ssize_t) stage.size())
            die("short read of the drafter's " + std::string(ggml_get_name(t)));
        ggml_backend_tensor_set(t, stage.data(), 0, stage.size());
        total += stage.size();
    }
    ::close(fd);
    ggml_init_params sp = {ggml_tensor_overhead() * 64, nullptr, true};
    ggml_context* sc = ggml_init(sp);
    for (DfLayer& l : d->L) {
        l.k_ring = ggml_new_tensor_2d(sc, GGML_TYPE_F16, d->kv_dim(), d->ring);
        l.v_ring = ggml_new_tensor_2d(sc, GGML_TYPE_F16, d->kv_dim(), d->ring);
    }
    ggml_backend_buffer_t rb = ggml_backend_alloc_ctx_tensors(sc, m.gpu[dev]);
    if (!rb) die("cannot allocate the drafter's KV ring");
    ggml_backend_buffer_clear(rb, 0);
    m.bufs.push_back(rb);
    d->ring_pos.assign(d->ring, -1);
    std::fprintf(stderr, "qwen35: %s drafter %d layers, block %d (%d drafts), top-%d %s, window %d (ring %d), reads "
                         "layers", d->dspine ? "DSpine" : "DFlash2", d->n_layer, d->block, m.o.dflash_n, d->top_k,
                 d->dspine ? "lattice" : "selector", d->swa, d->ring);
    for (int t : d->target_layers) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "; %.2f GiB\n", total / 1073741824.0);
    m.df = std::move(d);
    if (!m.o.spec_auto) m.o.mtp = 0;   // one drafter at a time, unless --spec auto keeps both
    else if (m.o.mtp <= 0) m.o.spec_auto = false;
}

// ---------------------------------------------------------------------------------------------------------------- graphs
ggml_tensor* rms(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

struct Inputs {
    ggml_tensor *tok = nullptr, *pos = nullptr, *mask = nullptr, *kv_idx = nullptr, *out_ids = nullptr;
    ggml_tensor *slot = nullptr, *cidx = nullptr;   // DeltaNet: the SSM slot [1] and the conv rows [3] to start from
    ggml_tensor *hidx = nullptr, *pidx = nullptr;   // MTP: the hbuf columns of its h rows [r], the next pending [1]
    ggml_tensor *ring_idx = nullptr, *anchor = nullptr;   // DFlash2: ring cells of the rows [n], the anchor id x top_k
    ggml_tensor *img = nullptr, *img_idx = nullptr;        // images: the encoder's rows [E, ni] and the rows they replace
};

// One gated full-attention block (trunk or nextn).  x: [E, n] normed.  Writes K/V rows at kv_idx, attends over the
// first n_kv cells with the mask.
ggml_tensor* attn_block(Model& m, ggml_context* c, ggml_cgraph* gf, Layer& l, ggml_tensor* x, const Inputs& in, int n,
                        int n_kv) {
    const Hp& h = m.hp;
    const int D = h.head_dim;
    ggml_tensor* qf = ggml_mul_mat(c, l.wq, x);   // [2*D*H, n]: per head the query then its gate
    ggml_tensor* q = ggml_view_3d(c, qf, D, h.n_head, n, ggml_element_size(qf) * D * 2,
                                  ggml_element_size(qf) * D * 2 * h.n_head, 0);
    ggml_tensor* gate = ggml_view_3d(c, qf, D, h.n_head, n, ggml_element_size(qf) * D * 2,
                                     ggml_element_size(qf) * D * 2 * h.n_head, ggml_element_size(qf) * D);
    gate = ggml_cont_2d(c, gate, D * h.n_head, n);
    q = rms(c, q, l.q_norm, h.eps);
    ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, l.wk, x), D, h.n_head_kv, n);
    k = rms(c, k, l.k_norm, h.eps);
    ggml_tensor* v = ggml_reshape_3d(c, ggml_mul_mat(c, l.wv, x), D, h.n_head_kv, n);
    int sec[4] = {h.sections[0], h.sections[1], h.sections[2], h.sections[3]};
    q = ggml_rope_multi(c, q, in.pos, nullptr, h.n_rot, sec, GGML_ROPE_TYPE_IMROPE, h.n_ctx_train, h.rope_base, 1.0f,
                        0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_multi(c, k, in.pos, nullptr, h.n_rot, sec, GGML_ROPE_TYPE_IMROPE, h.n_ctx_train, h.rope_base, 1.0f,
                        0.0f, 1.0f, 32.0f, 1.0f);
    ggml_build_forward_expand(gf, ggml_set_rows(c, l.k_cache, ggml_reshape_2d(c, k, h.kv_dim(), n), in.kv_idx));
    ggml_build_forward_expand(gf, ggml_set_rows(c, l.v_cache, ggml_reshape_2d(c, v, h.kv_dim(), n), in.kv_idx));
    const size_t row = ggml_row_size(m.o.kv_type, h.kv_dim()), hrow = ggml_row_size(m.o.kv_type, D);
    ggml_tensor* kk = ggml_view_3d(c, l.k_cache, D, h.n_head_kv, n_kv, hrow, row, 0);
    ggml_tensor* vv = ggml_view_3d(c, l.v_cache, D, h.n_head_kv, n_kv, hrow, row, 0);
    kk = ggml_permute(c, kk, 0, 2, 1, 3);
    vv = ggml_permute(c, vv, 0, 2, 1, 3);
    q = ggml_permute(c, q, 0, 2, 1, 3);
    ggml_tensor* o = ggml_flash_attn_ext(c, q, kk, vv, in.mask, 1.0f / std::sqrt((float) D), 0.0f, 0.0f);
    o = ggml_reshape_2d(c, o, D * h.n_head, n);
    o = ggml_mul(c, o, ggml_sigmoid(c, gate));
    return ggml_mul_mat(c, l.wo, o);
}

ggml_tensor* ffn(ggml_context* c, Layer& l, ggml_tensor* x) {
    ggml_tensor* g = ggml_mul_mat(c, l.ffn_gate, x);
    ggml_tensor* u = ggml_mul_mat(c, l.ffn_up, x);
    return ggml_mul_mat(c, l.ffn_down, ggml_swiglu_split(c, g, u));
}

// One Gated DeltaNet block.  Reads the conv state at rows in.cidx and the SSM state in slot in.slot; writes the conv
// window's tail and min(n, ksnap) snapshots.
ggml_tensor* gdn_block(Model& m, ggml_context* c, ggml_cgraph* gf, Layer& l, ggml_tensor* x, const Inputs& in, int n,
                       int ksnap) {
    const Hp& h = m.hp;
    const int S = h.ssm_state, Hk = h.ssm_groups, Hv = h.ssm_vheads, C = h.conv_ch();
    const int64_t Dst = h.ssm_d();
    ggml_tensor* qkv = ggml_mul_mat(c, l.wqkv, x);                  // [C, n]
    ggml_tensor* z = ggml_mul_mat(c, l.wgate, x);                   // [inner, n]
    ggml_tensor* beta = ggml_sigmoid(c, ggml_mul_mat(c, l.beta, x));
    beta = ggml_reshape_4d(c, beta, 1, Hv, n, 1);
    ggml_tensor* g = ggml_mul_mat(c, l.alpha, x);                   // [Hv, n]
    g = ggml_mul(c, ggml_softplus(c, ggml_add(c, g, l.dt)), l.ssm_a);
    g = ggml_reshape_4d(c, g, 1, Hv, n, 1);

    ggml_tensor* cst = ggml_transpose(c, ggml_get_rows(c, l.conv, in.cidx));   // [3, C]
    ggml_tensor* cin = ggml_concat(c, cst, ggml_transpose(c, qkv), 0);          // [3+n, C]
    const int mcols = std::min(3 + n, CONV_W);
    ggml_build_forward_expand(
        gf, ggml_cpy(c, ggml_transpose(c, ggml_view_2d(c, cin, mcols, C, cin->nb[1],
                                                       (size_t) (3 + n - mcols) * sizeof(float))),
                     ggml_view_2d(c, l.conv, C, mcols, l.conv->nb[1], (size_t) (CONV_W - mcols) * l.conv->nb[1])));
    ggml_tensor* co = ggml_silu(c, ggml_ssm_conv(c, ggml_reshape_3d(c, cin, 3 + n, C, 1), l.conv1d));   // [C, n, 1]
    const size_t es = ggml_element_size(co), nb1 = es * C;
    ggml_tensor* q = ggml_view_4d(c, co, S, Hk, n, 1, es * S, nb1, nb1 * n, 0);
    ggml_tensor* k = ggml_view_4d(c, co, S, Hk, n, 1, es * S, nb1, nb1 * n, es * S * Hk);
    ggml_tensor* v = ggml_view_4d(c, co, S, Hv, n, 1, es * S, nb1, nb1 * n, es * 2 * S * Hk);
    auto l2 = [&](ggml_tensor* t) { return ggml_scale(c, ggml_rms_norm(c, t, h.eps / S), 1.0f / std::sqrt((float) S)); };
    q = l2(q);
    k = l2(k);
    ggml_tensor* st = ggml_reshape_4d(c, ggml_get_rows(c, l.ssm, in.slot), S, S, Hv, 1);
    ggml_tensor* r = ggml_gated_delta_net(c, q, k, v, g, beta, st, ksnap);
    const int64_t attn_el = (int64_t) S * Hv * n;
    const int nw = std::min(n, ksnap);
    ggml_build_forward_expand(gf, ggml_cpy(c, ggml_view_2d(c, r, Dst, nw, (size_t) Dst * 4, (size_t) attn_el * 4),
                                           ggml_view_2d(c, l.ssm, Dst, nw, l.ssm->nb[1], 0)));
    ggml_tensor* o = ggml_view_4d(c, r, S, Hv, n, 1, (size_t) S * 4, (size_t) S * Hv * 4, (size_t) attn_el * 4, 0);
    ggml_tensor* zz = ggml_reshape_4d(c, z, S, Hv, n, 1);
    o = ggml_mul(c, rms(c, o, l.ssm_norm, h.eps), ggml_silu(c, zz));
    return ggml_mul_mat(c, l.ssm_out, ggml_reshape_2d(c, o, (int64_t) S * Hv, n));
}

int pad_kv(int64_t n, int64_t cap) { return (int) std::min<int64_t>(cap, (n + 255) / 256 * 256); }

// A built graph that can run again with new inputs while its key is the same.
struct Built {
    std::string key;
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_backend_sched_t sched = nullptr;   // its own: its compute buffers stay allocated for the next run
    Inputs in;
    ggml_tensor *logits = nullptr, *argmax = nullptr, *pmax = nullptr, *probs = nullptr;
    uint64_t used = 0;
    bool own_sched = true;
    ~Built() {
        if (ctx) ggml_free(ctx);
        if (sched && own_sched) ggml_backend_sched_free(sched);
    }
};

// One live graph per kind, each kind with its own scheduler: 0 the trunk (prompt windows, verify, decode), 1 the MTP
// block's catch-up windows, 2 its one-row draft steps.  A scheduler keeps one compute buffer, as large as the largest
// graph it ran (the prompt windows: with a q8_0 KV the attention also converts the cache it reads to f16 there, 0.8
// GiB at 200K), so the kinds do not each hold one per shape.  The graph a kind holds runs again with new inputs (no
// build, no allocation; ggml-hip replays its HIP graph); another shape is built anew - not re-allocated: splitting a
// graph points its nodes at input copies in the scheduler's own context, which the next split frees.
struct Graphs {
    ggml_backend_sched_t sched[4] = {nullptr, nullptr, nullptr, nullptr};   // 3: the DFlash2 block
    std::unique_ptr<Built> live[4];
    // -> the graph, and whether it is the live one (built and allocated); else a fresh one to build, then alloc()
    std::pair<Built*, bool> get(Model& m, const std::string& key, int kind) {
        if (!sched[kind])
            sched[kind] = ggml_backend_sched_new(m.bk.data(), nullptr, (int) m.bk.size(), GRAPH_NODES, false, false);
        if (live[kind] && live[kind]->key == key) return {live[kind].get(), true};
        ggml_backend_sched_reset(sched[kind]);
        live[kind] = std::make_unique<Built>();
        live[kind]->key = key;
        live[kind]->sched = sched[kind];
        live[kind]->own_sched = false;
        return {live[kind].get(), false};
    }
    void alloc(int kind, Built* b) {
        if (!ggml_backend_sched_alloc_graph(sched[kind], b->gf)) die("cannot allocate a graph (" + b->key + ")");
    }
    void clear() {
        for (auto& l : live) l.reset();
    }
    ~Graphs() {
        clear();
        for (auto s : sched)
            if (s) ggml_backend_sched_free(s);
    }
};

ggml_context* graph_ctx() {
    ggml_init_params ip = {ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false),
                           nullptr, true};
    return ggml_init(ip);
}

// The head's outputs: logits rows, and on the device their argmax and its probability (the greedy path reads 8
// bytes a row instead of the whole vocabulary).
void head_outputs(ggml_context* c, ggml_cgraph* gf, Built& b, ggml_tensor* logits, int m_rows, bool want_logits,
                  bool want_ppl) {
    b.logits = logits;
    if (want_ppl) {
        b.probs = ggml_soft_max(c, logits);
        ggml_set_output(b.probs);
        ggml_build_forward_expand(gf, b.probs);
        return;
    }
    b.argmax = ggml_argmax(c, logits);                                   // [m] i32
    if (m_rows == 1) {   // the draft's probability: softmax, then the argmax's entry of the flat [1, V] view
        ggml_tensor* p = ggml_soft_max(c, logits);
        b.pmax = ggml_get_rows(c, ggml_reshape_2d(c, p, 1, p->ne[0]), b.argmax);
        ggml_set_output(b.pmax);
        ggml_build_forward_expand(gf, b.pmax);
    }
    ggml_set_output(b.argmax);
    ggml_build_forward_expand(gf, b.argmax);
    if (want_logits) {
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }
}

// The trunk over n tokens at positions pos0..pos0+n-1.  The h_nextn rows go to hbuf[0..n); the head runs on m_rows
// rows: the last one (m_rows == 1) or all n (m_rows == n).
Built& trunk_graph(Model& m, Graphs& G, int n, int64_t pos0, int m_rows, int ksnap, bool want_logits, bool want_ppl,
                   int n_img = 0) {
    const Hp& h = m.hp;
    const int n_kv = pad_kv(pos0 + n, m.o.max_ctx);
    char key[160];
    const bool inject = m.df && !want_ppl;
    std::snprintf(key, sizeof key, "t n%d kv%d m%d k%d L%d P%d D%d I%d", n, n_kv, m_rows, ksnap, want_logits, want_ppl,
                  inject, n_img);
    auto [bp, hit] = G.get(m, key, 0);
    Built& b = *bp;
    if (hit) return b;
    b.ctx = graph_ctx();
    ggml_context* c = b.ctx;
    b.gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_cgraph* gf = b.gf;
    Inputs& in = b.in;
    in.tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    in.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) n * 4);
    in.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n);
    in.kv_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, n);
    in.out_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, m_rows);
    in.slot = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    in.cidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, 3);
    for (ggml_tensor* t : {in.tok, in.pos, in.mask, in.kv_idx, in.out_ids, in.slot, in.cidx}) ggml_set_input(t);
    if (inject) {
        in.ring_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, n);
        ggml_set_input(in.ring_idx);
    }
    std::vector<ggml_tensor*> feats(inject ? m.df->target_layers.size() : 0, nullptr);
    ggml_tensor* x = ggml_get_rows(c, m.tok_embd, in.tok);
    if (n_img > 0) {   // the picture's rows take the place of their <|image_pad|> tokens' embeddings
        in.img = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_embd, n_img);
        in.img_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, n_img);
        ggml_set_input(in.img);
        ggml_set_input(in.img_idx);
        x = ggml_set_rows(c, x, in.img, in.img_idx);
    }
    for (int il = 0; il < h.n_layer; ++il) {
        Layer& l = m.L[il];
        for (size_t k = 0; k < feats.size(); ++k)
            if (m.df->target_layers[k] == il) feats[k] = x;   // the residual stream entering layer il
        ggml_tensor* cur = rms(c, x, l.attn_norm, h.eps);
        cur = h.recurrent(il) ? gdn_block(m, c, gf, l, cur, in, n, ksnap) : attn_block(m, c, gf, l, cur, in, n, n_kv);
        x = ggml_add(c, cur, x);
        x = ggml_add(c, ffn(c, l, rms(c, x, l.post_norm, h.eps)), x);
    }
    if (inject) {
        // the drafter's context: fc(the five layers' inputs) -> norm -> each layer's K / V at the rows' ring cells
        const Drafter& d = *m.df;
        ggml_tensor* f = feats[0];
        for (size_t k = 1; k < feats.size(); ++k) f = ggml_concat(c, f, feats[k], 0);
        ggml_tensor* g = rms(c, ggml_mul_mat(c, d.fc, f), d.enc_norm, d.eps);
        int sec[4] = {d.head_dim / 2, 0, 0, 0};
        for (const DfLayer& dl : d.L) {
            ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, dl.wk, g), d.head_dim, d.n_head_kv, n);
            k = rms(c, k, dl.k_norm, d.eps);
            k = ggml_rope_multi(c, k, in.pos, nullptr, d.head_dim, sec, GGML_ROPE_TYPE_MROPE, h.n_ctx_train,
                                d.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            ggml_tensor* v = ggml_mul_mat(c, dl.wv, g);
            ggml_build_forward_expand(gf, ggml_set_rows(c, dl.k_ring, ggml_reshape_2d(c, k, d.kv_dim(), n), in.ring_idx));
            ggml_build_forward_expand(gf, ggml_set_rows(c, dl.v_ring, v, in.ring_idx));
        }
    }
    ggml_tensor* hn = rms(c, x, m.out_norm, h.eps);   // h_nextn: what the MTP block reads
    if (m.o.mtp > 0 && !want_ppl)
        ggml_build_forward_expand(gf, ggml_cpy(c, hn, ggml_view_2d(c, m.hbuf, h.n_embd, n, m.hbuf->nb[1], 0)));
    ggml_tensor* rows = m_rows == n ? hn : ggml_get_rows(c, hn, in.out_ids);
    head_outputs(c, gf, b, ggml_mul_mat(c, m.output, rows), m_rows, want_logits, want_ppl);
    G.alloc(0, &b);
    return b;
}

// The MTP block over r rows (tokens at pos0.., the last cell for padding rows).  Its h rows are hbuf columns hidx
// (catch-up windows, kind 1: afterwards pending <- hbuf column pidx), or the block's own last h (a draft step, r == 1,
// kind 2).  The head reads row out_ids.  The catch-up graph of a decode round always has the same r (padded), so it
// is built once and replayed.
Built& mtp_graph(Model& m, Graphs& G, int r, int64_t pos0, bool step, int n_img = 0) {
    const Hp& h = m.hp;
    Layer& l = m.L[h.n_layer];
    const int n_kv = pad_kv(pos0 + r, m.o.max_ctx);
    char key[160];
    std::snprintf(key, sizeof key, "m r%d kv%d s%d I%d", r, n_kv, step, n_img);
    const int kind = step ? 2 : 1;
    auto [bp, hit] = G.get(m, key, kind);
    Built& b = *bp;
    if (hit) return b;
    b.ctx = graph_ctx();
    ggml_context* c = b.ctx;
    b.gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_cgraph* gf = b.gf;
    Inputs& in = b.in;
    in.tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, r);
    in.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) r * 4);
    in.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, r);
    in.kv_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, r);
    in.out_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    in.hidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, r);
    in.pidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    for (ggml_tensor* t : {in.tok, in.pos, in.mask, in.kv_idx, in.out_ids, in.hidx, in.pidx}) ggml_set_input(t);
    ggml_tensor* e = ggml_get_rows(c, m.tok_embd, in.tok);
    if (n_img > 0) {   // a picture's rows: the block reads them as the trunk did, not the <|image_pad|> embedding
        in.img = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_embd, n_img);
        in.img_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, n_img);
        ggml_set_input(in.img);
        ggml_set_input(in.img_idx);
        e = ggml_set_rows(c, e, in.img, in.img_idx);
    }
    ggml_tensor* hr = step ? m.mtph : ggml_get_rows(c, m.hbuf, in.hidx);
    ggml_tensor* cur = ggml_concat(c, rms(c, e, l.enorm, h.eps), rms(c, hr, l.hnorm, h.eps), 0);
    cur = ggml_mul_mat(c, l.eh_proj, cur);
    ggml_tensor* a = attn_block(m, c, gf, l, rms(c, cur, l.attn_norm, h.eps), in, r, n_kv);
    cur = ggml_add(c, a, cur);
    cur = ggml_add(c, ffn(c, l, rms(c, cur, l.post_norm, h.eps)), cur);
    ggml_tensor* hn = rms(c, cur, l.head_norm ? l.head_norm : m.out_norm, h.eps);
    ggml_tensor* last = r == 1 ? hn : ggml_get_rows(c, hn, in.out_ids);
    ggml_build_forward_expand(gf, ggml_cpy(c, last, m.mtph));
    head_outputs(c, gf, b, ggml_mul_mat(c, m.output, last), 1, false, false);
    if (!step)
        ggml_build_forward_expand(
            gf, ggml_cpy(c, ggml_get_rows(c, m.hbuf, in.pidx),
                         ggml_view_2d(c, m.hbuf, h.n_embd, 1, m.hbuf->nb[1], (size_t) m.pend_col() * m.hbuf->nb[1])));
    G.alloc(kind, &b);
    return b;
}

// DFlash2's two-tap dynamic convolution over the block (one block of B rows): w_t = base[:, t, side] + the row's own
// coefficients (one per group of conv_g channels), out_i = w_0 * x_i + w_1 * x_{i-1} (x_{-1} = 0).
ggml_tensor* df_conv(ggml_context* c, const Drafter& d, ggml_tensor* x, ggml_tensor* dyn, ggml_tensor* base, int side,
                     int B) {
    const int64_t E = x->ne[0], G = E / d.conv_g, K = d.conv_k;
    x = ggml_cont(c, x);
    ggml_tensor* co = ggml_reshape_4d(c, ggml_cont(c, dyn), G, K, 2, B);
    co = ggml_cont(c, ggml_view_3d(c, co, G, K, B, co->nb[1], co->nb[3], side * co->nb[2]));
    co = ggml_repeat_4d(c, ggml_reshape_4d(c, co, 1, G, K, B), d.conv_g, G, K, B);
    ggml_tensor* bs = ggml_reshape_4d(c, ggml_view_1d(c, base, E * K, side * base->nb[2]), d.conv_g, G, K, 1);
    ggml_tensor* w = ggml_add(c, co, bs);   // [conv_g, G, K, B]
    ggml_tensor* out = nullptr;
    for (int t = 0; t < std::min<int64_t>(K, B); ++t) {
        ggml_tensor* vals = x;
        if (t > 0) vals = ggml_pad_ext(c, ggml_view_2d(c, x, E, B - t, x->nb[1], 0), 0, 0, t, 0, 0, 0, 0, 0);
        ggml_tensor* wt = ggml_reshape_2d(c, ggml_cont(c, ggml_view_4d(c, w, d.conv_g, G, 1, B, w->nb[1], w->nb[2],
                                                                       w->nb[3], t * w->nb[2])), E, B);
        ggml_tensor* term = ggml_mul(c, wt, vals);
        out = out ? ggml_add(c, out, term) : term;
    }
    return out;
}

// The DFlash2 block: B rows [anchor, MASK...] at pos0..; outputs the selector's candidates [top_k, B] and the
// transition scores [top_k (successor), top_k (predecessor), B-1] of rows 1..B-1.
Built& df_graph(Model& m, Graphs& G, int B) {
    const Drafter& d = *m.df;
    char key[64];
    std::snprintf(key, sizeof key, "df B%d", B);
    auto [bp, hit] = G.get(m, key, 3);
    Built& b = *bp;
    if (hit) return b;
    b.ctx = graph_ctx();
    ggml_context* c = b.ctx;
    b.gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_cgraph* gf = b.gf;
    Inputs& in = b.in;
    in.tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, B);
    in.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) B * 4);
    in.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, d.ring, B);
    in.ring_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, B);
    in.anchor = ggml_new_tensor_2d(c, GGML_TYPE_I32, d.top_k, 1);
    for (ggml_tensor* t : {in.tok, in.pos, in.mask, in.ring_idx, in.anchor}) ggml_set_input(t);
    int sec[4] = {d.head_dim / 2, 0, 0, 0};
    const int D = d.head_dim;
    const float sq_code = std::sqrt((float) d.sp_code);
    // DSpine's code bridge after a layer (see the Drafter comment); the last layer's pre-bridge state is kept for the
    // lattice: its rows, their rms and their own codes' gate projection
    ggml_tensor *st_raw = nullptr, *st_rms = nullptr, *st_q = nullptr;
    auto bridge = [&](const DfLayer& l, ggml_tensor* x, bool last) -> ggml_tensor* {
        if (B < 2) return x;
        ggml_tensor* rv = ggml_sqrt(c, ggml_scale(c, ggml_sum_rows(c, ggml_sqr(c, x)), 1.0f / d.n_embd));   // [1, B]
        ggml_tensor* codes = ggml_mul_mat(c, l.sp_read, ggml_rms_norm(c, x, d.eps));                       // [code, B]
        codes = ggml_scale(c, ggml_l2_norm(c, codes, d.eps), sq_code);
        ggml_tensor* anc = ggml_scale(c, ggml_get_rows(c, d.sp_codebook, ggml_view_1d(c, in.tok, 1, 0)), sq_code);
        codes = ggml_concat(c, anc, ggml_cont(c, ggml_view_2d(c, codes, d.sp_code, B - 1, codes->nb[1], codes->nb[1])), 1);
        ggml_tensor* q = ggml_mul_mat(c, d.sp_gate, codes);        // [rank, B]
        ggml_tensor* sg = ggml_mul_mat(c, d.sp_pred_gate, codes);
        ggml_tensor* v = ggml_mul_mat(c, d.sp_prev, codes);
        auto rows = [&](ggml_tensor* t, int from) {
            return ggml_cont(c, ggml_view_2d(c, t, t->ne[0], B - 1, t->nb[1], (size_t) from * t->nb[1]));
        };
        // message(t) = sigmoid(Wg c_t + Wp c_(t-1) + b) * (Wv c_(t-1)); none for the first row
        ggml_tensor* gate = ggml_sigmoid(c, ggml_add(c, ggml_add(c, rows(q, 1), rows(sg, 0)), d.sp_gate_b));
        ggml_tensor* delta = ggml_mul_mat(c, l.sp_write, ggml_mul(c, gate, rows(v, 0)));   // [E, B-1]
        delta = ggml_mul(c, ggml_pad_ext(c, delta, 0, 0, 1, 0, 0, 0, 0, 0), rv);
        if (last) { st_raw = x; st_rms = rv; st_q = q; }
        return ggml_add(c, x, delta);
    };
    ggml_tensor* x = ggml_get_rows(c, m.tok_embd, in.tok);
    for (size_t il = 0; il < d.L.size(); ++il) {
        const DfLayer& l = d.L[il];
        ggml_tensor* nn = rms(c, x, l.attn_norm, d.eps);
        ggml_tensor* adyn = nullptr;
        if (l.attn_conv_proj) {
            adyn = ggml_mul_mat(c, l.attn_conv_proj, nn);
            nn = df_conv(c, d, nn, adyn, l.attn_conv_base, 0, B);
        }
        ggml_tensor* q = ggml_reshape_3d(c, ggml_mul_mat(c, l.wq, nn), D, d.n_head, B);
        ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, l.wk, nn), D, d.n_head_kv, B);
        ggml_tensor* v = ggml_mul_mat(c, l.wv, nn);
        q = rms(c, q, l.q_norm, d.eps);
        k = rms(c, k, l.k_norm, d.eps);
        q = ggml_rope_multi(c, q, in.pos, nullptr, D, sec, GGML_ROPE_TYPE_MROPE, m.hp.n_ctx_train, d.rope_base, 1.0f,
                            0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_multi(c, k, in.pos, nullptr, D, sec, GGML_ROPE_TYPE_MROPE, m.hp.n_ctx_train, d.rope_base, 1.0f,
                            0.0f, 1.0f, 32.0f, 1.0f);
        ggml_build_forward_expand(gf, ggml_set_rows(c, l.k_ring, ggml_reshape_2d(c, k, d.kv_dim(), B), in.ring_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(c, l.v_ring, v, in.ring_idx));
        const size_t row = ggml_row_size(GGML_TYPE_F16, d.kv_dim()), hrow = ggml_row_size(GGML_TYPE_F16, D);
        ggml_tensor* kk = ggml_permute(c, ggml_view_3d(c, l.k_ring, D, d.n_head_kv, d.ring, hrow, row, 0), 0, 2, 1, 3);
        ggml_tensor* vv = ggml_permute(c, ggml_view_3d(c, l.v_ring, D, d.n_head_kv, d.ring, hrow, row, 0), 0, 2, 1, 3);
        ggml_tensor* o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), kk, vv, in.mask,
                                             1.0f / std::sqrt((float) D), 0.0f, 0.0f);
        o = ggml_mul_mat(c, l.wo, ggml_reshape_2d(c, o, D * d.n_head, B));
        if (adyn) o = df_conv(c, d, o, adyn, l.attn_conv_base, 1, B);
        ggml_tensor* fi = ggml_add(c, o, x);
        ggml_tensor* cur = rms(c, fi, l.ffn_norm, d.eps);
        ggml_tensor* fdyn = nullptr;
        if (l.ffn_conv_proj) {
            fdyn = ggml_mul_mat(c, l.ffn_conv_proj, cur);
            cur = df_conv(c, d, cur, fdyn, l.ffn_conv_base, 0, B);
        }
        cur = ggml_mul_mat(c, l.ffn_down, ggml_swiglu_split(c, ggml_mul_mat(c, l.ffn_gate, cur),
                                                           ggml_mul_mat(c, l.ffn_up, cur)));
        if (fdyn) cur = df_conv(c, d, cur, fdyn, l.ffn_conv_base, 1, B);
        x = ggml_add(c, cur, fi);
        if (l.sp_write) x = bridge(l, x, il + 1 == d.L.size());
    }
    ggml_tensor* emb = rms(c, x, d.out_norm, d.eps);
    ggml_tensor* logits = ggml_mul_mat(c, m.output, emb);                      // [V, B]
    const int K = d.top_k, R = d.rank, P = B - 1;
    ggml_tensor* cand = ggml_top_k(c, logits, K);                               // [K, B] i32
    // rows 1..B-1: successors = their own candidates, predecessors = row 1: the anchor; rows 2..: row-1's candidates
    ggml_tensor* succ_ids = ggml_cont(c, ggml_view_2d(c, cand, K, P, cand->nb[1], cand->nb[1]));
    ggml_tensor* pred_ids = P > 1 ? ggml_concat(c, in.anchor, ggml_cont(c, ggml_view_2d(c, cand, K, P - 1, cand->nb[1],
                                                                                      cand->nb[1])), 1)
                                  : in.anchor;
    ggml_tensor* score = nullptr;
    if (d.dspine && st_raw) {
        // the last bridge again, once per exact predecessor candidate; table[k][j] = the head's logit of candidate k
        // after the bridge with predecessor candidate j
        const int Rk = d.sp_rank;
        ggml_tensor* pc = ggml_scale(c, ggml_get_rows(c, d.sp_codebook, ggml_reshape_1d(c, ggml_cont(c, pred_ids), K * P)),
                                     sq_code);                                  // [code, K*P]
        ggml_tensor* gp = ggml_reshape_3d(c, ggml_mul_mat(c, d.sp_pred_gate, pc), Rk, K, P);
        ggml_tensor* pv = ggml_reshape_3d(c, ggml_mul_mat(c, d.sp_prev, pc), Rk, K, P);
        auto rows3 = [&](ggml_tensor* t) {   // rows 1..B-1 as [ne0, 1, P]
            return ggml_reshape_3d(c, ggml_cont(c, ggml_view_2d(c, t, t->ne[0], P, t->nb[1], t->nb[1])), t->ne[0], 1, P);
        };
        ggml_tensor* gate = ggml_sigmoid(c, ggml_add(c, ggml_add(c, gp, rows3(st_q)), d.sp_gate_b));
        ggml_tensor* delta = ggml_mul_mat(c, d.L.back().sp_write, ggml_reshape_2d(c, ggml_mul(c, gate, pv), Rk, K * P));
        delta = ggml_reshape_3d(c, delta, d.n_embd, K, P);
        ggml_tensor* refined = ggml_add(c, ggml_mul(c, delta, rows3(st_rms)), rows3(st_raw));
        refined = rms(c, refined, d.out_norm, d.eps);                          // [E, K(pred), P]
        ggml_tensor* head = ggml_reshape_3d(c, ggml_get_rows(c, m.output, ggml_reshape_1d(c, succ_ids, K * P)),
                                            d.n_embd, K, P);                    // [E, K(succ), P]
        score = ggml_mul_mat(c, head, refined);                                 // [K(succ), K(pred), P]
    } else {
        ggml_tensor* unary = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, logits, 1, logits->ne[0], B), cand),
                                             K, B);                              // [K, B]
        ggml_tensor* gate = ggml_mul_mat(c, d.sel_hidden, emb);                // [R, B]
        ggml_tensor* succ = ggml_reshape_3d(c, ggml_get_rows(c, d.sel_next, ggml_reshape_1d(c, succ_ids, K * P)), R, K, P);
        ggml_tensor* pred = ggml_reshape_3d(c, ggml_get_rows(c, d.sel_prev, ggml_reshape_1d(c, ggml_cont(c, pred_ids), K * P)),
                                            R, K, P);
        ggml_tensor* g1 = ggml_reshape_3d(c, ggml_cont(c, ggml_view_2d(c, gate, R, P, gate->nb[1], gate->nb[1])), R, 1, P);
        ggml_tensor* cond = ggml_mul(c, pred, g1);                              // [R, K(pred), P]
        score = ggml_mul_mat(c, succ, cond);                                    // [K(succ), K(pred), P]
        ggml_tensor* u1 = ggml_reshape_3d(c, ggml_cont(c, ggml_view_2d(c, unary, K, P, unary->nb[1], unary->nb[1])), K, 1, P);
        score = ggml_add(c, score, u1);
    }
    b.argmax = cand;   // the candidates
    b.probs = score;   // the transition scores
    ggml_set_output(cand);
    ggml_set_output(score);
    ggml_build_forward_expand(gf, cand);
    ggml_build_forward_expand(gf, score);
    G.alloc(3, &b);
    return b;
}

// Rows from n_real on are padding: their K/V go to the context's last cell (never attended: the usable context ends
// before it) and their outputs are not read.
// rp3: each row's M-RoPE position (t, h, w); null: the cache index of the row (text from position 0).
void set_inputs(const Inputs& in, const int32_t* tok, int n, int64_t pos0, int n_kv, int m_rows, int slot = 0,
                int conv_off = CONV_W - 3, int n_real = -1, int64_t scratch = 0, const int32_t* rp3 = nullptr) {
    ggml_backend_tensor_set(in.tok, tok, 0, (size_t) n * 4);
    if (in.slot) {
        const int32_t sl[1] = {slot}, ci[3] = {conv_off, conv_off + 1, conv_off + 2};
        ggml_backend_tensor_set(in.slot, sl, 0, 4);
        ggml_backend_tensor_set(in.cidx, ci, 0, 12);
    }
    std::vector<int32_t> pos((size_t) n * 4);
    std::vector<int64_t> idx(n);
    for (int i = 0; i < n; ++i) {
        if (rp3) {
            pos[i] = rp3[3 * i];
            pos[n + i] = rp3[3 * i + 1];
            pos[2 * n + i] = rp3[3 * i + 2];
        } else {
            pos[i] = pos[n + i] = pos[2 * n + i] = (int32_t) (pos0 + i);
        }
        pos[3 * n + i] = 0;
        idx[i] = (n_real >= 0 && i >= n_real) ? scratch : pos0 + i;
    }
    ggml_backend_tensor_set(in.pos, pos.data(), 0, pos.size() * 4);
    ggml_backend_tensor_set(in.kv_idx, idx.data(), 0, idx.size() * 8);
    std::vector<ggml_fp16_t> mask((size_t) n_kv * n);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n; ++i) {
        const int64_t lim = pos0 + i;
        ggml_fp16_t* row = mask.data() + (size_t) i * n_kv;
        for (int j = 0; j < n_kv; ++j) row[j] = j <= lim ? zero : ninf;
    }
    ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * 2);
    if (in.out_ids->buffer) {   // unused (unallocated) when the head reads every row
        std::vector<int32_t> out(m_rows);
        for (int i = 0; i < m_rows; ++i) out[i] = m_rows == 1 ? n - 1 : i;
        ggml_backend_tensor_set(in.out_ids, out.data(), 0, out.size() * 4);
    }
}

// ---------------------------------------------------------------------------------------------------------------- run
struct Sampling {
    float temp = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    float rep = 1.0f, freq = 0.0f, pres = 0.0f;
    int last_n = 64;
    uint64_t seed = 0;
};

struct Sampler {
    std::mt19937_64 rng{12345};
    std::vector<std::pair<float, int>> cand;
    // Penalised, tempered, truncated sampling over one row of logits (modified in place for the penalties).
    int sample(float* lg, int V, const Sampling& s, const std::vector<int32_t>& hist) {
        if (s.rep != 1.0f || s.freq != 0.0f || s.pres != 0.0f) {
            std::map<int, int> cnt;
            const size_t from = hist.size() > (size_t) s.last_n ? hist.size() - s.last_n : 0;
            for (size_t i = from; i < hist.size(); ++i) cnt[hist[i]]++;
            for (auto [t, k] : cnt) {
                if (t < 0 || t >= V) continue;
                float& x = lg[t];
                if (s.rep != 1.0f) x = x > 0 ? x / s.rep : x * s.rep;
                x -= s.freq * k + s.pres;
            }
        }
        if (s.temp <= 0.0f) return (int) (std::max_element(lg, lg + V) - lg);
        const int k = std::clamp(s.top_k, 1, 64);
        cand.clear();
        // top-k by a running threshold: one pass over the row
        float thr = -INFINITY;
        for (int i = 0; i < V; ++i) {
            if (lg[i] <= thr) continue;
            cand.emplace_back(lg[i], i);
            if ((int) cand.size() >= 4 * k) {
                std::nth_element(cand.begin(), cand.begin() + (k - 1), cand.end(),
                                 [](auto& a, auto& b) { return a.first > b.first; });
                cand.resize(k);
                thr = std::min_element(cand.begin(), cand.end())->first;
            }
        }
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        if ((int) cand.size() > k) cand.resize(k);
        const float mx = cand[0].first;
        double sum = 0;
        std::vector<double> p(cand.size());
        for (size_t i = 0; i < cand.size(); ++i) sum += p[i] = std::exp((cand[i].first - mx) / s.temp);
        for (auto& x : p) x /= sum;
        size_t keep = cand.size();
        if (s.top_p < 1.0f) {
            double acc = 0;
            for (size_t i = 0; i < cand.size(); ++i) {
                acc += p[i];
                if (acc >= s.top_p) { keep = i + 1; break; }
            }
        }
        if (s.min_p > 0.0f)
            for (size_t i = 1; i < keep; ++i)
                if (p[i] < s.min_p * p[0]) { keep = i; break; }
        double tot = 0;
        for (size_t i = 0; i < keep; ++i) tot += p[i];
        double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
        for (size_t i = 0; i < keep; ++i) {
            u -= p[i];
            if (u <= 0) return cand[i].second;
        }
        return cand[keep - 1].second;
    }
};

struct Engine {
    Model m;
    Graphs G;                           // the trunk's and the MTP block's graphs, by shape
    Sampler smp;
    std::vector<int32_t> cache;         // the tokens the trunk's state holds (cache cells 0..size-1)
    std::vector<int64_t> ckey;          // what a prompt is matched on: the token id, or an image row's key
    std::vector<int32_t> rp;            // each cached token's M-RoPE position (t, h, w)
    int32_t next_p = 0;                 // the position of the next text token (after a picture: p + max(nx, ny))
    // the request being read: its image rows and, per prompt token, the row it takes (-1: text)
    std::vector<float> img_buf;
    std::vector<int32_t> img_row;
    // M-RoPE positions of cache indices i0..i0+n-1: cached ones as kept, later ones text after next_p
    std::vector<int32_t> rope3(int64_t i0, int n) const {
        std::vector<int32_t> v((size_t) n * 3);
        for (int i = 0; i < n; ++i) {
            const int64_t idx = i0 + i;
            if (idx < (int64_t) cache.size()) std::copy(rp.begin() + idx * 3, rp.begin() + idx * 3 + 3, v.begin() + i * 3);
            else v[i * 3] = v[i * 3 + 1] = v[i * 3 + 2] = next_p + (int32_t) (idx - (int64_t) cache.size());
        }
        return v;
    }
    bool mtp_ok = false;                // the MTP KV and pending row follow `cache`
    int kept_prev = 0;                  // the last trunk window's rows whose MTP pairs are not made yet (hbuf 0..)
    std::atomic<bool>* stop = nullptr;
    int64_t drafts_offered = 0, drafts_accepted = 0;
    int64_t auto_rounds[2] = {0, 0};    // --spec auto: rounds drafted by MTP, by the block drafter
    // --spec auto follows the generated text: inside a ``` block or a <tool_call> the block drafter drafts, elsewhere
    // MTP.  The tokens that hold a backtick (68 of the vocabulary), as their text, and the tool-call markers.
    std::unordered_map<int32_t, std::string> tick_tok;
    int32_t tool_open = -1, tool_close = -1;
    bool in_fence = false, in_tool = false;
    int tick_run = 0;
    void code_scan_init() {
        const int64_t id = gguf_find_key(m.gguf, "tokenizer.ggml.tokens");
        if (id < 0) return;
        const size_t n = gguf_get_arr_n(m.gguf, id);
        for (size_t i = 0; i < n; ++i) {
            const char* t = gguf_get_arr_str(m.gguf, id, i);
            if (std::strchr(t, '`')) tick_tok[(int32_t) i] = t;
            if (!std::strcmp(t, "<tool_call>")) tool_open = (int32_t) i;
            if (!std::strcmp(t, "</tool_call>")) tool_close = (int32_t) i;
        }
    }
    void code_scan(int32_t y) {   // a run of three or more backticks opens or closes a fence
        if (y == tool_open) in_tool = true;
        if (y == tool_close) in_tool = false;
        auto it = tick_tok.find(y);
        if (it == tick_tok.end()) {
            if (tick_run >= 3) in_fence = !in_fence;
            tick_run = 0;
            return;
        }
        for (char ch : it->second) {
            if (ch == '`') { ++tick_run; continue; }
            if (tick_run >= 3) in_fence = !in_fence;
            tick_run = 0;
        }
    }
    bool in_code() const { return in_fence || in_tool || tick_run >= 3; }
    // ALMOND_Q35_PROF=1: time per trunk window size and per MTP call, printed at the end of a request
    bool prof = std::getenv("ALMOND_Q35_PROF") != nullptr;
    std::map<std::string, std::pair<int64_t, double>> pt;
    void prof_add(const std::string& k, Clock::time_point t0) {
        if (!prof) return;
        auto& e = pt[k];
        e.first++;
        e.second += ms_since(t0);
    }
    void prof_print() {
        if (!prof) return;
        for (auto& [k, v] : pt) std::fprintf(stderr, "prof %-10s %6lld x %7.3f ms\n", k.c_str(), (long long) v.first, v.second / v.first);
        pt.clear();
    }

    // Runs the trunk over tok[0..n) at the end of the cache.  m_rows: 1 or n.  Returns argmax rows; logits to `lg`
    // when asked.
    // The prompt's own rows (keys, positions, image rows) come from `rq_*` when it reads a prompt window; generated
    // tokens are text at next_p...
    const int64_t* rq_key = nullptr;
    const int32_t* rq_rp = nullptr;
    const int32_t* rq_img = nullptr;   // per row: the image row index into img_buf, -1 text
    void trunk(const int32_t* tok, int n, int m_rows, int ksnap, std::vector<int32_t>* am, std::vector<float>* lg,
               float* pmax = nullptr) {
        const auto tp = Clock::now();
        const int64_t pos0 = (int64_t) cache.size();
        if (pos0 + n > m.o.max_ctx - 1) die("the context is full (" + std::to_string(m.o.max_ctx) + " tokens)");
        std::vector<int32_t> r3 = rq_rp ? std::vector<int32_t>(rq_rp, rq_rp + 3 * n) : rope3(pos0, n);
        std::vector<int64_t> iidx;
        std::vector<float> irows;
        if (rq_img)
            for (int i = 0; i < n; ++i)
                if (rq_img[i] >= 0) {
                    iidx.push_back(i);
                    irows.insert(irows.end(), img_buf.begin() + (size_t) rq_img[i] * m.hp.n_embd,
                                 img_buf.begin() + (size_t) (rq_img[i] + 1) * m.hp.n_embd);
                }
        Built& b = trunk_graph(m, G, n, pos0, m_rows, ksnap, lg != nullptr, false, (int) iidx.size());
        set_inputs(b.in, tok, n, pos0, pad_kv(pos0 + n, m.o.max_ctx), m_rows, m.ssm_slot, m.conv_off, -1, 0, r3.data());
        if (!iidx.empty()) {
            ggml_backend_tensor_set(b.in.img, irows.data(), 0, irows.size() * 4);
            ggml_backend_tensor_set(b.in.img_idx, iidx.data(), 0, iidx.size() * 8);
        }
        if (b.in.ring_idx) {   // the drafter's context rows go to the ring cells of their positions
            std::vector<int64_t> ri(n);
            for (int i = 0; i < n; ++i) {
                ri[i] = (pos0 + i) % m.df->ring;
                m.df->ring_pos[ri[i]] = pos0 + i;
            }
            ggml_backend_tensor_set(b.in.ring_idx, ri.data(), 0, ri.size() * 8);
        }
        if (ggml_backend_sched_graph_compute(b.sched, b.gf) != GGML_STATUS_SUCCESS) die("the trunk graph failed");
        if (am) {
            am->resize(m_rows);
            ggml_backend_tensor_get(b.argmax, am->data(), 0, (size_t) m_rows * 4);
        }
        if (pmax && b.pmax) ggml_backend_tensor_get(b.pmax, pmax, 0, 4);
        if (lg) {
            lg->resize((size_t) m_rows * m.hp.n_vocab);
            ggml_backend_tensor_get(b.logits, lg->data(), 0, lg->size() * 4);
        }
        prof_add("trunk" + std::to_string(n), tp);
        cache.insert(cache.end(), tok, tok + n);
        rp.insert(rp.end(), r3.begin(), r3.end());
        if (rq_key) ckey.insert(ckey.end(), rq_key, rq_key + n);
        else {
            ckey.insert(ckey.end(), tok, tok + n);
            next_p = r3[3 * (n - 1)] + 1;   // generated text: the next position follows the last
        }
        // a full window kept: the state is the newest snapshot and the conv window's tail
        m.ssm_slot = 0;
        m.conv_off = CONV_W - 3;
    }

    // After a verify window of n tokens of which `keep` stay: roll the recurrent state back and drop the rest.
    void keep_window(int n, int keep) {
        cache.resize(cache.size() - (n - keep));
        ckey.resize(cache.size());
        rp.resize(cache.size() * 3);
        next_p -= n - keep;
        m.ssm_slot = n - keep;
        m.conv_off = CONV_W - (n + 3) + keep;
    }

    // The MTP block over r rows (see mtp_graph); returns its draft (argmax) and the draft's probability.
    // An MTP catch-up window: k pairs (tok[i], hbuf column hcols[i]) at pos0.., padded to R rows; pending <- hbuf
    // column pcol.  Returns the draft after the last pair (argmax) and its probability.
    int mtp_catch(const int32_t* tok, int k, int64_t pos0, const std::vector<int32_t>& hcols, int pcol, int R,
                  float* p, const int32_t* img = nullptr) {   // img: per row, its image row in img_buf (-1 text)
        const auto tp = Clock::now();
        std::vector<int64_t> iidx;
        std::vector<float> irows;
        if (img)
            for (int i = 0; i < k; ++i)
                if (img[i] >= 0) {
                    iidx.push_back(i);
                    irows.insert(irows.end(), img_buf.begin() + (size_t) img[i] * m.hp.n_embd,
                                 img_buf.begin() + (size_t) (img[i] + 1) * m.hp.n_embd);
                }
        Built& b = mtp_graph(m, G, R, pos0, false, (int) iidx.size());
        if (!iidx.empty()) {
            ggml_backend_tensor_set(b.in.img, irows.data(), 0, irows.size() * 4);
            ggml_backend_tensor_set(b.in.img_idx, iidx.data(), 0, iidx.size() * 8);
        }
        std::vector<int32_t> t(R, 0), hc(R, hcols.empty() ? 0 : hcols[0]);
        std::copy(tok, tok + k, t.begin());
        std::copy(hcols.begin(), hcols.end(), hc.begin());
        const std::vector<int32_t> r3 = rope3(pos0, R);
        set_inputs(b.in, t.data(), R, pos0, pad_kv(pos0 + R, m.o.max_ctx), 1, 0, CONV_W - 3, k, m.o.max_ctx - 1,
                   r3.data());
        const int32_t out[1] = {k - 1}, pc[1] = {pcol};
        if (b.in.out_ids->buffer) ggml_backend_tensor_set(b.in.out_ids, out, 0, 4);
        ggml_backend_tensor_set(b.in.hidx, hc.data(), 0, (size_t) R * 4);
        ggml_backend_tensor_set(b.in.pidx, pc, 0, 4);
        if (ggml_backend_sched_graph_compute(b.sched, b.gf) != GGML_STATUS_SUCCESS) die("the MTP graph failed");
        int32_t a = 0;
        ggml_backend_tensor_get(b.argmax, &a, 0, 4);
        if (p) ggml_backend_tensor_get(b.pmax, p, 0, 4);
        prof_add("mtpc" + std::to_string(R), tp);
        return a;
    }
    // A draft step: (tok, the block's own last h) at pos.
    int mtp_step(int32_t tok, int64_t pos, float* p) {
        const auto tp = Clock::now();
        Built& b = mtp_graph(m, G, 1, pos, true);
        const std::vector<int32_t> r3 = rope3(pos, 1);
        set_inputs(b.in, &tok, 1, pos, pad_kv(pos + 1, m.o.max_ctx), 1, 0, CONV_W - 3, -1, 0, r3.data());
        if (ggml_backend_sched_graph_compute(b.sched, b.gf) != GGML_STATUS_SUCCESS) die("the MTP graph failed");
        int32_t a = 0;
        ggml_backend_tensor_get(b.argmax, &a, 0, 4);
        if (p) ggml_backend_tensor_get(b.pmax, p, 0, 4);
        prof_add("mtps", tp);
        return a;
    }
    std::vector<int32_t> cols(int first_pending, int from, int count) {   // [P?] from, from+1, ...
        std::vector<int32_t> v;
        if (first_pending) v.push_back(m.pend_col());
        for (int i = 0; i < count; ++i) v.push_back(from + i);
        return v;
    }

    // A DFlash2 block at pos L: [t, MASK x n] -> up to n drafts, walked through the selector's lattice (greedy).
    void df_draft(int32_t t, int64_t L, std::vector<int32_t>& out) {
        const auto tp = Clock::now();
        Drafter& d = *m.df;
        const int B = m.o.dflash_n + 1, K = d.top_k;
        Built& b = df_graph(m, G, B);
        std::vector<int32_t> tok(B, d.mask_id), pos((size_t) B * 4), anc(K, t);
        std::vector<int64_t> ri(B);
        tok[0] = t;
        for (int i = 0; i < B; ++i) {
            pos[i] = pos[B + i] = pos[2 * B + i] = next_p + i;
            pos[3 * B + i] = 0;
            ri[i] = (L + i) % d.ring;
            d.ring_pos[ri[i]] = L + i;
        }
        std::vector<ggml_fp16_t> mask((size_t) d.ring * B);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < B; ++i) {
            const int64_t q = L + i;
            ggml_fp16_t* row = mask.data() + (size_t) i * d.ring;
            for (int cidx = 0; cidx < d.ring; ++cidx) {
                const int64_t p = d.ring_pos[cidx];
                // DFlash2: non-causal in the block; DSpine: causal
                row[cidx] = (p >= 0 && p <= (d.causal ? q : L + B - 1) && q - p < d.swa) ? zero : ninf;
            }
        }
        ggml_backend_tensor_set(b.in.tok, tok.data(), 0, tok.size() * 4);
        ggml_backend_tensor_set(b.in.pos, pos.data(), 0, pos.size() * 4);
        ggml_backend_tensor_set(b.in.ring_idx, ri.data(), 0, ri.size() * 8);
        ggml_backend_tensor_set(b.in.anchor, anc.data(), 0, anc.size() * 4);
        ggml_backend_tensor_set(b.in.mask, mask.data(), 0, mask.size() * 2);
        if (ggml_backend_sched_graph_compute(b.sched, b.gf) != GGML_STATUS_SUCCESS) die("the DFlash2 graph failed");
        std::vector<int32_t> cand((size_t) K * B);
        std::vector<float> score((size_t) K * K * (B - 1));
        ggml_backend_tensor_get(b.argmax, cand.data(), 0, cand.size() * 4);
        ggml_backend_tensor_get(b.probs, score.data(), 0, score.size() * 4);
        out.clear();
        int pred = 0;
        for (int i = 1; i < B; ++i) {
            const float* sc = score.data() + (size_t) (i - 1) * K * K + (size_t) pred * K;
            pred = (int) (std::max_element(sc, sc + K) - sc);
            out.push_back(cand[(size_t) i * K + pred]);
        }
        prof_add("dflash", tp);
    }

    // ---- Conversations on disk: SAVE <path> / RESTORE <path> between requests (serve_qwen.py's conversation_dir).
    // The file holds what the engine holds for the conversation: its tokens, match keys, M-RoPE positions and next
    // position, each DeltaNet layer's live SSM state and conv rows, the KV rows of every attention layer (the MTP
    // block's too) up to its length, the MTP rows not paired yet, and the DFlash2 ring.  Written to <path>.tmp, then
    // renamed; a restore checks the header and the file's size before it touches the device.
    static constexpr uint32_t kSessMagic = 0x53353351;   // "Q35S"
    struct SessHead {
        uint32_t magic, version;
        int32_t n_layer, n_embd, n_vocab, kv_type, mtp, df_layers, conv_ch, kv_dim;
        int64_t ssm_d, len, next_p, kept_prev, mtp_ok;
    };
    std::string sess_err;
    uint64_t sess_bytes(const SessHead& h) const {
        uint64_t n = sizeof(SessHead) + (uint64_t) h.len * (4 + 8 + 12);
        for (const Layer& l : m.L) {
            if (l.conv) n += (uint64_t) h.ssm_d * 4 + (uint64_t) 3 * h.conv_ch * 4;
            if (l.k_cache) n += 2 * (uint64_t) h.len * l.k_cache->nb[1];
        }
        n += (uint64_t) (h.kept_prev + 1) * h.n_embd * 4;
        if (m.df) n += (uint64_t) m.df->ring * 8 + (uint64_t) m.df->L.size() * 2 * m.df->ring * m.df->kv_dim() * 2;
        return n;
    }
    SessHead sess_head() const {
        SessHead h{};
        h.magic = kSessMagic;
        h.version = 1;
        h.n_layer = m.hp.n_layer;
        h.n_embd = m.hp.n_embd;
        h.n_vocab = m.hp.n_vocab;
        h.kv_type = (int32_t) m.o.kv_type;
        h.mtp = m.o.mtp > 0;
        h.df_layers = m.df ? (int32_t) m.df->L.size() : 0;
        h.conv_ch = m.hp.conv_ch();
        h.kv_dim = m.hp.kv_dim();
        h.ssm_d = m.hp.ssm_d();
        h.len = (int64_t) cache.size();
        h.next_p = next_p;
        h.kept_prev = kept_prev;
        h.mtp_ok = mtp_ok;
        return h;
    }
    // moves `bytes` between a device tensor (from byte `off`) and the file through a 16 MiB stage
    bool sess_io(FILE* f, ggml_tensor* t, size_t off, size_t bytes, bool save, uint64_t& done, uint64_t total) {
        static std::vector<uint8_t> stage(16u << 20);
        for (size_t at = 0; at < bytes;) {
            const size_t n = std::min(stage.size(), bytes - at);
            if (save) {
                ggml_backend_tensor_get(t, stage.data(), off + at, n);
                if (std::fwrite(stage.data(), 1, n, f) != n) return false;
            } else {
                if (std::fread(stage.data(), 1, n, f) != n) return false;
                ggml_backend_tensor_set(t, stage.data(), off + at, n);
            }
            at += n;
            done += n;
            std::printf("SESSION %llu %llu\n", (unsigned long long) done, (unsigned long long) total);
            std::fflush(stdout);
        }
        return true;
    }
    bool sess_host(FILE* f, void* p, size_t bytes, bool save, uint64_t& done) {
        const bool ok = save ? std::fwrite(p, 1, bytes, f) == bytes : std::fread(p, 1, bytes, f) == bytes;
        done += bytes;
        return ok;
    }
    // the body after the header, both ways (the state goes to / comes from the live slot and conv rows)
    bool sess_body(FILE* f, const SessHead& h, bool save, uint64_t total) {
        uint64_t done = sizeof(SessHead);
        const size_t L = (size_t) h.len;
        if (!save) {
            cache.resize(L);
            ckey.resize(L);
            rp.resize(L * 3);
        }
        if (!sess_host(f, cache.data(), L * 4, save, done) || !sess_host(f, ckey.data(), L * 8, save, done) ||
            !sess_host(f, rp.data(), L * 12, save, done))
            return false;
        const int C = h.conv_ch;
        const int slot = save ? m.ssm_slot : 0, coff = save ? m.conv_off : CONV_W - 3;
        for (Layer& l : m.L) {
            if (l.conv) {
                if (!sess_io(f, l.ssm, (size_t) slot * h.ssm_d * 4, (size_t) h.ssm_d * 4, save, done, total) ||
                    !sess_io(f, l.conv, (size_t) coff * C * 4, (size_t) 3 * C * 4, save, done, total))
                    return false;
            }
            if (l.k_cache) {
                if (!sess_io(f, l.k_cache, 0, L * l.k_cache->nb[1], save, done, total) ||
                    !sess_io(f, l.v_cache, 0, L * l.v_cache->nb[1], save, done, total))
                    return false;
            }
        }
        for (int64_t c = 0; c < h.kept_prev; ++c)
            if (!sess_io(f, m.hbuf, (size_t) c * m.hbuf->nb[1], (size_t) h.n_embd * 4, save, done, total)) return false;
        if (!sess_io(f, m.hbuf, (size_t) m.pend_col() * m.hbuf->nb[1], (size_t) h.n_embd * 4, save, done, total))
            return false;
        if (m.df) {
            if (!sess_host(f, m.df->ring_pos.data(), (size_t) m.df->ring * 8, save, done)) return false;
            for (DfLayer& l : m.df->L)
                if (!sess_io(f, l.k_ring, 0, ggml_nbytes(l.k_ring), save, done, total) ||
                    !sess_io(f, l.v_ring, 0, ggml_nbytes(l.v_ring), save, done, total))
                    return false;
        }
        return true;
    }
    // -> the answer line (SAVED / RESTORED <tokens> <bytes> <ms>, or SERR <kind> <published> <reason>)
    std::string session(bool save, const std::string& path) {
        const auto t0 = Clock::now();
        char buf[512];
        auto serr = [&](const char* kind, const std::string& why) {
            std::snprintf(buf, sizeof buf, "SERR %s 0 %s", kind, why.c_str());
            return std::string(buf);
        };
        if (path.empty()) return serr("invalid", "no path");
        if (save) {
            const SessHead h = sess_head();
            const uint64_t total = sess_bytes(h);
            const std::string tmp = path + ".tmp";
            FILE* f = std::fopen(tmp.c_str(), "wb");
            if (!f) return serr("storage", "cannot write " + tmp);
            bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && sess_body(f, h, true, total);
            ok = (std::fflush(f) == 0) && ok;
            ok = (std::fclose(f) == 0) && ok;
            if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
                std::remove(tmp.c_str());
                return serr("io", "writing " + path + " failed");
            }
            std::snprintf(buf, sizeof buf, "SAVED %lld %llu %.1f", (long long) h.len, (unsigned long long) total,
                          ms_since(t0));
            return buf;
        }
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return serr("invalid", "cannot open " + path);
        SessHead h{};
        const SessHead me = sess_head();
        if (std::fread(&h, sizeof h, 1, f) != 1 || h.magic != kSessMagic || h.version != 1 || h.n_layer != me.n_layer ||
            h.n_embd != me.n_embd || h.n_vocab != me.n_vocab || h.kv_type != me.kv_type || h.mtp != me.mtp ||
            h.df_layers != me.df_layers || h.conv_ch != me.conv_ch || h.kv_dim != me.kv_dim || h.ssm_d != me.ssm_d) {
            std::fclose(f);
            return serr("invalid", path + " was saved by another model or configuration");
        }
        if (h.len <= 0 || h.len + KSNAP + 1 >= m.o.max_ctx) {
            std::fclose(f);
            return serr("invalid", path + " holds more tokens than this context");
        }
        const uint64_t total = sess_bytes(h);
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fseek(f, sizeof h, SEEK_SET);
        if (size < 0 || (uint64_t) size != total) {
            std::fclose(f);
            return serr("invalid", path + " is not the size its header says");
        }
        const bool ok = sess_body(f, h, false, total);
        std::fclose(f);
        if (!ok) {   // the device state is partly written: start from nothing rather than from half a conversation
            clear();
            return serr("io", "reading " + path + " failed; the conversation was dropped");
        }
        m.ssm_slot = 0;
        m.conv_off = CONV_W - 3;
        next_p = (int32_t) h.next_p;
        kept_prev = (int) h.kept_prev;
        mtp_ok = h.mtp_ok != 0;
        std::snprintf(buf, sizeof buf, "RESTORED %lld %llu %.1f", (long long) h.len, (unsigned long long) total,
                      ms_since(t0));
        return buf;
    }

    // Reset to an empty sequence.
    void clear() {
        cache.clear();
        ckey.clear();
        rp.clear();
        next_p = 0;
        mtp_ok = false;
        kept_prev = 0;
        if (m.df) std::fill(m.df->ring_pos.begin(), m.df->ring_pos.end(), -1);
        m.conv_off = CONV_W - 3;
        m.ssm_slot = 0;
        for (Layer& l : m.L)
            if (l.conv) {
                ggml_backend_tensor_memset(l.conv, 0, 0, ggml_nbytes(l.conv));
                ggml_backend_tensor_memset(l.ssm, 0, 0, ggml_nbytes(l.ssm));
            }
    }

    // Reads the prompt (from where the cache stops matching), in ubatch windows; the MTP block reads its pairs.
    // Returns the last row's logits (or argmax when greedy).
    struct PromptOut {
        int64_t reused = 0;
        double ms = 0;
    };
    // Checkpoints of the recurrent state in host RAM: the DeltaNet layers' SSM state and conv rows and the MTP
    // block's pending row after a prefix of a prompt.  Everything else the engine keeps (the KV caches, the MTP
    // block's KV, the DFlash2 ring) is indexed by position, so it is still right for any prefix of what was read.  A
    // chat client resends the conversation with the last answer rendered again (its thinking dropped), so the new
    // prompt leaves the cached tokens inside the last assistant turn: the checkpoint in front of the prompt's last
    // <|im_start|> is the point the next turn shares.
    struct Ckpt {
        std::vector<int32_t> tok, rp;
        std::vector<int64_t> key;
        int32_t next_p = 0;
        std::vector<float> ssm, conv, pend;
        uint64_t used = 0;
    };
    static constexpr size_t kCkMax = 4;   // ~150 MB each
    static constexpr int32_t kImStart = 248045;
    std::vector<Ckpt> cks;
    uint64_t ck_tick = 0;
    // after a trunk window (state in slot 0, the conv window's tail) and its MTP pairs
    void save_ck(size_t n, int32_t np) {   // the first n cached tokens; np: the next text position after them
        for (Ckpt& c : cks)
            if (c.key.size() == n && std::equal(c.key.begin(), c.key.end(), ckey.begin())) { c.used = ++ck_tick; return; }
        const auto tp = Clock::now();
        const int C = m.hp.conv_ch();
        const size_t D = (size_t) m.hp.ssm_d();
        Ckpt* c;
        if (cks.size() < kCkMax) c = &cks.emplace_back();
        else c = &*std::min_element(cks.begin(), cks.end(), [](const Ckpt& a, const Ckpt& b) { return a.used < b.used; });
        c->tok.assign(cache.begin(), cache.begin() + n);
        c->key.assign(ckey.begin(), ckey.begin() + n);
        c->rp.assign(rp.begin(), rp.begin() + n * 3);
        c->next_p = np;
        c->used = ++ck_tick;
        int nl = 0;
        for (Layer& l : m.L) nl += l.conv ? 1 : 0;
        c->ssm.resize(D * nl);
        c->conv.resize((size_t) 3 * C * nl);
        int i = 0;
        for (Layer& l : m.L) {
            if (!l.conv) continue;
            ggml_backend_tensor_get(l.ssm, c->ssm.data() + D * i, 0, D * 4);
            ggml_backend_tensor_get(l.conv, c->conv.data() + (size_t) 3 * C * i, (size_t) (CONV_W - 3) * C * 4,
                                    (size_t) 3 * C * 4);
            ++i;
        }
        c->pend.clear();
        if (m.o.mtp > 0) {
            c->pend.resize(m.hp.n_embd);
            ggml_backend_tensor_get(m.hbuf, c->pend.data(), (size_t) m.pend_col() * m.hbuf->nb[1], (size_t) m.hp.n_embd * 4);
        }
        prof_add("ckpt_save", tp);
    }
    void restore_ck(Ckpt& c) {
        const int C = m.hp.conv_ch();
        const size_t D = (size_t) m.hp.ssm_d();
        int i = 0;
        for (Layer& l : m.L) {
            if (!l.conv) continue;
            ggml_backend_tensor_set(l.ssm, c.ssm.data() + D * i, 0, D * 4);
            ggml_backend_tensor_set(l.conv, c.conv.data() + (size_t) 3 * C * i, (size_t) (CONV_W - 3) * C * 4,
                                    (size_t) 3 * C * 4);
            ++i;
        }
        if (!c.pend.empty())
            ggml_backend_tensor_set(m.hbuf, c.pend.data(), (size_t) m.pend_col() * m.hbuf->nb[1], (size_t) m.hp.n_embd * 4);
        m.ssm_slot = 0;
        m.conv_off = CONV_W - 3;
        cache = c.tok;
        ckey = c.key;
        rp = c.rp;
        next_p = c.next_p;
        mtp_ok = m.o.mtp > 0 && !c.pend.empty();
        kept_prev = 0;
        c.used = ++ck_tick;
    }

    // A prompt: its token ids, what each is matched on (the id, or an image row's key), each one's M-RoPE position,
    // the next text position after it, and per token the image row it takes (-1: text; rows in img_buf).
    struct Req {
        std::vector<int32_t> ids, rp, img_row;
        std::vector<int64_t> key;
        int32_t next_p = 0;
        void text(const std::vector<int32_t>& t) {
            ids = t;
            key.assign(t.begin(), t.end());
            rp.resize(t.size() * 3);
            for (size_t i = 0; i < t.size(); ++i) rp[i * 3] = rp[i * 3 + 1] = rp[i * 3 + 2] = (int32_t) i;
            img_row.assign(t.size(), -1);
            next_p = (int32_t) t.size();
        }
    };
    PromptOut prompt(const std::vector<int32_t>& ids_in, bool greedy, std::vector<float>& last_logits, int32_t& last_am,
                     bool report, const Req* rq = nullptr) {
        Req text_rq;
        if (!rq) {
            text_rq.text(ids_in);
            rq = &text_rq;
        }
        const std::vector<int32_t>& ids = rq->ids;
        const std::vector<int64_t>& keys = rq->key;
        PromptOut po;
        const auto t0 = Clock::now();
        // reuse: the cache when it is a prefix of the prompt, or the longest checkpoint that is (the recurrent state
        // cannot go back by itself)
        size_t common = 0;
        if (cache.size() < ids.size() && std::equal(ckey.begin(), ckey.end(), keys.begin())) common = cache.size();
        Ckpt* best = nullptr;
        for (Ckpt& c : cks)
            if (c.key.size() > common && c.key.size() < ids.size() && std::equal(c.key.begin(), c.key.end(), keys.begin()) &&
                (!best || c.key.size() > best->key.size()))
                best = &c;
        if (best) {
            restore_ck(*best);
            common = cache.size();
        } else if (common == 0 && !cache.empty()) {
            clear();
        }
        po.reused = (int64_t) common;
        // where this prompt's checkpoint goes: in front of its last <|im_start|> (the turn the next request renders
        // again), when that is past what is reused
        size_t ck_at = 0;
        for (size_t i = ids.size(); i-- > common + 1;)
            if (ids[i] == kImStart) { ck_at = i; break; }
        size_t at = cache.size();
        const bool use_mtp = m.o.mtp > 0;
        if (use_mtp && at > 0 && mtp_ok && kept_prev > 0) {
            // the last generation's kept rows: their pairs first (the trunk window below overwrites hbuf)
            mtp_catch(cache.data() + at - kept_prev, kept_prev, (int64_t) at - kept_prev, cols(1, 0, kept_prev - 1),
                      kept_prev - 1, kept_prev, nullptr);
        }
        kept_prev = 0;
        const int64_t total = (int64_t) ids.size();
        while (at < ids.size()) {
            // the window shrinks with the depth: its mask and attention scratch grow with rows x context, and past
            // 64K a 1024-row window of the MTP block alone wanted 0.8-1.2 GiB more than one card had left at 256K
            const int64_t depth = pad_kv((int64_t) at + m.o.ubatch, m.o.max_ctx);
            int win = m.o.ubatch;
            if (depth > 65536) win = std::max(256, (int) ((int64_t) m.o.ubatch * 65536 / depth) / 256 * 256);
            int n = (int) std::min<size_t>(win, ids.size() - at);
            if (ck_at > at && ck_at < at + n) n = (int) (ck_at - at);   // a window ends at the checkpoint
            const bool lastw = at + n == ids.size();
            std::vector<int32_t> am;
            rq_key = keys.data() + at;
            rq_rp = rq->rp.data() + at * 3;
            rq_img = rq->img_row.data() + at;
            trunk(ids.data() + at, n, 1, 1, lastw ? &am : nullptr, lastw && !greedy ? &last_logits : nullptr);
            rq_key = nullptr;
            rq_rp = nullptr;
            rq_img = nullptr;
            next_p = at + n < ids.size() ? rq->rp[(at + n) * 3] : rq->next_p;
            if (lastw) last_am = am[0];
            if (use_mtp) {
                // pairs (h_{p-1}, x_p) at position p for the window's tokens; the first window skips x_0
                if (at == 0) {
                    if (n > 1)
                        mtp_catch(ids.data() + 1, n - 1, 1, cols(0, 0, n - 1), n - 1, n - 1, nullptr,
                                  rq->img_row.data() + 1);
                    else {
                        // a one-token prompt: pending <- h_0 through a 1-row pass that writes nothing useful
                        mtp_catch(ids.data(), 1, 0, cols(0, 0, 1), 0, 1, nullptr);
                    }
                } else if (mtp_ok) {
                    mtp_catch(ids.data() + at, n, (int64_t) at, cols(1, 0, n - 1), n - 1, n, nullptr,
                              rq->img_row.data() + at);
                }
                mtp_ok = true;
            }
            at += n;
            if (at == ck_at || at == ids.size()) save_ck(at, next_p);
            if (report) {
                const double ms = ms_since(t0);
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) at, (long long) total, ms,
                            (at - common) / std::max(1e-3, ms / 1000));
                std::fflush(stdout);
            }
            if (stop && stop->load()) break;
        }
        po.ms = ms_since(t0);
        return po;
    }
};

// ---------------------------------------------------------------------------------------------------------------- decode
struct GenResult {
    int64_t produced = 0;
    std::string finish = "length";
    double ms = 0;
};

// Generates up to max_new tokens after the prompt; `emit` gets each token.  With MTP: draft d tokens on the block,
// verify them in one trunk window, keep the agreeing prefix plus the trunk's own next token.
template <class Emit>
GenResult generate(Engine& E, int32_t first, const Sampling& s, int max_new, Emit emit) {
    Model& m = E.m;
    const int V = m.hp.n_vocab;
    GenResult gr;
    const auto t0 = Clock::now();
    const bool greedy = s.temp <= 0.0f && s.rep == 1.0f && s.freq == 0.0f && s.pres == 0.0f;
    auto is_eos = [&](int32_t t) { return std::find(m.o.eos.begin(), m.o.eos.end(), t) != m.o.eos.end(); };
    std::vector<int32_t> hist = E.cache;
    int32_t t = first;   // the next token: produced, not yet in the trunk
    auto out = [&](int32_t y) -> bool {
        emit(y);
        hist.push_back(y);
        ++gr.produced;
        if (is_eos(y)) { gr.finish = "stop"; return false; }
        if (gr.produced >= max_new) { gr.finish = "length"; return false; }
        if (E.stop && E.stop->load()) { gr.finish = "cancel"; return false; }
        return true;
    };
    if (!out(t)) { gr.ms = ms_since(t0); return gr; }
    std::vector<int32_t> am;
    std::vector<float> lg;
    if (m.df && m.o.spec_auto && m.o.mtp > 0) {
        // Both drafters, chosen by the text: inside a ``` block or a <tool_call> (Engine::code_scan) the block drafter
        // drafts, elsewhere MTP.  The MTP block is caught up every round (its pairs must follow the cache whichever
        // drafter drafts; the catch-up also gives its first draft); the block drafter's ring is filled by every trunk
        // window.  Each drafter's rate (tokens kept per millisecond of the round, a moving average) is kept: inside
        // code, when the block drafter's falls below MTP's, MTP drafts until the code ends.
        const int D = m.o.mtp;
        const int R = std::max(D, m.o.dflash_n) + 2;   // catch-up rows: up to dflash_n + 1 kept, then t
        static const double alpha = 0.25;
        static const double give_up = std::getenv("ALMOND_Q35_AUTO_GIVEUP") ? std::atof(std::getenv("ALMOND_Q35_AUTO_GIVEUP")) : 0.9;
        double rate[2] = {-1.0, -1.0};                 // 0 = MTP, 1 = the block drafter
        bool gave_up = false;
        if (E.tick_tok.empty()) E.code_scan_init();
        E.in_fence = E.in_tool = false;
        E.tick_run = 0;
        E.code_scan(t);
        int mode = E.in_code() ? 1 : 0;
        std::fprintf(stderr, "qwen35: --spec auto: starts with %s\n", mode == 1 ? (m.df->dspine ? "DSpine" : "DFlash2") : "MTP");
        // the server's console shows the drafter: a SPEC line on the protocol (servers that do not know it skip it)
        std::printf("SPEC %s\n", mode == 1 ? (m.df->dspine ? "DSpine" : "DFlash2") : "MTP");
        std::vector<int32_t> win, catchup, acc, drafts;
        for (;;) {
            const auto tr = Clock::now();
            const int64_t L = (int64_t) E.cache.size();
            const int kp = E.kept_prev;
            catchup.assign(E.cache.end() - kp, E.cache.end());
            catchup.push_back(t);
            float p = 1.0f;
            const int32_t d = E.mtp_catch(catchup.data(), (int) catchup.size(), L - kp, E.cols(1, 0, kp),
                                          kp > 0 ? kp - 1 : m.pend_col(), R, &p);
            win.assign(1, t);
            if (mode == 1) {
                E.df_draft(t, L, drafts);
                win.insert(win.end(), drafts.begin(), drafts.end());
            } else {
                if (p >= m.o.spec_min_p) win.push_back(d);
                while ((int) win.size() <= D && win.size() > 1) {
                    const int32_t d2 = E.mtp_step(win.back(), L + (int64_t) win.size() - 1, &p);
                    if (p < m.o.spec_min_p) break;
                    win.push_back(d2);
                }
            }
            const int n = (int) win.size();
            E.trunk(win.data(), n, n, n > 1 ? KSNAP : 1, &am, greedy ? nullptr : &lg);
            acc.clear();
            for (int i = 0; i < n; ++i) {
                const int32_t y = greedy ? am[i] : E.smp.sample(lg.data() + (size_t) i * V, V, s, hist);
                acc.push_back(y);
                if (!(i + 1 < n && y == win[i + 1])) break;
                hist.push_back(y);
            }
            hist.resize(hist.size() - (acc.size() - 1));
            const int keep = (int) acc.size();
            E.drafts_offered += n - 1;
            E.drafts_accepted += keep - 1;
            if (n > 1) E.keep_window(n, keep);
            E.kept_prev = keep;
            E.auto_rounds[mode]++;
            // the round's rate, into its drafter's average
            const double r = keep / std::max(1e-3, ms_since(tr));
            rate[mode] = rate[mode] < 0 ? r : (1 - alpha) * rate[mode] + alpha * r;
            for (int i = 0; i < keep; ++i) E.code_scan(acc[i]);
            // the next round's drafter: the text's, unless the block drafter did worse than MTP inside this code
            const bool code = E.in_code();
            const double rate_blk = rate[1];   // before a block's end resets it
            if (!code) {
                gave_up = false;
                rate[1] = -1.0;   // each block of code measures it again
            } else if (mode == 1 && rate[0] > 0 && rate[1] > 0 && rate[1] < give_up * rate[0]) {
                gave_up = true;
            }
            const bool was_gave_up = gave_up && mode == 1;
            const int next = code && !gave_up ? 1 : 0;
            if (next != mode) {
                // the engine log (stderr): every switch, at which generated token and why
                const char* why = next == 1 ? (E.in_tool ? "a tool call opened" : "a ``` block opened")
                                  : was_gave_up ? "the block drafter fell below MTP's rate inside this code"
                                  : "the code ended";
                std::fprintf(stderr, "qwen35: --spec auto: %s -> %s at token %lld (%s; rates MTP %.3f, %s %.3f tok/ms)\n",
                             mode == 1 ? (m.df->dspine ? "DSpine" : "DFlash2") : "MTP",
                             next == 1 ? (m.df->dspine ? "DSpine" : "DFlash2") : "MTP", (long long) (gr.produced + keep), why,
                             rate[0], m.df->dspine ? "DSpine" : "DFlash2", rate_blk);
                std::printf("SPEC %s\n", next == 1 ? (m.df->dspine ? "DSpine" : "DFlash2") : "MTP");
            }
            mode = next;
            bool go = true;
            for (int i = 0; i < keep && go; ++i) go = out(acc[i]);
            if (!go) break;
            t = acc.back();
        }
        gr.ms = ms_since(t0);
        return gr;
    }
    if (m.df) {
        // DFlash2 rounds: a block of drafts from (t, L), one verify window, the agreeing prefix plus the trunk's token
        std::vector<int32_t> drafts, win, acc;
        for (;;) {
            const int64_t L = (int64_t) E.cache.size();
            E.df_draft(t, L, drafts);
            win.assign(1, t);
            win.insert(win.end(), drafts.begin(), drafts.end());
            const int n = (int) win.size();
            E.trunk(win.data(), n, n, KSNAP, &am, greedy ? nullptr : &lg);
            acc.clear();
            for (int i = 0; i < n; ++i) {
                const int32_t y = greedy ? am[i] : E.smp.sample(lg.data() + (size_t) i * V, V, s, hist);
                acc.push_back(y);
                if (!(i + 1 < n && y == win[i + 1])) break;
                hist.push_back(y);
            }
            hist.resize(hist.size() - (acc.size() - 1));
            const int keep = (int) acc.size();
            E.drafts_offered += n - 1;
            E.drafts_accepted += keep - 1;
            E.keep_window(n, keep);
            bool go = true;
            for (int i = 0; i < keep && go; ++i) go = out(acc[i]);
            if (!go) break;
            t = acc.back();
        }
        gr.ms = ms_since(t0);
        return gr;
    }
    const int D = m.o.mtp;
    if (D <= 0) {
        for (;;) {
            E.trunk(&t, 1, 1, 1, &am, greedy ? nullptr : &lg);
            t = greedy ? am[0] : E.smp.sample(lg.data(), V, s, hist);
            if (!out(t)) break;
        }
        gr.ms = ms_since(t0);
        return gr;
    }
    // MTP rounds.  Invariant at the top: cache = confirmed tokens (positions 0..L-1), t at position L, pending =
    // h_{L-1}; the MTP KV holds pairs up to position L-1.  `kept` = the previous verify's accepted rows (their pairs
    // are made in this round's first MTP pass).
    std::vector<int32_t> win, catchup, acc;
    for (;;) {
        const int64_t L = (int64_t) E.cache.size();
        const int kp = E.kept_prev;
        // first draft step, merged with the catch-up: the previous window's kept tokens (their h rows are hbuf
        // 0..kp-1, the row before them is pending) and then (t, h_{L-1}); pending <- h_{L-1}
        catchup.assign(E.cache.end() - kp, E.cache.end());
        catchup.push_back(t);
        float p = 1.0f;
        const int32_t d = E.mtp_catch(catchup.data(), (int) catchup.size(), L - kp, E.cols(1, 0, kp),
                                      kp > 0 ? kp - 1 : m.pend_col(), D + 2, &p);   // kp <= D+1
        win.assign(1, t);
        if (p >= m.o.spec_min_p) win.push_back(d);
        while ((int) win.size() <= D && win.size() > 1) {
            const int32_t d2 = E.mtp_step(win.back(), L + (int64_t) win.size() - 1, &p);
            if (p < m.o.spec_min_p) break;
            win.push_back(d2);
        }
        const int n = (int) win.size();
        E.trunk(win.data(), n, n, n > 1 ? (std::getenv("Q35_K1") ? 1 : KSNAP) : 1, &am, greedy ? nullptr : &lg);
        // row i's own token against the draft at i+1: the agreeing drafts stay, then the trunk's own token
        acc.clear();
        for (int i = 0; i < n; ++i) {
            const int32_t y = greedy ? am[i] : E.smp.sample(lg.data() + (size_t) i * V, V, s, hist);
            acc.push_back(y);
            if (!(i + 1 < n && y == win[i + 1])) break;
            hist.push_back(y);   // the next row's penalties see it (out() below pushes it again: pop first)
        }
        hist.resize(hist.size() - (acc.size() - 1));
        const int keep = (int) acc.size();   // window rows kept: t and the keep-1 agreeing drafts
        E.drafts_offered += n - 1;
        E.drafts_accepted += keep - 1;
        if (n > 1) E.keep_window(n, keep);
        E.kept_prev = keep;
        bool go = true;
        for (int i = 0; i < keep && go; ++i) go = out(acc[i]);
        if (!go) break;
        t = acc.back();
    }
    gr.ms = ms_since(t0);
    return gr;
}

// ---------------------------------------------------------------------------------------------------------------- serve
struct LineQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool closed = false;
    std::atomic<bool> stop{false};
    void run() {
        char* line = nullptr;
        size_t cap = 0;
        ssize_t n;
        while ((n = getline(&line, &cap, stdin)) >= 0) {
            std::string s(line, n);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (s == "STOP") { stop = true; continue; }
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(s);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        closed = true;
        cv.notify_one();
    }
    bool pop(std::string& s) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return closed || !q.empty(); });
        if (q.empty()) return false;
        s = q.front();
        q.pop_front();
        return true;
    }
};

bool parse_ids(const std::string& s, std::vector<int32_t>& out) {
    out.clear();
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        long v = std::strtol(p, &e, 10);
        if (e == p) return false;
        out.push_back((int32_t) v);
        p = e;
        if (*p == ',') ++p;
        else if (*p) return false;
    }
    return !out.empty();
}

// A GENI request's pictures: the embeddings file holds one record per picture in prompt order (int32 'SVE1', n,
// nx, ny, n_embd, then n x n_embd floats, qwen-vision's / almond-vision's format); picture k's rows take the k-th run
// of <|image_pad|> tokens, in order.  Their M-RoPE positions are mtmd's for Qwen3.5: t = p, h = p + y, w = p + x
// (p the position of the run's first token, rows in row-major x fastest), and the text after the picture goes on at
// p + max(nx, ny).  Each row is matched on a key of its picture (the server's img_keys=, or a hash of its rows) and
// its index, so two pictures of the same size are never taken for each other.
bool image_request(Engine& E, const std::string& file, const std::vector<std::string>& f, Engine::Req& rq,
                   std::string& why) {
    constexpr int32_t kImagePad = 248056;
    const int E_ = E.m.hp.n_embd;
    std::vector<std::string> keys;
    for (const std::string& x : f)
        if (x.rfind("img_keys=", 0) == 0)
            for (size_t p = 9; p <= x.size();) {
                size_t q = x.find(',', p);
                if (q == std::string::npos) q = x.size();
                keys.push_back(x.substr(p, q - p));
                p = q + 1;
            }
    FILE* fp = std::fopen(file.c_str(), "rb");
    if (!fp) { why = "cannot open the embeddings file " + file; return false; }
    struct Rec { int32_t n, nx, ny; size_t row0; uint64_t h; };
    std::vector<Rec> recs;
    E.img_buf.clear();
    for (;;) {
        int32_t hd[5];
        if (std::fread(hd, 4, 5, fp) != 5) break;
        if (std::memcmp(hd, "SVE1", 4) != 0 || hd[4] != E_ || hd[1] <= 0 || (int64_t) hd[2] * hd[3] != hd[1]) {
            std::fclose(fp);
            why = "the embeddings file is not a qwen-vision record set for this model (n_embd " + std::to_string(hd[4]) + ")";
            return false;
        }
        const size_t row0 = E.img_buf.size() / E_;
        E.img_buf.resize(E.img_buf.size() + (size_t) hd[1] * E_);
        if (std::fread(E.img_buf.data() + row0 * E_, 4, (size_t) hd[1] * E_, fp) != (size_t) hd[1] * E_) {
            std::fclose(fp);
            why = "the embeddings file is cut short";
            return false;
        }
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the picture's key, or its rows
        const size_t k = recs.size();
        if (k < keys.size()) {
            for (char ch : keys[k]) h = (h ^ (uint8_t) ch) * 1099511628211ull;
        } else {
            const uint8_t* b = (const uint8_t*) (E.img_buf.data() + row0 * E_);
            for (size_t i = 0; i < (size_t) hd[1] * E_ * 4; i += 61) h = (h ^ b[i]) * 1099511628211ull;
        }
        recs.push_back({hd[1], hd[2], hd[3], row0, h});
    }
    std::fclose(fp);
    // the runs of <|image_pad|>
    size_t k = 0;
    int32_t p = 0;
    for (size_t i = 0; i < rq.ids.size();) {
        if (rq.ids[i] != kImagePad) {
            rq.rp[i * 3] = rq.rp[i * 3 + 1] = rq.rp[i * 3 + 2] = p++;
            rq.key[i] = rq.ids[i];
            ++i;
            continue;
        }
        size_t j = i;
        while (j < rq.ids.size() && rq.ids[j] == kImagePad) ++j;
        if (k >= recs.size() || (size_t) recs[k].n != j - i) {
            why = "the prompt's picture " + std::to_string(k + 1) + " has " + std::to_string(j - i) +
                  " image tokens, the embeddings file " + (k < recs.size() ? std::to_string(recs[k].n) : "none");
            return false;
        }
        const Rec& r = recs[k];
        for (int32_t t = 0; t < r.n; ++t) {
            const size_t q = i + t;
            rq.rp[q * 3] = p;
            rq.rp[q * 3 + 1] = p + t / r.nx;
            rq.rp[q * 3 + 2] = p + t % r.nx;
            rq.img_row[q] = (int32_t) (r.row0 + t);
            rq.key[q] = (int64_t) (((r.h ^ ((uint64_t) t * 0x9E3779B97F4A7C15ull)) | (1ull << 62)) & ~(1ull << 63));
        }
        p += std::max(r.nx, r.ny);
        ++k;
        i = j;
    }
    if (k != recs.size()) {
        why = "the embeddings file has " + std::to_string(recs.size()) + " pictures, the prompt " + std::to_string(k);
        return false;
    }
    rq.next_p = p;
    return true;
}

int serve(Engine& E) {
    LineQueue lq;
    E.stop = &lq.stop;
    std::thread rd([&] { lq.run(); });
    rd.detach();
    const Model& m = E.m;
    std::printf("INFO engine=%s model=qwen35 context=%lld kv=%s mtp=%d dflash=%d\n", ALMOND_VERSION,
                (long long) m.o.max_ctx, ggml_type_name(m.o.kv_type), m.o.mtp, m.df ? m.o.dflash_n : 0);
    std::printf("READY %lld stop\n", (long long) m.o.max_ctx);
    std::fflush(stdout);
    std::string line;
    while (lq.pop(line)) {
        if (line.empty()) continue;
        if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
            const bool save = line[0] == 'S';
            std::printf("%s\n", E.session(save, line.substr(save ? 5 : 8)).c_str());
            std::fflush(stdout);
            continue;
        }
        if (line == "CONVCLEAR") {
            E.clear();
            std::printf("CLEARED\n");
            std::fflush(stdout);
            continue;
        }
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (geni) line = "GEN " + line.substr(5);   // GENI <max_new> [keys] [img_keys=..] <embeddings file> <ids>
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unknown command\n");
            std::fflush(stdout);
            continue;
        }
        // GEN <max_new> [key=value ...] <ids>
        std::vector<std::string> f;
        for (size_t i = 4; i < line.size();) {
            size_t j = line.find(' ', i);
            if (j == std::string::npos) j = line.size();
            if (j > i) f.push_back(line.substr(i, j - i));
            i = j + 1;
        }
        std::vector<int32_t> ids;
        if (f.size() < (geni ? 3u : 2u) || !parse_ids(f.back(), ids)) {
            std::printf("ERR bad GEN line\n");
            std::fflush(stdout);
            continue;
        }
        Engine::Req rq;
        rq.text(ids);
        if (geni) {
            std::string why;
            if (!image_request(E, f[f.size() - 2], f, rq, why)) {
                std::printf("ERR %s\n", why.c_str());
                std::fflush(stdout);
                continue;
            }
            f.erase(f.end() - 2);   // the file is not a sampling key
        }
        const int max_new = std::max(1, std::atoi(f[0].c_str()));
        Sampling s;
        for (size_t i = 1; i + 1 < f.size(); ++i) {
            const size_t eq = f[i].find('=');
            if (eq == std::string::npos) continue;
            const std::string k = f[i].substr(0, eq), v = f[i].substr(eq + 1);
            const double x = std::atof(v.c_str());
            if (k == "temperature") s.temp = (float) x;
            else if (k == "top_p") s.top_p = (float) x;
            else if (k == "top_k") s.top_k = (int) x;
            else if (k == "min_p") s.min_p = (float) x;
            else if (k == "penalty_repeat") s.rep = (float) x;
            else if (k == "penalty_freq") s.freq = (float) x;
            else if (k == "penalty_present") s.pres = (float) x;
            else if (k == "penalty_last_n") s.last_n = (int) x;
            else if (k == "seed") s.seed = (uint64_t) x;
        }
        if (s.seed) E.smp.rng.seed(s.seed);
        if ((int64_t) ids.size() + 1 > m.o.max_ctx) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(),
                        (long long) m.o.max_ctx);
            std::fflush(stdout);
            continue;
        }
        lq.stop = false;
        E.drafts_offered = E.drafts_accepted = 0;
        const bool greedy = s.temp <= 0.0f && s.rep == 1.0f && s.freq == 0.0f && s.pres == 0.0f;
        // a prompt that is exactly the cache plus nothing new: read its last token again
        if (E.cache.size() >= ids.size() && std::equal(ids.begin(), ids.end(), E.cache.begin())) E.clear();
        std::vector<float> lg;
        int32_t am = 0;
        auto po = E.prompt(ids, greedy, lg, am, true, &rq);
        if (po.reused > 0) {
            std::printf("RESUME %lld\n", (long long) po.reused);
        }
        if (lq.stop) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %lld\n", ids.size(), po.ms, (long long) po.reused);
            std::fflush(stdout);
            continue;
        }
        std::vector<int32_t> hist = E.cache;
        const int32_t first = greedy ? am : E.smp.sample(lg.data(), m.hp.n_vocab, s, hist);
        const int64_t ctx_left = m.o.max_ctx - (int64_t) E.cache.size() - KSNAP;
        auto gr = generate(E, first, s, (int) std::min<int64_t>(max_new, std::max<int64_t>(1, ctx_left)),
                           [&](int32_t y) {
                               std::printf("T %d\n", y);
                               std::fflush(stdout);
                           });
        if (m.o.spec_auto) {
            std::fprintf(stderr, "qwen35: --spec auto rounds: MTP %lld, DFlash2 %lld\n", (long long) E.auto_rounds[0],
                         (long long) E.auto_rounds[1]);
            E.auto_rounds[0] = E.auto_rounds[1] = 0;
        }
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %lld\n", (long long) gr.produced, ids.size(), po.ms, gr.ms,
                    gr.finish.c_str(), (long long) E.drafts_accepted, (long long) E.drafts_offered,
                    (long long) po.reused);
        std::fflush(stdout);
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------- ppl
int ppl(Engine& E) {
    Model& m = E.m;
    FILE* f = std::fopen(m.o.ppl_file.c_str(), "rb");
    if (!f) die("cannot open " + m.o.ppl_file);
    std::vector<int32_t> ids;
    int32_t v;
    while (std::fread(&v, 4, 1, f) == 1) ids.push_back(v);
    std::fclose(f);
    const int n_ctx = m.o.ppl_ctx, V = m.hp.n_vocab;
    const int n_chunks = (int) (ids.size() / n_ctx);
    if (n_chunks < 1) die("the file has fewer tokens than --ppl-ctx");
    const int ub = std::min(m.o.ubatch, 512);   // the probabilities of a window are ub x vocab floats
    double nll = 0;
    int64_t count = 0;
    std::vector<float> probs;
    const auto t0 = Clock::now();
    for (int ch = 0; ch < n_chunks; ++ch) {
        E.clear();
        const int32_t* tok = ids.data() + (size_t) ch * n_ctx;
        for (int at = 0; at < n_ctx; at += ub) {
            const int n = std::min(ub, n_ctx - at);
            const int64_t pos0 = (int64_t) E.cache.size();
            Built& b = trunk_graph(m, E.G, n, pos0, n, 1, false, true);
            set_inputs(b.in, tok + at, n, pos0, pad_kv(pos0 + n, m.o.max_ctx), n, m.ssm_slot, m.conv_off);
            if (ggml_backend_sched_graph_compute(b.sched, b.gf) != GGML_STATUS_SUCCESS) die("the trunk graph failed");
            probs.resize((size_t) n * V);
            ggml_backend_tensor_get(b.probs, probs.data(), 0, probs.size() * 4);
            E.cache.insert(E.cache.end(), tok + at, tok + at + n);
            m.ssm_slot = 0;
            m.conv_off = CONV_W - 3;
            // llama-perplexity's rule: score the second half of each chunk
            for (int i = 0; i < n; ++i) {
                const int p = at + i;
                if (p < n_ctx / 2 || p + 1 >= n_ctx) continue;
                const double pr = std::max(1e-30, (double) probs[(size_t) i * V + tok[p + 1]]);
                nll -= std::log(pr);
                ++count;
            }
        }
        std::fprintf(stderr, "[%d] %.4f\n", ch + 1, std::exp(nll / count));
    }
    std::printf("PPL %.4f over %lld tokens (%d chunks of %d) in %.1f s\n", std::exp(nll / count), (long long) count,
                n_chunks, n_ctx, ms_since(t0) / 1000);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------- bench
int bench(Engine& E) {
    Model& m = E.m;
    std::vector<int32_t> ids;
    if (!m.o.prompt_ids.empty() && m.o.prompt_ids[0] == '@') {   // @file.i32[:N]: N int32 ids from a file
        std::string fn = m.o.prompt_ids.substr(1);
        size_t want = SIZE_MAX, c = fn.rfind(':');
        if (c != std::string::npos) { want = std::stoull(fn.substr(c + 1)); fn = fn.substr(0, c); }
        FILE* f = std::fopen(fn.c_str(), "rb");
        if (!f) die("cannot open " + fn);
        std::vector<int32_t> all;
        int32_t v;
        while (std::fread(&v, 4, 1, f) == 1) all.push_back(v);
        std::fclose(f);
        while (ids.size() < want && !all.empty())
            for (int32_t x : all) { if (ids.size() >= want) break; ids.push_back(x); }
    } else if (!m.o.prompt_ids.empty()) {
        if (!parse_ids(m.o.prompt_ids, ids)) die("bad --prompt-ids");
    } else {
        for (int i = 0; i < 512; ++i) ids.push_back(1000 + (i * 7919) % 50000);
    }
    const int reps = ids.size() > 8192 ? 1 : 2;
    for (int rep = 0; rep < reps; ++rep) {
        E.clear();
        E.drafts_offered = E.drafts_accepted = 0;
        E.auto_rounds[0] = E.auto_rounds[1] = 0;
        std::vector<float> lg;
        int32_t am = 0;
        auto po = E.prompt(ids, true, lg, am, false);
        Sampling s;
        std::vector<int32_t> outv;
        auto gr = generate(E, am, s, m.o.bench, [&](int32_t y) { outv.push_back(y); });
        std::printf("bench: prompt %zu tok in %.1f ms (%.1f tok/s), decode %lld tok in %.1f ms (%.2f tok/s), drafts "
                    "%lld/%lld\n",
                    ids.size(), po.ms, ids.size() / (po.ms / 1000), (long long) gr.produced, gr.ms,
                    (gr.produced - 1) / (gr.ms / 1000), (long long) E.drafts_accepted, (long long) E.drafts_offered);
        E.prof_print();
        if (m.o.spec_auto)
            std::printf("bench: --spec auto rounds: MTP %lld, DFlash2 %lld\n", (long long) E.auto_rounds[0],
                        (long long) E.auto_rounds[1]);
        if (rep == 0) {
            std::printf("ids:");
            for (size_t i = 0; i < std::min<size_t>(outv.size(), 48); ++i) std::printf(" %d", outv[i]);
            std::printf("\n");
        }
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int qwen35_main(int argc, char** argv) {
    // ggml's debug lines (a "CUDA Graph id N reused" per token) stay out of the log
    ggml_log_set([](ggml_log_level lv, const char* text, void*) {
        if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr);
    }, nullptr);
    Engine E;
    Opts& o = E.m.o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "-m" || a == "--model") o.model = next();
        else if (a == "--gpus" || a == "--gpu") {
            o.gpus.clear();
            std::string s = next();
            for (size_t p = 0; p <= s.size();) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                o.gpus.push_back(std::atoi(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else if (a == "--split") o.split = std::atoi(next().c_str());
        else if (a == "--max-context") o.max_ctx = std::atoll(next().c_str());
        else if (a == "--kv") {
            const std::string k = next();
            if (k == "f16") o.kv_type = GGML_TYPE_F16;
            else if (k == "q8_0" || k == "int8") o.kv_type = GGML_TYPE_Q8_0;
            else die("--kv: f16 or q8_0");
        } else if (a == "--ubatch") o.ubatch = std::atoi(next().c_str());
        else if (a == "--mtp") o.mtp = std::atoi(next().c_str());
        else if (a == "--spec-min-p") o.spec_min_p = (float) std::atof(next().c_str());
        else if (a == "--dflash") o.dflash = next();
        else if (a == "--dflash-n") o.dflash_n = std::atoi(next().c_str());
        else if (a == "--spec") {
            const std::string k = next();
            if (k == "auto") o.spec_auto = true;
            else if (k != "fixed") die("--spec: auto or fixed");
        }
        else if (a == "--serve") o.serve = true;
        else if (a == "--bench") o.bench = std::atoi(next().c_str());
        else if (a == "--prompt-ids") o.prompt_ids = next();
        else if (a == "--ppl") o.ppl_file = next();
        else if (a == "--ppl-ctx") o.ppl_ctx = std::atoi(next().c_str());
        else if (a == "--eos-ids") {
            std::vector<int32_t> e;
            if (!parse_ids(next(), e)) die("bad --eos-ids");
            o.eos = e;
        } else {
            std::fprintf(stderr, "qwen35: ignoring argument %s\n", a.c_str());
        }
    }
    if (o.model.empty()) die("-m <qwen35 GGUF> is required");
    if (o.gpus.empty() || o.gpus.size() > 2) die("--gpus takes one or two cards");
    o.max_ctx = (o.max_ctx + 255) / 256 * 256;
    if (!o.ppl_file.empty()) {
        o.mtp = 0;
        o.max_ctx = std::max<int64_t>(o.max_ctx, (o.ppl_ctx + 255) / 256 * 256);
    }
    load(E.m);
    if (!o.dflash.empty() && o.ppl_file.empty()) load_drafter(E.m);
    if (!o.ppl_file.empty()) return ppl(E);
    if (o.bench > 0) return bench(E);
    if (o.serve) return serve(E);
    die("nothing to do: --serve, --bench N or --ppl FILE");
}

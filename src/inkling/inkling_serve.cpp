// src/inkling/inkling_serve.cpp - almond-inkling: Inkling 2 Mini (GGUF architecture inkling2_loop) behind the engine's
// serve protocol (the lines of src/hy4/hy4_serve.cpp: READY, GEN, PP, T, DONE, SAVE / RESTORE, CONVCLEAR, STOP, QUIT),
// on llama.cpp's own graph for the model (src/models/inkling2*.cpp at the commit build.sh pins, built in
// build-inkling/). A separate executable: libllama carries its own ggml, which cannot link beside the vendored one
// in almond-engine. docs/inkling/PLAN.md has the plan; docs/inkling/NOTES.md what was measured.
//
//   almond-inkling -m MODEL.gguf [-c N] [--gpu 0 | --gpus 0,1] [--kv-q8] [--ubatch N] --serve
//
// The model keeps a short-convolution state per layer visit beside its KV cache, so the cache cannot be cut back to
// an arbitrary length: a prompt that shares a prefix with the last turn keeps the cache only when it extends it
// exactly (the usual chat turn). Otherwise it goes back to the last checkpoint inside the shared prefix: after each
// prompt is read, the state that cannot be cut back (the convolution state and the window layers' cache:
// LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) is copied to RAM, as llama-server's context checkpoints do. An agent's next
// prompt re-renders the model's last turn (its thinking and tool call as the template writes them), which is not the
// tokens the model produced; it does share everything up to the end of the last prompt, so only the new part is read.
// With no usable checkpoint it starts over.
//
// Built with -DALMOND_DSPARK (build-inkling/almond-inkling-dspark, on llama.cpp PR #32218 in
// build-inkling/llama.cpp-dspark), `--dspark DRAFT.gguf [--dspark-n N]` adds speculative decoding with Unsloth's DSpark
// drafter (opt-in). The drafter proposes its argmax (greedy drafts); the target samples every position with the
// request's own sampler and keeps the draft up to the first token it did not sample itself, so the output follows the
// same distribution as without the drafter. A rejected tail is cut back with llama.cpp's per-step recurrent snapshots
// (n_rs_seq = N): no checkpoint copy and no re-decode. Measured in docs/inkling/NOTES.md.
#include "llama.h"
#ifdef ALMOND_DSPARK
#include "common.h"
#include "speculative.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#ifndef ALMOND_VERSION
#define ALMOND_VERSION "dev"
#endif

namespace {

struct LineQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool closed = false;
    std::atomic<bool> stop{false};
    void run() {
        char * line = nullptr;
        size_t cap = 0;
        ssize_t n;
        while ((n = getline(&line, &cap, stdin)) >= 0) {
            std::string s(line, (size_t) n);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (s == "STOP") { stop = true; continue; }
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(s);
            cv.notify_one();
        }
        std::free(line);
        std::lock_guard<std::mutex> lk(mu);
        closed = true;
        cv.notify_one();
    }
    bool pop(std::string & s) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return closed || !q.empty(); });
        if (q.empty()) return false;
        s = q.front();
        q.pop_front();
        return true;
    }
};

bool parse_ids(const std::string & s, std::vector<llama_token> & out) {
    out.clear();
    const char * p = s.c_str();
    while (*p) {
        char * e = nullptr;
        const long long v = std::strtoll(p, &e, 10);
        if (e == p) return false;
        out.push_back((llama_token) v);
        p = e;
        if (*p == ',') ++p;
        else if (*p) return false;
    }
    return !out.empty();
}

void say(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void say(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

struct Sampling {
    float temp = 1.0f, top_p = 1.0f, min_p = 0.0f, rep = 1.0f, freq = 0.0f, pres = 0.0f;
    int top_k = 0, last_n = 64;
    uint64_t seed = 0;
};

// the same sampler as hy4_sample (src/hy4/hy4_serve.cpp)
llama_token sample(const float * logits, int64_t V, const Sampling & s, const std::vector<llama_token> & hist,
                   std::mt19937_64 & rng) {
    std::vector<float> l(logits, logits + V);
    if (s.rep != 1.0f || s.freq != 0.0f || s.pres != 0.0f) {
        std::unordered_map<llama_token, int> cnt;
        const size_t from = s.last_n > 0 && hist.size() > (size_t) s.last_n ? hist.size() - (size_t) s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto & [t, c] : cnt) {
            if (t < 0 || t >= V) continue;
            float & x = l[(size_t) t];
            if (s.rep != 1.0f) x = x > 0 ? x / s.rep : x * s.rep;
            x -= s.freq * (float) c + s.pres;
        }
    }
    if (s.temp <= 0.0f) return (llama_token) (std::max_element(l.begin(), l.end()) - l.begin());
    std::vector<int32_t> idx((size_t) V);
    for (int64_t i = 0; i < V; ++i) idx[(size_t) i] = (int32_t) i;
    const size_t cut = std::min<size_t>(s.top_k > 0 ? (size_t) s.top_k : (size_t) V, 4096);
    std::partial_sort(idx.begin(), idx.begin() + (ptrdiff_t) cut, idx.end(),
                      [&](int32_t a, int32_t b) { return l[(size_t) a] > l[(size_t) b]; });
    const size_t k = cut;
    const float mx = l[(size_t) idx[0]];
    std::vector<double> p(k);
    for (size_t i = 0; i < k; ++i) p[i] = std::exp((double) (l[(size_t) idx[i]] - mx) / s.temp);
    size_t keep = k;
    if (s.min_p > 0.0f)
        for (size_t i = 0; i < k; ++i) if (p[i] < s.min_p * p[0]) { keep = std::max<size_t>(1, i); break; }
    if (s.top_p < 1.0f) {
        double acc = 0, tot = 0;
        for (size_t i = 0; i < keep; ++i) tot += p[i];
        for (size_t i = 0; i < keep; ++i) {
            acc += p[i];
            if (acc >= s.top_p * tot) { keep = i + 1; break; }
        }
    }
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng), acc = 0;
    for (size_t i = 0; i < keep; ++i) {
        acc += p[i];
        if (r <= acc) return idx[i];
    }
    return idx[0];
}

struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_batch batch{};
    int n_batch = 512;
    int64_t n_vocab = 0;
    std::vector<llama_token> hist;   // the tokens in the cache
    struct Ckpt { size_t n; std::vector<uint8_t> data; };
    std::deque<Ckpt> ckpts;          // the newest last; at most kMaxCkpt
    static constexpr size_t kMaxCkpt = 4;
#ifdef ALMOND_DSPARK
    common_speculative * spec = nullptr;     // null: no drafter
    llama_context * ctx_dft = nullptr;
    int n_draft = 0;
    common_batch cb;
    long long n_drafted = 0, n_accepted = 0;
#endif

    int64_t n_ctx() const { return llama_n_ctx(ctx); }
    void clear() {
        llama_memory_clear(llama_get_memory(ctx), true);
#ifdef ALMOND_DSPARK
        if (ctx_dft) llama_memory_clear(llama_get_memory(ctx_dft), true);
#endif
        hist.clear();
        ckpts.clear();
    }
    // copy the state that cannot be cut back, at the cache's current end
    double save_ckpt() {
        const auto t0 = std::chrono::steady_clock::now();
        if (!ckpts.empty() && ckpts.back().n == hist.size()) return 0.0;
        const auto fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        Ckpt c{hist.size(), std::vector<uint8_t>(llama_state_seq_get_size_ext(ctx, 0, fl))};
        if (c.data.empty() || llama_state_seq_get_data_ext(ctx, c.data.data(), c.data.size(), 0, fl) != c.data.size())
            return -1.0;
        while (ckpts.size() >= kMaxCkpt) ckpts.pop_front();
        ckpts.push_back(std::move(c));
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    // go back to the newest checkpoint at or before `limit` tokens; 0 when there is none
    size_t restore_ckpt(size_t limit) {
        while (!ckpts.empty() && ckpts.back().n > limit) ckpts.pop_back();
        if (ckpts.empty() || ckpts.back().n == 0) return 0;
        const Ckpt & c = ckpts.back();
        const auto fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        if (llama_state_seq_set_data_ext(ctx, c.data.data(), c.data.size(), 0, fl) != c.data.size()) return 0;
        llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) c.n, -1);
#ifdef ALMOND_DSPARK
        if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, (llama_pos) c.n, -1);
#endif
        hist.resize(c.n);
        return c.n;
    }
#ifdef ALMOND_DSPARK
    // cut the cache (and the drafter's) back to n tokens: within the last batch only (the recurrent snapshots)
    void cut(size_t n) {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) n, -1);
        if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, (llama_pos) n, -1);
        hist.resize(n);
    }
    // one batch through common_batch, so the drafter reads the target's features
    bool run(const llama_token * ids, int n, bool all_logits, std::string & err) {
        cb.clear();
        for (int i = 0; i < n; ++i) cb.add(ids[i], (llama_pos) (hist.size() + i), 0, all_logits || i == n - 1);
        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, cb.get());
        if (rc != 0) { err = "llama_process failed (" + std::to_string(rc) + ")"; return false; }
        if (spec && !common_speculative_process(spec, cb)) { err = "the drafter could not read the batch"; return false; }
        hist.insert(hist.end(), ids, ids + n);
        return true;
    }
#endif
    // feed ids at the end of the cache; logits of the last token into lg
    bool decode(const llama_token * ids, int n, std::vector<float> & lg, std::string & err) {
#ifdef ALMOND_DSPARK
        for (int at = 0; at < n;) {
            const int m = std::min(n_batch, n - at);
            if (!run(ids + at, m, false, err)) return false;
            at += m;
        }
        const float * l0 = llama_get_logits_ith(ctx, -1);
        lg.assign(l0, l0 + n_vocab);
        return true;
#endif
        for (int at = 0; at < n;) {
            const int m = std::min(n_batch, n - at);
            batch.n_tokens = m;
            for (int i = 0; i < m; ++i) {
                batch.token[i] = ids[at + i];
                batch.pos[i] = (llama_pos) (hist.size() + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (at + i == n - 1);
            }
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) { err = "llama_decode failed (" + std::to_string(rc) + ")"; return false; }
            hist.insert(hist.end(), ids + at, ids + at + m);
            at += m;
        }
        const float * l = llama_get_logits_ith(ctx, -1);
        lg.assign(l, l + n_vocab);
        return true;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    std::string path;
    int n_ctx = 65536, gpu = 0, ubatch = 512;
    bool split = false;      // --gpus 0,1: the layers over both cards (llama.cpp's layer split)
    bool kv_q8 = false, serve = false;
    std::vector<int> eos;
    Sampling defaults;
    std::string dspark;      // --dspark DRAFT.gguf (almond-inkling-dspark only)
    int n_draft = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") path = next();
        else if (a == "-c") n_ctx = std::atoi(next());
        else if (a == "--gpu") gpu = std::atoi(next());
        else if (a == "--gpus") split = std::strchr(next(), ',') != nullptr;
        else if (a == "--ubatch") ubatch = std::atoi(next());
        else if (a == "--kv-q8") kv_q8 = true;
        else if (a == "--serve") serve = true;
        else if (a == "--inkling") {}            // the model's flag, as --hy4 / --qwen35 name theirs
        else if (a == "--eos") eos.push_back(std::atoi(next()));
        else if (a == "--temp") defaults.temp = (float) std::atof(next());
        else if (a == "--top-p") defaults.top_p = (float) std::atof(next());
        else if (a == "--dspark") dspark = next();
        else if (a == "--dspark-n") n_draft = std::max(1, std::atoi(next()));
        else { std::fprintf(stderr, "almond-inkling: unknown argument %s\n", a.c_str()); return 2; }
    }
#ifndef ALMOND_DSPARK
    if (!dspark.empty()) {
        std::fprintf(stderr, "almond-inkling: --dspark needs build-inkling/almond-inkling-dspark (build.sh, INKLING_DSPARK=1)\n");
        return 2;
    }
#endif
    if (path.empty() || !serve) {
        std::fprintf(stderr, "usage: almond-inkling -m MODEL.gguf [-c N] [--gpu N] [--kv-q8] [--ubatch N] --serve\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    mp.split_mode = split ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu = gpu;
    Engine eng;
    eng.model = llama_model_load_from_file(path.c_str(), mp);
    if (!eng.model) { std::fprintf(stderr, "almond-inkling: cannot load %s\n", path.c_str()); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) std::max(ubatch, 512);
    cp.n_ubatch = (uint32_t) ubatch;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // the 20 window layers (40 of the 48 visits) keep only their window, not the whole context (llama.cpp's default
    // keeps every token: 85 KB a token at q8_0, 21.7 GB at 256K, against ~17 KB for the 8 full-attention visits)
    cp.swa_full = false;
    if (kv_q8) { cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; }
#ifdef ALMOND_DSPARK
    common_params P;
    if (!dspark.empty()) {
        // per-step snapshots of the recurrent state for the last n_draft tokens of a batch: the rejected tail of a
        // verify batch is cut back without a checkpoint
        cp.n_rs_seq = (uint32_t) n_draft;
        const auto lim = common_speculative_get_output_limits((int32_t) cp.n_batch, 1, n_draft);
        cp.n_outputs_max = (uint32_t) std::max(lim.total, 0);
        cp.n_outputs_max_per_seq = (uint32_t) std::max(lim.per_seq, 0);
    }
#endif
    eng.ctx = llama_init_from_model(eng.model, cp);
    if (!eng.ctx) { std::fprintf(stderr, "almond-inkling: cannot make the context\n"); return 1; }
#ifdef ALMOND_DSPARK
    common_speculative_init_result_ptr spec_init;
    eng.cb = common_batch(eng.ctx);
    if (!dspark.empty()) {
        P.model.path = path;
        P.n_ctx = (int32_t) cp.n_ctx;
        P.n_batch = (int32_t) cp.n_batch;
        P.n_ubatch = (int32_t) cp.n_ubatch;
        P.n_parallel = 1;
        P.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        P.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
        P.speculative.draft.mparams.path = dspark;
        P.speculative.draft.n_max = n_draft;
        P.speculative.draft.n_gpu_layers = 999;
        P.speculative.draft.probabilistic = false;
        P.main_gpu = gpu;
        common_params pd = common_base_params_to_speculative(P);
        spec_init = common_speculative_init_from_params(pd, eng.model, eng.ctx);
        P.speculative.draft.ctx_tgt = eng.ctx;
        P.speculative.draft.ctx_dft = spec_init ? spec_init->context() : nullptr;
        eng.ctx_dft = P.speculative.draft.ctx_dft;
        try { eng.spec = common_speculative_init(P.speculative, 1); } catch (const std::exception & e) {
            std::fprintf(stderr, "almond-inkling: DSpark: %s\n", e.what());
        }
        if (!eng.spec || !eng.ctx_dft) { std::fprintf(stderr, "almond-inkling: cannot load the drafter %s\n", dspark.c_str()); return 1; }
        if (common_context_can_seq_rm(eng.ctx) != COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
            common_context_can_seq_rm(eng.ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            std::fprintf(stderr, "almond-inkling: DSpark needs partial cut-back on both contexts\n");
            return 1;
        }
        eng.n_draft = n_draft;
    }
#endif
    eng.vocab = llama_model_get_vocab(eng.model);
    eng.n_vocab = llama_vocab_n_tokens(eng.vocab);
    eng.n_batch = (int) cp.n_batch;
    eng.batch = llama_batch_init(eng.n_batch, 0, 1);
    if (eos.empty()) eos.push_back(llama_vocab_eos(eng.vocab));

    LineQueue lq;
    std::thread rd([&] { lq.run(); });
    rd.detach();
    std::mt19937_64 rng(std::random_device{}());
#ifdef ALMOND_DSPARK
    say("INFO engine=%s model=inkling context=%lld mtp=0 dspark=%d\n", ALMOND_VERSION, (long long) eng.n_ctx(),
        eng.n_draft);
#else
    say("INFO engine=%s model=inkling context=%lld mtp=0\n", ALMOND_VERSION, (long long) eng.n_ctx());
#endif
    say("READY %lld stop\n", (long long) eng.n_ctx());
    std::string line;
    while (lq.pop(line)) {
        if (line.empty()) continue;
        if (line == "QUIT") break;
        if (line == "CONVCLEAR") { eng.clear(); say("CLEARED 0 0\n"); continue; }
        if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
            const bool save = line[0] == 'S';
            const std::string file = line.substr(save ? 5 : 8);
            const auto t0 = std::chrono::steady_clock::now();
            size_t bytes = 0;
            if (save) {
                bytes = llama_state_seq_save_file(eng.ctx, file.c_str(), 0, eng.hist.data(), eng.hist.size());
            } else {
                std::vector<llama_token> t((size_t) eng.n_ctx());
                size_t n = 0;
                eng.clear();
                bytes = llama_state_seq_load_file(eng.ctx, file.c_str(), 0, t.data(), t.size(), &n);
                if (bytes) eng.hist.assign(t.begin(), t.begin() + (ptrdiff_t) n);
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (bytes) say("%s %zu %zu %.1f\n", save ? "SAVED" : "RESTORED", eng.hist.size(), bytes, ms);
            else { if (!save) eng.clear(); say("SERR %s 0 llama_state_seq_%s_file failed\n", save ? "write" : "read", save ? "save" : "load"); }
            continue;
        }
        if (line.rfind("GENI ", 0) == 0) { say("ERR images are not wired for Inkling yet\n"); continue; }
        if (line.rfind("GEN ", 0) != 0) { say("ERR unknown command\n"); continue; }
        std::vector<std::string> f;
        for (size_t i = 4; i < line.size();) {
            size_t j = line.find(' ', i);
            if (j == std::string::npos) j = line.size();
            if (j > i) f.push_back(line.substr(i, j - i));
            i = j + 1;
        }
        std::vector<llama_token> ids;
        if (f.size() < 2 || !parse_ids(f.back(), ids)) { say("ERR bad GEN line\n"); continue; }
        const int max_new = std::max(1, std::atoi(f[0].c_str()));
        Sampling s = defaults;
        for (size_t i = 1; i + 1 < f.size(); ++i) {
            const size_t eq = f[i].find('=');
            if (eq == std::string::npos) continue;
            const std::string k = f[i].substr(0, eq);
            const double x = std::atof(f[i].c_str() + eq + 1);
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
        if (s.seed) rng.seed(s.seed);
        if ((int64_t) ids.size() + 1 > eng.n_ctx()) {
            say("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), (long long) eng.n_ctx());
            continue;
        }
        lq.stop = false;
        // the cache is kept only when the prompt extends it (the convolution state cannot be cut back)
        size_t keep = 0;
        if (!eng.hist.empty() && eng.hist.size() < ids.size() &&
            std::equal(eng.hist.begin(), eng.hist.end(), ids.begin())) keep = eng.hist.size();
        else {
            size_t same = 0;
            while (same < eng.hist.size() && same < ids.size() && eng.hist[same] == ids[same]) ++same;
            keep = eng.restore_ckpt(std::min(same, ids.size() - 1));
            if (keep == 0) eng.clear();
            else say("INFO checkpoint restored at %zu (shared prefix %zu)\n", keep, same);
        }
        if (keep > 0) say("RESUME %zu\n", keep);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> lg;
        std::string err;
        bool ok = true, cancelled = false;
        const size_t chunk = (size_t) eng.n_batch;
        for (size_t at = keep; at < ids.size();) {
            const size_t n = std::min(chunk, ids.size() - at);
            if (!eng.decode(ids.data() + at, (int) n, lg, err)) { ok = false; break; }
            at += n;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            say("PP %zu %zu %.0f %.1f\n", at, ids.size(), ms, (double) (at - keep) / std::max(ms, 1e-3) * 1000.0);
            if (lq.stop && at < ids.size()) { cancelled = true; break; }
        }
        const double pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) { eng.clear(); say("ERR %s\n", err.c_str()); continue; }
        // the prompt's end: where the next prompt of this conversation will part from the cache
        if (!cancelled && !lq.stop && eng.save_ckpt() < 0) say("INFO checkpoint not saved\n");
        if (cancelled || lq.stop) {
            eng.clear();
            say("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu\n", ids.size(), pp_ms, keep);
            continue;
        }
        const auto t1 = std::chrono::steady_clock::now();
        int produced = 0;
        std::string finish = "length";
        std::vector<llama_token> hist(eng.hist);
        auto emit = [&](llama_token y) {
            say("T %d\n", y);
            ++produced;
            hist.push_back(y);
            if (std::find(eos.begin(), eos.end(), y) != eos.end()) { finish = "stop"; return false; }
            if (produced >= max_new) return false;
            if (lq.stop) { finish = "cancel"; return false; }
            return true;
        };
        llama_token y = sample(lg.data(), eng.n_vocab, s, hist, rng);
        bool more = emit(y);
#ifdef ALMOND_DSPARK
        long long drafted = 0, accepted = 0;
        if (eng.spec && more) {
            common_speculative_begin(eng.spec, 0, eng.hist);
            llama_tokens draft;
            while (more) {
                const size_t base = eng.hist.size();
                if ((int64_t) base + 1 > eng.n_ctx()) break;
                int n_max = (int) std::min<int64_t>({(int64_t) eng.n_draft, eng.n_ctx() - (int64_t) base - 2,
                                                     (int64_t) (max_new - produced - 1)});
                draft.clear();
                if (n_max > 0) {
                    common_speculative_get_draft_params(eng.spec, 0) = {
                        /* .drafting = */ true, /* .n_max = */ n_max, /* .pos0 = */ (llama_pos) base,
                        /* .id_last = */ y, /* .prompt = */ &eng.hist, /* .result = */ &draft,
                    };
                    common_speculative_draft(eng.spec);
                    if ((int) draft.size() > n_max) draft.resize((size_t) n_max);
                    // the drafter's own tokens past the cache are dropped before the target checks them
                    llama_memory_seq_rm(llama_get_memory(eng.ctx_dft), 0, (llama_pos) base, -1);
                }
                std::vector<llama_token> b(1, y);
                b.insert(b.end(), draft.begin(), draft.end());
                if (!eng.run(b.data(), (int) b.size(), true, err)) { ok = false; break; }
                // the target samples each position; the draft holds up to the first token it did not sample
                size_t i = 0, kept = 0;
                for (; i < b.size(); ++i) {
                    const float * l = llama_get_logits_ith(eng.ctx, (int32_t) i);
                    y = sample(l, eng.n_vocab, s, hist, rng);
                    more = emit(y);
                    kept = i + 1;
                    if (!more || i == draft.size() || y != draft[i]) break;
                }
                drafted += (long long) draft.size();
                accepted += (long long) (kept - 1);
                common_speculative_accept(eng.spec, 0, (uint16_t) (kept - 1));
                // the cache keeps the batch's first `kept` tokens; y (the last sampled) goes in with the next batch
                if (base + kept < eng.hist.size()) eng.cut(base + kept);
            }
            eng.n_drafted += drafted;
            eng.n_accepted += accepted;
            if (drafted > 0) say("INFO dspark drafted=%lld accepted=%lld\n", drafted, accepted);
            more = false;
        }
#endif
        while (more) {
            if ((int64_t) eng.hist.size() + 1 > eng.n_ctx()) break;
            if (!eng.decode(&y, 1, lg, err)) { ok = false; break; }
            y = sample(lg.data(), eng.n_vocab, s, hist, rng);
            more = emit(y);
        }
        const double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        if (!ok) { eng.clear(); say("ERR %s\n", err.c_str()); continue; }
        say("DONE %d %zu %.1f %.1f %s 0 0 %zu\n", produced, ids.size(), pp_ms, gen_ms, finish.c_str(), keep);
    }
#ifdef ALMOND_DSPARK
    if (eng.spec) common_speculative_free(eng.spec);
#endif
    llama_batch_free(eng.batch);
    llama_free(eng.ctx);
    llama_model_free(eng.model);
    llama_backend_free();
    return 0;
}

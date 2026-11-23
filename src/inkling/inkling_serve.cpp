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
// exactly (the usual chat turn); otherwise it starts over.
#include "llama.h"

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

    int64_t n_ctx() const { return llama_n_ctx(ctx); }
    void clear() {
        llama_memory_clear(llama_get_memory(ctx), true);
        hist.clear();
    }
    // feed ids at the end of the cache; logits of the last token into lg
    bool decode(const llama_token * ids, int n, std::vector<float> & lg, std::string & err) {
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
        else { std::fprintf(stderr, "almond-inkling: unknown argument %s\n", a.c_str()); return 2; }
    }
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
    eng.ctx = llama_init_from_model(eng.model, cp);
    if (!eng.ctx) { std::fprintf(stderr, "almond-inkling: cannot make the context\n"); return 1; }
    eng.vocab = llama_model_get_vocab(eng.model);
    eng.n_vocab = llama_vocab_n_tokens(eng.vocab);
    eng.n_batch = (int) cp.n_batch;
    eng.batch = llama_batch_init(eng.n_batch, 0, 1);
    if (eos.empty()) eos.push_back(llama_vocab_eos(eng.vocab));

    LineQueue lq;
    std::thread rd([&] { lq.run(); });
    rd.detach();
    std::mt19937_64 rng(std::random_device{}());
    say("INFO engine=%s model=inkling context=%lld mtp=0\n", ALMOND_VERSION, (long long) eng.n_ctx());
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
        else eng.clear();
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
    llama_batch_free(eng.batch);
    llama_free(eng.ctx);
    llama_model_free(eng.model);
    llama_backend_free();
    return 0;
}

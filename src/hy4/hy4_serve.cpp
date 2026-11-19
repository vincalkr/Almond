// src/hy4/hy4_serve.cpp - see hy4_serve.hpp. The lines are those of src/qwen35/qwen35.cpp's serve().
#include "hy4_serve.hpp"

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
#include <string>
#include <thread>
#include <unordered_map>

namespace almond::hy4 {

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

bool parse_ids(const std::string & s, std::vector<int64_t> & out) {
    out.clear();
    const char * p = s.c_str();
    while (*p) {
        char * e = nullptr;
        const long long v = std::strtoll(p, &e, 10);
        if (e == p) return false;
        out.push_back(v);
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

/// GENI's pictures: each SVE1 record's rows take the next run of `pad` tokens (the run must be as long); those ids
/// become negative keys from the picture's key (img_keys=) or its bytes and the row index, so the cache never takes one
/// picture for another, and `rows` points each such position at its row.
bool image_rows(const std::string & file, const std::vector<std::string> & f, int64_t n_embd, int64_t pad,
                std::vector<int64_t> & ids, std::vector<float> & buf, std::vector<const float *> & rows,
                std::vector<int32_t> & yx, std::string & why) {
    std::vector<std::string> keys;
    for (const std::string & x : f)
        if (x.rfind("img_keys=", 0) == 0)
            for (size_t p = 9; p <= x.size();) {
                size_t q = x.find(',', p);
                if (q == std::string::npos) q = x.size();
                keys.push_back(x.substr(p, q - p));
                p = q + 1;
            }
    std::FILE * fp = std::fopen(file.c_str(), "rb");
    if (!fp) { why = "cannot open the embeddings file " + file; return false; }
    struct Rec { int64_t n, nx; size_t row0; uint64_t h; };
    std::vector<Rec> recs;
    for (;;) {
        int32_t hd[5];
        if (std::fread(hd, 4, 5, fp) != 5) break;
        if (std::memcmp(hd, "SVE1", 4) != 0 || hd[4] != n_embd || hd[1] <= 0) {
            std::fclose(fp);
            why = "the embeddings file is not an almond-vision record set for this model";
            return false;
        }
        const size_t row0 = buf.size() / (size_t) n_embd;
        buf.resize(buf.size() + (size_t) hd[1] * (size_t) n_embd);
        if (std::fread(buf.data() + row0 * n_embd, 4, (size_t) hd[1] * n_embd, fp) != (size_t) hd[1] * n_embd) {
            std::fclose(fp);
            why = "the embeddings file is cut short";
            return false;
        }
        uint64_t h = 1469598103934665603ull;
        if (recs.size() < keys.size()) for (char ch : keys[recs.size()]) h = (h ^ (uint8_t) ch) * 1099511628211ull;
        else {
            const uint8_t * b = (const uint8_t *) (buf.data() + row0 * n_embd);
            for (size_t i = 0; i < (size_t) hd[1] * n_embd * 4; i += 61) h = (h ^ b[i]) * 1099511628211ull;
        }
        recs.push_back({ hd[1], hd[2] > 0 && hd[1] % hd[2] == 0 ? hd[2] : hd[1], row0, h });
    }
    std::fclose(fp);
    rows.assign(ids.size(), nullptr);
    yx.assign(2 * ids.size(), -1);
    size_t k = 0;
    for (size_t i = 0; i < ids.size();) {
        if (ids[i] != pad) { ++i; continue; }
        size_t j = i;
        while (j < ids.size() && ids[j] == pad) ++j;
        if (k >= recs.size() || (int64_t) (j - i) != recs[k].n) {
            why = "the prompt's image tokens do not match the embeddings file";
            return false;
        }
        for (size_t r = 0; r < j - i; ++r) {
            rows[i + r] = buf.data() + (recs[k].row0 + r) * (size_t) n_embd;
            yx[2 * (i + r)] = (int32_t) (r / recs[k].nx);        // its line and column (almond-vision's W + 1 by H)
            yx[2 * (i + r) + 1] = (int32_t) (r % recs[k].nx);
            ids[i + r] = -(int64_t) (((recs[k].h * 4099u + r) & 0x3fffffffffffffffull) + 1);
        }
        ++k;
        i = j;
    }
    if (k != recs.size()) { why = "the prompt's image tokens do not match the embeddings file"; return false; }
    return true;
}

}  // namespace

int64_t hy4_sample(const float * logits, int64_t V, const Hy4Sampling & s, const std::vector<int64_t> & hist,
                   std::mt19937_64 & rng) {
    std::vector<float> l(logits, logits + V);
    if (s.rep != 1.0f || s.freq != 0.0f || s.pres != 0.0f) {
        std::unordered_map<int64_t, int> cnt;
        const size_t from = s.last_n > 0 && hist.size() > (size_t) s.last_n ? hist.size() - (size_t) s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto & [t, c] : cnt) {
            if (t < 0 || t >= V) continue;
            float & x = l[(size_t) t];
            if (s.rep != 1.0f) x = x > 0 ? x / s.rep : x * s.rep;
            x -= s.freq * (float) c + s.pres;
        }
    }
    if (s.temp <= 0.0f) return (int64_t) (std::max_element(l.begin(), l.end()) - l.begin());
    std::vector<int64_t> idx((size_t) V);
    for (int64_t i = 0; i < V; ++i) idx[(size_t) i] = i;
    size_t k = s.top_k > 0 ? std::min<size_t>((size_t) s.top_k, (size_t) V) : (size_t) V;
    // the candidates in order: a full sort only when no top_k narrows them
    const size_t cut = std::min<size_t>(k, 4096);
    std::partial_sort(idx.begin(), idx.begin() + (ptrdiff_t) cut, idx.end(),
                      [&](int64_t a, int64_t b) { return l[(size_t) a] > l[(size_t) b]; });
    k = cut;
    const float mx = l[(size_t) idx[0]];
    std::vector<double> p(k);
    double sum = 0;
    for (size_t i = 0; i < k; ++i) { p[i] = std::exp((double) (l[(size_t) idx[i]] - mx) / s.temp); sum += p[i]; }
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

int hy4_serve(Hy4Engine & eng, const std::vector<int64_t> & eos, const Hy4Sampling & defaults, int mtp_n,
              int64_t image_pad) {
    LineQueue lq;
    std::thread rd([&] { lq.run(); });
    rd.detach();
    const Hy4Geometry & G = eng.geometry();
    std::mt19937_64 rng(std::random_device{}());
    say("INFO engine=%s model=hy4 context=%lld mtp=%d\n", ALMOND_VERSION, (long long) eng.n_ctx(), eng.has_mtp() ? 1 : 0);
    say("READY %lld stop\n", (long long) eng.n_ctx());
    std::string line;
    while (lq.pop(line)) {
        if (line.empty()) continue;
        if (line == "QUIT") break;
        if (line == "CONVCLEAR") { eng.truncate(0); say("CLEARED 0 0\n"); continue; }
        if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
            const bool save = line[0] == 'S';
            const std::string path = line.substr(save ? 5 : 8);
            const auto t0 = std::chrono::steady_clock::now();
            int64_t n = 0;
            uint64_t bytes = 0;
            std::string err;
            const bool ok = save ? eng.save(path, n, bytes, err) : eng.restore(path, n, bytes, err);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ok) say("%s %lld %llu %.1f\n", save ? "SAVED" : "RESTORED", (long long) n, (unsigned long long) bytes, ms);
            else say("SERR %s 0 %s\n", save ? "write" : "read", err.c_str());
            continue;
        }
        // GENI <max_new> [keys] [img_keys=a,b] <embeddings file> <ids>: the pictures' rows (almond-vision's SVE1
        // records, in prompt order) take the runs of <｜hy_image▁pad｜> tokens
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (geni) line = "GEN " + line.substr(5);
        if (line.rfind("GEN ", 0) != 0) { say("ERR unknown command\n"); continue; }
        std::vector<std::string> f;
        for (size_t i = 4; i < line.size();) {
            size_t j = line.find(' ', i);
            if (j == std::string::npos) j = line.size();
            if (j > i) f.push_back(line.substr(i, j - i));
            i = j + 1;
        }
        std::vector<int64_t> ids;
        if (f.size() < (geni ? 3u : 2u) || !parse_ids(f.back(), ids)) { say("ERR bad GEN line\n"); continue; }
        std::vector<float> img_buf;
        std::vector<const float *> rows;
        std::vector<int32_t> yx;
        if (geni) {
            std::string why;
            if (!image_rows(f[f.size() - 2], f, eng.geometry().n_embd, image_pad, ids, img_buf, rows, yx, why)) {
                say("ERR %s\n", why.c_str());
                continue;
            }
            f.erase(f.end() - 2);
        }
        const int max_new = std::max(1, std::atoi(f[0].c_str()));
        Hy4Sampling s = defaults;
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
        // reuse the cached prefix; at least the prompt's last token is read again for its logits
        const auto & h = eng.history();
        size_t common = 0;
        while (common < h.size() && common < ids.size() && h[common] == ids[common]) ++common;
        const size_t keep = std::min(common, ids.size() - 1);
        eng.truncate((int64_t) keep);
        if (keep > 0) say("RESUME %zu\n", keep);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> lg;
        std::string err;
        bool ok = true, cancelled = false;
        const int64_t chunk = 512;
        for (size_t at = keep; at < ids.size();) {
            const size_t n = std::min<size_t>((size_t) chunk, ids.size() - at);
            if (!eng.decode(ids.data() + at, (int64_t) n, lg, err, rows.empty() ? nullptr : rows.data() + at,
                            yx.empty() ? nullptr : yx.data() + 2 * at)) {
                ok = false;
                break;
            }
            at += n;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            say("PP %zu %zu %.0f %.1f\n", at, ids.size(), ms, (double) (at - keep) / std::max(ms, 1e-3) * 1000.0);
            if (lq.stop && at < ids.size()) { cancelled = true; break; }
        }
        const double pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) { eng.truncate(0); say("ERR %s\n", err.c_str()); continue; }
        if (cancelled || lq.stop) {
            say("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu\n", ids.size(), pp_ms, keep);
            continue;
        }
        const auto t1 = std::chrono::steady_clock::now();
        int produced = 0;
        std::string finish = "length";
        std::vector<int64_t> hist(eng.history());
        int64_t offered = 0, accepted = 0;
        // emits one token; true when the turn goes on
        auto emit = [&](int64_t y) {
            say("T %lld\n", (long long) y);
            ++produced;
            hist.push_back(y);
            if (std::find(eos.begin(), eos.end(), y) != eos.end()) { finish = "stop"; return false; }
            if (produced >= max_new) return false;
            if (lq.stop) { finish = "cancel"; return false; }
            return true;
        };
        int64_t y = hy4_sample(lg.data(), G.n_vocab, s, hist, rng);
        bool more = emit(y);
        while (more) {
            if (eng.has_mtp() && eng.n_past() + 2 <= eng.n_ctx()) {
                // --mtp: the block's guesses d1..dk after y, all run at once with y; each kept while the token
                // sampled at its place is that guess (exact for any sampling), the rest cut from the cache
                std::vector<int64_t> d;
                if (!(eng.has_dfly() ? eng.dfly_draft(y, mtp_n, d, err) : eng.draft_chain(y, mtp_n, d, err))) { ok = false; break; }
                std::vector<int64_t> run{ y };
                run.insert(run.end(), d.begin(), d.end());
                if (!eng.decode_all(run.data(), (int64_t) run.size(), lg, err)) { ok = false; break; }
                offered += (int64_t) d.size();
                size_t i = 0;
                for (;;) {
                    const int64_t c = hy4_sample(lg.data() + i * (size_t) G.n_vocab, G.n_vocab, s, hist, rng);
                    if (i < d.size() && c == d[i]) {
                        ++accepted;
                        if (!(more = emit(c))) break;
                        ++i;
                        continue;
                    }
                    eng.truncate(eng.n_past() - (int64_t) (d.size() - i));
                    y = c;
                    more = emit(y);
                    break;
                }
                continue;
            }
            if (eng.n_past() + 1 > eng.n_ctx()) break;
            if (!eng.decode(&y, 1, lg, err)) { ok = false; break; }
            y = hy4_sample(lg.data(), G.n_vocab, s, hist, rng);
            more = emit(y);
        }
        const double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        if (!ok) { eng.truncate(0); say("ERR %s\n", err.c_str()); continue; }
        say("DONE %d %zu %.1f %.1f %s %lld %lld %zu\n", produced, ids.size(), pp_ms, gen_ms, finish.c_str(),
            (long long) accepted, (long long) offered, keep);
    }
    return 0;
}

}  // namespace almond::hy4

// src/hy4/hy4_generate.cpp - the Hy4-flash path of almond-engine (--hy4).
//
// Opens the GGUF, checks geometry and tensor map, reports them, and (with --prompt-ids) runs a greedy/temperature
// generation loop over the single-pass forward.  Token ids are the interface: tokenization, chat template and the
// serve protocol belong in the Python shim, as in the other engine paths.
#include "hy4_attn.hpp"
#include "hy4_engine.hpp"
#include "hy4_serve.hpp"
#include "hy4_forward.hpp"
#include "hy4_geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace almond;
using namespace almond::hy4;

namespace {

int parse_csv(const char * s, std::vector<int64_t> & out);

struct Args {
    bool mtp_n_set = false;
    std::string model;
    bool info = false, list = false;
    std::vector<int64_t> prompt;
    int n_predict = 0;
    float temp = -1.0f, top_p = -1.0f;   // -1 = the GGUF's metadata (0.9 / 1.0)
    uint64_t seed = 1;
    bool no_cache = false;               // rerun the whole prefix each step (the old path, for comparison)
    std::string ppl;                     // a file of token ids: perplexity over chunks of -c tokens
    std::string ppl_save, ppl_ref;       // the scored rows' top-64 log-probs to a file / KL and top-1 against such a file
    bool serve = false;                  // the serve protocol on stdin/stdout (serve/server.py)
    std::vector<int64_t> eos{ 120025 };  // <｜hy_end｜> (tokenizer.ggml.eos_token_id)
    bool cpu = false;                    // the CPU reference path (hy4_forward.cpp) instead of the engine
    Hy4EngineOptions eo;
};

std::vector<int> parse_ints(const char * s) {
    std::vector<int64_t> v;
    std::vector<int> out;
    if (parse_csv(s, v)) return out;
    for (auto x : v) out.push_back((int) x);
    return out;
}

int parse_csv(const char * s, std::vector<int64_t> & out) {
    const char * p = s;
    while (*p) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;
        char * end = nullptr;
        long long v = std::strtoll(p, &end, 10);
        if (end == p) return 2;
        out.push_back(v);
        p = end;
    }
    return out.empty() ? 2 : 0;
}

int parse(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        const char * s = argv[i];
        if (!std::strcmp(s, "-m") || !std::strcmp(s, "--model")) {
            if (++i >= argc) return 2;
            a.model = argv[i];
        } else if (!std::strcmp(s, "--info")) a.info = true;
        else if (!std::strcmp(s, "--list-tensors")) a.list = true;
        else if (!std::strcmp(s, "--prompt-ids")) {
            if (++i >= argc) return 2;
            if (parse_csv(argv[i], a.prompt)) return 2;
        } else if (!std::strcmp(s, "-n")) {
            if (++i >= argc) return 2;
            a.n_predict = std::atoi(argv[i]);
        } else if (!std::strcmp(s, "--temp")) {
            if (++i >= argc) return 2;
            a.temp = (float) std::atof(argv[i]);
        } else if (!std::strcmp(s, "--top-p")) {
            if (++i >= argc) return 2;
            a.top_p = (float) std::atof(argv[i]);
        } else if (!std::strcmp(s, "--seed")) {
            if (++i >= argc) return 2;
            a.seed = strtoull(argv[i], nullptr, 10);
        } else if (!std::strcmp(s, "--no-kv-cache")) a.no_cache = true;
        else if (!std::strcmp(s, "--cpu")) a.cpu = true;
        else if (!std::strcmp(s, "--serve")) a.serve = true;
        else if (!std::strcmp(s, "--vision")) {}           // serve/server.py: pictures come as GENI lines
        else if (!std::strcmp(s, "--vram-reserve-mib")) {  // (the encoder runs on the CPU: nothing to keep free)
            if (++i >= argc) return 2;
        }
        else if (!std::strcmp(s, "--layer-split")) {      // serve/server.py adds it for two cards: the engine
            if (++i >= argc) return 2;                    // places the layers itself (Hy4EngineOptions)
            if (std::strcmp(argv[i], "auto")) std::fprintf(stderr, "hy4: --layer-split %s ignored (placement is automatic)\n", argv[i]);
        }
        else if (!std::strcmp(s, "--eos-ids")) {
            if (++i >= argc) return 2;
            a.eos.clear();
            if (parse_csv(argv[i], a.eos)) return 2;
        }
        else if (!std::strcmp(s, "--dense-q8")) a.eo.dense_q8 = true;
        else if (!std::strcmp(s, "--mtp")) a.eo.mtp = true;
        else if (!std::strcmp(s, "--fused-hc")) a.eo.fused_hc = true;
        else if (!std::strcmp(s, "--fused-proj")) a.eo.fused_proj = true;
        else if (!std::strcmp(s, "--down-requant")) {
            if (++i >= argc) return 2;
            a.eo.down_requant = argv[i];
        }
        else if (!std::strcmp(s, "--reserve-mib")) {
            if (++i >= argc) return 2;
            a.eo.reserve_mib = std::atoll(argv[i]);
        }
        else if (!std::strcmp(s, "--dfly")) {
            if (++i >= argc) return false;
            a.eo.dfly = argv[i];
            a.eo.mtp = true;
            if (!a.mtp_n_set) a.eo.mtp_n = 5;
        }
        else if (!std::strcmp(s, "--mtp-n")) {
            a.mtp_n_set = true;
            if (++i >= argc) return 2;
            a.eo.mtp_n = std::max(1, std::atoi(argv[i]));
        }
        else if (!std::strcmp(s, "--engram-place")) {
            if (++i >= argc) return 2;
            a.eo.engram = argv[i];
            if (a.eo.engram != "auto" && a.eo.engram != "ram" && a.eo.engram != "ssd") return 2;
        }
        else if (!std::strcmp(s, "--ppl-save")) {
            if (++i >= argc) return 2;
            a.ppl_save = argv[i];
        } else if (!std::strcmp(s, "--ppl-ref")) {
            if (++i >= argc) return 2;
            a.ppl_ref = argv[i];
        }
        else if (!std::strcmp(s, "--ppl")) {
            if (++i >= argc) return 2;
            a.ppl = argv[i];
        }
        else if (!std::strcmp(s, "--gpus")) {
            if (++i >= argc) return 2;
            a.eo.gpus = parse_ints(argv[i]);
            if (a.eo.gpus.empty()) return 2;
        } else if (!std::strcmp(s, "-c") || !std::strcmp(s, "--ctx")) {
            if (++i >= argc) return 2;
            a.eo.n_ctx = std::atoll(argv[i]);
        } else if (!std::strcmp(s, "-t") || !std::strcmp(s, "--threads")) {
            if (++i >= argc) return 2;
            a.eo.threads = std::atoi(argv[i]);
        } else if (!std::strcmp(s, "--expert-layers-gpu")) {
            if (++i >= argc) return 2;
            a.eo.expert_layers_gpu = std::atoi(argv[i]);
        } else if (!std::strcmp(s, "--ubatch")) {
            if (++i >= argc) return 2;
            a.eo.n_ubatch = std::atoll(argv[i]);
        }
        else if (!std::strcmp(s, "--help")) {
            std::printf("usage: almond-engine --hy4 -m <gguf> [--info] [--list-tensors]\n"
                        "       [--prompt-ids 1,2,3 -n 8 [--temp 0.9] [--top-p 1.0] [--seed 1]]\n"
                        "       [--gpus 0,1] [-c 4096] [-t N] [--expert-layers-gpu N] [--ubatch 512] [--dense-q8] [--engram-place auto|ram|ssd] [--mtp [--mtp-n 1]] [--dfly DFLY.gguf [--mtp-n 5]]\n"
                        "       [--serve [--eos-ids 120025]]   serve/server.py's protocol on stdin/stdout\n"
                        "       [--cpu [--no-kv-cache]]   the CPU reference path\n");
            return 1;
        } else return 2;
    }
    return 0;
}

void report(const GgufModel & m, const Hy4Geometry & G) {
    const TensorInfo * table = m.find("per_layer_token_embd.weight");
    std::printf("hy4: architecture hy_v4_flash\n");
    std::printf("hy4: %lld blocks (%lld dense, %lld draft), hidden %lld, ff %lld, vocab %lld, context %lld\n",
            (long long) G.n_layer, (long long) G.n_dense, (long long) G.n_nextn, (long long) G.n_embd,
            (long long) G.n_ff, (long long) G.n_vocab, (long long) G.n_ctx);
    std::printf("hy4: MLA q_lora %lld kv_lora %lld key %lld/%lld value %lld rope %lld (%.0f/%.0f)\n",
            (long long) G.q_lora, (long long) G.kv_lora, (long long) G.d_key, (long long) G.d_key_mla,
            (long long) G.d_val, (long long) G.d_rope, (double) G.rope_base, (double) G.rope_base_swa);
    std::printf("hy4: heads %lld sparse / %lld sliding, swa %lld, indexer %lldx%lld top %lld kpool %lld tail %lld\n",
            (long long) G.n_head, (long long) G.n_head_swa, (long long) G.swa, (long long) G.idx_heads,
            (long long) G.idx_dim, (long long) G.idx_topk, (long long) G.idx_kpool, (long long) G.idx_tail);
    std::printf("hy4: experts %lld, %lld used, %lld shared, latent %lld, ff %lld, scale %.4f, norm %lld\n",
            (long long) G.n_expert, (long long) G.n_used, (long long) G.n_shared, (long long) G.d_latent,
            (long long) G.d_expert, G.weights_scale, (long long) G.weights_norm);
    if (table)
        std::printf("hy4: Engram table %s: %llu elements, type %u, %.2f GB at 1 byte/element\n",
                table->name.c_str(), (unsigned long long) table->elements(), table->type, table->elements() / 1e9);
}

int64_t sample(const float * logits, int64_t V, float temp, float top_p, std::mt19937_64 & rng) {
    if (temp <= 0.0f) {
        return (int64_t) (std::max_element(logits, logits + V) - logits);
    }
    std::vector<std::pair<float, int64_t>> v((size_t) V);
    float mx = *std::max_element(logits, logits + V);
    for (int64_t i = 0; i < V; ++i) v[(size_t) i] = { std::exp((logits[i] - mx) / temp), i };
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    float sum = 0.0f;
    size_t keep = 0;
    for (; keep < v.size(); ++keep) {
        sum += v[keep].first;
        if (top_p < 1.0f && sum >= top_p) { ++keep; break; }
    }
    std::uniform_real_distribution<float> u(0.0f, sum);
    float r = u(rng), acc = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        acc += v[i].first;
        if (r <= acc) return v[i].second;
    }
    return v[0].second;
}

}  // namespace

int hy4_main(int argc, char ** argv) {
    Args a;
    int rc = parse(argc, argv, a);
    if (rc) return rc == 1 ? 0 : 2;
    if (a.model.empty()) { std::fprintf(stderr, "hy4: -m <gguf> is required\n"); return 2; }

    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(a.model)); }
    catch (const std::exception & e) { std::fprintf(stderr, "hy4: open: %s\n", e.what()); return 1; }
    const GgufModel & m = *mp;

    Hy4Geometry G;
    std::string e = hy4_read_geometry(m, G);
    if (!e.empty()) { std::fprintf(stderr, "hy4: geometry: %s\n", e.c_str()); return 1; }
    e = hy4_check_tensors(m, G);
    if (!e.empty()) { std::fprintf(stderr, "hy4: tensors: %s\n", e.c_str()); return 1; }
    if (a.info) {
        report(m, G);
        if (a.list) for (size_t i = 0; i < m.size(); ++i)
            for (const auto & t : m.shard(i).tensors())
                std::printf("%s  type %u  %llu\n", t.name.c_str(), t.type, (unsigned long long) t.elements());
        return 0;
    }
    if (a.serve) {
        Hy4Engine eng;
        if (!eng.load(a.model, a.eo, e)) { std::printf("ERR %s\n", e.c_str()); std::fflush(stdout); return 1; }
        // a GEN line without a key means that key's neutral value, as for the other engines: serve/server.py leaves
        // temperature out for a greedy request (temperature 0) and sends the configuration's defaults (the model
        // card's 0.9 / 1.0, local/hy4-local.json) for a request that names none
        Hy4Sampling d;
        d.temp = a.temp < 0 ? 0.0f : a.temp;
        d.top_p = a.top_p < 0 ? 1.0f : a.top_p;
        return hy4_serve(eng, a.eos, d, a.eo.mtp_n);
    }
    if (!a.ppl.empty()) {
        // perplexity: the file holds token ids (commas or whitespace); each chunk of n_ctx tokens starts an empty
        // cache and scores its second half, as llama-perplexity does
        std::FILE * f = std::fopen(a.ppl.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "hy4: cannot open %s\n", a.ppl.c_str()); return 1; }
        std::string txt;
        char buf[65536];
        for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) txt.append(buf, n);
        std::fclose(f);
        for (char & ch : txt) if (ch == '\n' || ch == '\t' || ch == '[' || ch == ']') ch = ' ';
        std::vector<int64_t> ids;
        if (parse_csv(txt.c_str(), ids)) { std::fprintf(stderr, "hy4: no ids in %s\n", a.ppl.c_str()); return 1; }
        Hy4Engine eng;
        if (!eng.load(a.model, a.eo, e)) { std::fprintf(stderr, "hy4: %s\n", e.c_str()); return 1; }
        const int64_t n_ctx = a.eo.n_ctx, V = G.n_vocab;
        double nll = 0; int64_t cnt = 0;
        // --ppl-save / --ppl-ref: per scored row the reference's 64 most likely ids and their log-probs; against it, the
        // KL over those 64 (the reference renormalized there, as Lucebox scores against its teacher's top 256) and how
        // often both rank the same token first
        constexpr int TOPK = 64;
        std::FILE * fsave = a.ppl_save.empty() ? nullptr : std::fopen(a.ppl_save.c_str(), "wb");
        std::FILE * fref = a.ppl_ref.empty() ? nullptr : std::fopen(a.ppl_ref.c_str(), "rb");
        double kl = 0; int64_t agree = 0, nref = 0;
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> lg;
        for (int64_t s0 = 0; s0 + n_ctx <= (int64_t) ids.size(); s0 += n_ctx) {
            eng.truncate(0);
            if (!eng.decode_all(ids.data() + s0, n_ctx, lg, e)) { std::fprintf(stderr, "hy4: %s\n", e.c_str()); return 1; }
            for (int64_t i = n_ctx / 2; i < n_ctx - 1; ++i) {
                const float * r = lg.data() + (size_t) (i * V);
                float mx = r[0];
                for (int64_t k = 1; k < V; ++k) mx = std::max(mx, r[k]);
                double se = 0;
                for (int64_t k = 0; k < V; ++k) se += std::exp((double) (r[k] - mx));
                const double lse = std::log(se) + mx;
                nll += lse - r[ids[(size_t) (s0 + i + 1)]];
                ++cnt;
                if (fsave || fref) {
                    int64_t am = 0;
                    for (int64_t k = 1; k < V; ++k) if (r[k] > r[am]) am = k;
                    if (fsave) {
                        std::vector<int32_t> top((size_t) V);
                        for (int64_t k = 0; k < V; ++k) top[(size_t) k] = (int32_t) k;
                        std::partial_sort(top.begin(), top.begin() + TOPK, top.end(), [&](int32_t x, int32_t y) { return r[x] > r[y]; });
                        float lp[TOPK];
                        for (int k = 0; k < TOPK; ++k) lp[k] = (float) (r[top[(size_t) k]] - lse);
                        std::fwrite(top.data(), 4, TOPK, fsave);
                        std::fwrite(lp, 4, TOPK, fsave);
                    }
                    if (fref) {
                        int32_t ti[TOPK]; float tl[TOPK];
                        if (std::fread(ti, 4, TOPK, fref) == TOPK && std::fread(tl, 4, TOPK, fref) == TOPK) {
                            double z = 0, ours = 0;
                            for (int k = 0; k < TOPK; ++k) { z += std::exp((double) tl[k]); ours += std::exp((double) r[ti[k]] - lse); }
                            for (int k = 0; k < TOPK; ++k) {
                                const double pt = std::exp((double) tl[k]) / z;
                                const double po = std::exp((double) r[ti[k]] - lse) / ours;
                                kl += pt * (std::log(pt) - std::log(std::max(po, 1e-30)));
                            }
                            agree += ti[0] == am;
                            ++nref;
                        }
                    }
                }
            }
            std::printf("hy4: chunk %lld: ppl %.4f\n", (long long) (s0 / n_ctx + 1), std::exp(nll / cnt));
            std::fflush(stdout);
        }
        if (fsave) std::fclose(fsave);
        if (fref) {
            std::fclose(fref);
            std::printf("hy4: against %s: KL %.5f, top-1 agreement %.2f%% over %lld rows\n", a.ppl_ref.c_str(),
                        kl / std::max<int64_t>(nref, 1), 100.0 * agree / std::max<int64_t>(nref, 1), (long long) nref);
        }
        std::printf("hy4: ppl %.4f over %lld tokens (%.1f s)\n", std::exp(nll / std::max<int64_t>(cnt, 1)), (long long) cnt,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return 0;
    }
    if (a.prompt.empty() || a.n_predict <= 0) {
        std::fprintf(stderr, "hy4: use --info, or --prompt-ids and -n to generate\n"); return 2;
    }
    const float temp = a.temp < 0 ? 0.9f : a.temp;
    const float top_p = a.top_p < 0 ? 1.0f : a.top_p;
    std::mt19937_64 rng(a.seed);
    std::vector<int64_t> toks = a.prompt;
    std::printf("hy4: generate %d tokens, temp %.2f top_p %.2f\n", a.n_predict, temp, top_p);
    const int64_t ng = G.ple_ngram - 1;
    const int64_t max_ctx = hy4_max_exact_ctx(G);
    if ((int64_t) toks.size() + a.n_predict > max_ctx)
        std::fprintf(stderr, "hy4: warning: past %lld tokens the sparse layers need the k-pool indexer, which is not "
                     "implemented; attention there is full, not the model's\n", (long long) max_ctx);
    // Rows [from, to) of the sequence: their positions and their Engram histories (raw ids, newest first, -1 before
    // the start).
    auto run = [&](int64_t from, int64_t to, Hy4KvCache * cache, std::vector<float> & logits) {
        const int64_t nt = to - from;
        std::vector<int64_t> prev((size_t) (nt * ng), -1);
        std::vector<int32_t> pos((size_t) nt);
        for (int64_t i = 0; i < nt; ++i) {
            const int64_t t = from + i;
            pos[(size_t) i] = (int32_t) t;
            for (int64_t s = 0; s < ng; ++s)
                prev[(size_t) (i * ng + s)] = t - 1 - s >= 0 ? toks[(size_t) (t - 1 - s)] : -1;
        }
        return hy4_forward_cpu(m, G, nt, toks.data() + from, prev.data(), pos.data(), logits, e, cache);
    };
    if (!a.cpu) {
        Hy4Engine eng;
        auto t0 = std::chrono::steady_clock::now();
        if (!eng.load(a.model, a.eo, e)) { std::fprintf(stderr, "hy4: %s\n", e.c_str()); return 1; }
        std::printf("hy4: loaded in %.1f s, context %lld\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (long long) eng.n_ctx());
        std::vector<float> logits;
        std::vector<int64_t> pending = toks;
        double t_prompt = 0, t_gen = 0;
        if (eng.has_mtp() && !std::getenv("HY4_NO_SPEC")) {        // one draft per step from the MTP block, checked with the token before it
            auto t1 = std::chrono::steady_clock::now();
            if (!eng.decode(toks.data(), (int64_t) toks.size(), logits, e)) { std::fprintf(stderr, "hy4: %s\n", e.c_str()); return 1; }
            t_prompt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            int64_t y = sample(logits.data(), G.n_vocab, temp, top_p, rng);
            std::vector<int64_t> out{ y };
            int rounds = 0, accepted = 0;
            auto t2 = std::chrono::steady_clock::now();
            int offered = 0;
            while ((int) out.size() < a.n_predict) {
                std::vector<int64_t> d;
                if (!(eng.has_dfly() ? eng.dfly_draft(y, a.eo.mtp_n, d, e) : eng.draft_chain(y, a.eo.mtp_n, d, e))) {
                    std::fprintf(stderr, "hy4: draft: %s\n", e.c_str()); return 1;
                }
                std::vector<int64_t> run{ y };
                run.insert(run.end(), d.begin(), d.end());
                if (!eng.decode_all(run.data(), (int64_t) run.size(), logits, e)) { std::fprintf(stderr, "hy4: %s\n", e.c_str()); return 1; }
                ++rounds;
                offered += (int) d.size();
                size_t i = 0;
                for (;;) {
                    const int64_t cnd = sample(logits.data() + i * (size_t) G.n_vocab, G.n_vocab, temp, top_p, rng);
                    if (i < d.size() && cnd == d[i]) { ++accepted; out.push_back(cnd); ++i; continue; }
                    eng.truncate(eng.n_past() - (int64_t) (d.size() - i));
                    y = cnd;
                    break;
                }
                out.push_back(y);
            }
            t_gen = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
            std::printf("hy4: ids:");
            for (size_t i = 0; i < out.size() && i < 32; ++i) std::printf(" %lld", (long long) out[i]);
            std::printf("\nhy4: %s: %d rounds, %d of %d drafts accepted (%.2f tokens a round)\n", eng.has_dfly() ? "DFly2" : "MTP", rounds,
                        accepted, offered, (double) (accepted + rounds) / std::max(rounds, 1));
            std::printf("hy4: prompt %zu tokens %.2f s (%.1f tok/s); %zu generated tokens %.2f s (%.2f tok/s)\n", toks.size(),
                        t_prompt, toks.size() / std::max(t_prompt, 1e-9), out.size() - 1, t_gen,
                        (out.size() - 1) / std::max(t_gen, 1e-9));
            return 0;
        }
        for (int step = 0; step < a.n_predict; ++step) {
            auto t1 = std::chrono::steady_clock::now();
            if (!eng.decode(pending.data(), (int64_t) pending.size(), logits, e)) {
                std::fprintf(stderr, "hy4: decode: %s\n", e.c_str()); return 1;
            }
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            (step == 0 ? t_prompt : t_gen) += sec;
            const int64_t next = sample(logits.data(), G.n_vocab, temp, top_p, rng);
            std::printf("hy4: step %d: token %lld (%.3fs, %lld new / %lld context tokens)\n", step, (long long) next, sec,
                        (long long) pending.size(), (long long) eng.n_past());
            std::fflush(stdout);
            toks.push_back(next);
            pending.assign(1, next);
        }
        std::printf("hy4: prompt %zu tokens %.2f s (%.1f tok/s); %d generated tokens %.2f s (%.2f tok/s)\n", a.prompt.size(),
                    t_prompt, a.prompt.size() / std::max(t_prompt, 1e-9), a.n_predict - 1, t_gen,
                    (a.n_predict - 1) / std::max(t_gen, 1e-9));
        return 0;
    }
    Hy4KvCache cache;
    for (int step = 0; step < a.n_predict; ++step) {
        const int64_t n = (int64_t) toks.size();
        const int64_t from = a.no_cache ? 0 : cache.n_past;
        std::vector<float> logits;
        auto t0 = std::chrono::steady_clock::now();
        if (!run(from, n, a.no_cache ? nullptr : &cache, logits)) {
            std::fprintf(stderr, "hy4: forward: %s\n", e.c_str()); return 1;
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        int64_t next = sample(logits.data() + (size_t) (n - from - 1) * G.n_vocab, G.n_vocab, temp, top_p, rng);
        std::printf("hy4: step %d: token %lld (%.1fs, %lld new / %lld context tokens)\n", step, (long long) next, sec,
                    (long long) (n - from), (long long) n);
        std::fflush(stdout);
        toks.push_back(next);
    }
    return 0;
}

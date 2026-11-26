// hy4_pr_dump <gguf> <outdir> <prompt> : run llama.cpp PR #32517 on one prompt and save named activations as raw f32.
// Build against the PR checkout P: g++ -O2 -std=c++17 hy4_pr_dump.cpp -I$P/include -I$P/ggml/include -L$P/build/bin \
//   -lllama -lggml -lggml-base -Wl,-rpath,$P/build/bin -o hy4_pr_dump
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string g_dir;

static bool wanted(const char * n) {
    static const char * pref[] = { "hc_init", "engram_out-", "hc_attn_out-", "attn_out-", "ffn_out-", "l_out-",
                                   "hc_head", "result_norm", "result_output" };
    for (const char * p : pref) if (!std::strncmp(n, p, std::strlen(p))) return true;
    return false;
}

static bool cb(ggml_tensor * t, bool ask, void *) {
    if (ask) return wanted(t->name);
    if (!wanted(t->name) || t->type != GGML_TYPE_F32) return true;
    std::vector<float> buf(ggml_nelements(t));
    if (ggml_is_contiguous(t)) ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    else return true;
    std::string p = g_dir + "/" + t->name + ".bin";
    FILE * f = std::fopen(p.c_str(), "wb");
    std::fwrite(buf.data(), 4, buf.size(), f);
    std::fclose(f);
    std::fprintf(stderr, "dump %s [%lld %lld %lld %lld]\n", t->name, (long long) t->ne[0], (long long) t->ne[1],
                 (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) return 2;
    g_dir = argv[2];
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(256);
    int n = llama_tokenize(vocab, argv[3], (int) std::strlen(argv[3]), toks.data(), 256, true, true);
    toks.resize(n);
    FILE * f = std::fopen((g_dir + "/tokens.txt").c_str(), "w");
    for (int i = 0; i < n; ++i) std::fprintf(f, "%d%s", toks[i], i + 1 < n ? "," : "\n");
    std::fclose(f);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    cp.n_threads = 12; cp.n_threads_batch = 12;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = n;
    int rc = llama_decode(ctx, b);
    std::fprintf(stderr, "decode rc=%d tokens=%d\n", rc, n);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return rc;
}

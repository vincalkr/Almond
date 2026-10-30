// src/ds41/glm_trunk.hpp - GLM-5.3-Flash (`glm5-next`): the trunk (everything but the routed experts).
//
// The math is llama.cpp's src/models/glm5-next.cpp (the reference this file is checked against, logits on the same
// GGUF): 45 layers of 4 hyper-connection streams (mHC, Sinkhorn mixing); 34 Kimi Delta Attention layers (linear,
// a recurrent state per layer) and 11 nope-MLA layers (a 512-wide latent cache, sparse attention over the cells a
// k-pool indexer picks); layers 0-2 a dense FFN, 3-44 a shared expert plus 8 of 288 routed experts (sigmoid router,
// selection bias, weights normalised and x 2.5).  The routed experts run in the expert tier (Ds41MoeTier).
//
// One sequence, cells = positions.  This first version is the exact path, built for correctness: graphs are built per
// call, the attention is the plain soft_max path, and the conversation-state features of MoeTrunk (park, the
// conversation stores, the layer-major prompt batch) are refused so the driver falls back to the chunked prompt.
#pragma once

#include "moe_trunk.hpp"

#include <memory>
#include <string>

class GlmTrunk : public MoeTrunk {
public:
    struct Impl;
    GlmTrunk();
    ~GlmTrunk() override;
    GlmTrunk(const GlmTrunk &) = delete;
    GlmTrunk & operator=(const GlmTrunk &) = delete;

    bool init(const std::string & gguf, const TrunkConfig & cfg, std::string & err) override;
    const TrunkInfo & info() const override;
    ggml_backend_t backend() const override;
    const std::string & last_error() const override;

    bool begin(const int * tids, int n, int pos0) override;
    bool attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev = nullptr) override;
    bool finish(int il, const float * routed_sum) override;
    bool logits(int row0, int nrows, float * out) override;

    bool predict(int il, int * top_ids, int n_top) override;
    bool predict_multi(int il, int * top_ids) override;
    bool finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) override;
    bool finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                              int * ahead_ids) override;

    bool begin_batch(const int * tids, int B, int pos0) override;
    bool attn_router_span(int il, int col0, int n) override;
    bool attn_router_span_kv(int il, int col0, int n) override;
    const int32_t * batch_ids() const override;
    const float * batch_wts() const override;
    const float * batch_fn() const override;
    int batch_size() const override;
    bool upload_routed(const float * routed_host, int B) override;
    float * routed_dev() override;
    bool finish_span(int il, int col0, int n) override;
    bool set_columns(int nc) override;
    int columns() const override;
    void release_pass() override;
    void set_oom_hook(std::function<bool()> hook) override;

    bool snapshot(int pos) override;
    bool restore() override;
    int snapshot_pos() const override;
    bool park(int64_t positions) override;
    bool unpark() override;
    void drop_park() override;
    bool parked() const override;
    void rearm(int pos) override;
    std::vector<StateRegion> state_regions(int64_t positions) const override;
    void pool_regions(bool snap, std::vector<PoolRegion> & streams, std::vector<PoolRegion> & regions) const override;
    void conv_tensors(std::vector<StreamTensor> & streams, std::vector<ggml_tensor *> & snap_regions) const override;

    bool read_state(std::vector<float> & x, std::vector<float> & pre) override;
    bool read_state_cols(int c0, int n, std::vector<float> & x) override;
    ggml_tensor * weight(const std::string & name) const override;

private:
    std::unique_ptr<Impl> p_;
};

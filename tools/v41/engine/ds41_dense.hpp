// tools/v41/engine/ds41_dense.hpp - DeepSeek-V4.1-Flash (`deepseek41`): the dense (non-routed-expert) half.
//
// Math = DeepSeek's inference/model.py (tools/v41/engine/ds41_golden.py runs it on the GGUF's own weights; that is
// this file's gate).  The structure follows tools/ds4/ds4_dense (ggml graphs, CPU or CUDA backend, per-layer graphs
// built once and reused), but V4.1 differs enough that it is its own class:
//
//   - one PASS = n consecutive tokens at positions pos0..pos0+n-1 (n = 1: decode; n = a prompt chunk: prefill);
//     every graph is the same code for both (state rings are read as [ring | the pass's rows] and written back)
//   - hyper-connections single pass: a sublayer's hc_mixes give the NEXT sublayer's pre-mix (pre_state carries the
//     attention pre-mix from layer l's FFN to layer l+1)
//   - no per-head q norm; window K/V ring of 128; compressed KV (ratio 2 / 1) only on kv source layers, read by the
//     later layers of the same ratio; indexers on index source layers, their top-k reused by the layers after them;
//     layer `cand_layer` (20) picks 2048 candidate blocks of 8 that the later indexers rank inside
//   - attention = each query's own key set (its window + its top-k compressed rows) gathered and attended in one
//     flash-attn with the query as the batch axis (sinks added)
//   - Engram at layers 1 / 14: n-gram hash ids (host, from the token history) -> rows read from the HF shards
//     (Ds41Engram) -> wkv -> per-hc gated add, before the block
//
// Per layer per pass:  attn_router(l) -> [routed experts on fn] -> finish(l).  Then logits(...).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "ds41_geometry.hpp"
#include "moe_trunk.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

/// Ds41Dense's configuration is the trunk's (moe_trunk.hpp); the field docs for DeepSeek are kept there.
using Ds41DenseConfig = TrunkConfig;

class Ds41Dense : public MoeTrunk {
public:
    struct Impl;
    Ds41Dense();
    ~Ds41Dense();
    Ds41Dense(const Ds41Dense &) = delete;
    Ds41Dense & operator=(const Ds41Dense &) = delete;

    bool init(const std::string & gguf, const Ds41DenseConfig & cfg, std::string & err) override;
    const almond::v41::Ds41Geometry & geom() const;
    const TrunkInfo & info() const override;
    /// Blocks present in the file (all 40 for the real model; a prefix for a partial test file).
    int64_t n_layer() const override;
    ggml_backend_t backend() const override;

    /// Start a pass: tokens tids[0..n) at positions pos0.., n <= max_pass.  Embeddings + Engram rows (the hash
    /// needs the 3 tokens before pos0: the history is kept by position, so a re-decoded position overwrites it).
    bool begin(const int * tids, int n, int pos0) override;
    /// Layer-major prompt batch (cfg.prompt_batch > 0): the whole batch's inputs are prepared once - embeddings,
    /// positions, window / compressed / indexer index maps, Engram rows and the initial x/pre state - into the state
    /// tensors' columns 0..B-1; then attn_router_span / finish_span walk layer il's sub-batches in order.  B <=
    /// cfg.prompt_batch (the state columns) and B <= ctx_max - pos0.  Every cross-layer dependency of the sequential
    /// chunk path (the per-layer KV ring / compressed rows / indexer state advances in sub-batch order; layer l's
    /// residual and pre-mix are the columns the sub-batch used) is preserved, so the outputs match --prefill-chunk.
    bool begin_batch(const int * tids, int B, int pos0) override;
    /// Dense half of layer il for the batch's sub-batch starting at column col0 (n tokens, col0 % max_pass == 0):
    /// attention, hc, router, shared expert + the Engram add.  ids / wts / fn land in the batch buffers (batch_ids /
    /// batch_wts / batch_fn) at column col0.  Uses (and advances) layer il's own KV state in sub-batch order.
    bool attn_router_span(int il, int col0, int n) override;
    /// Encoder-only skip (enc-skip): build and run ONLY the persistent-KV half of layer il for the batch's sub-batch
    /// at column col0 - hc_pre / attn_norm, the raw window K/V ring write and, on a kv source layer, the compressed-KV
    /// row + indexer-key row + ratio-2 compressor state (all the state a LATER query can read across positions).  No
    /// attention output, no router, no shared expert, no hc_post / finish, no indexer selection / candidates: nothing
    /// that only ever flows into this layer's own output columns.  For prompt tokens whose layers past the replay
    /// window are never read (the model's cross-position reach is the 128-token window per layer, so the last
    /// `enc_replay` tokens are all that matters).  Writes no host ids/wts/fn, so batch_ids()/batch_wts()/batch_fn()
    /// are left untouched at these columns.
    bool attn_router_span_kv(int il, int col0, int n) override;
    /// The batch's host buffers filled by attn_router_span: ids / wts are n_used * B (token-major), fn is n_embd * B.
    const int32_t * batch_ids() const override;
    const float * batch_wts() const override;
    const float * batch_fn() const override;
    int batch_size() const override;
    /// finish for the batch's sub-batch at column col0: x_state[.., col0+n) = hc_post(shexp + routed, hap, post_f,
    /// comb_f).  `routed_host` is n_embd * B (token-major); call once per layer, before the sub-batches' finishes.
    bool upload_routed(const float * routed_host, int B) override;
    /// The routed-sum input tensor's device memory (columns of n_embd floats; null on the CPU backend): the tier can
    /// write a layer-major batch's routed sums straight into it (run_chunk_dev) instead of upload_routed's host trip.
    float * routed_dev() override;
    bool finish_span(int il, int col0, int n) override;
    /// ALMOND_DENSE_PROF=1: print + reset the batch dense split (build / alloc / compute / Engram / readback) - diagnostic
    void prof_report(const char * tag) override;
    /// Give the prompt passes' compute buffer back to the device (rebuilt by the next multi-token pass): the deferred
    /// VRAM expert cache opens into it after the prompt.
    void release_pass() override;
    /// Called when a graph's buffer cannot be allocated; return true if it freed device memory (the allocation is
    /// then retried once).  E.g. the expert tier's prestage region.
    void set_oom_hook(std::function<bool()> hook) override;
    /// Selection-only routing bias for layer il (n_expert floats): added to the top-k score, not to the mixing weights.
    /// Soft pruning (-penalty on cold experts) / cache-aware routing (+bonus on VRAM-resident ones).  Default 0.
    void set_route_bias(int il, const float * bias) override;
    /// Layer il of the current pass: Engram (layers 1/14), attention, hc, ffn_norm, router, shared expert.
    /// ids / wts: n * n_used (token-major); *fn_host: n * n_embd floats (the routed experts' input, valid until the
    /// next call).  `fn_dev` (optional): the same activations on the backend.
    bool attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev = nullptr) override;
    /// Prefetch hint for layer il of a one-token pass (call before attn_router(il)): the layer's router applied to
    /// rms_norm(hc_pre(state, pre)) * ffn_norm.weight, + the selection bias, ranked; top_ids gets n_top (<= 16).
    /// A hint only: the result does not depend on it.
    bool predict(int il, int * top_ids, int n_top) override;
    /// predict() for EVERY column of the current pass (call before attn_router(il)): top_ids[t * 16 + i] = token t's
    /// i-th predicted expert of layer il (16 per token).  For the multi-token verify pass's expert prefetch; a hint only.
    bool predict_multi(int il, int * top_ids) override;
    /// routed_sum: n * n_embd floats, the routed experts' weighted sum (host).
    bool finish(int il, const float * routed_sum) override;
    /// Decode (n == 1) fusion: finish(il) and predict(next_il) in ONE cached graph - the router prediction for the next
    /// layer reads the state finish just wrote, so the result is identical to finish() then predict(), with one graph
    /// launch / sync / readback less per layer.  top_ids gets the next layer's top n_top (<= 16) predicted experts.
    bool finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) override;
    /// finish_predict + the routing predictions of the `ahead` layers after next_il, from the same state, in the same
    /// graph: ahead_ids[a * 16 ..] = layer next_il + 1 + a's top 16 (-1 past the last layer).  For speculative
    /// file-tier reads; a hint only.
    bool finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                              int * ahead_ids);
    /// Logits of rows row0..row0+nrows-1 of the pass after the last layer's finish: nrows * n_vocab floats.
    bool logits(int row0, int nrows, float * out) override;
    /// Conversation reuse: every cache is position-indexed except the window ring and the ratio-2 compressor state
    /// ring, so a snapshot is those rings (~11 MB) + its position; restore() rewinds to it (decoding continues at
    /// snapshot_pos()).  Positions past a rewind point are overwritten when decoded again (rows past a query's
    /// visible count are never attended).
    bool snapshot(int pos) override;
    bool restore() override;
    int snapshot_pos() const override;
    /// Serve: keep a conversation through an UNRELATED request.  An unrelated request writes its own positions from 0, so
    /// it overwrites the start of every position-indexed cache (compressed KV, indexer keys) and the snapshot rings.
    /// park(n) copies the snapshot rings + the first n positions' rows of those caches (and their spare rows) to host
    /// memory; unpark() writes them back and re-arms the snapshot, so restore() resumes the parked conversation.  Valid
    /// only while no request has written past position n since park (the caller checks).
    bool park(int64_t positions) override;
    bool unpark() override;
    void drop_park() override;
    bool parked() const override;
    /// Engram history: the n-gram hash of a pass's first positions reads the tokens before it from the history, which
    /// is kept BY POSITION - another request that wrote those positions since (a parked conversation's side request)
    /// leaves its own tokens there.  Rewrite [pos0, pos0 + n) with tids (DEAD where dead[i] != 0: an image position).
    /// A no-op without Engram; false only when a pending Engram read failed.
    bool set_engram_history(int64_t pos0, int n, const int * tids, const uint8_t * dead) override;
    /// A conversation's reusable device state at its snapshot, as regions of the device tensors: the snapshot rings
    /// (window K/V, ratio-2 compressor state) + rows [0, positions] of every compressed-KV / indexer-key cache and their
    /// spare rows - exactly what park() copies.  A conversation store saves them right after snapshot(); writing them
    /// back and rearm(pos) + restore() resumes that conversation at pos.
    std::vector<StateRegion> state_regions(int64_t positions) const override;
    /// The snapshot tensors now hold position `pos`'s state (a conversation store wrote them): restore() resumes there.
    void rearm(int pos) override;
    /// Conversation pool (ds41_conv_pool, ds41_conv_pool_dense.hpp): where a conversation's device state at a turn
    /// boundary lives.  `streams`: the position-indexed caches (compressed KV + indexer keys of every kv source layer;
    /// row r holds positions [r * ratio, (r + 1) * ratio), `rows` = cmax, the spare row excluded).  `regions`: the window
    /// rings and the ratio-2 compressor states - the *_snap copies (snap = true: what snapshot() / park() keep) or the
    /// live ones.  Pointers into the state buffer (device memory on the GPU); the sizes never change after init.
    void pool_regions(bool snap, std::vector<PoolRegion> & streams, std::vector<PoolRegion> & regions) const override;
    /// The same state as tensors (ds41_conv_disk.hpp copies it with ggml_backend_tensor_get / set): the position-indexed
    /// streams in pool_regions' order (row r holds positions [r * ratio, (r + 1) * ratio), `rows` without the spare
    /// row) and the snapshot rings (raw_snap, st_kv_snap, st_sc_snap of every layer).
    void conv_tensors(std::vector<StreamTensor> & streams, std::vector<ggml_tensor *> & snap_regions) const override;
    /// Debug / gate taps: the hc state after the last finish (n * hc * n_embd floats, token-major) and the pre mix.
    bool read_state(std::vector<float> & x, std::vector<float> & pre) override;
    /// The hc state of state columns [c0, c0 + n) (token-major, n * hc * n_embd floats) - a layer-major batch's columns
    /// after finish_span (the serve DSpark seed reads its main_hidden there instead of re-running the tail as passes).
    bool read_state_cols(int c0, int n, std::vector<float> & x) override;
    /// Resize the state columns to max(S, min(nc, prompt_batch)) between passes (drops every cached graph); columns()
    /// is the current count.  A no-op when unchanged.
    bool set_columns(int nc) override;
    int columns() const override;
    /// Gate tap: an index source layer's top-k rows of the current pass (n * K ints, token-major; K = min(cap, top-k))
    /// and their visibility mask; empty for other layers.
    bool read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) override;
    /// The x state after begin (+ the Engram add of the layer `il`, when il is an Engram layer) - gate tap.
    const std::string & last_error() const override;
    /// Image input (cfg.vision): the request's image span rows by absolute position - `rows` n_rows * n_embd floats,
    /// `pos` the n_rows ascending positions.  A pass overwrites those positions' embeddings with the rows, routes them
    /// with exp_probs_b_vl, shuts the Engram gate there and marks them DEAD in the n-gram history (model.py).  The
    /// arrays must stay alive until clear_vision(); the positions the request reuses are simply never fed.
    bool vision() const override;
    void set_vision(const float * rows, const int64_t * pos, int n_rows) override;
    void clear_vision() override;
    /// A loaded weight on the backend (nullptr if absent) - for sharing the trunk head with the DSpark draft stack.
    ggml_tensor * weight(const std::string & name) const override;

private:
    std::unique_ptr<Impl> p_;
};

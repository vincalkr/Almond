// src/ds41/moe_trunk.hpp - what the engine needs from a model: its trunk (the dense, non-routed-expert half).
//
// Almond runs several MoE models through one expert tier (Ds41MoeTier: VRAM cache, peer card, RAM arena, NVMe) and one
// driver (ds41_generate.cpp: prompt, decode, serve, conversation reuse).  What differs between models is the trunk:
// attention, norms, the router, shared experts, whatever a model adds (DeepSeek's Engram and hyper-connections).  A
// model is a MoeTrunk; the driver calls it per layer:
//
//   begin(tokens) -> for every layer: attn_router(l) -> [the tier runs the routed experts] -> finish(l) -> logits()
//
// plus the prompt's layer-major batch, the next-layer prediction the tier prefetches from, and the state a
// conversation keeps (snapshot / park / the regions the conversation stores copy).  The first implementation is
// Ds41Dense (DeepSeek-V4.1-Flash); the method contracts are documented there (ds41_dense.hpp).
//
// A model without a feature keeps the default: no route bias, no Engram history, no image input, no indexer taps.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/// How a trunk is built.  The fields after `prompt_batch` are model features; a model without them ignores them.
struct TrunkConfig {
    ggml_backend_t backend = nullptr;   ///< NULL = create and own a CPU backend
    int     n_threads = 8;
    int64_t ctx_max = 65536;            ///< positions the caches hold
    int     max_pass = 1;               ///< tokens a pass may carry (>= 1; prefill chunk size = the sub-batch size S)
    /// Layer-major prompt batch (0 = off): see Ds41DenseConfig.
    int     prompt_batch = 0;
    bool    skip_routed_experts = true; ///< do not load the routed experts (the tier owns them)
    /// KV cache element type (GGML_TYPE_*), or -1 = the model's default.
    int     kv_type = -1;
    /// --serve: start with max_pass state columns and let set_columns() grow them for a layer-major batch.
    bool    elastic_cols = false;
    // ---- DeepSeek-V4.1 (Ds41Dense)
    std::vector<std::string> engram_dirs;   ///< the Engram tables (empty = no Engram add)
    int     engram_threads = 32;
    bool    act_quant = true;               ///< model.py's activation rounding
    bool    kv_fp4 = false;                 ///< compressed KV as NVFP4, indexer keys as MXFP4
    bool    vision = false;                 ///< image input (needs the image routing bias)
};

/// The numbers the driver reads from a model.
struct TrunkInfo {
    std::string arch;          ///< "deepseek41", ...
    int64_t n_embd = 0;
    int64_t n_layer = 0;       ///< blocks present in the file
    int64_t n_used = 0;        ///< routed experts a token takes per layer
    int64_t vocab = 0;
    int64_t eos = -1;
    /// The prompt's encoder-only skip (--enc-skip): the layer from which every later layer reads only its own window
    /// of `n_swa` positions, so a long prompt's earlier tokens can stop there.  -1 = the model has no such cut.
    int     enc_cut_layer = -1;
    int64_t n_swa = 0;
};

class MoeTrunk {
public:
    virtual ~MoeTrunk() = default;

    virtual bool init(const std::string & gguf, const TrunkConfig & cfg, std::string & err) = 0;
    virtual const TrunkInfo & info() const = 0;
    virtual int64_t n_layer() const { return info().n_layer; }
    virtual ggml_backend_t backend() const = 0;
    virtual const std::string & last_error() const = 0;

    // ---- one pass (decode: n = 1; a prompt chunk: n <= max_pass)
    virtual bool begin(const int * tids, int n, int pos0) = 0;
    virtual bool attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev = nullptr) = 0;
    virtual bool finish(int il, const float * routed_sum) = 0;
    virtual bool logits(int row0, int nrows, float * out) = 0;

    // ---- the next layer's experts, for the tier's prefetch (hints: the result never depends on them)
    virtual bool predict(int il, int * top_ids, int n_top) = 0;
    virtual bool predict_multi(int il, int * top_ids) = 0;
    virtual bool finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) = 0;
    virtual bool finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                                      int * ahead_ids) = 0;

    // ---- the prompt's layer-major batch
    virtual bool begin_batch(const int * tids, int B, int pos0) = 0;
    virtual bool attn_router_span(int il, int col0, int n) = 0;
    virtual bool attn_router_span_kv(int il, int col0, int n) = 0;
    virtual const int32_t * batch_ids() const = 0;
    virtual const float * batch_wts() const = 0;
    virtual const float * batch_fn() const = 0;
    virtual int batch_size() const = 0;
    virtual bool upload_routed(const float * routed_host, int B) = 0;
    virtual float * routed_dev() = 0;
    virtual bool finish_span(int il, int col0, int n) = 0;
    virtual bool set_columns(int nc) = 0;
    virtual int columns() const = 0;
    virtual void release_pass() = 0;
    virtual bool reset_columns() { return true; }   // (Ds41Dense: the column state and graphs rebuilt, after images)
    virtual void set_oom_hook(std::function<bool()> hook) = 0;
    virtual void prof_report(const char * tag) { (void) tag; }

    // ---- a conversation's state
    virtual bool snapshot(int pos) = 0;
    virtual bool restore() = 0;
    virtual int snapshot_pos() const = 0;
    virtual bool park(int64_t positions) = 0;
    virtual bool unpark() = 0;
    virtual void drop_park() = 0;
    virtual bool parked() const = 0;
    virtual void rearm(int pos) = 0;
    /// The device state a conversation store copies (see Ds41Dense: state_regions / pool_regions / conv_tensors).
    struct StateRegion { ggml_tensor * t; size_t off, bytes; };
    virtual std::vector<StateRegion> state_regions(int64_t positions) const = 0;
    struct PoolRegion {
        std::string name;
        void * data = nullptr;
        size_t row_bytes = 0, bytes = 0;
        int64_t rows = 0;
        int ratio = 0;
    };
    virtual void pool_regions(bool snap, std::vector<PoolRegion> & streams, std::vector<PoolRegion> & regions) const = 0;
    struct StreamTensor { ggml_tensor * t; size_t row_bytes; int64_t rows; int ratio; };
    virtual void conv_tensors(std::vector<StreamTensor> & streams, std::vector<ggml_tensor *> & snap_regions) const = 0;

    // ---- debug / gate taps
    virtual bool read_state(std::vector<float> & x, std::vector<float> & pre) = 0;
    virtual bool read_state_cols(int c0, int n, std::vector<float> & x) = 0;
    virtual bool read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
        (void) il; sel.clear(); mask.clear(); if (K) *K = 0; return true;
    }
    /// A loaded weight on the backend (nullptr if absent): the draft model shares the trunk's output head.
    virtual ggml_tensor * weight(const std::string & name) const { (void) name; return nullptr; }

    // ---- model features (no-ops for a model without them)
    /// Selection-only routing bias (lossy; the default path never sets it).
    virtual void set_route_bias(int il, const float * bias) { (void) il; (void) bias; }
    virtual bool set_engram_history(int64_t pos0, int n, const int * tids, const uint8_t * dead) {
        (void) pos0; (void) n; (void) tids; (void) dead; return true;
    }
    virtual bool vision() const { return false; }
    virtual void set_vision(const float * rows, const int64_t * pos, int n_rows) { (void) rows; (void) pos; (void) n_rows; }
    virtual void clear_vision() {}
};

/// The trunk for the GGUF's architecture (general.architecture, or the DS41_META overlay's): "deepseek41" -> Ds41Dense.
/// Null with `err` for an architecture Almond does not run.
std::unique_ptr<MoeTrunk> make_trunk(const std::string & gguf, std::string & err);

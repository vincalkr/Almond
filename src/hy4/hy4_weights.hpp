// src/hy4/hy4_weights.hpp - GGUF weights as ggml tensors whose data points into the mapped file, made once per name.
//
// The CPU path used to copy every weight (the stacked experts are about 1.3 GB per layer) into a fresh ggml context
// at every step. These tensors live in a no_alloc context for the life of the process and are never written: the
// CPU backend only reads `data`, so a graph can mix them with tensors of its own context.
#pragma once

#include "hy4_geometry.hpp"

#include "ggml.h"

#include <map>
#include <string>
#include <utility>

namespace almond::hy4 {

/// The weight `name` with ggml shape [ne0, ne1, ne2] (ne2 = 0 for 2-D, ne1 = ne2 = 0 for 1-D) and the file's own
/// type. The byte count must match the file's; the shape must be the same at every call for the same name.
inline ggml_tensor * hy4_weight(const GgufModel & m, const std::string & name, int64_t ne0, int64_t ne1,
                                int64_t ne2, std::string & err) {
    static ggml_context * ctx = nullptr;
    static std::map<std::pair<const GgufModel *, std::string>, ggml_tensor *> made;
    auto key = std::make_pair(&m, name);
    if (auto it = made.find(key); it != made.end()) {
        ggml_tensor * g = it->second;
        if (g->ne[0] != ne0 || (ne1 && g->ne[1] != ne1) || (ne2 && g->ne[2] != ne2)) {
            err = name + ": asked with two shapes"; return nullptr;
        }
        return g;
    }
    if (!ctx) {
        ggml_init_params ip = { 4096 * ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
    }
    size_t which = 0;
    const TensorInfo * t = m.find(name, &which);
    if (!t) { err = "missing " + name; return nullptr; }
    const auto type = (enum ggml_type) t->type;
    ggml_tensor * g = ne2 ? ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)
                    : ne1 ? ggml_new_tensor_2d(ctx, type, ne0, ne1)
                          : ggml_new_tensor_1d(ctx, type, ne0);
    if (!g) { err = "tensor overhead exhausted at " + name; return nullptr; }
    size_t expect = (size_t) ggml_row_size(type, t->shape[0]);
    for (size_t i = 1; i < t->shape.size(); ++i) expect *= t->shape[i];
    if ((size_t) ggml_nbytes(g) != expect) { err = name + ": ggml and GGUF byte counts disagree"; return nullptr; }
    g->data = const_cast<uint8_t *>(m.shard(which).tensor_data(*t));
    ggml_set_name(g, name.c_str());
    made.emplace(key, g);
    return g;
}

}  // namespace almond::hy4

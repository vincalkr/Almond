#!/usr/bin/env python3
"""sensenova/SenseNova-Loop-9B-DFlash2 (safetensors) -> the dflash GGUF that src/qwen35 reads with --dflash.

The same tensors and keys as llama.cpp's convert_hf_to_gguf.py DFlashModel writes (checked byte for byte on the
Q8_0), without its need for a target directory whose architecture llama.cpp knows: the drafter shares the target's
embeddings and output head, so it needs no vocabulary, only the mask token id.

target_layer_ids are the target's *execution* indices whose outputs it reads (3, 27, 51, 54 over the loop's 56
executions); the GGUF keeps llama.cpp's convention, the layer whose input that is (+1). The fifth feature, the final
post-norm hidden ("final_postnorm"), is not a key: the engine sees it from fc's width ((targets + 1) x hidden).

    python tools/sensenova/dflash2_gguf.py <drafter dir> <out.gguf> [--outtype q8_0|bf16|f16]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp",
                                "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import quantize  # noqa: E402


def read_safetensors(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        raw = mm[base + a:base + b]
        if v["dtype"] == "BF16":
            x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif v["dtype"] == "F32":
            x = raw.view(np.float32)
        elif v["dtype"] == "F16":
            x = raw.view(np.float16).astype(np.float32)
        else:
            raise SystemExit(f"unexpected dtype {v['dtype']} for {k}")
        out[k] = x.reshape(v["shape"])
    return out


NAMES = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector.successor_codebook": "selector_successor.weight",
}
LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "attention_conv.base_kernel": "attn_conv_base",
    "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
    "mlp_conv.base_kernel": "ffn_conv_base",
    "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--outtype", default="q8_0", choices=["q8_0", "bf16", "f16"])
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    dc = cfg["dflash_config"]
    if cfg.get("architectures") != ["DFlash2DraftModel"]:
        raise SystemExit("not a DFlash2 drafter")
    sources = cfg.get("target_feature_sources", dc["target_layer_ids"])
    layers = [int(s) for s in sources if s != "final_postnorm"]
    if layers != dc["target_layer_ids"] or sources[len(layers):] not in ([], ["final_postnorm"]):
        raise SystemExit(f"unexpected feature sources {sources}")
    T = read_safetensors(os.path.join(a.src, "model.safetensors"))
    width = T["fc.weight"].shape[1]
    if width != len(sources) * cfg["hidden_size"]:
        raise SystemExit(f"fc reads {width}, not {len(sources)} x {cfg['hidden_size']}")

    w = gguf.GGUFWriter(a.out, "dflash")
    arch = "dflash"
    w.add_type("model")
    w.add_name("SenseNova-Loop-9B-DFlash2")
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_causal_attention(bool(cfg.get("is_causal", False)))
    w.add_rope_freq_base(float(cfg["rope_parameters"]["rope_theta"]))
    w.add_layer_norm_rms_eps(float(cfg["rms_norm_eps"]))
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_block_size(dc["block_size"])
    w.add_conv_kernel_size(dc["conv_kernel_size"])
    w.add_conv_group_size(dc["conv_group_size"])
    w.add_selector_rank(dc["selector_rank"])
    w.add_selector_top_k(dc["selector_top_k"])
    w.add_target_layers([i + 1 for i in layers])
    w.add_sliding_window(cfg["sliding_window"])
    w.add_sliding_window_pattern([t == "sliding_attention" for t in cfg["layer_types"]])
    w.add_rope_dimension_sections([cfg["head_dim"] // 2, 0, 0, 0])
    w.add_mask_token_id(dc["mask_token_id"])
    qt = {"q8_0": gguf.GGMLQuantizationType.Q8_0, "bf16": gguf.GGMLQuantizationType.BF16,
          "f16": gguf.GGMLQuantizationType.F16}[a.outtype]
    w.add_file_type({"q8_0": gguf.LlamaFileType.MOSTLY_Q8_0, "bf16": gguf.LlamaFileType.MOSTLY_BF16,
                     "f16": gguf.LlamaFileType.MOSTLY_F16}[a.outtype])
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)

    for k, x in T.items():
        if k in NAMES:
            name = NAMES[k]
        elif k.startswith("layers."):
            _, bid, rest = k.split(".", 2)
            if rest not in LAYER:
                raise SystemExit(f"unknown tensor {k}")
            name = f"blk.{bid}.{LAYER[rest]}"
        else:
            raise SystemExit(f"unknown tensor {k}")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 2:   # matrices in the out type; norms and the conv base kernels in F32
            data = quantize(x, qt)
            w.add_tensor(name, data, raw_dtype=qt)
        else:
            w.add_tensor(name, x)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()

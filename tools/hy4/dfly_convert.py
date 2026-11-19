#!/usr/bin/env python3
"""Hy4-flash's DFly2 drafter (tencent/Hy4-flash, model-00148-of-00148.safetensors, `model.dfly_layers.*`) -> a GGUF
the engine reads with --dfly (src/hy4/hy4_engine.cpp). BF16 kept as is; kv_b_proj split into the per-head key and
value halves in the layout of the trunk's attn_k_b / attn_v_b.

  PYTHONPATH=third_party/llama.cpp/gguf-py .venv/bin/python tools/hy4/dfly_convert.py \
      models/hy4-dfly2/model-00148-of-00148.safetensors models/hy4-dfly2/hy4-dfly2-bf16.gguf
"""
import json
import struct
import sys

import numpy as np
import gguf

H, NOPE, VD, KR = 32, 192, 256, 512
P = "model.dfly_layers."
LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "self_attn.q_a_proj.weight": "attn_q_a.weight",
    "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
    "self_attn.q_b_proj.weight": "attn_q_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
    "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
    "self_attn.linear_gate.weight": "attn_gate.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.post_proj_norm.weight": "post_proj_norm.weight",
    "self_attn.learnable_sink_param": "attn_sinks.weight",
}
TOP = {
    "final_norm.weight": "dfly.final_norm.weight",
    "mtp_fusion.eh_proj.weight": "dfly.fuse.eh_proj.weight",
    "mtp_fusion.enorm.weight": "dfly.fuse.enorm.weight",
    "mtp_fusion.hnorm.weight": "dfly.fuse.hnorm.weight",
    "mtp_fusion.gate": "dfly.fuse.gate.weight",
    "hidden_correction.hidden_norm.weight": "dfly.corr.hidden_norm.weight",
    "hidden_correction.embed_norm.weight": "dfly.corr.embed_norm.weight",
    "hidden_correction.gate_proj.weight": "dfly.corr.gate.weight",
    "hidden_correction.up_proj.weight": "dfly.corr.up.weight",
    "hidden_correction.down_proj.weight": "dfly.corr.down.weight",
}


def main(src, dst):
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        raw = np.memmap(src, dtype=np.uint8, mode="r")
    w = gguf.GGUFWriter(dst, "hy4-dfly")
    w.add_uint32("hy4-dfly.layer_count", 5)
    w.add_uint32("hy4-dfly.mask_token_id", 120009)
    w.add_array("hy4-dfly.target_layers", [25, 29, 33, 37, 41])
    n_out = 0

    def put(name, a):
        nonlocal n_out
        w.add_tensor(name, np.ascontiguousarray(a), raw_dtype=gguf.GGMLQuantizationType.BF16)
        n_out += 1

    for k, v in sorted(head.items()):
        if k == "__metadata__" or not k.startswith(P):
            continue
        assert v["dtype"] == "BF16", k
        a, b = v["data_offsets"]
        t = raw[base + a: base + b].view(np.uint16).reshape(v["shape"])
        r = k[len(P):]
        if r.startswith("confidence_head."):
            continue                       # vLLM leaves it out too
        if r in TOP:
            put(TOP[r], t)
            continue
        i, rest = r.split(".", 1)
        pre = f"dfly.{int(i)}."
        if rest == "self_attn.kv_b_proj.weight":
            kv = t.reshape(H, NOPE + VD, KR)
            put(pre + "attn_k_b.weight", kv[:, :NOPE, :].transpose(0, 2, 1))   # ggml [NOPE, KR, H]
            put(pre + "attn_v_b.weight", kv[:, NOPE:, :])                      # ggml [KR, VD, H]
            continue
        put(pre + LAYER[rest], t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{n_out} tensors -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

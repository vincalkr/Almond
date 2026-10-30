#!/usr/bin/env python3
"""The draft (NextN / MTP) block of GLM-5.3-Flash, blk.45.*, out of a GGUF that has one (project-maya's Maya-M, its
third shard) into a GGUF of its own: the GSQ-RCO 3.0bit GGUF was published without it, and the engine takes one from
another quant of the same model (STRATA_GLM_MTP_GGUF, engines/maya/src/core/glm_model.cu: the block reads the trunk's
hidden state and the embedding, so any quant's block fits).  The metadata is copied as it is; only blk.45's tensors.

  ALMOND_GGUF_PY=third_party/llama.cpp/gguf-py .venv/bin/python local/glm-mtp-extract.py SRC.gguf OUT.gguf
"""
import os
import sys

sys.path.insert(0, os.environ.get("ALMOND_GGUF_PY", os.path.join(os.path.dirname(__file__), "..", "third_party",
                                                                 "llama.cpp", "gguf-py")))
import gguf  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
# a later shard of a split GGUF carries no architecture key: the first shard's (the name with -00001-of-)
if "general.architecture" in r.fields:
    arch = r.fields["general.architecture"].contents()
elif os.environ.get("GGUF_ARCH"):              # (local/fetch.sh: only the shard 3 is downloaded)
    arch = os.environ["GGUF_ARCH"]
else:
    head = gguf.GGUFReader(__import__("re").sub(r"-\d{5}-of-", "-00001-of-", src))
    arch = head.fields["general.architecture"].contents()
w = gguf.GGUFWriter(out, arch)
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture" or name.startswith("split."):
        continue
    t = f.types[0]
    if t == gguf.GGUFValueType.ARRAY:
        w.add_key_value(name, f.contents(), t, sub_type=f.types[-1])
    else:
        w.add_key_value(name, f.contents(), t)
n = 0
for t in r.tensors:
    if t.name.startswith("blk.45."):
        w.add_tensor(t.name, t.data, raw_shape=list(reversed([int(x) for x in t.shape])) if t.tensor_type in
                     (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)
                     else None, raw_dtype=t.tensor_type)
        n += 1
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{out}: {n} tensors of blk.45")

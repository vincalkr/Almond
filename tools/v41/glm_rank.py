#!/usr/bin/env python3
"""glm_rank.py - rank every routed (layer, expert) of a GLM-5.3 GGUF for ds41_generate --rank.

The same order as v41_rank.py (routing count per BYTE of blob, hottest first), with GLM's geometry: the counts are
ds41_generate --dump-routes DIR's LNN.json, the blob bytes come from the GGUF's per-layer expert tensor types (the
GSQ-RCO packs mix Q2_K, Q3_K and Q4_K by layer).  Dense layers (no ffn_*_exps) are left out.

    python3 tools/v41/glm_rank.py --gguf M.gguf --stats local/glm-ref/routes17k --out local/glm-rank.bin
writes int32 (layer, expert) pairs and prints the routed share covered by the first N GiB.
"""
import argparse
import json
import os
import re
import struct

import numpy as np

# bytes per 256-value block of the expert formats the GLM packs use (ggml type id -> bytes)
BLOCK = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 8: 272}


def tensor_infos(path):
    """(name, dims, ggml type) of every tensor: GGUF v3's header, the KVs skipped."""
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]
    u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    gstr = lambda: f.read(u64()).decode()
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(t):
        if t == 8:
            f.seek(u64(), 1)
        elif t == 9:
            et, n = u32(), u64()
            if et in size:
                f.seek(size[et] * n, 1)
            else:
                for _ in range(n):
                    skip(et)
        else:
            f.seek(size[t], 1)

    assert f.read(4) == b"GGUF"
    u32()
    nt, nkv = u64(), u64()
    for _ in range(nkv):
        gstr()
        skip(u32())
    out = []
    for _ in range(nt):
        name = gstr()
        dims = [u64() for _ in range(u32())]
        ty = u32()
        u64()
        out.append((name, dims, ty))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blob = {}   # layer -> bytes of one expert (gate + up + down)
    ne = 0
    for name, dims, ty in tensor_infos(a.gguf):
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
        if not m:
            continue
        L = int(m.group(1))
        ne = int(dims[-1])
        blob[L] = blob.get(L, 0) + dims[0] * dims[1] // 256 * BLOCK[ty]
    rows, total = [], 0
    for L in sorted(blob):
        p = os.path.join(a.stats, f"L{L:02d}.json")
        ex = json.load(open(p))["experts"] if os.path.exists(p) else {}
        for e in range(ne):
            n = ex.get(str(e), {}).get("n", 0) if isinstance(ex, dict) else ex[e].get("n", 0)
            rows.append((n / blob[L], L, e, n, blob[L]))
            total += n
    rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    np.array([(L, e) for _, L, e, _, _ in rows], dtype=np.int32).tofile(a.out)
    acc_b = acc_n = 0
    marks = [10, 20, 30, 40, 60, 80, 100]
    for _, L, e, n, b in rows:
        acc_b += b
        acc_n += n
        while marks and acc_b >= marks[0] * 2**30:
            print(f"first {marks.pop(0):3d} GiB: {100.0 * acc_n / max(total, 1):5.1f}% of the routed selections")
    print(f"{len(rows)} experts, {acc_b / 2**30:.1f} GiB, {total} selections -> {a.out}")


if __name__ == "__main__":
    main()

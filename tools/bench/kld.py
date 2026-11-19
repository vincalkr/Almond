#!/usr/bin/env python3
"""KL divergence of a quantization against a reference build of the same model, from logit dumps of the same
token stream: no answers and no judge, the most sensitive single number for "how much did the quantization lose".

  .venv/bin/python tools/bench/kld.py REF.f32 CAND.f32 --vocab V --name qwen27-iq3s --ref-name qwen27-q8

The dumps are raw little-endian float32, one row of V logits per position, as the engines' --dump-logits writes
them (ds41_generate; glm and qwen generate write one row per generated position). Both files must cover the same
tokens. Reported, as llama-perplexity --kl-divergence does: mean KLD(ref || cand) with its standard error,
the 99th percentile and maximum, and the share of positions where both builds put the same token first.
The result goes to local/bench/kld/<name>.json and shows up in the UI's Compare tab.
"""
import argparse
import json
import math
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "local", "bench", "kld")


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def compare(ref_path, cand_path, vocab, chunk=256):
    a = np.memmap(ref_path, dtype="<f4", mode="r")
    b = np.memmap(cand_path, dtype="<f4", mode="r")
    if a.size % vocab or b.size % vocab:
        raise ValueError(f"file size is not a multiple of the vocabulary ({vocab})")
    n = min(a.size, b.size) // vocab
    if a.size != b.size:
        print(f"note: the dumps differ in length, comparing the first {n} positions")
    a, b = a[:n * vocab].reshape(n, vocab), b[:n * vocab].reshape(n, vocab)
    kl, same = [], 0
    for i in range(0, n, chunk):
        la, lb = log_softmax(np.asarray(a[i:i + chunk])), log_softmax(np.asarray(b[i:i + chunk]))
        kl.append((np.exp(la) * (la - lb)).sum(axis=1))
        same += int((la.argmax(1) == lb.argmax(1)).sum())
    kl = np.concatenate(kl)
    return {"positions": n, "kld_mean": float(kl.mean()), "kld_se": float(kl.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "kld_median": float(np.median(kl)), "kld_p99": float(np.percentile(kl, 99)), "kld_max": float(kl.max()),
            "same_top1": same / n}


def load_all():
    out = []
    if os.path.isdir(OUT):
        for fn in sorted(os.listdir(OUT)):
            if fn.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(OUT, fn)))
                    d["path"] = os.path.join(OUT, fn)
                    out.append(d)
                except (OSError, ValueError):
                    pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--name", required=True, help="the quantization being measured")
    ap.add_argument("--ref-name", default="reference")
    ap.add_argument("--text", default="", help="what the token stream was (for the record)")
    a = ap.parse_args()
    r = compare(a.ref, a.cand, a.vocab)
    r.update(name=a.name, ref=a.ref_name, text=a.text, when=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{a.name}-vs-{a.ref_name}.json".replace("/", "_"))
    json.dump(r, open(p, "w"), indent=1)
    print(f"{a.name} vs {a.ref_name}: KLD {r['kld_mean']:.5f} +- {r['kld_se']:.5f} (p99 {r['kld_p99']:.4f}, "
          f"max {r['kld_max']:.3f}), same top-1 {r['same_top1'] * 100:.2f}% over {r['positions']} positions")
    print("saved", p)


if __name__ == "__main__":
    main()

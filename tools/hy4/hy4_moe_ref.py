#!/usr/bin/env python3
"""Verify Hy4-flash's LatentMoE stages against a numpy recomputation from dumped intermediates.

Usage: hy4_moe_ref.py GGUF DUMP_DIR
The dump comes from test_hy4_moe with HY4_DUMP set. Expert weights stay in ggml; what is checked here is the router,
top-k and weight normalization, the SwiGLU combination, and the final projections/addition.
"""
import sys

import numpy as np

sys.path.insert(0, 'tools/hy4')
from hy4_attn_ref import G

m = G(sys.argv[1])
D = '/tmp/opencode/hy4/dumpmoe/' if len(sys.argv) < 3 else sys.argv[2] + '/'
ld = lambda n, dt='<f4': np.fromfile(D + n + ('.f32' if dt == '<f4' else '.i32'), dtype=dt)
p = 'blk.1.'
bias = m.raw(p + 'exp_probs_b.bias')
rup = m.raw(p + 'ffn_routed_up.weight').reshape(3072, 1536)   # [out, in] via ggml [ne1, ne0]
sdown = m.raw(p + 'ffn_down_shexp.weight').reshape(3072, 1536)

logits = ld('logits'); probs = ld('probs'); selection = ld('selection'); ids = ld('ids', '<i4')
weights = ld('weights'); routed = ld('routed'); g = ld('g'); u = ld('u'); h = ld('h')
d = ld('d'); summed = ld('summed'); moe = ld('moe'); sg = ld('sg'); su = ld('su')
shexp = ld('shexp'); out = ld('out')
x = np.fromfile('/tmp/opencode/hy4/moe.bin', dtype='<f4', count=3072)

want_probs = 1.0 / (1.0 + np.exp(-logits))
want_ids = np.argsort(-selection, axis=0)[:16]
want_w = probs[want_ids]
want_w = want_w / np.clip(want_w.sum(), 6.103515625e-5, np.inf) * 3.998

silu = lambda v: v / (1.0 + np.exp(-v))
want_h = silu(g) * u
want_sum = np.sum(d.reshape(16, 1536), axis=0)  # d is already weighted at dump time
want_moe = rup @ want_sum
want_shexp = sdown @ (silu(sg) * su)
want_out = want_moe + want_shexp

for name, got, ref in [
    ('probs', probs, want_probs), ('ids', ids, want_ids), ('weights', weights.reshape(-1), want_w),
    ('h', h, want_h), ('summed', summed, want_sum), ('moe', moe, want_moe),
    ('shexp', shexp, want_shexp), ('out', out, want_out),
]:
    err = np.abs(got - ref)
    print('%-8s max %.6g mean %.6g' % (name, err.max(), err.mean()))

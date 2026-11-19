#!/usr/bin/env python3
"""Recompute three-token gated MLA attention with NORM RoPE, causal mask and sinks, using numpy.

Usage: hy4_attn_multi_ref.py GGUF OUT.BIN (OUT.BIN from test_hy4_attn_multi: x then output, ggml order).
"""
import struct
import sys

import numpy as np

sys.path.insert(0, 'tools/hy4')
from hy4_attn_ref import G, rms


def rope_norms(x, pos, base, n_dims=64):
    """NORM rope: rotate consecutive pairs (ggml mode 0). x is [n_dims, n_head, n_tokens]."""
    out = x.copy()
    half = n_dims // 2
    i = np.arange(half)
    theta = np.asarray(pos, dtype=np.float64)[:, None] * (base ** (-2.0 * i / n_dims))[None, :]
    c, s = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)   # [nt, half]
    a, b = out[0::2, :, :].copy(), out[1::2, :, :].copy()                             # [dims/2, head, nt]
    out[0::2, :, :] = a * c.T[:, None, :] - b * s.T[:, None, :]
    out[1::2, :, :] = a * s.T[:, None, :] + b * c.T[:, None, :]
    return out


def main():
    gg, path = sys.argv[1], sys.argv[2]
    g = G(gg)
    D = int(g.meta['hy_v4_flash.embedding_length'])
    RQ = int(g.meta['hy_v4_flash.attention.q_lora_rank'])
    KR = int(g.meta['hy_v4_flash.attention.kv_lora_rank'])
    R = int(g.meta['hy_v4_flash.rope.dimension_count'])
    H = int(g.meta['hy_v4_flash.attention.head_count'][0])
    qhead = int(g.meta['hy_v4_flash.attention.key_length_mla'])
    nope = qhead - R
    VD = int(g.meta['hy_v4_flash.attention.value_length_mla'])
    base = float(g.meta['hy_v4_flash.rope.freq_base'])
    eps = float(g.meta['hy_v4_flash.attention.layer_norm_rms_epsilon'])
    b = open(path, 'rb').read()
    nt = D and len(b) // (D * 8)
    x = np.frombuffer(b, dtype='<f4', count=D * nt, offset=0).reshape(nt, D).T
    got = np.frombuffer(b, dtype='<f4', count=D * nt, offset=D * nt * 4).reshape(nt, D).T
    pos = np.arange(nt, dtype=np.int32)
    p = 'blk.0.'
    wqa = g.raw(p + 'attn_q_a.weight').reshape(RQ, D)
    qan = g.raw(p + 'attn_q_a_norm.weight')
    wqb = g.raw(p + 'attn_q_b.weight').reshape(H * qhead, RQ)
    wkv = g.raw(p + 'attn_kv_a_mqa.weight').reshape(KR + R, D)
    kvn = g.raw(p + 'attn_kv_a_norm.weight')
    wkb = g.raw(p + 'attn_k_b.weight').reshape(H, KR, nope).transpose(2, 1, 0)   # [nope, KR, H]
    wvb = g.raw(p + 'attn_v_b.weight').reshape(H, VD, KR).transpose(2, 1, 0)     # [KR, VD, H]
    gatew = g.raw(p + 'attn_gate.weight').reshape(H * VD, D)
    wo = g.raw(p + 'attn_output.weight').reshape(D, H * VD)
    sinks = g.raw(p + 'attn_sinks.weight')
    post = g.raw(p + 'post_attention_norm.weight')

    qr = rms(wqa @ x, eps) * qan[:, None]
    q = (wqb @ qr).T.reshape(nt, H, qhead).transpose(2, 1, 0)        # [qhead, H, nt]
    qn = q[:nope]
    qp = rope_norms(q[nope:], pos, base).transpose(0, 2, 1)         # [R, nt, H]
    qa = np.einsum('kjh,kht->jht', wkb, qn).transpose(0, 2, 1)      # [KR, nt, H]
    Q = np.concatenate([qa, qp], axis=0)                            # [KR+R, nt, H]

    kv = wkv @ x
    lat = (rms(kv[:KR], eps) * kvn[:, None]).astype(np.float16).astype(np.float32)
    kp = rope_norms(kv[KR:][:, None, :], pos, base).transpose(0, 2, 1).astype(np.float16).astype(np.float32)
    K = np.concatenate([lat[:, :, None], kp], axis=0)               # [KR+R, nt, 1]

    scale = 1.0 / np.sqrt(qhead)
    scores = np.einsum('kth,ks->hst', Q, K[:, 0, :]) * scale        # [H, s, t]
    mask = np.tril(np.full((nt, nt), -np.inf, dtype=np.float32), k=-1)
    scores = scores + mask[None, :, :]
    mx = np.maximum(scores.max(axis=1, keepdims=True), sinks[:, None, None])
    e = np.exp(scores - mx)
    prob = e / (e.sum(axis=1, keepdims=True) + np.exp(sinks[:, None, None] - mx))

    att = np.einsum('hst,vs->vht', prob, lat)                       # [KR, H, nt]
    A = np.einsum('kvh,kht->vht', wvb, att)                         # [VD, H, nt]
    A = A.transpose(1, 0, 2).reshape(H * VD, nt)         # [H*VD, nt]
    gate = 1.0 / (1.0 + np.exp(-(gatew @ x)))
    A = A * gate
    y = rms(wo @ A, eps) * post[:, None]

    err = np.abs(y - got)
    print('nt=%d heads=%d khead=%d' % (nt, H, KR + R))
    print('max abs diff %.6g mean %.6g at %d/%d over 1e-3' % (err.max(), err.mean(), np.count_nonzero(err > 1e-3), err.size))
    print('want', y[:4, 0], '\ngot ', got[:4, 0])


if __name__ == '__main__':
    main()

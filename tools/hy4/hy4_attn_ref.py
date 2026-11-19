#!/usr/bin/env python3
"""Recompute Hy4-flash's one-token gated MLA attention with numpy and compare with test_hy4_attn's ggml output.

Usage: hy4_attn_ref.py GGUF OUT.BIN. OUT.BIN holds x then the ggml output, both f32 of length n_embd.
This is the single-key case: the softmax over one position is 1 and RoPE cancels in q·k, so only the projections,
norms, latent absorption and output projection are exercised here.
"""
import struct
import sys

import numpy as np

EPS = 1.0e-5


def q8_0(raw, k):
    """ggml Q8_0: per 32 values, one F16 scale then 32 signed bytes."""
    nb = k // 32
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 34)
    d = b[:, 0:2].copy().view('<f2').astype(np.float32)
    q = b[:, 2:34].view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def rms(v, eps=EPS):
    return v / np.sqrt(np.mean(v * v, axis=0, keepdims=True) + eps)


class G:
    def __init__(self, path):
        import mmap
        self.f = open(path, 'rb')
        self.buf = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.p = 0
        assert self.take(4) == b'GGUF'
        self.u32(); nt = self.u64(); nk = self.u64()
        self.meta = {}
        for _ in range(nk):
            k = self.s()
            self.meta[k] = self.val(self.u32())
        self.t = {}
        for _ in range(nt):
            n = self.s(); nd = self.u32(); sh = [self.u64() for _ in range(nd)]
            ty = self.u32(); off = self.u64()
            self.t[n] = (sh, ty, off)
        al = int(self.meta.get('general.alignment', 32))
        self.data_start = (self.p + al - 1) // al * al

    def take(self, n):
        b = self.buf[self.p:self.p + n]; self.p += n; return b

    def u32(self): return struct.unpack('<I', self.take(4))[0]
    def u64(self): return struct.unpack('<Q', self.take(8))[0]
    def s(self): return self.take(self.u64()).decode('utf-8', 'replace')
    def val(self, vt):
        if vt == 8: return self.s()
        if vt == 9:
            et, n = self.u32(), self.u64()
            if et == 8: return [self.s() for _ in range(n)]
            fmt = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<B',10:'<Q',11:'<q',12:'<d'}[et]
            sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[et]
            return [struct.unpack(fmt, self.take(sz))[0] for _ in range(n)]
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[vt]
        return struct.unpack({0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<B',10:'<Q',11:'<q',12:'<d'}[vt], self.take(sz))[0]

    def raw(self, name):
        sh, ty, off = self.t[name]
        n = int(np.prod(sh))
        b = self.buf[self.data_start + off:self.data_start + off + n * 4]
        if ty == 0: return np.frombuffer(b, dtype='<f4', count=n)
        if ty == 30: return (np.frombuffer(b[:n * 2], dtype='<u2').astype(np.uint32) << 16).view('<f4')
        if ty == 8: return q8_0(b[:n // 32 * 34], n)
        raise ValueError(name + ' type ' + str(ty))


def main():
    gg, out = sys.argv[1], sys.argv[2]
    g = G(gg)
    D = int(g.meta['hy_v4_flash.embedding_length'])
    RQ = int(g.meta['hy_v4_flash.attention.q_lora_rank'])
    KR = int(g.meta['hy_v4_flash.attention.kv_lora_rank'])
    R = int(g.meta['hy_v4_flash.rope.dimension_count'])
    heads = int(g.meta['hy_v4_flash.attention.head_count'][0])
    nope = int(g.meta['hy_v4_flash.attention.key_length_mla']) - R
    VD = int(g.meta['hy_v4_flash.attention.value_length_mla'])
    b = open(out, 'rb').read()
    x = np.frombuffer(b, dtype='<f4', count=D, offset=0)
    got = np.frombuffer(b, dtype='<f4', count=D, offset=D * 4)
    p = 'blk.0.'
    # every raw array is in ggml order (ne0 fastest); reshape as [ne1, ne0].T to get [ne0, ne1]
    wqa = g.raw(p + 'attn_q_a.weight').reshape(RQ, D)
    qan = g.raw(p + 'attn_q_a_norm.weight')
    wqb = g.raw(p + 'attn_q_b.weight').reshape(heads * (nope + R), RQ)
    wkv = g.raw(p + 'attn_kv_a_mqa.weight').reshape(KR + R, D)
    kvn = g.raw(p + 'attn_kv_a_norm.weight')
    wkb = g.raw(p + 'attn_k_b.weight').reshape(heads, KR, nope).transpose(2, 1, 0)
    wvb = g.raw(p + 'attn_v_b.weight').reshape(heads, VD, KR).transpose(2, 1, 0)
    wgate = g.raw(p + 'attn_gate.weight').reshape(heads * VD, D)
    wo = g.raw(p + 'attn_output.weight').reshape(D, heads * VD)
    post = g.raw(p + 'post_attention_norm.weight')

    qr = rms(wqa @ x) * qan
    q = (wqb @ qr).reshape(heads, nope + R).T          # [kdim, heads]
    q_nope = q[:nope]
    absorbed = np.einsum('kjh,kh->jh', wkb, q_nope)    # [KR, heads]
    kv = wkv @ x
    lat = rms(kv[:KR]) * kvn
    A = np.einsum('kvj,k->vj', wvb, lat)               # [VD, heads]
    Aflat = A.T.reshape(-1)
    gate = 1.0 / (1.0 + np.exp(-(wgate @ x)))
    Aflat = Aflat * gate
    y = rms(wo @ Aflat) * post

    err = np.abs(y - got)
    print('head=%d nope=%d VD=%d D=%d' % (heads, nope, VD, D))
    print('max abs diff %.6g, mean %.6g, at %d/%d over 1e-3' % (err.max(), err.mean(), np.count_nonzero(err > 1e-3), err.size))
    print('want', y[:5], '\ngot ', got[:5])


if __name__ == '__main__':
    main()

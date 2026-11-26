#!/usr/bin/env python3
"""Recompute the Hy4-flash Engram projection with numpy and compare with test_hy4_graph's ggml output.

Usage: hy4_engram_ref.py GGUF SHARD2 OUT.BIN
OUT.BIN is written by build-hip-fused/test_hy4_graph: five int64 dimensions (D, HC, nt, cols, dim), then x (f32),
ids (i32), valid (f32) and the ggml delta (f32). GGUF is opened only to read the six small F32/BF16 weight tensors
and the FP8 rows, so this is independent of the C++ loader.
"""
import mmap
import struct
import sys

import numpy as np

SCALARS = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2), 4: ('<I', 4), 5: ('<i', 4),
           6: ('<f', 4), 7: ('<B', 1), 10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8)}


class GGUF:
    def __init__(self, path):
        self.file = open(path, 'rb')
        self.buf = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.p = 0
        assert self.take(4) == b'GGUF'
        self.ver, nt, nk = self.u32(), self.u64(), self.u64()
        self.meta = {}
        for _ in range(nk):
            k = self.string()
            self.meta[k] = self.value(self.u32())
        self.tensors = {}
        for _ in range(nt):
            name = self.string()
            nd = self.u32()
            shape = [self.u64() for _ in range(nd)]
            tt = self.u32()
            off = self.u64()
            self.tensors[name] = (shape, tt, off)
        align = int(self.meta.get('general.alignment', 32))
        self.data_start = (self.p + align - 1) // align * align

    def take(self, n):
        b = self.buf[self.p:self.p + n]
        self.p += n
        return b

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode('utf-8', 'replace')

    def value(self, vt):
        if vt == 8:
            return self.string()
        if vt == 9:
            et, n = self.u32(), self.u64()
            if et == 8:
                return [self.string() for _ in range(n)]
            fmt, sz = SCALARS[et]
            return [struct.unpack(fmt, self.take(sz))[0] for _ in range(n)]
        fmt, sz = SCALARS[vt]
        return struct.unpack(fmt, self.take(sz))[0]

    def data(self, name):
        shape, tt, off = self.tensors[name]
        n = int(np.prod(shape))
        raw = self.buf[self.data_start + off:self.data_start + off + n * 4]
        if tt == 0:
            return np.frombuffer(raw[:n * 4], dtype='<f4').reshape(shape)
        if tt == 30:
            return (np.frombuffer(raw[:n * 2], dtype='<u2').astype(np.uint32) << 16).view('<f4').reshape(shape)
        raise ValueError('%s has type %d' % (name, tt))

    def fp8_rows(self, name, ids, dim):
        """Read only the requested FP8 rows of a huge table (1 byte per element)."""
        shape, tt, off = self.tensors[name]
        assert tt == 43 and shape[0] == dim
        base = self.data_start + off
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        out = np.empty((ids.size, dim), dtype=np.uint8)
        for i, r in enumerate(ids):
            out[i] = np.frombuffer(self.buf[base + int(r) * dim:base + (int(r) + 1) * dim], dtype=np.uint8)
        return self.fp8_byte(out)

    def fp8_byte(self, arr):
        b = arr.astype(np.uint8)
        sign = np.where((b & 0x80) != 0, -1.0, 1.0)
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        out = np.where(exp == 0,
                       sign * (mant / 512.0),
                       sign * np.ldexp(1.0 + mant / 8.0, exp.astype(int) - 7))
        out = np.where((exp == 15) & (mant == 7), np.nan, out)
        return out.astype(np.float32)


def rms(x, eps=1.1920929e-7):
    return x / np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)


def main():
    gguf, shard2, out = sys.argv[1], sys.argv[2], sys.argv[3]
    m = GGUF(gguf)
    m2 = GGUF(shard2)
    b = open(out, 'rb').read()
    D, HC, nt, cols, dim = struct.unpack_from('<5q', b, 0)
    off = 40
    # x and delta are written in ggml order (ne0 fastest), so the flat array is [nt, HC, D] in C order.
    x = np.frombuffer(b, dtype='<f4', count=D * HC * nt, offset=off).reshape(nt, HC, D).transpose(2, 1, 0)
    off += D * HC * nt * 4
    ids = np.frombuffer(b, dtype='<i4', count=nt * cols, offset=off).reshape(nt, cols); off += ids.nbytes
    valid = np.frombuffer(b, dtype='<f4', count=nt * cols, offset=off).reshape(nt, cols); off += valid.nbytes
    got = np.frombuffer(b, dtype='<f4', count=D * HC * nt, offset=off).reshape(nt, HC, D).transpose(2, 1, 0)

    layer = int(m.meta['hy_v4_flash.ple.layers'][0])
    p = 'blk.%d.' % layer
    # GGUF shape is [ne0, ne1] but numpy reads it C-order; transpose to put ne0 first in memory like ggml.
    ln = m.data(p + 'engram_lookup_norm.weight').reshape(cols, dim).T
    kv_w = m.data(p + 'engram_kv.weight')
    kv_b = m.data(p + 'engram_kv_b.weight').reshape(HC + 1, D).T
    gate_w = m.data(p + 'engram_gate.weight').reshape(HC, D).T
    gamma = m.data(p + 'engram_pre_gamma.weight')
    hc_scale = m.data(p + 'engram_hc_scale.weight').reshape(HC, D).T
    rows = m2.fp8_rows('per_layer_token_embd.weight', ids.reshape(-1), dim).reshape(nt, cols, dim).transpose(2, 1, 0)

    # ggml's memory order is ne0 fastest; the GGUF arrays are already in that order. rows here is [dim, cols, nt].
    r = (rms(rows) * ln[:, :, None] * valid.T[None, :, :]).transpose(2, 1, 0).reshape(nt, dim * cols).T
    kv = (kv_w.reshape(D * (HC + 1), dim * cols) @ r).T.reshape(nt, HC + 1, D).transpose(2, 1, 0)
    kv = kv + kv_b[:, :, None]
    key, value = kv[:, :HC, :], kv[:, HC, :]
    dot = np.sum(rms(x) * rms(key) * gate_w[:, :, None], axis=0) / np.sqrt(D)
    mag = np.sqrt(np.maximum(np.abs(dot), 1e-6))
    sign = np.where(dot >= 0, 1.0, -1.0)
    gate = 1.0 / (1.0 + np.exp(-mag * sign))
    signal = rms(value) * gamma[:, None]
    scale = gate[None, :, :] + hc_scale[:, :, None]
    d = signal[:, None, :] * scale
    any = (np.sum(valid, axis=1) > 0).astype(np.float32)
    d *= any[None, None, :]
    want = x + d

    err = np.abs(want - got)
    print('D=%d HC=%d nt=%d cols=%d dim=%d' % (D, HC, nt, cols, dim))
    print('reference: first delta %.6f %.6f %.6f' % (d[0, 0, 0], d[0, 1, 0], d[0, 0, 1]))
    print('ggml:      first delta %.6f %.6f %.6f' % (got[0, 0, 0] - x[0, 0, 0], got[0, 1, 0] - x[0, 1, 0], got[0, 0, 1] - x[0, 0, 1]))
    print('max abs diff %.6g, mean abs diff %.6g, at %d/%d elements over 1e-4'
          % (err.max(), err.mean(), np.count_nonzero(err > 1e-4), err.size))
    print('first row want:', np.array2string(want[:4, 0, 0], precision=5))
    print('first row got: ', np.array2string(got[:4, 0, 0], precision=5))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Recompute iHC pre/post with numpy and compare with test_hy4_hc's ggml output. Usage: hy4_hc_ref.py HC.BIN"""
import struct
import sys

import numpy as np


def main():
    b = open(sys.argv[1], 'rb').read()
    D, HC, nt = struct.unpack_from('<3q', b, 0)
    eps, mag = struct.unpack_from('<2d', b, 24)
    off = 40
    def rd(n, dt='<f4'):
        nonlocal off
        a = np.frombuffer(b, dtype=dt, count=n, offset=off)
        off += a.nbytes
        return a
    x = rd(D * HC * nt).reshape(nt, HC, D).transpose(2, 1, 0)   # ggml [D,HC,nt]
    fn = rd(HC * D * 2 * HC)  # raw ggml order [k, n] is n*(HC*D)+k           # ggml [HC*D,2HC] -> [2HC,HC*D]
    scale = rd(2)
    base = rd(2 * HC).reshape(2, HC)                          # ggml [2HC] as pre[HC],post[HC]
    got_red = rd(D * nt).reshape(nt, D).T
    got_post = rd(HC * nt).reshape(nt, HC).T
    got_mix = rd(2 * HC * nt).reshape(nt, 2 * HC).T
    got_pre = rd(HC * nt).reshape(nt, HC).T
    x2 = rd(D * nt).reshape(nt, D).T
    res = rd(D * HC * nt).reshape(nt, HC, D).transpose(2, 1, 0)
    got_out = rd(D * HC * nt).reshape(nt, HC, D).transpose(2, 1, 0)

    flat = x.transpose(2, 1, 0).reshape(nt, HC * D).T
    norm = flat / np.sqrt(np.mean(flat * flat, axis=0, keepdims=True) + eps)
    mixes = fn.reshape(2 * HC, HC * D) @ norm  # [n, k] @ [k, t]                                          # [2HC, nt]
    pre = 1.0 / (1.0 + np.exp(-(mixes[:HC] * scale[0] + base[0][:, None] + eps)))
    post = 1.0 / (1.0 + np.exp(-(mixes[HC:] * scale[1] + base[1][:, None] + eps)))
    post = post * mag + eps
    red = np.sum(x * pre[None, :, :], axis=1)
    out = res + x2[:, None, :] * post[None, :, :]

    for name, want, got in [('mixes', mixes, got_mix), ('pre', pre, got_pre), ('reduced', red, got_red), ('post', post, got_post), ('out', out, got_out)]:
        err = np.abs(want - got)
        print('%-8s max %.6g mean %.6g first want %.6f got %.6f' % (name, err.max(), err.mean(), want.flat[0], got.flat[0]))


if __name__ == '__main__':
    main()

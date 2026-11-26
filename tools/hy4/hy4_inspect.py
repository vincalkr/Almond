#!/usr/bin/env python3
"""hy4_inspect.py FILE [FILE ...] - dump a GGUF's metadata and tensor map.

Prints enough of an Hy4-flash GGUF header to check a download and to see which tensor types a build uses. Reads only
the header region, so a file can be inspected before it is fully downloaded: fetch the first 8 MB with

    curl -L -r 0-8388607 -o shard.head "<the HF resolve URL>"

then pass shard.head here. No third-party modules. See docs/hy4/GGUF_NOTES.md for the inspected 2-bit build.
"""
import collections
import json
import struct
import sys

SCALARS = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2), 4: ('<I', 4), 5: ('<i', 4),
           6: ('<f', 4), 7: ('<B', 1), 10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8)}
# the vendored ggml enum (third_party/llama.cpp/ggml/include/ggml.h); 43 is upstream's F8_E4M3
TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
         10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS',
         17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S',
         23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M',
         30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4', 40: 'NVFP4', 41: 'Q1_0', 42: 'Q2_0',
         43: 'F8_E4M3'}


class Reader:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        if self.p + n > len(self.d):
            raise EOFError('need %d bytes at %d, file has %d' % (n, self.p, len(self.d)))
        b = self.d[self.p:self.p + n]
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


def short(v, n=90):
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
    return s if len(s) <= n else s[:n] + '... (%d chars)' % len(s)


def dump(path):
    r = Reader(open(path, 'rb').read())
    if r.take(4) != b'GGUF':
        print('%s: not a GGUF' % path)
        return
    version, nt, nk = r.u32(), r.u64(), r.u64()
    print('\n=== %s: GGUF v%d, %d tensors, %d metadata keys' % (path, version, nt, nk))
    meta = {}
    for _ in range(nk):
        k = r.string()
        meta[k] = r.value(r.u32())
    for k in sorted(meta):
        if k.startswith('tokenizer.') or k.startswith('quantize.'):
            continue
        print('  %s = %s' % (k, short(meta[k])))
    tok = [k for k in meta if k.startswith('tokenizer.ggml.') and not k.endswith(('tokens', 'merges', 'token_type'))]
    for k in sorted(tok):
        print('  %s = %s' % (k, short(meta[k])))
    tensors = []
    for _ in range(nt):
        name = r.string()
        nd = r.u32()
        shape = [r.u64() for _ in range(nd)]
        tt = r.u32()
        r.u64()  # offset
        tensors.append((name, tt, shape))
    counts = collections.Counter(TYPES.get(t, 'type%d' % t) for _, t, _ in tensors)
    print('  types: %s' % dict(counts))
    for name, tt, shape in tensors:
        if name.startswith('blk.0.') or name.startswith('blk.1.') or 'engram' in name or 'nextn' in name \
                or name.startswith(('output', 'token_embd', 'per_layer_token_embd')):
            print('    %-9s %-30s %s' % (TYPES.get(tt, 'type%d' % tt), str(shape), name))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        dump(p)

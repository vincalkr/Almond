#!/usr/bin/env python3
"""gguf_tui: terminal browser for GGUF files (metadata, tensors, quantizations, layers).

Proof of concept. No dependencies beyond the standard library (curses); reads only the
header, so a 100 GB shard opens in a moment. A split model (-00001-of-0000N) is read as
one: every shard's tensors are listed with the shard they live in.

  python3 tools/gguf_tui.py models/qwen38-27b/Qwen3.8-27B-Q8_0.gguf
  python3 tools/gguf_tui.py FILE --dump        # plain-text report, no TUI

Keys: Tab / 1-5 switch view, arrows/PgUp/PgDn/Home/End move, / filter, s sort (tensors),
Enter details, q quit.
"""
import curses
import os
import re
import struct
import sys
from collections import OrderedDict, defaultdict

# type id -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}
# prefer the vendored gguf-py table when present (it knows any type added since)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "llama.cpp", "gguf-py"))
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType  # type: ignore
    for t, (bs, ts) in GGML_QUANT_SIZES.items():
        GGML_TYPES[int(t)] = (t.name, bs, ts)
except Exception:
    pass
finally:
    sys.path.pop(0)

VT = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool",
      8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64"}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def get(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated header")
        return struct.unpack(fmt, b)[0]

    def string(self):
        n = self.get("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, vt, arr_cap=64):
        if vt == 8:
            return self.string()
        if vt == 9:
            et = self.get("<I")
            n = self.get("<Q")
            if et in FMT and n > arr_cap:  # skip the bulk of big numeric arrays
                sz = struct.calcsize(FMT[et])
                head = [self.get(FMT[et]) for _ in range(arr_cap)]
                self.f.seek((n - arr_cap) * sz, 1)
                return ArrayVal(VT.get(et, "?"), n, head)
            items = [self.value(et, arr_cap) for _ in range(n)]
            return ArrayVal(VT.get(et, "?"), n, items[:arr_cap] if et == 8 else items, items if et == 8 else None)
        return self.get(FMT[vt])


class ArrayVal:
    def __init__(self, etype, n, head, full=None):
        self.etype, self.n, self.head, self.full = etype, n, head, full

    def __str__(self):
        shown = ", ".join(repr(x) if isinstance(x, str) else fmt_num(x) for x in self.head[:8])
        more = ", ..." if self.n > 8 else ""
        return f"[{self.etype} x {self.n}] [{shown}{more}]"


class Tensor:
    __slots__ = ("name", "shape", "tid", "offset", "shard", "nbytes", "nel")

    @property
    def tname(self):
        return GGML_TYPES.get(self.tid, (f"type{self.tid}",))[0]

    @property
    def bpw(self):
        return self.nbytes * 8 / self.nel if self.nel else 0


def fmt_num(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or u == "TiB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.2f} {u}"
        n /= 1024


def shard_paths(path):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(3))
    paths = [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, n + 1)]
    return [p for p in paths if os.path.exists(p)]


def parse(path):
    meta = OrderedDict()
    tensors = []
    info = []
    for si, p in enumerate(shard_paths(path)):
        with open(p, "rb") as f:
            r = Reader(f)
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"{p}: not a GGUF file (magic {magic!r})")
            ver = r.get("<I")
            nt, nkv = r.get("<Q"), r.get("<Q")
            for _ in range(nkv):
                k = r.string()
                vt = r.get("<I")
                v = r.value(vt)
                if k not in meta:
                    meta[k] = (VT.get(vt, "?"), v)
            align = meta.get("general.alignment", (0, 32))[1]
            loc = []
            for _ in range(nt):
                t = Tensor()
                t.name = r.string()
                nd = r.get("<I")
                t.shape = [r.get("<Q") for _ in range(nd)]
                t.tid = r.get("<I")
                t.offset = r.get("<Q")
                t.shard = si + 1
                t.nel = 1
                for d in t.shape:
                    t.nel *= d
                name, bs, ts = GGML_TYPES.get(t.tid, ("?", 1, 0))
                t.nbytes = t.nel // bs * ts
                loc.append(t)
            hdr = f.tell()
            data_start = (hdr + align - 1) // align * align
            fsize = os.path.getsize(p)
            # bytes from the real offsets when the type is unknown
            srt = sorted(loc, key=lambda t: t.offset)
            for i, t in enumerate(srt):
                if t.nbytes == 0:
                    end = srt[i + 1].offset if i + 1 < len(srt) else fsize - data_start
                    t.nbytes = end - t.offset
            tensors += loc
            info.append(dict(path=p, version=ver, n_tensors=nt, n_kv=nkv, header=hdr,
                             data_start=data_start, size=fsize))
    return meta, tensors, info


LAYER_RE = re.compile(r"^(?:blk|layers?|model\.layers)\.(\d+)\.(.*)$")


def summaries(tensors):
    by_type = defaultdict(lambda: [0, 0, 0])
    for t in tensors:
        s = by_type[t.tname]
        s[0] += 1; s[1] += t.nel; s[2] += t.nbytes
    layers = defaultdict(lambda: [0, 0, 0, defaultdict(int)])
    groups = defaultdict(lambda: [0, 0, 0, defaultdict(int)])
    for t in tensors:
        m = LAYER_RE.match(t.name)
        key = f"blk.{int(m.group(1)):03d}" if m else "(non-layer)"
        base = re.sub(r"\.\d+\.", ".N.", t.name)
        for d, k in ((layers, key), (groups, base)):
            e = d[k]
            e[0] += 1; e[1] += t.nel; e[2] += t.nbytes; e[3][t.tname] += 1
    return by_type, layers, groups


def build_views(path, meta, tensors, info):
    tot_b = sum(t.nbytes for t in tensors)
    tot_n = sum(t.nel for t in tensors)
    by_type, layers, groups = summaries(tensors)
    g = lambda k, d="-": meta[k][1] if k in meta else d
    arch = g("general.architecture", "?")
    ov = [
        ("File", path), ("Shards", str(len(info))),
        ("GGUF version", str(info[0]["version"])),
        ("Name", str(g("general.name"))), ("Architecture", str(arch)),
        ("Size label", str(g("general.size_label"))),
        ("File type", str(g("general.file_type"))),
        ("Quantized by", str(g("general.quantized_by"))),
        ("Tensors", f"{len(tensors)}"), ("Metadata keys", f"{len(meta)}"),
        ("Parameters", f"{tot_n:,} ({tot_n / 1e9:.3f} B)"),
        ("Tensor data", f"{human(tot_b)}  ({tot_b * 8 / tot_n:.3f} bits/weight avg)" if tot_n else "-"),
        ("Files on disk", human(sum(i["size"] for i in info))),
        ("Layers", str(sum(1 for k in layers if k.startswith("blk")))),
    ]
    for k, v in meta.items():
        if k.startswith(arch + ".") and not isinstance(v[1], ArrayVal):
            ov.append((k[len(arch) + 1:], fmt_num(v[1])))
    tok = [(k, v) for k, v in meta.items() if k.startswith("tokenizer.")]
    if tok:
        ov.append(("", ""))
        for k, (_, v) in tok:
            if k == "tokenizer.chat_template":
                v = f"<{len(v)} chars>"
            ov.append((k, str(v)[:200]))
    for i in info:
        ov.append(("", ""))
        ov.append(("shard", os.path.basename(i["path"])))
        ov.append(("  size / header", f"{human(i['size'])} / {human(i['header'])}, data at {i['data_start']}"))
    return dict(tot_b=tot_b, tot_n=tot_n, by_type=by_type, layers=layers, groups=groups, overview=ov)


# ---------------------------------------------------------------- TUI
VIEWS = ["Overview", "Metadata", "Tensors", "Quant types", "Layers", "Groups"]
SORTS = ["file", "name", "size", "type", "bpw", "params"]


class App:
    def __init__(self, path, meta, tensors, info):
        self.path, self.meta, self.tensors, self.info = path, meta, tensors, info
        self.v = build_views(path, meta, tensors, info)
        self.view = 0
        self.pos = defaultdict(int)
        self.top = defaultdict(int)
        self.filter = ""
        self.sort = 0
        self.rev = False

    def rows(self):
        """returns (header, rows, detail_fn); each row a str"""
        flt = self.filter.lower()
        keep = lambda s: not flt or flt in s.lower()
        V = VIEWS[self.view]
        if V == "Overview":
            rows = [f"{k:<28} {v}" if k else "" for k, v in self.v["overview"]]
            return "", [r for r in rows if keep(r)], None
        if V == "Metadata":
            items = [(k, t, v) for k, (t, v) in self.meta.items() if keep(k + " " + str(v))]
            rows = [f"{k:<48} {t:<5} {str(v).replace(chr(10), ' ')}" for k, t, v in items]
            return f"{'key':<48} {'type':<5} value", rows, lambda i: self.meta_detail(items[i])
        if V == "Tensors":
            ts = [t for t in self.tensors if keep(t.name + " " + t.tname)]
            key = SORTS[self.sort]
            if key != "file":
                kf = {"name": lambda t: t.name, "size": lambda t: t.nbytes, "type": lambda t: t.tname,
                      "bpw": lambda t: t.bpw, "params": lambda t: t.nel}[key]
                ts = sorted(ts, key=kf, reverse=self.rev)
            elif self.rev:
                ts = ts[::-1]
            rows = [f"{t.name:<52} {t.tname:<8} {'x'.join(map(str, t.shape)):<22} {t.nel:>14,} "
                    f"{human(t.nbytes):>11} {t.bpw:6.3f} {t.shard:>3}" for t in ts]
            hdr = (f"{'name':<52} {'type':<8} {'shape':<22} {'elements':>14} {'bytes':>11} {'bpw':>6} {'sh':>3}"
                   f"   [sort: {key}{' desc' if self.rev else ''}]")
            return hdr, rows, lambda i: self.tensor_detail(ts[i])
        if V == "Quant types":
            tb = self.v["tot_b"] or 1
            items = sorted(self.v["by_type"].items(), key=lambda kv: -kv[1][2])
            rows = []
            for name, (n, nel, nb) in items:
                bar = "#" * int(40 * nb / tb)
                rows.append(f"{name:<10} {n:>6} {nel:>16,} {human(nb):>12} {100 * nb / tb:6.2f}% "
                            f"{nb * 8 / nel if nel else 0:6.3f}  {bar}")
            return f"{'type':<10} {'count':>6} {'elements':>16} {'bytes':>12} {'share':>7} {'bpw':>6}", \
                [r for r in rows if keep(r)], None
        if V in ("Layers", "Groups"):
            d = self.v["layers" if V == "Layers" else "groups"]
            items = sorted(d.items())
            rows = []
            for k, (n, nel, nb, types) in items:
                ty = " ".join(f"{t}:{c}" for t, c in sorted(types.items(), key=lambda x: -x[1]))
                rows.append(f"{k:<46} {n:>5} {nel:>16,} {human(nb):>12} {nb * 8 / nel if nel else 0:6.3f}  {ty}")
            return f"{'layer' if V == 'Layers' else 'tensor name pattern':<46} {'n':>5} {'elements':>16} " \
                   f"{'bytes':>12} {'bpw':>6}  types", [r for r in rows if keep(r)], None
        return "", [], None

    def meta_detail(self, item):
        k, t, v = item
        lines = [f"key:  {k}", f"type: {t}", ""]
        if isinstance(v, ArrayVal):
            lines.append(f"array of {v.n} {v.etype}" + (" (first shown)" if v.full is None and v.n > len(v.head) else ""))
            src = v.full if v.full is not None else v.head
            lines += [f"{i:>8}: {x!r}" for i, x in enumerate(src)]
        else:
            lines += str(v).split("\n")
        return lines

    def tensor_detail(self, t):
        name, bs, ts = GGML_TYPES.get(t.tid, ("?", 0, 0))
        info = self.info[t.shard - 1]
        return [
            f"name:        {t.name}", f"type:        {name} (id {t.tid}), block {bs} elements / {ts} bytes",
            f"shape:       {t.shape}  (ggml order, ne0 first)", f"elements:    {t.nel:,}",
            f"bytes:       {t.nbytes:,} ({human(t.nbytes)})", f"bits/weight: {t.bpw:.4f}",
            f"shard:       {t.shard}  {os.path.basename(info['path'])}",
            f"offset:      {t.offset:,} in data, {info['data_start'] + t.offset:,} in file",
            f"share:       {100 * t.nbytes / (self.v['tot_b'] or 1):.4f}% of tensor data",
        ]

    def popup(self, scr, lines):
        top = 0
        while True:
            H, W = scr.getmaxyx()
            h, w = H - 4, W - 6
            win = curses.newwin(h, w, 2, 3)
            win.box()
            for i, l in enumerate(lines[top:top + h - 2]):
                win.addnstr(i + 1, 2, l, w - 4)
            win.addnstr(h - 1, 2, f" {top + 1}-{min(top + h - 2, len(lines))}/{len(lines)}  Esc/q close ", w - 4)
            win.refresh()
            c = scr.getch()
            if c in (27, ord("q"), 10, 13):
                return
            top = self.move(c, top, len(lines), h - 2)

    @staticmethod
    def move(c, p, n, page):
        if c in (curses.KEY_DOWN, ord("j")): p += 1
        elif c in (curses.KEY_UP, ord("k")): p -= 1
        elif c == curses.KEY_NPAGE: p += page
        elif c == curses.KEY_PPAGE: p -= page
        elif c in (curses.KEY_HOME, ord("g")): p = 0
        elif c in (curses.KEY_END, ord("G")): p = n - 1
        return max(0, min(p, max(n - 1, 0)))

    def prompt(self, scr, label):
        H, W = scr.getmaxyx()
        curses.echo(); curses.curs_set(1)
        scr.move(H - 1, 0); scr.clrtoeol(); scr.addstr(H - 1, 0, label)
        s = scr.getstr(H - 1, len(label), W - len(label) - 1).decode("utf-8", "replace")
        curses.noecho(); curses.curs_set(0)
        return s

    def run(self, scr):
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        hx = 0
        while True:
            H, W = scr.getmaxyx()
            scr.erase()
            hdr, rows, det = self.rows()
            # tab bar
            x = 0
            for i, name in enumerate(VIEWS):
                lab = f" {i + 1} {name} "
                scr.addnstr(0, x, lab, max(0, W - x - 1), curses.color_pair(1) if i == self.view else curses.A_DIM)
                x += len(lab) + 1
                if x >= W: break
            body = H - 4
            v = self.view
            n = len(rows)
            p = self.pos[v] = min(self.pos[v], max(n - 1, 0))
            if p < self.top[v]: self.top[v] = p
            if p >= self.top[v] + body: self.top[v] = p - body + 1
            if hdr:
                scr.addnstr(1, 0, hdr[hx:], W - 1, curses.color_pair(2) | curses.A_BOLD)
            for i, r in enumerate(rows[self.top[v]:self.top[v] + body]):
                attr = curses.A_REVERSE if self.top[v] + i == p else 0
                scr.addnstr(2 + i, 0, r[hx:].ljust(W - 1), W - 1, attr)
            st = (f" {os.path.basename(self.path)} | {p + 1 if n else 0}/{n}"
                  f"{' | filter: ' + self.filter if self.filter else ''}"
                  f" | Tab/1-6 view  / filter  s sort  r reverse  Enter detail  ←→ scroll  q quit")
            scr.addnstr(H - 1, 0, st.ljust(W - 1), W - 1, curses.color_pair(1))
            scr.refresh()
            c = scr.getch()
            if c in (ord("q"), 27): return
            elif c == 9: self.view = (self.view + 1) % len(VIEWS); hx = 0
            elif c == curses.KEY_BTAB: self.view = (self.view - 1) % len(VIEWS); hx = 0
            elif ord("1") <= c <= ord(str(len(VIEWS))): self.view = c - ord("1"); hx = 0
            elif c == ord("/"): self.filter = self.prompt(scr, "filter: "); self.pos[v] = 0
            elif c == ord("s"): self.sort = (self.sort + 1) % len(SORTS)
            elif c == ord("r"): self.rev = not self.rev
            elif c == curses.KEY_RIGHT: hx += 8
            elif c == curses.KEY_LEFT: hx = max(0, hx - 8)
            elif c in (10, 13, curses.KEY_ENTER) and det and n: self.popup(scr, det(p))
            else: self.pos[v] = self.move(c, p, n, body)


def dump(path, meta, tensors, info):
    a = App(path, meta, tensors, info)
    for i, name in enumerate(VIEWS):
        a.view = i
        hdr, rows, _ = a.rows()
        print(f"\n===== {name} =====")
        if hdr: print(hdr)
        for r in rows: print(r)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__); sys.exit(1)
    path = args[0]
    meta, tensors, info = parse(path)
    if "--dump" in sys.argv:
        dump(path, meta, tensors, info)
    else:
        os.environ.setdefault("ESCDELAY", "25")
        curses.wrapper(App(path, meta, tensors, info).run)


if __name__ == "__main__":
    main()

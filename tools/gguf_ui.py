#!/usr/bin/env python3
"""gguf_ui: Textual browser for GGUF files (the parser lives in gguf_tui.py).

  .venv/bin/python tools/gguf_ui.py MODEL.gguf

Views: Overview (cards + quantization mix), Tensors (filter, sort by clicking a column, detail
panel), Quantization, Layers (one stacked bar per block), Metadata (tree). Keys in the footer.

  .venv/bin/python tools/gguf_ui.py MODEL.gguf --export [DIR]   # Markdown + JSON report, no UI

`e` in the UI writes the same report into the current directory.
"""
import colorsys
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_tui import GGML_TYPES, LAYER_RE, ArrayVal, build_views, fmt_num, human, parse  # noqa: E402

from rich.text import Text  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll  # noqa: E402
from textual.widgets import DataTable, Footer, Input, Static, TabbedContent, TabPane, Tree  # noqa: E402


def type_bpw(tid):
    n, bs, ts = GGML_TYPES.get(tid, ("?", 1, 0))
    return ts * 8 / bs if bs else 0


class Palette:
    """one colour per type present, on a warm (few bits) -> cool (many bits) ramp"""

    def __init__(self, tensors):
        types = {}
        for t in tensors:
            types[t.tname] = t.bpw if t.tname not in types else types[t.tname]
        order = sorted(types, key=lambda k: types[k])
        self.c = {}
        ramp = ["#ff5370", "#ff8a5c", "#ffb454", "#ffd866", "#c3e88d", "#7fdb8a", "#4fd6be",
                "#5ccfe6", "#73a8ff", "#9d8cff", "#c792ea", "#f07fd0", "#8a93b8", "#5f6b8f"]
        for i, name in enumerate(order):
            if len(order) <= len(ramp):
                j = round(i * (len(ramp) - 1) / max(len(order) - 1, 1)) if len(order) > 1 else 6
                self.c[name] = ramp[j]
            else:
                r, g, b = colorsys.hls_to_rgb(0.85 * i / (len(order) - 1), 0.62, 0.75)
                self.c[name] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        self.order = order

    def __getitem__(self, k):
        return self.c.get(k, "#888888")

    def chip(self, name):
        return Text(f" {name} ", style=f"bold #111111 on {self[name]}")


def stacked_bar(parts, width, pal):
    """parts: [(type, bytes)] -> Text of `width` cells"""
    tot = sum(b for _, b in parts) or 1
    out, used = Text(), 0
    for i, (name, b) in enumerate(parts):
        w = round(width * b / tot) if i < len(parts) - 1 else width - used
        w = max(0, min(w, width - used))
        out.append("█" * w, style=pal[name])
        used += w
    return out


CSS = """
Screen { background: $background; }
#hero { height: auto; padding: 1 2; background: $boost; }
TabbedContent { height: 1fr; }
.cards { grid-size: 4; grid-gutter: 1 2; height: auto; padding: 1 2 0 2; }
.card { height: 5; padding: 0 2; background: $panel; border: round $primary 40%; }
.section { padding: 1 2 0 2; color: $text-muted; text-style: bold; }
#mix, #mixlegend { padding: 0 2; height: auto; }
#hparams { padding: 0 2 1 2; height: auto; }
#tensors-side { width: 40; padding: 1 2; background: $panel; border-left: tall $primary 40%; }
#filter { margin: 0 1; }
DataTable { height: 1fr; }
#layers { padding: 0 1; }
#meta-side { width: 1fr; padding: 1 2; background: $panel; border-left: tall $primary 40%; }
Tree { width: 1fr; }
"""


class GGUFApp(App):
    CSS = CSS
    TITLE = "gguf"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("1", "tab('ov')", "Overview"), Binding("2", "tab('ts')", "Tensors"),
        Binding("3", "tab('qt')", "Quant"), Binding("4", "tab('ly')", "Layers"),
        Binding("5", "tab('md')", "Metadata"),
        Binding("e", "export", "Export"),
    ]

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.meta, self.tensors, self.info = parse(path)
        self.v = build_views(path, self.meta, self.tensors, self.info)
        self.pal = Palette(self.tensors)
        self.sort_key, self.sort_rev = None, False
        self.shown = []

    # ------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        g = lambda k, d="—": self.meta[k][1] if k in self.meta else d
        arch = g("general.architecture", "?")
        hero = Text()
        hero.append(str(g("general.name", os.path.basename(self.path))), style="bold")
        hero.append(f"   {arch}", style="bold magenta")
        if g("general.size_label", None):
            hero.append(f" · {g('general.size_label')}", style="cyan")
        hero.append(f"   {os.path.basename(self.path)}", style="dim")
        if len(self.info) > 1:
            hero.append(f"  ({len(self.info)} shards)", style="dim")
        yield Static(hero, id="hero")
        with TabbedContent(initial="ov"):
            with TabPane("Overview", id="ov"):
                yield VerticalScroll(*self.overview())
            with TabPane("Tensors", id="ts"):
                with Horizontal():
                    with Vertical():
                        yield Input(placeholder="filter tensors by name or type  (e.g. ffn_down, IQ2, blk.3.)", id="filter")
                        yield DataTable(id="tt", cursor_type="row", zebra_stripes=True)
                    yield Static(id="tensors-side")
            with TabPane("Quantization", id="qt"):
                yield DataTable(id="qtt", cursor_type="row", zebra_stripes=True)
            with TabPane("Layers", id="ly"):
                yield VerticalScroll(Static(id="layers"))
            with TabPane("Metadata", id="md"):
                with Horizontal():
                    yield Tree(os.path.basename(self.path), id="mt")
                    yield VerticalScroll(Static(id="meta-side"))
        yield Footer()

    def card(self, label, value, sub=""):
        t = Text()
        t.append(label.upper() + "\n", style="dim")
        t.append(value + "\n", style="bold")
        t.append(sub, style="dim italic")
        return Static(t, classes="card")

    def overview(self):
        g = lambda k, d="—": self.meta[k][1] if k in self.meta else d
        arch = g("general.architecture", "?")
        n, b = self.v["tot_n"], self.v["tot_b"]
        nlay = sum(1 for k in self.v["layers"] if k.startswith("blk"))
        ctx = g(f"{arch}.context_length", None)
        experts = g(f"{arch}.expert_count", None)
        cards = [
            self.card("Parameters", f"{n / 1e9:,.2f} B", f"{n:,}"),
            self.card("Weights", human(b), f"{human(sum(i['size'] for i in self.info))} on disk"),
            self.card("Bits / weight", f"{b * 8 / n:.3f}" if n else "—", f"{len(self.pal.order)} tensor types"),
            self.card("Tensors", f"{len(self.tensors):,}", f"{len(self.meta)} metadata keys"),
            self.card("Layers", str(nlay), f"embd {g(f'{arch}.embedding_length')}"),
            self.card("Context", f"{ctx:,}" if isinstance(ctx, int) else "—", f"rope base {fmt_num(g(f'{arch}.rope.freq_base'))}"),
            self.card("Attention", f"{g(f'{arch}.attention.head_count')} / {g(f'{arch}.attention.head_count_kv')}", "heads / kv heads"),
            self.card("Experts", f"{experts} ({g(f'{arch}.expert_used_count')} used)" if experts else "dense", f"GGUF v{self.info[0]['version']}"),
        ]
        yield Grid(*cards, classes="cards")
        yield Static("QUANTIZATION MIX  (share of bytes)", classes="section")
        parts = sorted(((k, v[2]) for k, v in self.v["by_type"].items()), key=lambda x: -x[1])
        self.mix_parts = parts
        yield Static(id="mix")
        leg = Text()
        for k, nb in parts:
            leg.append("● ", style=self.pal[k])
            leg.append(f"{k} ", style="bold")
            leg.append(f"{100 * nb / (b or 1):.1f}%   ", style="dim")
        yield Static(leg, id="mixlegend")
        yield Static("HYPERPARAMETERS", classes="section")
        hp = Text()
        items = [(k[len(arch) + 1:], v) for k, (_, v) in self.meta.items()
                 if k.startswith(arch + ".") and not isinstance(v, ArrayVal)]
        for k, v in items:
            hp.append(f"{k:<40}", style="cyan")
            hp.append(f"{fmt_num(v)}\n")
        yield Static(hp, id="hparams")

    def on_mount(self):
        self.theme = "tokyo-night"
        self.query_one("#mix", Static).update(stacked_bar(self.mix_parts, 100, self.pal))
        self.fill_tensors()
        self.fill_quant()
        self.fill_layers()
        self.fill_meta()

    def on_resize(self, event):
        w = max(20, self.size.width - 6)
        self.query_one("#mix", Static).update(stacked_bar(self.mix_parts, w, self.pal))
        self.fill_layers()

    # ------------------------------------------------------------ tensors
    COLS = [("name", "Name"), ("type", "Type"), ("shape", "Shape"), ("params", "Elements"),
            ("size", "Size"), ("bpw", "bpw"), ("shard", "Shard")]

    def fill_tensors(self, flt=""):
        tt = self.query_one("#tt", DataTable)
        if not tt.columns:
            for k, label in self.COLS:
                tt.add_column(label, key=k)
        tt.clear()
        f = flt.lower()
        ts = [t for t in self.tensors if not f or f in t.name.lower() or f in t.tname.lower()]
        kf = {"name": lambda t: t.name, "type": lambda t: (t.bpw, t.tname), "shape": lambda t: t.shape,
              "params": lambda t: t.nel, "size": lambda t: t.nbytes, "bpw": lambda t: t.bpw,
              "shard": lambda t: t.shard}.get(self.sort_key)
        if kf:
            ts.sort(key=kf, reverse=self.sort_rev)
        self.shown = ts
        maxb = max((t.nbytes for t in self.tensors), default=1)
        for i, t in enumerate(ts):
            size = Text(f"{human(t.nbytes):>10} ")
            size.append("▮" * max(1, round(8 * t.nbytes / maxb)), style=self.pal[t.tname])
            tt.add_row(Text(t.name), self.pal.chip(t.tname), Text(" × ".join(map(str, t.shape)), style="dim"),
                       Text(f"{t.nel:,}", justify="right"), size,
                       Text(f"{t.bpw:.3f}", justify="right"), Text(str(t.shard), justify="center"), key=str(i))
        self.query_one("#tensors-side", Static).update(
            Text(f"{len(ts):,} of {len(self.tensors):,} tensors\n{human(sum(t.nbytes for t in ts))}", style="dim"))

    def on_input_changed(self, ev: Input.Changed):
        if ev.input.id == "filter":
            self.fill_tensors(ev.value)

    def on_data_table_header_selected(self, ev: DataTable.HeaderSelected):
        if ev.data_table.id != "tt":
            return
        k = ev.column_key.value
        self.sort_rev = (not self.sort_rev) if self.sort_key == k else k in ("params", "size", "bpw")
        self.sort_key = k
        self.fill_tensors(self.query_one("#filter", Input).value)

    def on_data_table_row_highlighted(self, ev: DataTable.RowHighlighted):
        if ev.data_table.id != "tt" or ev.row_key is None or ev.row_key.value is None:
            return
        t = self.shown[int(ev.row_key.value)]
        name, bs, tsz = GGML_TYPES.get(t.tid, ("?", 0, 0))
        info = self.info[t.shard - 1]
        m = LAYER_RE.match(t.name)
        x = Text()
        x.append(t.name + "\n\n", style="bold")
        x.append_text(self.pal.chip(t.tname))
        x.append(f"  {t.bpw:.3f} bits/weight\n\n", style="bold")

        def row(k, v):
            x.append(f"{k:<11}", style="dim")
            x.append(f"{v}\n")
        row("layer", m.group(1) if m else "—")
        row("shape", " × ".join(map(str, t.shape)))
        row("elements", f"{t.nel:,}")
        row("bytes", f"{t.nbytes:,}")
        row("", human(t.nbytes))
        row("block", f"{bs} el / {tsz} B" if bs else "—")
        row("share", f"{100 * t.nbytes / (self.v['tot_b'] or 1):.3f}%")
        row("shard", f"{t.shard}")
        row("offset", f"{info['data_start'] + t.offset:,}")
        self.query_one("#tensors-side", Static).update(x)

    def action_focus_filter(self):
        self.query_one(TabbedContent).active = "ts"
        self.query_one("#filter", Input).focus()

    def action_export(self):
        try:
            paths = export(self.path, self.meta, self.tensors, self.info, os.getcwd())
            self.notify("\n".join(os.path.relpath(p) for p in paths), title="Exported")
        except Exception as e:
            self.notify(str(e), title="Export failed", severity="error")

    def action_tab(self, tab):
        self.query_one(TabbedContent).active = tab

    # ------------------------------------------------------------ quant
    def fill_quant(self):
        qt = self.query_one("#qtt", DataTable)
        for c in ("Type", "Tensors", "Elements", "Size", "Share", "bpw", ""):
            qt.add_column(c)
        tot = self.v["tot_b"] or 1
        for k, (n, nel, nb) in sorted(self.v["by_type"].items(), key=lambda kv: -kv[1][2]):
            bar = Text("█" * max(1, round(40 * nb / tot)), style=self.pal[k])
            qt.add_row(self.pal.chip(k), Text(f"{n:,}", justify="right"), Text(f"{nel:,}", justify="right"),
                       Text(human(nb), justify="right"), Text(f"{100 * nb / tot:.2f}%", justify="right"),
                       Text(f"{nb * 8 / nel if nel else 0:.3f}", justify="right"), bar)

    # ------------------------------------------------------------ layers
    def fill_layers(self):
        per = defaultdict(lambda: defaultdict(int))
        for t in self.tensors:
            m = LAYER_RE.match(t.name)
            per[int(m.group(1)) if m else -1][t.tname] += t.nbytes
        biggest = max((sum(d.values()) for k, d in per.items() if k >= 0), default=1)
        w = max(20, self.size.width - 40)
        out = Text()
        out.append("\nbytes per block, coloured by tensor type; bar length relative to the largest block (non-layer tensors clipped ▸)\n\n", style="dim")
        for k in sorted(per):
            d = per[k]
            tot = sum(d.values())
            parts = sorted(d.items(), key=lambda x: -x[1])
            label = f"blk {k:>3}" if k >= 0 else "other  "
            out.append(f" {label}  ", style="bold" if k >= 0 else "bold magenta")
            bw = max(1, round(w * tot / biggest))
            out.append_text(stacked_bar(parts, min(bw, w), self.pal))
            if bw > w:
                out.append("▸", style="bold")
            out.append(f"  {human(tot)}\n", style="dim")
        out.append("\n ")
        for name in self.pal.order:
            out.append("● ", style=self.pal[name])
            out.append(f"{name}  ")
        self.query_one("#layers", Static).update(out)

    # ------------------------------------------------------------ metadata
    def fill_meta(self):
        tr = self.query_one("#mt", Tree)
        tr.root.expand()
        nodes = {}
        for k, (ty, v) in self.meta.items():
            parts = k.split(".")
            parent, path = tr.root, ""
            for p in parts[:-1][:2]:
                path += p + "."
                if path not in nodes:
                    nodes[path] = parent.add(Text(p, style="bold cyan"), expand=path.count(".") == 1)
                parent = nodes[path]
            leaf = ".".join(parts[len(path.split(".")) - 1:])
            preview = str(v).replace("\n", " ")
            lab = Text(leaf)
            lab.append(f"  {preview[:50]}{'…' if len(preview) > 50 else ''}", style="dim")
            parent.add_leaf(lab, data=k)

    def on_tree_node_highlighted(self, ev: Tree.NodeHighlighted):
        k = ev.node.data
        if not k:
            return
        ty, v = self.meta[k]
        x = Text()
        x.append(k + "\n", style="bold")
        x.append(f"{ty}\n\n", style="magenta")
        if isinstance(v, ArrayVal):
            src = v.full if v.full is not None else v.head
            x.append(f"array of {v.n:,} {v.etype}" + ("" if len(src) == v.n else f", first {len(src)} shown") + "\n\n", style="dim")
            for i, e in enumerate(src[:2000]):
                x.append(f"{i:>7}  ", style="dim")
                x.append(f"{e!r}\n")
        else:
            x.append(str(v))
        self.query_one("#meta-side", Static).update(x)


def _jsonable(v):
    if isinstance(v, ArrayVal):
        src = v.full if v.full is not None else v.head
        return {"array_type": v.etype, "length": v.n, "values": src, "truncated": len(src) < v.n}
    return v


def export(path, meta, tensors, info, out_dir):
    """writes <model>.gguf-report.md and <model>.gguf-report.json in out_dir; returns their paths"""
    import json
    v = build_views(path, meta, tensors, info)
    base = os.path.join(out_dir, os.path.basename(path).removesuffix(".gguf") + ".gguf-report")
    tot_b = v["tot_b"] or 1
    data = {
        "file": path,
        "shards": [{k: i[k] for k in ("path", "version", "n_tensors", "n_kv", "header", "data_start", "size")} for i in info],
        "summary": {"parameters": v["tot_n"], "tensor_bytes": v["tot_b"],
                    "bits_per_weight": v["tot_b"] * 8 / v["tot_n"] if v["tot_n"] else None,
                    "tensors": len(tensors), "metadata_keys": len(meta)},
        "quant_types": [{"type": k, "tensors": n, "elements": nel, "bytes": nb, "share": nb / tot_b,
                         "bpw": nb * 8 / nel if nel else None}
                        for k, (n, nel, nb) in sorted(v["by_type"].items(), key=lambda kv: -kv[1][2])],
        "layers": [{"layer": k, "tensors": n, "elements": nel, "bytes": nb, "types": dict(ty)}
                   for k, (n, nel, nb, ty) in sorted(v["layers"].items())],
        "groups": [{"pattern": k, "tensors": n, "elements": nel, "bytes": nb, "types": dict(ty)}
                   for k, (n, nel, nb, ty) in sorted(v["groups"].items())],
        "metadata": {k: {"type": t, "value": _jsonable(x)} for k, (t, x) in meta.items()},
        "tensors": [{"name": t.name, "type": t.tname, "shape": t.shape, "elements": t.nel, "bytes": t.nbytes,
                     "bpw": t.bpw, "shard": t.shard, "offset": t.offset} for t in tensors],
    }
    with open(base + ".json", "w") as f:
        json.dump(data, f, indent=1, default=str)

    L = [f"# {os.path.basename(path)}", ""]
    L += ["| | |", "|---|---|"] + [f"| {k} | {str(x).replace('|', '/')} |" for k, x in v["overview"] if k]
    L += ["", "## Quantization types", "", "| type | tensors | elements | bytes | share | bpw |", "|---|--:|--:|--:|--:|--:|"]
    L += [f"| {q['type']} | {q['tensors']} | {q['elements']:,} | {human(q['bytes'])} | {100 * q['share']:.2f}% | "
          f"{q['bpw']:.3f} |" for q in data["quant_types"]]
    L += ["", "## Tensor groups", "", "| pattern | n | bytes | types |", "|---|--:|--:|---|"]
    L += [f"| `{g['pattern']}` | {g['tensors']} | {human(g['bytes'])} | "
          f"{', '.join(f'{a}:{b}' for a, b in g['types'].items())} |" for g in data["groups"]]
    L += ["", "## Layers", "", "| layer | n | bytes | types |", "|---|--:|--:|---|"]
    L += [f"| {l['layer']} | {l['tensors']} | {human(l['bytes'])} | "
          f"{', '.join(f'{a}:{b}' for a, b in l['types'].items())} |" for l in data["layers"]]
    L += ["", "## Metadata", "", "| key | type | value |", "|---|---|---|"]
    for k, (t, x) in meta.items():
        val = f"<{len(x)} chars>" if k == "tokenizer.chat_template" else str(x)
        L.append(f"| `{k}` | {t} | {val.replace('|', '/').replace(chr(10), ' ')[:300]} |")
    L += ["", "## Tensors", "", "| name | type | shape | elements | bytes | bpw | shard |", "|---|---|---|--:|--:|--:|--:|"]
    L += [f"| `{t.name}` | {t.tname} | {' x '.join(map(str, t.shape))} | {t.nel:,} | {human(t.nbytes)} | "
          f"{t.bpw:.3f} | {t.shard} |" for t in tensors]
    with open(base + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    return [base + ".md", base + ".json"]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    if "--export" in sys.argv:
        out = args[1] if len(args) > 1 else os.getcwd()
        for p in export(args[0], *parse(args[0]), out):
            print(p)
    else:
        GGUFApp(args[0]).run()

#!/usr/bin/env python3
"""almond-bench: Textual UI for the local benchmark (core in bench_core.py, data from datasets.py).

  ./almond-bench [--url http://127.0.0.1:8150/v1]

Tabs: Run (endpoint, model, judge, preset, suites; live results with the running score), Results (one saved run:
score +- standard error per suite, every task with the model's answer and why it was graded so), Compare (all
runs side by side, e.g. the quantizations of one model, and the KL divergences measured with kld.py).
When nothing answers at a local endpoint the router is started as ./almond does, and stopped on quit.
Keys in the footer; docs/BENCH.md.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_core as bc  # noqa: E402
import datasets as ds  # noqa: E402
import kld  # noqa: E402

from rich.text import Text  # noqa: E402
from textual import work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll  # noqa: E402
from textual.screen import ModalScreen  # noqa: E402
from textual.widgets import (Button, Checkbox, DataTable, Footer, Header, Input, Label, ProgressBar,  # noqa: E402
                             Select, Static, TabbedContent, TabPane)


def score_color(s):
    if s is None:
        return "#8a93b8"
    return "#ff5370" if s < 0.34 else "#ffb454" if s < 0.6 else "#c3e88d" if s < 0.85 else "#4fd6be"


def bar(s, se=0.0, width=28):
    """a bar with the +-1 standard error drawn as a lighter band"""
    if s is None:
        return Text("  n/a", style="#8a93b8")
    lo, hi = max(0.0, s - se), min(1.0, s + se)
    t = Text()
    for i in range(width):
        x = (i + 0.5) / width
        t.append("█" if x <= lo else "▓" if x <= hi else "░",
                 style=score_color(s) if x <= hi else "#3a3f58")
    t.append(f" {s * 100:5.1f}", style="bold")
    t.append(f" ±{se * 100:.1f}" if se else "", style="#8a93b8")
    return t


def cell(s, se=None):
    if s is None:
        return Text("  -  ", style="#8a93b8")
    t = Text(f"{s * 100:5.1f}", style=f"bold {score_color(s)}")
    if se:
        t.append(f"±{se * 100:.1f}", style="#8a93b8")
    return t


def sid(suite):
    return "su-" + suite.replace("+", "-plus")


def sget(summary, k):
    v = summary.get(k)
    return (v["score"], v["se"]) if v else (None, None)


TRASH = os.path.join(bc.ROOT, "local", "bench", "trash")


def to_trash(path):
    """runs and KL results are moved to local/bench/trash (recoverable), not deleted"""
    os.makedirs(TRASH, exist_ok=True)
    dst = os.path.join(TRASH, f"{time.strftime('%Y%m%d-%H%M%S')}-{os.path.basename(path)}")
    shutil.move(path, dst)


class Confirm(ModalScreen):
    CSS = """
    Confirm { align: center middle; }
    #box { grid-size: 2; grid-gutter: 1 2; grid-rows: auto 3; padding: 1 2; width: 72; height: auto;
           border: thick #ff5370; background: #24273a; }
    #q { column-span: 2; }
    #box Button { width: 100%; }
    """
    BINDINGS = [Binding("y", "yes", "Yes"), Binding("n,escape", "no", "No")]

    def __init__(self, text):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Grid(id="box"):
            yield Static(self.text, id="q")
            yield Button("Move to trash (y)", variant="error", id="yes")
            yield Button("Cancel (n)", variant="primary", id="no")

    def on_button_pressed(self, ev):
        self.dismiss(ev.button.id == "yes")

    def action_yes(self):
        self.dismiss(True)

    def action_no(self):
        self.dismiss(False)


class BenchApp(App):
    TITLE = "almond-bench"
    CSS = """
    Screen { background: #1a1c27; }
    #side { width: 48; padding: 0 1; border-right: tall #3a3f58; }
    #side Label { margin-top: 1; color: #9d8cff; text-style: bold; }
    #side Checkbox { padding: 0; border: none; height: 1; }
    #buttons { height: 3; margin-top: 1; }
    #buttons Button { width: 1fr; min-width: 8; }
    #status { height: 2; padding: 0 1; color: #c3e88d; }
    #running { height: auto; padding: 0 1; }
    #summary { height: auto; padding: 0 1; border: round #3a3f58; margin: 0 1; }
    #detail { height: 14; border: round #3a3f58; padding: 0 1; margin: 0 1; }
    #runs { width: 58; }
    #kld { height: auto; max-height: 14; }
    .title { color: #9d8cff; text-style: bold; padding: 1 1 0 1; }
    DataTable { height: 1fr; }
    """
    BINDINGS = [Binding("q", "quit", "Quit"), Binding("s", "start", "Start"), Binding("x", "stop", "Stop"),
                Binding("m", "models", "Models"), Binding("d", "fetch", "Download missing data"),
                Binding("r", "refresh", "Reload runs"), Binding("c", "continue_run", "Resume selected run"),
                Binding("delete", "delete", "Delete selected"), Binding("D", "delete_all", "Delete all results")]

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.runs = []
        self.cur = None
        self.stop_flag = threading.Event()
        self.busy = False
        self.router = None
        self.kld_rows = []

    def compose(self) -> ComposeResult:
        have = ds.available()
        yield Header()
        with TabbedContent():
            with TabPane("Run", id="t-run"):
                with Horizontal():
                    with VerticalScroll(id="side"):
                        yield Label("Endpoint")
                        yield Input(self.url, id="url")
                        yield Label("Model under test")
                        yield Select([], id="model", prompt="m: fetch the list")
                        yield Label("Judge (arena-hard only)")
                        yield Select([], id="judge", prompt="none: arena-hard skipped")
                        yield Label("Preset")
                        yield Select([("quick: fixed sample per suite", "quick"), ("full: every task", "full")],
                                     value="quick", allow_blank=False, id="preset")
                        yield Input(placeholder="tasks per suite (optional)", id="limit", type="integer")
                        yield Label("Suites")
                        for name, (kind, quick, _, what) in bc.SUITES.items():
                            ok = have.get(name)
                            yield Checkbox(f"{name:<11}{what}" + ("" if ok else "  (missing)"),
                                           bool(ok) and name != "arena-hard", id=sid(name), disabled=not ok)
                        with Horizontal(id="buttons"):
                            yield Button("Start", id="start", variant="success")
                            yield Button("Stop", id="stop", variant="error", disabled=True)
                            yield Button("Data", id="fetch")
                    with Vertical():
                        yield ProgressBar(id="prog", show_eta=True)
                        yield Static("idle", id="status")
                        yield Static(id="running")
                        yield DataTable(id="live", cursor_type="row", zebra_stripes=True)
            with TabPane("Results", id="t-res"):
                with Horizontal():
                    yield DataTable(id="runs", cursor_type="row")
                    with Vertical():
                        yield Static(id="summary")
                        yield DataTable(id="tasks", cursor_type="row", zebra_stripes=True)
                        yield VerticalScroll(Static(id="detail-text"), id="detail")
            with TabPane("Compare", id="t-cmp"):
                with Vertical():
                    yield Static("Benchmark scores (% ± standard error)", classes="title")
                    yield DataTable(id="cmp", zebra_stripes=True)
                    yield Static("KL divergence against the reference build (kld.py; lower is better)", classes="title")
                    yield DataTable(id="kld", zebra_stripes=True)
        yield Footer()

    def on_mount(self):
        self.query_one("#live", DataTable).add_columns("suite", "task", "score", "s", "tok", "note")
        self.query_one("#runs", DataTable).add_columns("when", "model", "preset", "n", "overall")
        self.query_one("#tasks", DataTable).add_columns("suite", "task", "category", "score", "s", "tok", "note")
        self.action_refresh()
        self.action_models()

    def set_status(self, msg):
        self.query_one("#status", Static).update(msg)

    # ------------------------------------------------ router, models and data
    def start_router(self, url):
        """as ./almond does: nothing answers on this machine's address, so start the router (launch.py --router);
        it is stopped on quit, and only if this UI started it"""
        u = urllib.parse.urlparse(url)
        if u.hostname not in ("127.0.0.1", "localhost") or self.router:
            return False
        log = os.path.join(bc.ROOT, "local", "router.log")
        self.call_from_thread(self.set_status, f"nothing answers at {u.netloc}: starting the router (log local/router.log) …")
        self.router = subprocess.Popen([sys.executable, os.path.join(bc.ROOT, "tools/cli/launch.py"), "--router",
                                        "--port", str(u.port or 8150)], cwd=bc.ROOT, start_new_session=True,
                                       stdout=open(log, "a"), stderr=subprocess.STDOUT)
        for _ in range(240):
            if self.router.poll() is not None:
                self.call_from_thread(self.set_status, f"[red]the router exited with {self.router.returncode} (local/router.log)")
                self.router = None
                return False
            try:
                bc.list_models(url)
                return True
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        return False

    def on_unmount(self):
        if self.router and self.router.poll() is None:
            os.killpg(self.router.pid, signal.SIGINT)
            try:
                self.router.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(self.router.pid, signal.SIGKILL)

    @work(thread=True, exclusive=True, group="models")
    def action_models(self):
        url = self.query_one("#url", Input).value
        try:
            ms = bc.list_models(url)
        except Exception as ex:  # noqa: BLE001
            if not self.start_router(url):
                self.call_from_thread(self.set_status, f"[red]no model list from the endpoint: {ex}")
                return
            ms = bc.list_models(url)
        opts = [(m, m) for m in ms]

        def apply():
            self.query_one("#model", Select).set_options(opts)
            self.query_one("#judge", Select).set_options(opts)
            self.set_status(f"{len(ms)} models at the endpoint" + (" (router started here, stopped on quit)" if self.router else ""))
        self.call_from_thread(apply)

    @work(thread=True, exclusive=True, group="fetch")
    def action_fetch(self):
        self.call_from_thread(self.set_status, "downloading the missing suites …")
        ds.fetch(log=lambda m: self.call_from_thread(self.set_status, m))
        have = ds.available()

        def apply():
            for n, ok in have.items():
                cb = self.query_one("#" + sid(n), Checkbox)
                cb.disabled = not ok
                if ok:
                    cb.label = str(cb.label).replace("  (missing)", "")
            self.set_status("data: " + ", ".join(f"{n} {'ok' if ok else 'missing'}" for n, ok in have.items()))
        self.call_from_thread(apply)

    # ------------------------------------------------ run
    def on_button_pressed(self, ev):
        {"start": self.action_start, "stop": self.action_stop, "fetch": self.action_fetch}[ev.button.id]()

    def action_stop(self):
        if self.busy:
            self.stop_flag.set()
            self.set_status("stopping after the current task …")

    def action_start(self, resume=None):
        if self.busy:
            return
        model = self.query_one("#model", Select).value
        if resume is None and not isinstance(model, str):
            self.set_status("[red]pick a model first (m fetches the list)")
            return
        judge = self.query_one("#judge", Select).value
        judge = judge if isinstance(judge, str) else None
        pick = [n for n in bc.SUITES if self.query_one("#" + sid(n), Checkbox).value]
        lim = self.query_one("#limit", Input).value
        url = self.query_one("#url", Input).value
        if resume:
            run = bc.Run.load(resume["path"])
            pick = list(dict.fromkeys(r.suite for r in run.results)) or pick
            judge = run.judge or judge
        else:
            run = bc.Run(model, url, time.strftime("%Y-%m-%dT%H:%M:%S"), self.query_one("#preset", Select).value, judge)
        self.busy = True
        self.stop_flag.clear()
        self.query_one("#start", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.query_one("#live", DataTable).clear()
        self.query_one(TabbedContent).active = "t-run"
        self.bench(run, pick, {"url": run.url, "model": run.model, "judge_model": judge}, int(lim) if lim else None)

    def action_delete(self):
        """the selected run (Results), or the selected KL result (Compare's KLD table focused)"""
        if self.busy:
            self.set_status("[red]stop the run first")
            return
        k = self.query_one("#kld", DataTable)
        if self.query_one(TabbedContent).active == "t-cmp" and k.has_focus and k.row_count:
            x = self.kld_rows[k.cursor_row]
            self.push_screen(Confirm(f"Delete the KL result {x['name']} vs {x['ref']}?"),
                             lambda ok: ok and self.trash([x["path"]]))
            return
        if not self.cur:
            return
        r = self.cur
        self.push_screen(Confirm(f"Delete the run {r['model']} ({r.get('preset', '?')}, {r['started']}, "
                                 f"{len(r['results'])} tasks)?"), lambda ok: ok and self.trash([r["path"]]))

    def action_delete_all(self):
        if self.busy:
            self.set_status("[red]stop the run first")
            return
        runs = [r["path"] for r in self.runs]
        kl = [x["path"] for x in kld.load_all()]
        if not runs and not kl:
            return
        self.push_screen(Confirm(f"Delete every result: {len(runs)} runs and {len(kl)} KL results?"),
                         lambda ok: ok and self.trash(runs + kl))

    def trash(self, paths):
        for p in paths:
            if os.path.exists(p):
                to_trash(p)
        self.cur = None
        self.query_one("#summary", Static).update("")
        self.query_one("#tasks", DataTable).clear()
        self.query_one("#detail-text", Static).update("")
        self.action_refresh()
        self.set_status(f"{len(paths)} moved to local/bench/trash")

    def action_continue_run(self):
        if self.cur:
            self.action_start(resume=self.cur)

    @work(thread=True, exclusive=True, group="bench")
    def bench(self, run, pick, opts, limit):
        prog = self.query_one("#prog", ProgressBar)
        live = self.query_one("#live", DataTable)

        def on_result(i, n, r):
            row = (r.suite, r.id, cell(r.score), f"{r.seconds:.1f}", str(r.tokens), r.note[:90])
            summ = run.summary()
            txt = Text()
            for k, v in summ.items():
                txt.append(f"{k:>11} ", style="bold" if k == "overall" else "")
                txt.append_text(bar(v["score"], v["se"] if k != "overall" else 0, 30))
                txt.append(f"  n={v['n']}", style="#8a93b8")
                txt.append(f"  {v['truncated']} truncated\n" if v.get("truncated") else "\n", style="#ffb454")

            def ui():
                prog.update(total=n, progress=i)
                live.add_row(*row)
                live.move_cursor(row=live.row_count - 1)
                self.query_one("#running", Static).update(txt)
                self.set_status(f"[{i}/{n}] {run.model} · {r.suite} / {r.id}")
            self.call_from_thread(ui)
        bc.run_bench(run, pick, opts, limit, on_result, stop=self.stop_flag.is_set)
        o = run.summary().get("overall", {}).get("score", 0)

        def done():
            self.busy = False
            self.set_status(("stopped" if self.stop_flag.is_set() else "done") +
                            f": overall {o * 100:.1f}%  →  {os.path.relpath(run.path or '', bc.ROOT)}  (c resumes it)")
            self.query_one("#start", Button).disabled = False
            self.query_one("#stop", Button).disabled = True
            self.action_refresh()
        self.call_from_thread(done)

    # ------------------------------------------------ results
    def action_refresh(self):
        self.runs = list(reversed(bc.load_runs()))
        t = self.query_one("#runs", DataTable)
        t.clear()
        for i, r in enumerate(self.runs):
            s, _ = sget(r["summary"], "overall")
            t.add_row(r["started"].replace("T", " ")[:16], r["model"], r.get("preset", "?"), str(len(r["results"])),
                      cell(s), key=str(i))
        self.fill_compare()
        if self.runs:
            self.show_run(self.runs[0])

    def on_data_table_row_highlighted(self, ev):
        if ev.row_key is None:
            return
        if ev.data_table.id == "runs":
            self.show_run(self.runs[int(ev.row_key.value)])
        elif ev.data_table.id == "tasks" and self.cur:
            r = self.cur["results"][int(ev.row_key.value)]
            txt = Text()
            txt.append(f"{r['suite']} / {r['id']}  ", style="bold #9d8cff")
            txt.append(f"{r['note']}\n\n", style=score_color(r["score"]))
            txt.append(r["answer"] or "(no text)")
            self.query_one("#detail-text", Static).update(txt)

    def show_run(self, run):
        self.cur = run
        s = run["summary"]
        txt = Text()
        txt.append(run["model"], style="bold #73a8ff")
        txt.append(f"   {run['started']}   {run.get('preset', '?')}   judge: {run.get('judge') or '-'}\n", style="#8a93b8")
        cur = ds.versions()
        stale = [k for k, v in (run.get("data") or {}).items() if k in s and cur.get(k) and cur[k] != v]
        if stale:
            txt.append(f"data changed since this run: {', '.join(stale)} (scores not comparable with newer runs)\n",
                       style="#ffb454")
        for k in [k for k in s if k != "overall"] + ["overall"]:
            if k in s:
                what = bc.SUITES.get(k, ("", 0, 0, ""))[3]
                txt.append(f"{k:>11} ", style="bold" if k == "overall" else "")
                txt.append_text(bar(s[k]["score"], s[k]["se"], 30))
                txt.append(f"  n={s[k]['n']:<5} {what}", style="#8a93b8")
                txt.append(f"  {s[k]['truncated']} truncated\n" if s[k].get("truncated") else "\n", style="#ffb454")
        self.query_one("#summary", Static).update(txt)
        t = self.query_one("#tasks", DataTable)
        t.clear()
        for i, r in enumerate(run["results"]):
            t.add_row(r["suite"], r["id"], str(r["category"])[:18], cell(r["score"]), f"{r['seconds']:.1f}",
                      str(r["tokens"]), r["note"][:70], key=str(i))

    def fill_compare(self):
        t = self.query_one("#cmp", DataTable)
        t.clear(columns=True)
        suites = [s for s in bc.SUITES if any(s in r["summary"] for r in self.runs)]
        t.add_columns("model", "preset", "when", *suites, "overall", "tok/s")
        best = {}
        for s in suites:  # the best score in each column, to mark it
            vals = [r["summary"][s]["score"] for r in self.runs if s in r["summary"]]
            best[s] = max(vals) if vals else None
        for r in sorted(self.runs, key=lambda r: (r["model"], r["started"])):
            res = r["results"]
            secs = sum(x["seconds"] for x in res) or 1
            cells = []
            for s in suites:
                v, se = sget(r["summary"], s)
                c = cell(v, se)
                if v is not None and v == best[s] and len(self.runs) > 1:
                    c.stylize("underline")
                cells.append(c)
            t.add_row(r["model"], r.get("preset", "?"), r["started"][5:16].replace("T", " "), *cells,
                      cell(sget(r["summary"], "overall")[0]), f"{sum(x['tokens'] for x in res) / secs:.1f}")
        k = self.query_one("#kld", DataTable)
        k.clear(columns=True)
        k.add_columns("quantization", "reference", "KLD mean", "± se", "p99", "max", "same top-1", "positions", "", "text")
        rows = sorted(kld.load_all(), key=lambda x: x["kld_mean"])
        self.kld_rows = rows
        top = max([x["kld_mean"] for x in rows] + [1e-9])
        for x in rows:
            frac = x["kld_mean"] / top
            b = Text("█" * max(1, round(frac * 20)), style=score_color(1 - frac))
            k.add_row(x["name"], x["ref"], f"{x['kld_mean']:.5f}", f"{x['kld_se']:.5f}", f"{x['kld_p99']:.4f}",
                      f"{x['kld_max']:.3f}", cell(x["same_top1"]), str(x["positions"]), b, x.get("text", "")[:30])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8150/v1")
    BenchApp(ap.parse_args().url).run()

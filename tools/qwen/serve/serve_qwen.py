#!/usr/bin/env python3
"""Almond's API server (serve/server.py) in front of Qwen3.8-Flash-Next on the engine (`almond-engine --qwen --serve`,
src/qwen).  The server's defaults are this model's own: the qwen35 tokenizer (tokenizer.ggml.pre), its template and
its tool calls (`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`).  This shim sets what the
model card (Qwen/Qwen3.8-Flash-Next) says: the thinking levels under their names (xhigh, the default, medium, low; off
= enable_thinking false; a client's "high" or "max" is xhigh) and the sampling without thinking (0.7 / 0.8 / top_k 20 /
presence 1.5; with it the configuration's 1.0 / 0.95 / top_k 20).

Conversations on disk (opt-in, "conversation_dir" in the configuration, e.g. "local/qwen-slots"): after each reply the
conversation the engine holds is written there (the engine's SAVE), and a prompt that continues one of those files
while the engine has nothing of it in RAM (after a restart, a model switch, or once it left the RAM cache) is resumed
from it (RESTORE) instead of being read again.  At most "conversation_dir_gib" (default 16) on disk, oldest out first.

  serve_qwen.py --engine almond --config local/qwen-local.json --port 8150   (./run.sh --model qwen does this)
"""
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ALMOND = Path(os.environ.get("ALMOND_HOME") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ALMOND))
sys.path.insert(0, str(ALMOND / "tools"))

from serve import server as S  # noqa: E402

# the template's own name for its top level: xhigh (a client's "high" or "max" is the same level)
S.REASONING_LEVELS = [["off", "Off", "answers right away"], ["low", "Low", "short"], ["medium", "Medium", "medium"],
                      ["xhigh", "XHigh", "thorough (the default; high and max are the same)"]]
S.REASONING_DEFAULT = "xhigh"

# The model card's sampling (Qwen/Qwen3.8-Flash-Next, "Best Practices"): thinking temperature 1.0, top_p 0.95, top_k
# 20 (the configuration's); without thinking ("instruct mode") temperature 0.7, top_p 0.8, top_k 20, presence penalty
# 1.5.  Without thinking those replace the configuration's; a request's own values and the chat's /set still win.
INSTRUCT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "repetition_penalty": 1.0}
_run = S.Service.run


def _run_qwen(self, ids, thinking, tools, max_new, sampling, cancel, force=None):
    if not thinking:
        req = {k: v for k, v in (sampling or {}).items() if v is not None}
        sampling = {**self.sampling_defaults, **INSTRUCT, **self.shared, **req}
    return _run(self, ids, thinking, tools, max_new, sampling, cancel, force)


S.Service.run = _run_qwen


class Disk:
    """The conversations kept in `folder`: <hash>.bin (the engine's session file) and <hash>.json (the prompt's ids,
    which a later prompt must start with, and when it was written)."""

    def __init__(self, folder: Path, budget: int):
        self.folder, self.budget = folder, budget
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pid, self.seen = None, []          # the engine process and the prompts it has read since it started

    def entries(self):
        out = []
        for f in self.folder.glob("*.json"):
            try:
                meta = json.loads(f.read_text())
                b = f.with_suffix(".bin")
                out.append((meta.get("at", 0), list(meta["prompt"]), b, f, b.stat().st_size))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return out

    @staticmethod
    def starts(ids, prefix):
        return len(prefix) <= len(ids) and ids[:len(prefix)] == prefix

    def engine_has(self, ids, n):
        """The engine read a prompt sharing at least n tokens with ids since it started: its RAM cache has it."""
        return any(self.starts(ids, p[:n]) for p in self.seen if len(p) >= n)

    def best(self, ids):
        """The longest saved prompt ids starts with (and is longer than nothing the engine holds)."""
        hits = [e for e in self.entries() if len(e[1]) and self.starts(ids, e[1])]
        return max(hits, key=lambda e: len(e[1]), default=None)

    def drop(self, e):
        for f in (e[2], e[3]):
            try:
                f.unlink()
            except OSError:
                pass

    def note_saved(self, prompt):
        """After a SAVE of this prompt's conversation: its earlier turns go, then the oldest past the budget."""
        es = self.entries()
        for e in es:
            if len(e[1]) < len(prompt) and self.starts(prompt, e[1]):
                self.drop(e)
        es = sorted(self.entries(), key=lambda e: e[0])
        total = sum(e[4] for e in es)
        while es and total > self.budget:
            e = es.pop(0)
            total -= e[4]
            self.drop(e)


def _engine_pid(svc):
    proc = getattr(svc.engine, "proc", None)
    return getattr(proc, "pid", None)


def _session(svc, action, path):
    """SAVE / RESTORE between requests, holding the FIFO; None when it was refused or failed (logged)."""
    with svc.fifo:
        if not svc.loaded():
            return None
        try:
            return svc.engine.session_file(action, str(path))
        except Exception as e:                          # a refusal or a lost engine: the request goes on without it
            print(f"[almond] conversation disk: {action} {path.name}: {e}", file=sys.stderr, flush=True)
            return None


_run_disk_inner = S.Service.run


def _run_disk(self, ids, thinking, tools, max_new, sampling, cancel, force=None):
    disk = getattr(self, "qwen_disk", None)
    if disk is None or getattr(self.engine, "batch", 0) or getattr(self.embeddings, "path", None):
        yield from _run_disk_inner(self, ids, thinking, tools, max_new, sampling, cancel, force)
        return
    ids = list(ids)
    with self.fifo:
        self.ensure_loaded()
    with disk.lock:
        if _engine_pid(self) != disk.pid:
            disk.pid, disk.seen = _engine_pid(self), []
        e = disk.best(ids)
        if e is not None and not disk.engine_has(ids, len(e[1])):
            t0 = time.perf_counter()
            r = _session(self, "restore", e[2])
            if r is not None:
                print(f"[almond] conversation disk: resumed {r['tokens']} tokens from {e[2].name} in "
                      f"{(time.perf_counter() - t0) * 1000:.0f} ms", file=sys.stderr, flush=True)
                disk.seen.append(e[1])
    done = False
    for item in _run_disk_inner(self, ids, thinking, tools, max_new, sampling, cancel, force):
        if item[0] == "done":
            done = True
        yield item
    with disk.lock:
        disk.seen.append(ids)
        if not done:
            return
        name = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:24]
        b = disk.folder / f"{name}.bin"
        t0 = time.perf_counter()
        r = _session(self, "save", b)
        if r is None:
            return
        (disk.folder / f"{name}.json").write_text(json.dumps({"prompt": ids, "at": time.time()}))
        disk.note_saved(ids)
        print(f"[almond] conversation disk: saved {r['tokens']} tokens, {r['bytes'] / 2**20:.0f} MiB in "
              f"{(time.perf_counter() - t0) * 1000:.0f} ms", file=sys.stderr, flush=True)


S.Service.run = _run_disk


def _config():
    a = sys.argv
    if "--config" in a and a.index("--config") + 1 < len(a):
        try:
            return json.loads(Path(a[a.index("--config") + 1]).read_text())
        except (OSError, ValueError):
            return {}
    return {}


_init = S.Service.__init__


def _init_qwen(self, *args, **kw):
    _init(self, *args, **kw)
    cfg = _config()
    d = cfg.get("conversation_dir")
    self.qwen_disk = None
    if d:
        d = Path(d) if os.path.isabs(d) else ALMOND / d
        self.qwen_disk = Disk(d, int(float(cfg.get("conversation_dir_gib", 16)) * 2**30))
        print(f"[almond] conversations on disk: {d}", file=sys.stderr, flush=True)


S.Service.__init__ = _init_qwen

if __name__ == "__main__":
    sys.exit(S.main())

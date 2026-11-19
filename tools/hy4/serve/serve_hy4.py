#!/usr/bin/env python3
"""Almond's API server (serve/server.py) in front of Hy4-flash on the engine (`almond-engine --hy4 --serve`, src/hy4).
The server is generic and the tokenizer directory carries the model's own Jinja chat template (extracted from the GGUF
by tools/almond_tokenizer.py); this shim swaps in Hy4's differences:

- tokenizer: the GGUF's BPE with llama.cpp's `hy_v4` pre-split (the same three regexes as `joyai-llm`)
- end of turn: <｜hy_end:6124c78e｜> (id 120025, the GGUF's eos)
- reasoning effort: the levels of Tencent's model card and template, no_think / high (the default) / max; a client's
  none / off / disabled is no_think, minimal / low / medium is high, xhigh is max
- tags: the model writes `<think:6124c78e>` ... `</think:6124c78e>` and its calls as
  `<tool_calls:6124c78e><tool_call:6124c78e>NAME<arg_key:6124c78e>K</arg_key:6124c78e><arg_value:6124c78e>V
  </arg_value:6124c78e></tool_call:6124c78e></tool_calls:6124c78e>`.  The stream is rewritten to the server's
  `<think>` and `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`, as serve_glm.py does
  for GLM's calls (the same key/value shape), so every heuristic of the server's parser keeps working.

  serve_hy4.py --engine almond --config local/hy4-local.json --port 8150   (./almond serve --model hy4 does this)
"""
import os
import re
import sys
from pathlib import Path

ALMOND = Path(os.environ.get("ALMOND_HOME") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ALMOND))
sys.path.insert(0, str(ALMOND / "tools"))

from serve import frontend as F  # noqa: E402
from serve import server as S  # noqa: E402

HYTK = ":6124c78e"
END = "<｜hy_end" + HYTK + "｜>"
EOS_ID = 120025

# ---- tokenizer: the server builds ST.Tokenizer(tokens, merges, types); Hy4's pre-split is hy_v4
import almond_tokenizer as ST  # noqa: E402
_Tok = ST.Tokenizer


class Hy4Tokenizer(_Tok):
    def __init__(self, tokens, merges, token_types=None, pre="hy_v4", **k):
        super().__init__(tokens, merges, token_types, pre="hy_v4", **k)


ST.Tokenizer = Hy4Tokenizer

# ---- images: the template writes <｜hy_image▁start｜><｜hy_image▁pad｜><｜hy_image▁end｜> per picture; the server's own
# expansion makes the pad one token per encoder row (build-vision-hy4/bin/almond-vision, llama.cpp PR #32517's mtmd)
S.IMAGE_PAD = "<｜hy_image▁pad" + HYTK + "｜>"
S.VISION_START = "<｜hy_image▁start" + HYTK + "｜>"

# ---- end of turn
S.IM_END = END
_svc_init = S.Service.__init__


def _svc_init_hy4(self, *a, **k):
    _svc_init(self, *a, **k)
    self.stop_ids = {EOS_ID}
    self.effort_end = False      # the effort is the template's reasoning_mode line in the first system turn


S.Service.__init__ = _svc_init_hy4

# ---- reasoning effort
# The model card (tencent/Hy4-flash) and Tencent's own chat_template.jinja know high (the default), max and no_think;
# the GGUF's template also takes "low", which the official one refuses, so low / minimal / medium are high here.
HY4_EFFORT = {"none": "no_think", "off": "no_think", "disabled": "no_think", "false": "no_think",
              "no_think": "no_think", "minimal": "high", "low": "high", "medium": "high", "high": "high",
              "xhigh": "max", "max": "max", "maximum": "max", "auto": "high", "default": "high"}


def effort_kwargs(value) -> dict:
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key not in HY4_EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, high or max (low / medium = high)")
    level = HY4_EFFORT[key]
    return {"enable_thinking": False} if level == "no_think" else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "high" if n < 8192 else "max"}


for mod in (F, S):
    if hasattr(mod, "effort_kwargs"):
        mod.effort_kwargs = effort_kwargs
    if hasattr(mod, "budget_effort"):
        mod.budget_effort = budget_effort
S.REASONING_LEVELS = [["none", "Off", "answers right away (no_think)"], ["high", "High", "thorough (the default)"],
                      ["max", "Max", "the most thorough"]]
S.REASONING_DEFAULT = "high"

_render = F.ChatTemplate.render


def _render_hy4(self, messages, tools=None, add_generation_prompt=True, **kwargs):
    off = kwargs.pop("enable_thinking", True) is False
    level = kwargs.pop("reasoning_effort", None)
    kwargs["reasoning_effort"] = "no_think" if off else level if level in ("high", "max") else "high"
    # the server's own <think> marks inside earlier turns: the template wants the reasoning in its own field
    out = _render(self, messages, tools, add_generation_prompt, **kwargs)
    return out


F.ChatTemplate.render = _render_hy4

# ---- output: Hy4's tags -> the server's
TAG_MAP = [("<think" + HYTK + ">", "<think>"), ("</think" + HYTK + ">", "</think>"),
           ("<tool_calls" + HYTK + ">", ""), ("</tool_calls" + HYTK + ">", ""),
           ("<tool_call" + HYTK + ">", "<tool_call>"), ("</tool_call" + HYTK + ">", "</tool_call>"),
           ("<arg_key" + HYTK + ">", "<arg_key>"), ("</arg_key" + HYTK + ">", "</arg_key>"),
           ("<arg_value" + HYTK + ">", "<arg_value>"), ("</arg_value" + HYTK + ">", "</arg_value>")]
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
KEY_OPEN, KEY_CLOSE, VAL_OPEN, VAL_CLOSE = "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"
_PAIR_RE = re.compile(re.escape(KEY_OPEN) + r"(.*?)" + re.escape(KEY_CLOSE) + r"\s*" + re.escape(VAL_OPEN) +
                      r"(.*?)" + re.escape(VAL_CLOSE), re.DOTALL)


def hy4_tags(buf: str, final: bool) -> tuple[str, str]:
    """(text with Hy4's tags renamed, text held back): a tail that may be the start of a tag waits for more."""
    for a, b in TAG_MAP:
        buf = buf.replace(a, b)
    if not final:
        i = buf.rfind("<")
        if i >= 0 and ">" not in buf[i:] and any(a.startswith(buf[i:]) for a, _ in TAG_MAP):
            return buf[:i], buf[i:]
    return buf, ""


def translate_call(body: str) -> str:
    k = body.find(KEY_OPEN)
    name = (body if k < 0 else body[:k]).strip()
    if not name or re.search(r"\s", name):
        return CALL_OPEN + body + CALL_CLOSE
    pairs = _PAIR_RE.findall(body[k:]) if k >= 0 else []
    params = "".join(f"<parameter={key.strip()}>\n{value}\n</parameter>\n" for key, value in pairs)
    return f"{CALL_OPEN}\n<function={name}>\n{params}</function>\n{CALL_CLOSE}"


def calls_rewrite(buf: str, final: bool) -> tuple[str, str]:
    """(text to pass on, text to hold): each complete call translated; an open one is held until it closes."""
    out = []
    while True:
        i = buf.find(CALL_OPEN)
        if i < 0:
            break
        j = buf.find(CALL_CLOSE, i + len(CALL_OPEN))
        if j < 0:
            if final:
                break
            out.append(buf[:i])
            return "".join(out), buf[i:]
        out.append(buf[:i])
        out.append(translate_call(buf[i + len(CALL_OPEN):j]))
        buf = buf[j + len(CALL_CLOSE):]
    out.append(buf)
    return "".join(out), ""


_op_init, _op_feed, _op_finish = F.OutputParser.__init__, F.OutputParser.feed, F.OutputParser.finish


def _op_init_hy4(self, *a, **k):
    _op_init(self, *a, **k)
    self.stream_tools = False          # a call reaches the parser whole
    self.hy4_tag_hold = ""
    self.hy4_call_hold = ""


def _rewrite(self, delta: str, final: bool) -> str:
    text, self.hy4_tag_hold = hy4_tags(self.hy4_tag_hold + delta, final)
    text, self.hy4_call_hold = calls_rewrite(self.hy4_call_hold + text, final)
    return text


def _op_feed_hy4(self, delta: str):
    text = _rewrite(self, delta, False)
    return _op_feed(self, text) if text else []


def _op_finish_hy4(self, reason=None):
    text = _rewrite(self, "", True)
    return (_op_feed(self, text) if text else []) + _op_finish(self, reason)


F.OutputParser.__init__ = _op_init_hy4
F.OutputParser.feed = _op_feed_hy4
F.OutputParser.finish = _op_finish_hy4
S.OutputParser = F.OutputParser

# ---- conversations on disk ("conversation_dir" in the configuration): the same as serve_qwen.py's (the engine's
# SAVE / RESTORE after each reply and for a prompt that continues a saved conversation the engine no longer holds),
# copied here because importing serve_qwen.py would also bring Qwen's sampling.
import hashlib  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

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
    disk = getattr(self, "hy4_disk", None)
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


def _init_hy4_disk(self, *args, **kw):
    _init(self, *args, **kw)
    cfg = _config()
    d = cfg.get("conversation_dir")
    self.hy4_disk = None
    if d:
        d = Path(d) if os.path.isabs(d) else ALMOND / d
        self.hy4_disk = Disk(d, int(float(cfg.get("conversation_dir_gib", 16)) * 2**30))
        print(f"[almond] conversations on disk: {d}", file=sys.stderr, flush=True)


S.Service.__init__ = _init_hy4_disk


if __name__ == "__main__":
    sys.exit(S.main())

#!/usr/bin/env python3
"""almond - Almond's terminal chat.

  ./almond                       a conversation (starts the router if nothing answers at --url, stops it on /quit)
  ./almond -p "question"         one prompt, the answer on stdout, then exit
  ./almond -m glm                the model (a name from /models, or the start of one: glm, deepseek)
  ./almond --ctx 65536           the context window of the models it starts (a running server keeps its own)

Thinking: --nothink, --think (the model's default), --think-max, --think-level LEVEL - the levels are the model's own
(GLM-5.3-Flash: low / high / max, it always thinks; DeepSeek-V4.1: off / low / high / max = 0 / 50 / 75 / 100).  In the conversation:

  /help  /quit  /models  /model NAME  /think [LEVEL]  /nothink  /clear  /read FILE  /system TEXT  /ctx
  /set [KEY VALUE]   sampling: temp, top_p, top_k, min_p, repeat_penalty, presence_penalty, seed, max_tokens
                     (/set alone shows them and the model's defaults; /set KEY default goes back to the model's)
  /stats             what the server's monitor shows: speed, GPUs, VRAM, CPU, RAM, disk, context, experts, requests
  /forget            clear the conversation cache the engine keeps
  /hide /show        the thinking
  /agent [on|off]    the agent: the model reads, writes and edits files and runs commands in the working directory;
                     a write, an edit, a command or a path outside it asks y / n / a (a: always, for that tool)

On a terminal the last line is a status bar: the prompt's reading, the answer's speed, idle the last request's numbers.

  ./almond --monitor   the same as /stats, refreshed every second (Ctrl+C leaves)

A line ending in \\ goes on to the next; \"\"\" starts and ends a block of several lines.  Ctrl+C stops the answer and
returns to the prompt; Ctrl+D (or /quit) leaves.  It speaks Almond's OpenAI API (the router's, ./run.sh --router, or
one model's server, ./run.sh): the history and the thinking level live here, the model there.
"""
import argparse
import difflib
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
TTY = sys.stdout.isatty()
DIM, BOLD, CYAN, RED, RESET = (("\x1b[2m", "\x1b[1m", "\x1b[36m", "\x1b[31m", "\x1b[0m") if TTY else ("",) * 5)
HISTORY = Path.home() / ".almond_history"


class Api:
    def __init__(self, url: str, api_key: str):
        u = urlsplit(url)
        self.host, self.port, self.key = u.hostname or "127.0.0.1", u.port or 80, api_key

    def _conn(self, timeout=3600):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def get(self, path: str, timeout=10) -> dict:
        c = self._conn(timeout)
        c.request("GET", path, headers=self._headers())
        r = c.getresponse()
        body = r.read()
        if r.status != 200:
            raise RuntimeError(f"GET {path}: {r.status} {body[:200]!r}")
        return json.loads(body)

    def post(self, path: str, obj: dict, timeout=3600) -> dict:
        c = self._conn(timeout)
        c.request("POST", path, body=json.dumps(obj), headers=self._headers())
        r = c.getresponse()
        body = r.read()
        out = json.loads(body or b"{}")
        if r.status != 200:
            raise RuntimeError((out.get("error") or {}).get("message") if isinstance(out.get("error"), dict)
                               else out.get("error") or f"{r.status}")
        return out

    def up(self) -> bool:
        try:
            self.get("/health", timeout=2)
            return True
        except (OSError, RuntimeError, ValueError, http.client.HTTPException):
            return False

    def stream(self, body: dict):
        """Yields ("reasoning" | "content" | "tool" | "end", payload) as the answer streams."""
        c = self._conn()
        c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=self._headers())
        r = c.getresponse()
        if r.status != 200:
            data = r.read()
            try:
                msg = json.loads(data)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                msg = data[:300].decode(errors="replace")
            raise RuntimeError(msg)
        self.live = c
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    yield "reasoning", d["reasoning_content"]
                if d.get("content"):
                    yield "content", d["content"]
                for tc in d.get("tool_calls") or []:
                    yield "tool", tc
                if ch.get("finish_reason"):
                    yield "end", {"finish": ch["finish_reason"], "usage": ev.get("usage"), "timings": ev.get("timings")}


class Bar:
    """The status line at the bottom of the terminal: a scroll region (DECSTBM) over every
    row but the last, so the conversation scrolls above it.  While a request runs a thread reads /metrics' live state
    four times a second (the prompt's reading: a bar, read / total, tok/s; the answer: tokens, tok/s, the time,
    thinking or answer); idle, the last request's numbers, the model, the thinking level and the context.  Only on a
    terminal: -p into a pipe never sees it."""

    def __init__(self, api: Api):
        self.api, self.on, self.text = api, False, ""
        self.lock = threading.RLock()
        self.busy = None                        # threading.Event while a request runs
        self.phase = "thinking"                 # what the stream shows: thinking / answer

    def rows(self):
        s = os.get_terminal_size(sys.stdout.fileno())
        return s.lines, s.columns

    def start(self):
        if not TTY:
            return
        self.on = True
        sys.stdout.write("\n\x1b[A")        # a free row under the cursor, so the last row is not the one it is on
        signal.signal(signal.SIGWINCH, lambda *_: self.region())
        self.region()

    def region(self):
        """The scroll region over rows 1 .. H-1 (again after a resize), then the line."""
        with self.lock:
            h, _ = self.rows()
            sys.stdout.write(f"\x1b7\x1b[1;{max(1, h - 1)}r\x1b8")
            self.draw()

    def stop(self):
        if not self.on:
            return
        with self.lock:
            self.on = False
            h, _ = self.rows()
            sys.stdout.write(f"\x1b7\x1b[r\x1b[{h};1H\x1b[2K\x1b8")
            sys.stdout.flush()
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)

    def draw(self, text: str | None = None):
        if not self.on:
            return
        with self.lock:
            if text is not None:
                self.text = text
            h, w = self.rows()
            line = self.text[:w - 1]
            sys.stdout.write(f"\x1b7\x1b[{h};1H\x1b[2K{DIM}{line}{RESET}\x1b8")
            sys.stdout.flush()

    def write(self, out, text: str):
        """The stream's text, under the same lock as the line, so neither cuts into the other."""
        with self.lock:
            out.write(text)
            out.flush()

    # ---- while a request runs
    def run(self, model: str):
        if not self.on:
            return
        self.busy, self.phase, t0 = threading.Event(), "thinking", time.time()

        def poll(busy):
            while not busy.is_set():
                try:
                    live = self.api.get("/metrics", timeout=2).get("live") or {}
                except (OSError, RuntimeError, ValueError, http.client.HTTPException):
                    live = {}
                if not busy.is_set():
                    self.draw(live_text(live, model, self.phase, time.time() - t0, self.rows()[1]))
                busy.wait(0.25)
        threading.Thread(target=poll, args=(self.busy,), daemon=True).start()

    def done(self):
        if self.busy is not None:
            self.busy.set()
            self.busy = None

    def idle(self, chat):
        """The last request's numbers (/metrics' requests[0]), the model, the thinking level, the context."""
        if not self.on:
            return
        try:
            last = (self.api.get("/metrics", timeout=2).get("requests") or [None])[0]
        except (OSError, RuntimeError, ValueError, http.client.HTTPException):
            last = None
        parts = [f"context {_f(chat.ctx_used)} / {_f(chat.model.get('max_context'))}", f"thinking {chat.level_text()}",
                 chat.model["id"]]
        if last and chat.messages:
            parts.insert(0, f"last: prompt {_f(last.get('prompt_tokens'))}, reused {_f(last.get('reused'))}, "
                         f"output {_f(last.get('output_tokens'))} at {_f(last.get('decode_tok_s'), 1)} tok/s")
        self.draw("  ".join(parts))


def live_text(live: dict, model: str, phase: str, secs: float, width: int) -> str:
    """The bottom line while a request runs, from /metrics' live state."""
    state = live.get("state")
    if state == "reading":
        read, total = live.get("prompt_read") or 0, live.get("prompt_total") or 0
        tail = (f" {_f(read)} / {_f(total)} tokens  {_f(live.get('prefill_tok_s_mean'), 0)} tok/s  {secs:.0f} s")
        head = f"{model}  reading the prompt "
        n = max(10, min(40, width - len(head) - len(tail) - 3))
        k = int(n * read / total) if total else 0
        return head + "[" + "#" * k + "." * (n - k) + "]" + tail
    if state == "generating":
        return (f"{model}  {phase}: {_f(live.get('generated'))} tokens  {_f(live.get('tok_s'), 1)} tok/s  "
                f"{secs:.1f} s" + (f"  ({live['queued']} queued)" if live.get("queued") else ""))
    return f"{model}  waiting for the engine  {secs:.0f} s" + (f"  ({live['queued']} queued)" if live.get("queued") else "")


AGENT_STEPS = 50
TOOL_OUT = 8000                                 # the most of a tool's output the model sees, in characters


def _tool(name, doc, **params):
    req = [k for k, v in params.items() if not v.pop("optional", False)]
    return {"type": "function", "function": {"name": name, "description": doc, "parameters": {
        "type": "object", "properties": params, "required": req}}}


TOOLS = [
    _tool("read_file", "Read a text file; lines numbered from 1.", path={"type": "string"},
          offset={"type": "integer", "description": "the first line (1)", "optional": True},
          limit={"type": "integer", "description": "how many lines (all)", "optional": True}),
    _tool("write_file", "Write a file (created, or replaced whole).", path={"type": "string"},
          content={"type": "string"}),
    _tool("edit_file", "Replace the one occurrence of `old` in a file with `new` (old must occur exactly once).",
          path={"type": "string"}, old={"type": "string"}, new={"type": "string"}),
    _tool("list_dir", "List a directory (subdirectories end in /).", path={"type": "string"}),
    _tool("run_command", "Run a bash command in the working directory; its output and exit code.",
          command={"type": "string"},
          timeout={"type": "integer", "description": "seconds (120)", "optional": True}),
]


class Agent:
    """The tools the model may call, and the asking: reads and listings inside the working directory go ahead;
    a write, an edit, a command or any path outside it asks y (yes) / n (no) / a (always, for that tool, until the
    end of the session); --yes asks only for a path outside the working directory, which always asks."""

    def __init__(self, yes: bool = False):
        self.root, self.yes, self.always = Path.cwd().resolve(), yes, set()

    def path(self, p: str) -> Path:
        return (self.root / os.path.expanduser(p)).resolve()

    def inside(self, p: Path) -> bool:
        return p == self.root or self.root in p.parents

    def allow(self, name: str, what: str, outside: bool = False) -> bool:
        if not outside and (self.yes or name in self.always):
            return True
        while True:
            try:
                a = input(f"{BOLD}{what}{RESET} - allow? [y/n/a] " if TTY else f"{what} - allow? [y/n/a] ").strip().lower()
            except EOFError:
                return False
            if a in ("y", "yes", "s", "si"):
                return True
            if a in ("n", "no", ""):
                return False
            if a in ("a", "always") and not outside:
                self.always.add(name)
                return True

    def run(self, call: dict) -> str:
        """One call: its line on the terminal, then the result the model gets."""
        name = call["name"]
        try:
            args = json.loads(call["arguments"] or "{}")
            if not isinstance(args, dict):
                raise ValueError("not an object")
        except ValueError as e:
            say(f"{CYAN}* {name}{RESET} {RED}bad arguments: {e}{RESET}")
            return f"error: the arguments are not a JSON object ({e})"
        fn = getattr(self, "t_" + name, None)
        brief = args.get("command") or args.get("path") or ""
        say(f"{CYAN}* {name}{RESET} {brief.splitlines()[0] if brief else ''}")
        if fn is None:
            return f"error: no tool {name}"
        try:
            text = fn(**args)
        except TypeError as e:
            text = f"error: {e}"
        except OSError as e:
            text = f"error: {e}"
        if len(text) > TOOL_OUT:
            text = text[:TOOL_OUT // 2] + f"\n[... {len(text) - TOOL_OUT} characters cut ...]\n" + text[-TOOL_OUT // 2:]
        first = text.splitlines()[0] if text else ""
        say(f"{DIM}  {first[:120]}{' ...' if len(text) > len(first) else ''}{RESET}")
        return text

    def t_read_file(self, path, offset=1, limit=None):
        p = self.path(path)
        if not self.inside(p) and not self.allow("read_file", f"read {p} (outside {self.root})", True):
            return "denied by the user"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(offset or 1))
        part = lines[start - 1:start - 1 + int(limit)] if limit else lines[start - 1:]
        return "\n".join(f"{start + i:6}\t{x}" for i, x in enumerate(part)) or "(empty)"

    def t_list_dir(self, path="."):
        p = self.path(path)
        if not self.inside(p) and not self.allow("list_dir", f"list {p} (outside {self.root})", True):
            return "denied by the user"
        return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())) or "(empty)"

    def t_write_file(self, path, content):
        p = self.path(path)
        old = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
        say(diff_text(old or "", content, str(p)) if old is not None else
            f"{DIM}  a new file, {len(content.splitlines())} lines{RESET}")
        where = "" if self.inside(p) else f" (outside {self.root})"
        if not self.allow("write_file", f"write {p}{where}", bool(where)):
            return "denied by the user"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {p} ({len(content)} characters)"

    def t_edit_file(self, path, old, new):
        p = self.path(path)
        text = p.read_text(encoding="utf-8")
        n = text.count(old)
        if n != 1:
            return f"error: `old` occurs {n} times in {p}; it must occur exactly once"
        after = text.replace(old, new)
        say(diff_text(text, after, str(p)))
        where = "" if self.inside(p) else f" (outside {self.root})"
        if not self.allow("edit_file", f"edit {p}{where}", bool(where)):
            return "denied by the user"
        p.write_text(after, encoding="utf-8")
        return f"edited {p}"

    def t_run_command(self, command, timeout=120):
        say(f"{BOLD}  $ {command}{RESET}")
        if not self.allow("run_command", "run it"):
            return "denied by the user"
        try:
            r = subprocess.run(["bash", "-c", command], cwd=self.root, capture_output=True, text=True,
                               timeout=float(timeout or 120), errors="replace")
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") + (e.stderr or "")
            return (out if isinstance(out, str) else out.decode(errors="replace")) + f"\n[timed out after {timeout} s]"
        return (r.stdout + r.stderr).rstrip() + f"\n[exit code {r.returncode}]"


def diff_text(old: str, new: str, name: str) -> str:
    """A short unified diff, coloured on a terminal, at most 40 lines."""
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), name, name, n=2, lineterm=""))[2:]
    more = len(lines) - 40
    out = []
    for x in lines[:40]:
        c = RED if x.startswith("-") else CYAN if x.startswith("+") else DIM
        out.append(f"  {c}{x}{RESET}")
    return "\n".join(out + ([f"  {DIM}... {more} more lines{RESET}"] if more > 0 else []))


class Chat:
    def __init__(self, api: Api, a):
        self.api, self.a = api, a
        self.messages, self.system = [], a.system
        self.show = not a.hide_thinking
        self.sampling = {k: v for k, v in (("temperature", a.temp), ("top_p", a.top_p), ("top_k", a.top_k),
                                           ("min_p", a.min_p), ("repetition_penalty", a.repeat_penalty),
                                           ("presence_penalty", a.presence_penalty), ("seed", a.seed),
                                           ("max_tokens", a.max_tokens)) if v is not None}
        self.models = []                        # the router's listing (one entry for a single model's server)
        self.model = None                       # the listing entry in use
        self.level = None                       # None: the model's default
        self.ctx_used = 0
        self.bar = Bar(api)
        self.agent = Agent(a.yes) if a.agent else None
        self.yes, self.stopped = a.yes, False

    # ---- models and levels
    def refresh(self):
        data = self.api.get("/v1/models").get("data") or []
        if data and "reasoning_levels" not in data[0]:  # one model's server (not the router): its levels from /health
            h = self.api.get("/health")
            data = [dict(data[0], status={"value": "loaded"}, reasoning_levels=h.get("reasoning_levels"),
                         reasoning_default=h.get("reasoning_default"), max_context=h.get("max_context"))]
        self.models = data

    def find(self, name: str):
        n = name.lower()
        alias = {"ds41": "deepseek", "ds4": "deepseek"}.get(n, n)
        for m in self.models:
            if m["id"].lower() == n:
                return m
        hits = [m for m in self.models if m["id"].lower().startswith(alias) or alias in m["id"].lower()]
        return hits[0] if len(hits) == 1 else None

    def pick(self, name: str | None):
        self.refresh()
        if name:
            m = self.find(name)
            if m is None:
                raise SystemExit(f"almond: no model {name!r}; there are: " + ", ".join(x["id"] for x in self.models))
        else:
            m = next((x for x in self.models if (x.get("status") or {}).get("value") == "loaded"), None) or \
                next((x for x in self.models if x.get("default")), None) or (self.models or [None])[0]
        if m is None:
            raise SystemExit("almond: the server lists no models")
        self.model = m
        if (m.get("status") or {}).get("value") != "loaded":
            say(f"{DIM}loading {m['id']} (a minute or two the first time) ...{RESET}")
            t0 = time.time()
            self.api.post("/models/load", {"model": m["id"]})
            say(f"{DIM}{m['id']} loaded in {time.time() - t0:.0f} s{RESET}")
            self.refresh()
            self.model = self.find(m["id"]) or m
        if self.level is not None and self.level not in self.values():
            say(f"{DIM}{m['id']} has no thinking level {self.level}: its default{RESET}")
            self.level = None

    def values(self) -> list:
        return [v for v, *_ in self.model.get("reasoning_levels") or []]

    def level_text(self) -> str:
        levels = self.model.get("reasoning_levels") or []
        v = self.level or self.model.get("reasoning_default") or (levels[-1][0] if levels else "?")
        label = next((x[1] for x in levels if x[0] == v), v)
        return label + ("" if self.level else " (default)")

    def set_level(self, word: str) -> bool:
        vals = self.values()
        w = {"0": "off", "none": "off", "nothink": "off"}.get(word.lower(), word.lower())
        if w == "off" and "off" not in vals and vals:      # a model that always thinks (GLM-5.3): its lowest level
            say(f"{DIM}{self.model['id']} always thinks: {vals[0]}, its lowest level{RESET}")
            w = vals[0]
        if w.isdigit() and vals == ["off", "low", "high", "max"]:   # 1-100: the nearest named level
            n = int(w)
            w = "off" if n == 0 else "low" if n <= 50 else "high" if n <= 75 else "max"
        if w not in vals:
            say(f"{RED}{self.model['id']}'s thinking levels: {', '.join(vals)}{RESET}")
            return False
        self.level = w
        return True

    # ---- one turn
    def ask(self, text: str, out=None) -> bool:
        out = out or sys.stdout
        self.messages.append({"role": "user", "content": text})
        for step in range(AGENT_STEPS if self.agent else 1):
            calls = self.reply(out)
            if calls is None:
                if step == 0:
                    self.messages.pop()
                return False
            if not calls or self.stopped:
                return True
            try:
                for c in calls:
                    self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": self.agent.run(c)})
            except KeyboardInterrupt:
                say(f"{RESET}\n{DIM}[stopped]{RESET}")
                for c in calls:                 # every call answered, so the history stays one the model accepts
                    if not any(m.get("tool_call_id") == c["id"] for m in self.messages):
                        self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": "stopped by the user"})
                return True
        say(f"{DIM}[the agent stopped after {AGENT_STEPS} steps]{RESET}")
        return True

    def reply(self, out):
        """One answer streamed; its tool calls ([] for none), None on an error."""
        body = {"model": self.model["id"], "stream": True, "messages": self.wire()}
        if self.agent:
            body["tools"] = TOOLS
        if self.level:
            body["reasoning_effort"] = self.level
        body.update(self.sampling)
        reasoning, content, state, end, calls = [], [], None, None, {}
        self.stopped = False
        # the thinking goes with the answer on a terminal; otherwise (-p into a pipe or a file) to stderr, so stdout
        # holds the answer alone
        think = sys.stdout if TTY else sys.stderr
        t0 = time.time()
        first = None
        bar = self.bar
        bar.run(self.model["id"])
        try:
            for kind, v in self.api.stream(body):
                if kind in ("reasoning", "content") and first is None:
                    first = time.time()
                if kind == "reasoning":
                    reasoning.append(v)
                    if self.show:
                        bar.write(think, (DIM if state != "reasoning" else "") + v)
                    state = "reasoning"
                elif kind == "content":
                    if state == "reasoning" and self.show:
                        bar.write(think, f"{RESET}\n\n" if TTY else "\n")
                    state = "content"
                    bar.phase = "answer"
                    content.append(v)
                    bar.write(out, v)
                elif kind == "tool":            # the pieces of each call, by index
                    c = calls.setdefault(v.get("index", len(calls)), {"id": None, "name": "", "arguments": ""})
                    fn = v.get("function") or {}
                    c["id"] = v.get("id") or c["id"]
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
                elif kind == "end":
                    end = v
        except KeyboardInterrupt:
            bar.done()
            try:
                self.api.live.close()           # the server sees the client go and stops the answer
            except Exception:
                pass
            if TTY:
                out.write(f"{RESET}\n{DIM}[stopped]{RESET}\n")
            end, self.stopped, calls = {"finish": "stopped"}, True, {}
        except (RuntimeError, OSError, http.client.HTTPException) as e:
            bar.done()
            say(f"{RESET}{RED}error: {e}{RESET}")
            return None
        bar.done()
        if state == "reasoning" and self.show:
            think.write(RESET if TTY else "\n")
        out.write("\n")
        msg = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        calls = [dict(c, id=c["id"] or f"call_{i}") for i, c in sorted(calls.items())]
        if calls:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
            if not self.agent:
                for c in calls:
                    say(f"{CYAN}[tool call {c['name']} {c['arguments']}]{RESET}")
                calls = []
        self.messages.append(msg)
        if end and end.get("usage"):
            u = end["usage"]
            self.ctx_used = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        if TTY and end:
            t = end.get("timings") or {}
            n = (end.get("usage") or {}).get("completion_tokens")
            speed = t.get("predicted_per_second") or (n / (time.time() - first) if n and first else None)
            parts = [f"{n} tokens" if n else None, f"{speed:.1f} tok/s" if speed else None,
                     f"prompt {t['prompt_per_second']:.0f} tok/s" if t.get("prompt_n", 0) > 1 else None,
                     f"{time.time() - t0:.1f} s", end.get("finish") if end.get("finish") not in ("stop", None) else None]
            say(f"{DIM}[{', '.join(p for p in parts if p)}]{RESET}")
        return calls

    def wire(self) -> list:
        sys_msg = [{"role": "system", "content": self.system}] if self.system else []
        return sys_msg + [{k: v for k, v in m.items() if k != "reasoning_content"} for m in self.messages]

    # ---- sampling
    KEYS = {"temp": "temperature", "temperature": "temperature", "top_p": "top_p", "top-p": "top_p", "top_k": "top_k",
            "top-k": "top_k", "min_p": "min_p", "min-p": "min_p", "repeat_penalty": "repetition_penalty",
            "repetition_penalty": "repetition_penalty", "presence_penalty": "presence_penalty", "seed": "seed",
            "max_tokens": "max_tokens", "max": "max_tokens"}
    INTS = ("top_k", "seed", "max_tokens")

    def defaults(self) -> dict:
        try:
            q = "/props?model=" + self.model["id"]
            return (self.api.get(q).get("default_generation_settings") or {}).get("params") or {}
        except (OSError, RuntimeError, ValueError, http.client.HTTPException):
            return {}

    def set(self, arg: str):
        if not arg:
            d = self.defaults()
            names = {"repetition_penalty": "repeat_penalty"}
            for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "seed",
                      "max_tokens"):
                mine = self.sampling.get(k)
                model = d.get(names.get(k, k), d.get(k)) if k != "max_tokens" else d.get("n_predict")
                say(f"  {k:<20} {mine if mine is not None else '-':<10} {DIM}model default {model}{RESET}")
            return
        key, _, val = arg.partition(" ")
        k = self.KEYS.get(key.lower())
        if k is None:
            say(f"{RED}/set takes: {', '.join(sorted(set(self.KEYS) - {'temperature', 'max', 'top-p', 'top-k', 'min-p', 'repetition_penalty'}))}{RESET}")
            return
        val = val.strip()
        if val in ("", "default", "-"):
            self.sampling.pop(k, None)
            say(f"{DIM}{k}: the model's default{RESET}")
            return
        try:
            self.sampling[k] = int(val) if k in self.INTS else float(val)
        except ValueError:
            say(f"{RED}{k} takes a number{RESET}")
            return
        say(f"{DIM}{k} = {self.sampling[k]}{RESET}")

    # ---- the conversation
    def banner(self):
        m = self.model
        say(f"{BOLD}almond{RESET} - {m['id']}, context {m.get('max_context') or '?'}, thinking {self.level_text()}"
            + (f", agent in {os.getcwd()}" if self.agent else "") + ". /help for the commands.")

    def command(self, line: str) -> bool:
        """A slash command; False to leave."""
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd == "help":
            say(__doc__.split("In the conversation:")[1].split("It speaks")[0].strip("\n:"))
        elif cmd == "models":
            self.refresh()
            for x in self.models:
                st = (x.get("status") or {}).get("value", "?")
                lv = "/".join(v for v, *_ in x.get("reasoning_levels") or [])
                mark = "*" if self.model and x["id"] == self.model["id"] else " "
                say(f" {mark} {x['id']:<40} {st:<9} context {x.get('max_context') or '?':<7} thinking {lv}")
        elif cmd == "model":
            if not arg:
                say(f"{self.model['id']} (thinking {self.level_text()})")
            else:
                try:
                    self.pick(arg)
                except SystemExit as e:
                    say(f"{RED}{e}{RESET}")
                    return True
                except RuntimeError as e:
                    say(f"{RED}could not load it: {e}{RESET}")
                    return True
                self.banner()
        elif cmd == "nothink":
            if self.set_level("off"):
                say(f"{DIM}thinking off{RESET}")
        elif cmd == "think":
            if not arg:
                self.level = None
                say(f"{DIM}thinking {self.level_text()}{RESET}")
            elif self.set_level(arg):
                say(f"{DIM}thinking {self.level_text()}{RESET}")
        elif cmd == "clear":
            self.messages, self.ctx_used = [], 0
            say(f"{DIM}a new conversation{RESET}")
        elif cmd == "system":
            self.system = arg or None
            say(f"{DIM}system prompt {'set' if arg else 'cleared'}{RESET}")
        elif cmd == "set":
            self.set(arg)
        elif cmd == "stats":
            try:
                say(monitor_text(self.api.get("/metrics"), self.model))
            except (OSError, RuntimeError, ValueError) as e:
                say(f"{RED}{e}{RESET}")
        elif cmd == "forget":
            try:
                r = self.api.post("/conversation-cache/clear", {})
                say(f"{DIM}conversation cache cleared: {r.get('pool', 0)} in RAM, {r.get('disk', 0)} on disk{RESET}")
            except (OSError, RuntimeError) as e:
                say(f"{RED}{e}{RESET}")
        elif cmd == "read":
            try:
                text = Path(os.path.expanduser(arg)).read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                say(f"{RED}{e}{RESET}")
                return True
            self.pending = f"{arg}:\n```\n{text}\n```\n"
            say(f"{DIM}{arg}: {len(text)} characters, sent with your next message{RESET}")
        elif cmd == "ctx":
            say(f"{DIM}{self.ctx_used} of {self.model.get('max_context') or '?'} tokens in use, "
                f"{len(self.messages)} messages{RESET}")
        elif cmd == "agent":
            if arg in ("on", "off"):
                self.agent = Agent(self.yes) if arg == "on" else None
            say(f"{DIM}agent {'on' if self.agent else 'off'}"
                + (f": tools read_file, write_file, edit_file, list_dir, run_command in {os.getcwd()}" if self.agent else "")
                + RESET)
        elif cmd in ("hide", "show"):
            self.show = cmd == "show"
            say(f"{DIM}thinking {'shown' if self.show else 'hidden'}{RESET}")
        else:
            say(f"{RED}unknown command /{cmd} (/help){RESET}")
        return True

    def loop(self):
        try:
            import readline
            try:
                readline.read_history_file(HISTORY)
            except OSError:
                pass
            readline.set_history_length(2000)
        except ImportError:
            readline = None
        self.pending = ""
        self.banner()
        self.bar.start()
        try:
            self.talk()
        finally:
            self.bar.stop()
        if readline:
            try:
                readline.write_history_file(HISTORY)
            except OSError:
                pass

    def talk(self):
        while True:
            self.bar.idle(self)
            try:
                line = read_input(f"{BOLD}>{RESET} " if TTY else "> ")
            except EOFError:
                say("")
                break
            except KeyboardInterrupt:
                say("")
                continue
            if not line.strip():
                continue
            if line.startswith("/"):
                if not self.command(line.strip()):
                    break
                continue
            text, self.pending = self.pending + line, ""
            self.ask(text)


def _f(x, nd=0):
    return "-" if x is None else f"{x:,.{nd}f}"


def monitor_text(m: dict, model: dict | None = None) -> str:
    """GET /metrics (what the web app's Monitor drew) as text."""
    live, hw, st, eng = m.get("live") or {}, m.get("hardware") or {}, m.get("hardware_static") or {}, m.get("engine") or {}
    reqs, last = m.get("requests") or [], (m.get("requests") or [None])[0]
    gb = lambda b: "-" if b is None else f"{b / 1e9:.1f}"
    lines = []
    state = live.get("state", "idle")
    if state == "reading":
        what = f"reading the prompt {_f(live.get('prompt_read'))} / {_f(live.get('prompt_total'))} tokens"
    elif state == "generating":
        what = f"generating: {_f(live.get('generated'))} tokens at {_f(live.get('tok_s'), 1)} tok/s"
    else:
        what = "idle" + (f" - last: {_f(last.get('output_tokens'))} tokens at {_f(last.get('decode_tok_s'), 1)} tok/s"
                         if last else "")
    q = f", {live['queued']} queued" if live.get("queued") else ""
    lines.append(f"{BOLD}{eng.get('model') or (model or {}).get('id', '?')}{RESET}  {what}{q}")
    ctx = eng.get("max_context") or 0
    used = ((live.get("prompt_tokens") or 0) + (live.get("generated") or 0)) if state != "idle" else \
        (((last or {}).get("prompt_tokens") or 0) + ((last or {}).get("output_tokens") or 0))
    lines.append(f"context    {_f(used)} / {_f(ctx)} tokens ({100 * used / ctx:.0f}%)" if ctx else "context    -")
    lines.append(f"experts    {_f(eng.get('expert_slots'))} in VRAM ({(eng.get('expert_cache_mib') or 0) / 1024:.1f} GiB)"
                 + (f", MTP drafts up to {max(0, (eng.get('mtp_max') or eng.get('spec')) - 1)}" if eng.get("spec") else ""))
    gpus = hw.get("gpus") or []
    if gpus:
        for g in gpus:
            lines.append(f"GPU {g.get('index')}      {_f(g.get('util'))}%  VRAM {gb(g.get('mem_used'))} GB  "
                         f"{_f(g.get('temp'))} °C" + (f"  {_f(g.get('power'))} W" if g.get("power") is not None else ""))
    lines.append(f"GPUs       {_f(hw.get('gpu_util'))}%  VRAM {gb(hw.get('gpu_mem_used'))} / {gb(hw.get('gpu_mem_total'))} GB  "
                 f"{_f(hw.get('gpu_temp'))} °C  {_f(hw.get('gpu_power'))} W"
                 + (f" of {_f(hw.get('gpu_power_limit'))}" if hw.get("gpu_power_limit") else "")
                 + (f"  PCIe Gen{hw.get('gpu_pcie_gen_max') or hw.get('gpu_pcie_gen')} x{hw.get('gpu_pcie_width')}"
                    f" {_f(hw.get('gpu_pcie_rx_mb'))} MB/s to the GPU" if hw.get("gpu_pcie_width") else ""))
    lines.append(f"CPU        {_f(hw.get('cpu'))}%" + (f" of {st.get('threads')} threads" if st.get("threads") else "")
                 + f"   RAM {gb(hw.get('ram_used'))} / {gb(hw.get('ram_total'))} GB"
                 + f"   disk read {_f(hw.get('disk_read_mb'))} MB/s, write {_f(hw.get('disk_write_mb'), 1)} MB/s")
    cc = m.get("conversation_cache") or {}
    if cc.get("requests"):
        lines.append(f"conv cache {_f(cc.get('requests_reused'))} of {_f(cc.get('requests'))} requests reused part of "
                     f"their prompt ({_f(cc.get('reused_tokens'))} tokens)"
                     + (f"; {_f(cc.get('parked'))} / {_f(cc.get('slots'))} parked" if cc.get("enabled") else ""))
    if reqs:
        lines.append(f"{DIM}{'time':<9}{'end':<8}{'prompt':>8}{'reused':>8}{'output':>8}{'tok/s':>7}{'VRAM hit':>9}{'secs':>7}{RESET}")
        for r in reqs[:8]:
            t = time.strftime("%H:%M:%S", time.localtime(r.get("time") or 0))
            hit = "-" if r.get("hit_rate") is None else f"{100 * r['hit_rate']:.0f}%"
            lines.append(f"{t:<9}{str(r.get('finish') or '-'):<8}{_f(r.get('prompt_tokens')):>8}{_f(r.get('reused')):>8}"
                         f"{_f(r.get('output_tokens')):>8}{_f(r.get('decode_tok_s'), 1):>7}{hit:>9}{_f(r.get('duration_s'), 1):>7}")
    t = m.get("totals") or {}
    if t.get("requests"):
        read = (t.get("prompt_tokens") or 0) - (t.get("reused") or 0)
        lines.append(f"{DIM}since the start: {_f(t['requests'])} requests, {_f(read)} prompt tokens read, "
                     f"{_f(t.get('output_tokens'))} written"
                     + (f" at {t['output_tokens'] / (t['decode_ms'] / 1000):.1f} tok/s" if t.get("decode_ms") else "") + RESET)
    return "\n".join(lines)


def session_dirs() -> list[tuple[str, Path]]:
    """Each model's folder of conversations kept on disk, from its configuration: DeepSeek's --conversation-disk,
    GLM's ALMOND_GLM_SLOT_DIR."""
    out = []
    for name, cfg, get in (("DeepSeek", "ds41-local.json",
                            lambda c: (c.get("args") or [])[c["args"].index("--conversation-disk") + 1]
                            if "--conversation-disk" in (c.get("args") or []) else None),
                           ("GLM", "glm-local.json", lambda c: (c.get("env") or {}).get("ALMOND_GLM_SLOT_DIR"))):
        try:
            d = get(json.loads((ROOT / "local" / cfg).read_text()))
        except (OSError, ValueError):
            d = None
        if d:
            out.append((name, (ROOT / d) if not os.path.isabs(d) else Path(d)))
    return out


def sessions_text() -> str:
    """The conversations kept on disk: per model, the files and their size."""
    lines = []
    for name, d in session_dirs():
        files = [f for f in d.rglob("*") if f.is_file()] if d.is_dir() else []
        n = sum(1 for f in files if f.suffix in (".meta", ".rec", ".json")) or len(files)
        size = sum(f.stat().st_size for f in files)
        lines.append(f"sessions   {name:<9} {n} on disk, {size / 1e9:.2f} GB  ({d})")
    return "\n".join(lines)


def clear_sessions() -> int:
    """Move every model's kept conversations to the trash (gio trash); only with no engine running, which holds them."""
    busy = subprocess.run(["pgrep", "-f", "almond-engine"], capture_output=True).returncode == 0
    if busy:
        raise SystemExit("almond: an engine is running and holds its sessions: ./stop.sh first")
    for name, d in session_dirs():
        files = list(d.iterdir()) if d.is_dir() else []
        if not files:
            print(f"{name}: nothing on disk")
            continue
        subprocess.run(["gio", "trash", *map(str, files)], check=True)
        print(f"{name}: {len(files)} file(s) moved to the trash ({d})")
    return 0


def idle_text(api: Api) -> str:
    """The router with no model loaded: its models, nothing else to show."""
    data = api.get("/v1/models").get("data") or []
    lines = [f"{BOLD}no model loaded{RESET} - the first request (or ./almond -m NAME) loads one"]
    for x in data:
        st = (x.get("status") or {}).get("value", "?")
        lines.append(f"  {x['id']:<40} {st:<9} context {x.get('max_context') or '?'}")
    return "\n".join(lines)


def run_monitor(api: Api):
    try:
        while True:
            try:
                h = api.get("/health")
                text = idle_text(api) if h.get("service") == "almond-router" and not h.get("loaded") else \
                    monitor_text(api.get("/metrics"))
                text += "\n" + sessions_text()
            except (OSError, RuntimeError, ValueError, http.client.HTTPException) as e:
                text = f"{RED}{e}{RESET}"
            sys.stdout.write("\x1b[H\x1b[2J" + text + f"\n{DIM}(Ctrl+C leaves){RESET}\n" if TTY else text + "\n")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print()
    except BrokenPipeError:             # (piped into head, say)
        pass


def read_input(prompt: str) -> str:
    """One message: a line, lines joined by a trailing \\, or a \"\"\" block."""
    line = input(prompt)
    if line.strip() == '"""':
        lines = []
        while True:
            x = input("")
            if x.strip() == '"""':
                return "\n".join(lines)
            lines.append(x)
    lines = [line]
    while lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(input(". "))
    return "\n".join(lines)


def say(text: str):
    print(text, file=sys.stderr if not TTY else sys.stdout, flush=True)


def start_router(api: Api, port: int, ctx: int | None = None, session_log: str | None = None) -> subprocess.Popen:
    log = ROOT / "local" / "router.log"
    say(f"{DIM}nothing answers at {api.host}:{api.port}: starting the router (./run.sh --router, log {log}) ...{RESET}")
    env = dict(os.environ)                               # (run.sh / the server: every model the router starts)
    if ctx:
        env["ALMOND_CTX"] = str(ctx)
    if session_log:
        env["ALMOND_SESSION_LOG"] = session_log
    p = subprocess.Popen([str(ROOT / "run.sh"), "--router", "--port", str(port)], cwd=ROOT, start_new_session=True,
                         stdout=open(log, "a"), stderr=subprocess.STDOUT, env=env)
    for _ in range(240):
        if p.poll() is not None:
            raise SystemExit(f"almond: the router exited with {p.returncode} (see {log})")
        if api.up():
            return p
        time.sleep(0.5)
    raise SystemExit(f"almond: the router did not answer in 2 minutes (see {log})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prompt", help="one prompt: the answer on stdout, then exit")
    ap.add_argument("-m", "--model", help="the model (a name from /models or its start: glm, deepseek)")
    ap.add_argument("--url", default=os.environ.get("ALMOND_URL", "http://127.0.0.1:8150"))
    ap.add_argument("--api-key", default=os.environ.get("ALMOND_API_KEY", ""))
    t = ap.add_mutually_exclusive_group()
    t.add_argument("--nothink", action="store_true", help="answer without thinking")
    t.add_argument("--think", action="store_true", help="the model's default thinking level")
    t.add_argument("--think-max", action="store_true", help="the most thorough level")
    t.add_argument("--think-level", metavar="LEVEL", help="one of the model's levels (off / low / high / max; "
                                                          "a number 0-100 picks the nearest)")
    ap.add_argument("--temp", type=float, default=None, help="temperature (0 = greedy; default: the model's)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--session-log", nargs="?", const="local/session-log", default=None, metavar="DIR",
                    help="the servers this starts write every request and its answer to DIR/<date>.jsonl "
                         "(default DIR: local/session-log), for debugging")
    ap.add_argument("--ctx", type=int, default=None, metavar="TOKENS",
                    help="the context window of the models this starts (default: each configuration's); a server "
                         "already running keeps its own")
    ap.add_argument("--monitor", action="store_true", help="what the server's monitor shows, refreshed every second")
    ap.add_argument("--sessions", action="store_true", help="the conversations kept on disk, per model")
    ap.add_argument("--clear-sessions", action="store_true",
                    help="move the conversations kept on disk to the trash (no engine may be running)")
    ap.add_argument("--system", default=None, help="a system prompt")
    ap.add_argument("--agent", action="store_true",
                    help="the model may read and edit files and run commands in the working directory (it asks first)")
    ap.add_argument("--yes", action="store_true", help="with --agent: do not ask before a write, an edit or a command (a path outside the directory still asks)")
    ap.add_argument("--hide-thinking", action="store_true", help="do not print the thinking")
    ap.add_argument("--no-start", action="store_true", help="never start the router")
    ap.add_argument("--keep", action="store_true", help="leave a router this started running on exit")
    a = ap.parse_args()

    if a.sessions:
        print(sessions_text())
        return 0
    if a.clear_sessions:
        return clear_sessions()
    api = Api(a.url, a.api_key)
    started = None
    if a.monitor:                       # a look, never a start: nothing is loaded for it
        if not api.up():
            raise SystemExit(f"almond: nothing answers at {a.url} (./run.sh --router, or ./almond, starts it)")
        run_monitor(api)
        return 0
    if not api.up():
        if a.no_start or api.host not in ("127.0.0.1", "localhost"):
            raise SystemExit(f"almond: nothing answers at {a.url}")
        started = start_router(api, api.port, a.ctx, a.session_log)
    chat = Chat(api, a)
    try:
        chat.pick(a.model)
        if a.ctx and started is None and (chat.model or {}).get("max_context") not in (None, a.ctx):
            say(f"{DIM}--ctx {a.ctx}: the server was already running with {chat.model.get('max_context')} tokens "
                f"(./stop.sh, then ./almond --ctx {a.ctx} starts it with yours){RESET}")
        if a.nothink:
            chat.set_level("off")
        elif a.think_max:
            chat.set_level("max")
        elif a.think_level and not chat.set_level(a.think_level):
            return 2
        if a.prompt is not None:
            if a.prompt == "-":
                a.prompt = sys.stdin.read()
            return 0 if chat.ask(a.prompt) else 1
        chat.loop()
        return 0
    except RuntimeError as e:
        say(f"{RED}almond: {e}{RESET}")
        return 1
    finally:
        if started is not None and not a.keep:
            say(f"{DIM}stopping the router ...{RESET}")
            os.killpg(started.pid, signal.SIGINT)
            try:
                started.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(started.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())

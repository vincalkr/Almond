#!/usr/bin/env python3
"""almond - the terminal chat, in the manner of antirez's ds4 CLI.

  ./almond                       a conversation (starts the router if nothing answers at --url, stops it on /quit)
  ./almond -p "question"         one prompt, the answer on stdout, then exit
  ./almond -m glm                the model (a name from /models, or the start of one: glm, deepseek)

Thinking: --nothink, --think (the model's default), --think-max, --think-level LEVEL - the levels are the model's own
(GLM-5.3-Flash: low / high / max, it always thinks; DeepSeek-V4.1: none / low / high / max = 0 / 50 / 75 / 100).  In the conversation:

  /help  /quit  /models  /model NAME  /think [LEVEL]  /nothink  /clear  /read FILE  /system TEXT  /ctx
  /set [KEY VALUE]   sampling: temp, top_p, top_k, min_p, repeat_penalty, presence_penalty, seed, max_tokens
                     (/set alone shows them and the model's defaults; /set KEY default goes back to the model's)
  /stats             what the server's monitor shows: speed, GPUs, VRAM, CPU, RAM, disk, context, experts, requests
  /forget            clear the conversation cache the engine keeps
  /hide /show        the thinking

  ./almond --monitor   the same as /stats, refreshed every second (Ctrl+C leaves)

A line ending in \\ goes on to the next; \"\"\" starts and ends a block of several lines.  Ctrl+C stops the answer and
returns to the prompt; Ctrl+D (or /quit) leaves.  It speaks Almond's OpenAI API (the router's, ./run.sh --router, or
one model's server, ./run.sh): the history and the thinking level live here, the model there.
"""
import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
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
        w = {"0": "none", "off": "none", "nothink": "none"}.get(word.lower(), word.lower())
        if w == "none" and "none" not in vals and vals:      # a model that always thinks (GLM-5.3): its lowest level
            say(f"{DIM}{self.model['id']} always thinks: {vals[0]}, its lowest level{RESET}")
            w = vals[0]
        if w.isdigit() and vals == ["none", "low", "high", "max"]:   # ds4's 1-100: the nearest named level
            n = int(w)
            w = "none" if n == 0 else "low" if n <= 50 else "high" if n <= 75 else "max"
        if w not in vals:
            say(f"{RED}{self.model['id']}'s thinking levels: {', '.join(vals)}{RESET}")
            return False
        self.level = w
        return True

    # ---- one turn
    def ask(self, text: str, out=None) -> bool:
        out = out or sys.stdout
        self.messages.append({"role": "user", "content": text})
        body = {"model": self.model["id"], "stream": True, "messages": self.wire()}
        if self.level:
            body["reasoning_effort"] = self.level
        body.update(self.sampling)
        reasoning, content, state, end = [], [], None, None
        # the thinking goes with the answer on a terminal; otherwise (-p into a pipe or a file) to stderr, so stdout
        # holds the answer alone
        think = sys.stdout if TTY else sys.stderr
        t0 = time.time()
        first = None
        try:
            for kind, v in self.api.stream(body):
                if kind in ("reasoning", "content") and first is None:
                    first = time.time()
                if kind == "reasoning":
                    reasoning.append(v)
                    if self.show:
                        if state != "reasoning":
                            think.write(f"{DIM}")
                        think.write(v)
                        think.flush()
                    state = "reasoning"
                elif kind == "content":
                    if state == "reasoning" and self.show:
                        think.write(f"{RESET}\n\n" if TTY else "\n")
                        think.flush()
                    state = "content"
                    content.append(v)
                    out.write(v)
                    out.flush()
                elif kind == "tool":
                    fn = v.get("function") or {}
                    out.write(f"{CYAN}[tool call {fn.get('name', '')} {fn.get('arguments', '')}]{RESET}")
                elif kind == "end":
                    end = v
        except KeyboardInterrupt:
            try:
                self.api.live.close()           # the server sees the client go and stops the answer
            except Exception:
                pass
            if TTY:
                out.write(f"{RESET}\n{DIM}[stopped]{RESET}\n")
            end = {"finish": "stopped"}
        except (RuntimeError, OSError, http.client.HTTPException) as e:
            self.messages.pop()
            say(f"{RESET}{RED}error: {e}{RESET}")
            return False
        if state == "reasoning" and self.show:
            think.write(RESET if TTY else "\n")
        out.write("\n")
        msg = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
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
        return True

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
        say(f"{BOLD}almond{RESET} - {m['id']}, context {m.get('max_context') or '?'}, thinking {self.level_text()}. "
            f"/help for the commands.")

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
            if self.set_level("none"):
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
        while True:
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
        if readline:
            try:
                readline.write_history_file(HISTORY)
            except OSError:
                pass


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


def start_router(api: Api, port: int) -> subprocess.Popen:
    log = ROOT / "local" / "router.log"
    say(f"{DIM}nothing answers at {api.host}:{api.port}: starting the router (./run.sh --router, log {log}) ...{RESET}")
    p = subprocess.Popen([str(ROOT / "run.sh"), "--router", "--port", str(port)], cwd=ROOT, start_new_session=True,
                         stdout=open(log, "a"), stderr=subprocess.STDOUT)
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
    t.add_argument("--think-level", metavar="LEVEL", help="one of the model's levels (none / low / high / max; "
                                                          "a number 0-100 picks the nearest)")
    ap.add_argument("--temp", type=float, default=None, help="temperature (0 = greedy; default: the model's)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--monitor", action="store_true", help="what the server's monitor shows, refreshed every second")
    ap.add_argument("--system", default=None, help="a system prompt")
    ap.add_argument("--hide-thinking", action="store_true", help="do not print the thinking")
    ap.add_argument("--no-start", action="store_true", help="never start the router")
    ap.add_argument("--keep", action="store_true", help="leave a router this started running on exit")
    a = ap.parse_args()

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
        started = start_router(api, api.port)
    chat = Chat(api, a)
    try:
        chat.pick(a.model)
        if a.nothink:
            chat.set_level("none")
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

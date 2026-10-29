#!/usr/bin/env python3
"""Almond's API server (serve/server.py) in front of GLM-5.3-Flash on project-maya's engine (`build-maya-hip/maya
--serve`, vendored in engines/maya).  The server is generic and the pack's tokenizer directory carries the model's own
Jinja chat template; this shim swaps in GLM's differences:

- reasoning effort and sampling as Z.ai's API has them for GLM-5.3-Flash (docs.z.ai, chat completions): thinking is
  always on, its levels Low, High and Max (the default), any client spelling mapped onto them; temperature 0-1
  (default 1.0), top_p 0.01-1 (0.95), no penalties, top_k or min_p, at most 128K tokens out

- tool calls: GLM writes `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`; the server's
  parser reads `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`.  The stream is
  translated, as serve_ds41.py does for DeepSeek's DSML, so every heuristic of that parser keeps working.

  serve_glm.py --engine almond --config local/glm-local.json --port 8150   (./run.sh --model glm does this)
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

CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
KEY_OPEN, KEY_CLOSE, VAL_OPEN, VAL_CLOSE = "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"
_PAIR_RE = re.compile(re.escape(KEY_OPEN) + r"(.*?)" + re.escape(KEY_CLOSE) + r"\s*" + re.escape(VAL_OPEN) +
                      r"(.*?)" + re.escape(VAL_CLOSE), re.DOTALL)


def glm_translate(body: str) -> str:
    """One whole GLM call body (between <tool_call> and </tool_call>) -> the server's XML form; a body that is not a
    call (no name, or text between the pairs) comes back unchanged with its wrapper, as the prose it is."""
    k = body.find(KEY_OPEN)
    name = (body if k < 0 else body[:k]).strip()
    pairs = _PAIR_RE.findall(body[k:]) if k >= 0 else []
    if not name or re.search(r"\s", name) or (k >= 0 and _PAIR_RE.sub("", body[k:]).strip()):
        return CALL_OPEN + body + CALL_CLOSE
    params = "".join(f"<parameter={key.strip()}>\n{value}\n</parameter>\n" for key, value in pairs)
    return f"{CALL_OPEN}\n<function={name}>\n{params}</function>\n{CALL_CLOSE}"


def glm_rewrite(buf: str, final: bool) -> tuple[str, str]:
    """(text to pass on, text to hold): each complete GLM call translated; an open call (or a partial `<tool_call>`
    at the end) is held until its close arrives - at the end of the turn whatever is held goes out as it is."""
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
        out.append(glm_translate(buf[i + len(CALL_OPEN):j]))
        buf = buf[j + len(CALL_CLOSE):]
    if not final:                                   # a `<tool_call>` still being written
        for n in range(min(len(CALL_OPEN) - 1, len(buf)), 0, -1):
            if CALL_OPEN.startswith(buf[-n:]):
                out.append(buf[:-n])
                return "".join(out), buf[-n:]
    out.append(buf)
    return "".join(out), ""


# ---- reasoning effort: GLM-5.3-Flash's own levels, as Z.ai's API takes them: "GLM-5.3 GLM-5.3-FLASH series can only
# be enabled, and the thinking depth is controlled by reasoning_effort" - low / high / max, default max.  Its template
# knows "low" and "high"; anything else is "Max".  A level the API does not have goes to the nearest: none / minimal
# (thinking off, which GLM-5.3 does not do) to Low, medium to High, xhigh to Max.  ALMOND_GLM_THINK_OFF=1 keeps the
# old way of no thinking (Low with the think block closed at once: not the API's, and not measured).
THINK_OFF = os.environ.get("ALMOND_GLM_THINK_OFF") == "1"
GLM_EFFORT = {"none": None if THINK_OFF else "low", "off": None if THINK_OFF else "low",
              "minimal": None if THINK_OFF else "low", "disabled": None if THINK_OFF else "low",
              "false": None if THINK_OFF else "low",
              "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max", "maximum": "max",
              "auto": "max", "default": "max"}


def effort_kwargs(value) -> dict:
    """A reasoning effort as a client gives it -> the template's kwargs; an unknown one is a 400."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False} if THINK_OFF else {"reasoning_effort": "low"}
    key = str(value).strip().lower()
    if key not in GLM_EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, high or max (medium = high, xhigh = max)")
    level = GLM_EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K Low, under 8K High, else Max."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "high" if n < 8192 else "max"}


for mod in (F, S):
    if hasattr(mod, "effort_kwargs"):
        mod.effort_kwargs = effort_kwargs
    if hasattr(mod, "budget_effort"):
        mod.budget_effort = budget_effort
S.REASONING_LEVELS = ([["none", "Off", "answers right away (not Z.ai's)"]] if THINK_OFF else []) + \
    [["low", "Low", "short"], ["high", "High", "thorough"], ["max", "Max", "the most thorough (the default)"]]
S.REASONING_DEFAULT = "max"

# ---- sampling: Z.ai's API for GLM-5.3-Flash - temperature [0, 1] (default 1.0), top_p [0.01, 1] (default 0.95), no
# penalties, top_k or min_p; max_tokens at most 131072 (no default given: an unset one is the context's rest, up to that).
# ALMOND_GLM_FREE_SAMPLING=1 passes a request's values as they are.
NEUTRAL = {"top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def api_sampling(sampling: dict | None) -> dict:
    s = dict(sampling or {})
    if os.environ.get("ALMOND_GLM_FREE_SAMPLING") == "1":
        return s
    s.update(NEUTRAL)
    for key, lo, hi, default in (("temperature", 0.0, 1.0, 1.0), ("top_p", 0.01, 1.0, 0.95)):
        try:
            s[key] = min(hi, max(lo, float(s.get(key, default))))
        except (TypeError, ValueError):
            s[key] = default
    return s


_run, _prepare = S.Service.run, S.Service.prepare


def _run_glm(self, ids, thinking, tools, max_new, sampling, cancel, force=None):
    merged = {**self.sampling_defaults, **self.shared,
              **{k: v for k, v in (sampling or {}).items() if v is not None}}
    return _run(self, ids, thinking, tools, max_new, api_sampling(merged), cancel, force)


def _prepare_glm(self, messages, tools, kwargs, max_new=None, force=None, req=None):
    if not THINK_OFF and kwargs.get("enable_thinking") is False:     # (Anthropic's disabled, chat_template_kwargs)
        kwargs = {**{k: v for k, v in kwargs.items() if k != "enable_thinking"}, "reasoning_effort": "low"}
    ids, thinking, n = _prepare(self, messages, tools, kwargs, max_new, force, req)
    return ids, thinking, min(n, 131072)


S.Service.run = _run_glm
S.Service.prepare = _prepare_glm

_render = F.ChatTemplate.render


def _render_glm(self, messages, tools=None, add_generation_prompt=True, **kwargs):
    """The template told the level in its own words; thinking off = Low with <think></think> (the way the template
    writes a turn without reasoning)."""
    off = kwargs.pop("enable_thinking", True) is False
    level = kwargs.pop("reasoning_effort", None)
    kwargs["reasoning_effort"] = "low" if off else level if level in ("low", "high") else "max"
    out = _render(self, messages, tools, add_generation_prompt, **kwargs)
    if off and add_generation_prompt and out.endswith("<think>"):
        out += "</think>"
    return out


F.ChatTemplate.render = _render_glm

_op_init, _op_feed, _op_finish = F.OutputParser.__init__, F.OutputParser.feed, F.OutputParser.finish


def _op_init_glm(self, *a, **k):
    _op_init(self, *a, **k)
    self.stream_tools = False          # a call reaches the parser whole
    self.glm_hold = ""


def _op_feed_glm(self, delta: str):
    text, self.glm_hold = glm_rewrite(self.glm_hold + delta, False)
    return _op_feed(self, text) if text else []


def _op_finish_glm(self, reason=None):
    text, self.glm_hold = glm_rewrite(self.glm_hold, True)
    return (_op_feed(self, text) if text else []) + _op_finish(self, reason)


F.OutputParser.__init__ = _op_init_glm
F.OutputParser.feed = _op_feed_glm
F.OutputParser.finish = _op_finish_glm
S.OutputParser = F.OutputParser

if __name__ == "__main__":
    sys.exit(S.main())

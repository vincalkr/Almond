#!/usr/bin/env python3
"""Almond's API server (serve/server.py) in front of Inkling 2 Mini on almond-inkling (src/inkling: llama.cpp's
inkling2_loop graph behind the engine's serve protocol).  The tokenizer directory carries the GGUF's own Jinja chat
template (tools/almond_tokenizer.py, pre-split `inkling2` = llama.cpp's GPT-4o split); this shim swaps in Inkling's
differences:

- end of turn: <|content_model_end_sampling|> (id 200006, the GGUF's eos)
- reasoning effort: the template's own levels none / minimal / low / medium / high / xhigh / max (0.0 .. 0.99); the
  template's default is 0.9 (high)
- messages: the model writes its turns as `<|message_model|><|content_thinking|>...<|end_message|>` (thinking),
  `<|message_model|><|content_text|>...<|end_message|>` (the answer) and its calls as
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>`.  The stream is
  rewritten to the server's `<think>...</think>` and `<tool_call><function=NAME><parameter=K>V</parameter></function>
  </tool_call>` (as serve_hy4.py does), so the server's parser stays the same.

  serve_inkling.py --engine almond --config local/inkling-local.json --port 8150   (./almond serve --model inkling)
"""
import json
import os
import re
import sys
from pathlib import Path

ALMOND = Path(os.environ.get("ALMOND_HOME") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ALMOND))
sys.path.insert(0, str(ALMOND / "tools"))

from serve import frontend as F  # noqa: E402
from serve import server as S  # noqa: E402

EOS_ID = 200006
MSG = "<|message_model|>"
END = "<|end_message|>"
THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
CALL = "<|content_invoke_tool_json|>"
STOP = "<|content_model_end_sampling|>"

S.IM_END = STOP
_svc_init = S.Service.__init__


def _svc_init_inkling(self, *a, **k):
    _svc_init(self, *a, **k)
    self.stop_ids = {EOS_ID}
    self.effort_end = False      # the effort is the template's own system turn


S.Service.__init__ = _svc_init_inkling

# top_k: the configuration leaves it out, which almond-inkling takes as off (Unsloth's recommended top_k 0; the
# server's own config check takes only 1..64, its engine's cap)

# ---- reasoning effort: the template's effort_map
LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
ALIASES = {"off": "none", "disabled": "none", "false": "none", "no_think": "none", "auto": "high",
           "default": "high", "maximum": "max"}


def effort_kwargs(value) -> dict:
    if value is None or value == "":
        return {}
    if value is False:
        return {"reasoning_effort": "none"}
    key = str(value).strip().lower()
    key = ALIASES.get(key, key)
    if key not in LEVELS:
        try:
            x = float(key)
        except ValueError:
            raise ValueError(f"unknown reasoning effort {value!r}: use {', '.join(LEVELS)} or 0.0-0.99")
        return {"reasoning_effort": min(max(x, 0.0), 0.99)}
    return {"reasoning_effort": key}


def budget_effort(tokens) -> dict:
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else "high"}


for mod in (F, S):
    if hasattr(mod, "effort_kwargs"):
        mod.effort_kwargs = effort_kwargs
    if hasattr(mod, "budget_effort"):
        mod.budget_effort = budget_effort
S.REASONING_LEVELS = [["none", "Off", "answers right away (little thinking)"], ["low", "Low", "short"],
                      ["medium", "Medium", "medium"], ["high", "High", "thorough (the default)"],
                      ["max", "Max", "the most thorough"]]
S.REASONING_DEFAULT = "high"

# the template marks the assistant turns with transformers' {% generation %} block (for training masks), which plain
# Jinja does not know: it is read without those two tags (the text between them stays)
_tpl_init = F.ChatTemplate.__init__
_GEN_RE = re.compile(r"\{%-?\s*(?:end)?generation\s*-?%\}")


def _tpl_init_inkling(self, path):
    import tempfile
    src = Path(path).read_text(encoding="utf-8")
    if not _GEN_RE.search(src):
        return _tpl_init(self, path)
    with tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False, encoding="utf-8") as f:
        f.write(_GEN_RE.sub("", src))
    try:
        _tpl_init(self, f.name)
    finally:
        os.unlink(f.name)


F.ChatTemplate.__init__ = _tpl_init_inkling

_render = F.ChatTemplate.render


def _render_inkling(self, messages, tools=None, add_generation_prompt=True, **kwargs):
    off = kwargs.pop("enable_thinking", True) is False
    level = kwargs.pop("reasoning_effort", None)
    kwargs["reasoning_effort"] = "none" if off else level if level is not None else "high"
    msgs = []
    for m in messages:
        m = dict(m)
        # the server keeps an earlier turn's thinking inside <think>: the template wants it in reasoning_content
        c = m.get("content")
        if m.get("role") == "assistant" and isinstance(c, str) and "<think>" in c and "</think>" in c:
            a, _, rest = c.partition("<think>")
            think, _, after = rest.partition("</think>")
            m["reasoning_content"] = think.strip()
            m["content"] = (a + after).strip()
        # the template wants the arguments as an object
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {}
                calls.append(tc)
            m["tool_calls"] = calls
        msgs.append(m)
    return _render(self, msgs, tools, add_generation_prompt, **kwargs)


F.ChatTemplate.render = _render_inkling


# ---- output: Inkling's message tokens -> the server's tags
def _param(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def translate_call(name: str, body: str) -> str:
    try:
        obj = json.loads(body)
        name = obj.get("name") or name
        args = obj.get("args") or obj.get("arguments") or {}
    except ValueError:
        return f"<tool_call>{body}</tool_call>"
    params = "".join(f"<parameter={k}>\n{_param(v)}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name.strip()}>\n{params}</function>\n</tool_call>"


TAGS = (MSG, END, THINK, TEXT, CALL, STOP)


class Rewriter:
    """Turns the model's message stream into the server's text; holds what may still be a tag or an open call."""

    def __init__(self, parser=None):
        self.hold = ""
        self.mode = None          # None, "think", "text", "call"
        self.call_name = ""
        # the server's parser starts inside the reasoning when the request thinks (it reads only `</think>`): the
        # rewriter closes it before the first answer or call, and opens it when the parser starts in the answer
        self.parser = parser
        self.reasoning = parser is None or getattr(parser, "state", "reasoning") == "reasoning"

    def _leave_reasoning(self, out):
        if self.reasoning:
            out.append("</think>\n\n")
            self.reasoning = False

    def feed(self, delta: str, final: bool) -> str:
        buf, out = self.hold + delta, []
        self.hold = ""
        while buf:
            if self.mode == "call":
                j = buf.find(END)
                if j < 0:
                    if final:
                        out.append(translate_call(self.call_name, buf))
                        buf = ""
                        break
                    self.hold = buf
                    return "".join(out)
                out.append(translate_call(self.call_name, buf[:j]))
                buf, self.mode = buf[j + len(END):], None
                continue
            i = buf.find("<|")
            if i < 0:
                if not final and buf.endswith("<"):        # maybe the start of a tag
                    out.append(self._text(buf[:-1]))
                    self.hold = "<"
                    return "".join(out)
                out.append(self._text(buf))
                buf = ""
                break
            if i > 0:
                out.append(self._text(buf[:i]))
                buf = buf[i:]
            k = buf.find("|>")
            if k < 0:
                if final:
                    out.append(self._text(buf))
                    buf = ""
                    break
                self.hold = buf
                return "".join(out)
            tag, buf = buf[:k + 2], buf[k + 2:]
            if tag == MSG:
                # a call's name runs up to <|content_invoke_tool_json|>
                j = buf.find("<|")
                if (j < 0 or "|>" not in buf[j:]) and not final:
                    self.hold = tag + buf
                    return "".join(out)
                if j >= 0 and buf[j:].startswith(CALL):
                    self._leave_reasoning(out)
                    self.call_name, buf, self.mode = buf[:j], buf[j + len(CALL):], "call"
                elif j >= 0 and "|>" not in buf[j:] and not final:
                    self.hold = tag + buf          # the tag after the name is not whole yet
                    return "".join(out)
                continue
            if tag == THINK:
                self.mode = "think"
                if not self.reasoning and self.parser is not None:
                    self.parser.state = "reasoning"      # a request without thinking whose model thinks anyway
                self.reasoning = True
            elif tag == TEXT:
                self.mode = "text"
                self._leave_reasoning(out)
            elif tag == END:
                if self.mode == "think":
                    self._leave_reasoning(out)
                self.mode = None
            elif tag == STOP:
                self.mode = None
            else:
                out.append(self._text(tag))
        if final and self.mode == "think":
            self._leave_reasoning(out)
            self.mode = None
        return "".join(out)

    def _text(self, s: str) -> str:
        return s


_op_init, _op_feed, _op_finish = F.OutputParser.__init__, F.OutputParser.feed, F.OutputParser.finish


def _op_init_inkling(self, *a, **k):
    _op_init(self, *a, **k)
    self.stream_tools = False
    self.inkling_rw = Rewriter(self)


def _op_feed_inkling(self, delta: str):
    text = self.inkling_rw.feed(delta, False)
    return _op_feed(self, text) if text else []


def _op_finish_inkling(self, reason=None):
    text = self.inkling_rw.feed("", True)
    return (_op_feed(self, text) if text else []) + _op_finish(self, reason)


F.OutputParser.__init__ = _op_init_inkling
F.OutputParser.feed = _op_feed_inkling
F.OutputParser.finish = _op_finish_inkling
S.OutputParser = F.OutputParser

if __name__ == "__main__":
    sys.exit(S.main())

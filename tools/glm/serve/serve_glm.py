#!/usr/bin/env python3
"""Almond's API server (serve/server.py) in front of GLM-5.3-Flash on project-maya's engine (`build-maya-hip/maya
--serve`, vendored in engines/maya).  The server is generic and the pack's tokenizer directory carries the model's own
Jinja chat template; this shim swaps in GLM's one difference:

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

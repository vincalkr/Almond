#!/usr/bin/env python3
"""Almond's API server (serve/server.py) in front of Qwen3.8-Flash-Next on the engine (`almond-engine --qwen --serve`,
src/qwen).  The server's defaults are this model's own: the qwen35 tokenizer (tokenizer.ggml.pre), its template and
its tool calls (`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`).  This shim sets what the
model card (Qwen/Qwen3.8-Flash-Next) says: the thinking levels under their names (xhigh, the default, medium, low; off
= enable_thinking false; a client's "high" or "max" is xhigh) and the sampling without thinking (0.7 / 0.8 / top_k 20 /
presence 1.5; with it the configuration's 1.0 / 0.95 / top_k 20).

  serve_qwen.py --engine almond --config local/qwen-local.json --port 8150   (./run.sh --model qwen does this)
"""
import os
import sys
from pathlib import Path

ALMOND = Path(os.environ.get("ALMOND_HOME") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ALMOND))
sys.path.insert(0, str(ALMOND / "tools"))

from serve import server as S  # noqa: E402

# the template's own name for its top level: xhigh (a client's "high" or "max" is the same level)
S.REASONING_LEVELS = [["none", "Off", "answers right away"], ["low", "Low", "short"], ["medium", "Medium", "medium"],
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

if __name__ == "__main__":
    sys.exit(S.main())

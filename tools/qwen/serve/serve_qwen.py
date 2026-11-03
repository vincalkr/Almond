#!/usr/bin/env python3
"""Almond's API server (serve/server.py) in front of Qwen3.8-Flash-Next on the engine (`almond-engine --qwen --serve`,
src/qwen).  The server's defaults are this model's own: the qwen35 tokenizer (tokenizer.ggml.pre), the template's
thinking levels (low, medium, xhigh; "high" is xhigh, the default; none = enable_thinking false) and its tool calls
(`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`), so nothing is swapped in here; the
sampling is the configuration's (the GGUF's general.sampling: temperature 1.0, top_p 0.95, top_k 20).

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

if __name__ == "__main__":
    sys.exit(S.main())

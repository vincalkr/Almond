"""tools/qwen/_paths.py - where the Qwen pack tools find llama.cpp's gguf-py: $ALMOND_GGUF_PY, else the engine's own
third_party/llama.cpp/gguf-py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ALMOND = Path(__file__).resolve().parents[2]


def add_gguf_py() -> None:
    p = os.environ.get("ALMOND_GGUF_PY") or str(ALMOND / "third_party" / "llama.cpp" / "gguf-py")
    if not (Path(p) / "gguf").is_dir():
        sys.exit(f"llama.cpp's gguf-py is not at {p}: set ALMOND_GGUF_PY to a llama.cpp checkout's gguf-py folder")
    if p not in sys.path:
        sys.path.insert(0, p)

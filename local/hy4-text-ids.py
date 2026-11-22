#!/usr/bin/env python3
"""The real-text prompt the hy4 measurements use (local/hy4-*-check.sh).

The ids must be real text: with pseudo-random ids every draft is "accepted" (the
drafter sees a continuation that is not random) and the acceptance number means
nothing - it read 99.7% once.  So the prompt is the repo's own prose and source.

    python3 local/hy4-text-ids.py [N] [out]     # defaults 16384, /tmp/hy4-p16k-text.ids
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "glm"))         # almond_tokenizer's gguf_reader
from almond_tokenizer import Tokenizer                                     # noqa: E402

CORPUS = ["docs/hy4/GGUF_NOTES.md", "docs/hy4/SPEED_WORK.md", "HANDOVER.md",
          "src/hy4/hy4_engine.cpp", "src/hy4/hy4_moe.cpp", "tools/router/almond_router.py"]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("/tmp/hy4-p16k-text.ids")
gguf = ROOT / "models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf"

tk = Tokenizer.from_gguf(str(gguf))
text = "\n".join((ROOT / p).read_text(encoding="utf-8", errors="replace") for p in CORPUS)
ids = tk.encode(text)[:n]
out.write_text(",".join(str(i) for i in ids))     # hy4's --prompt-ids @FILE is a csv (parse_csv)
print(f"{out}: {len(ids)} ids, from {len(CORPUS)} files")

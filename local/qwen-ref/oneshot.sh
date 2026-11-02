#!/bin/sh
# oneshot.sh [EXE] [NEW]: Qwen3.8-Flash-Next one-shot on local/qwen-ref/doc512.ids.txt (local/glm-ref/doc.txt's first
# 512 tokens), greedy, NEW tokens (default 64); prints the generated ids and the speeds.  EXE defaults to
# build-hip-fused/almond-engine; another build of the same engine (not named almond-engine) runs without --qwen.  Two
# builds that print the same ids read the prompt and decode alike (the native path dumps no logits to compare).
R=$(cd "$(dirname "$0")/../.." && pwd)
EXE=${1:-$R/build-hip-fused/almond-engine}; NEW=${2:-64}; M=$R/models
Q=""; [ "$(basename "$EXE")" = almond-engine ] && Q=--qwen
"$EXE" $Q --pack $M/qwen38-iq3s-pack --native $M/qwen38-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --ple-gguf $M/qwen38-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
    --expert-profile "$R/tools/qwen/data/expert-profile-qwen.bin" --expert-cache auto --prefill auto:32768 --spec 4 \
    --mtp $M/qwen38-mtp --max-context 8192 --kv int8 --vram-reserve-mib 400 \
    --tokens "$(cat "$R/local/qwen-ref/doc512.ids.txt")" --max-new "$NEW" 2>/dev/null | grep -E "^output|^decode|^prefill"

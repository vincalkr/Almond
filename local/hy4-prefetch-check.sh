#!/usr/bin/env bash
# local/hy4-prefetch-check.sh: does --prefetch-experts still pay now that place() charges its staging buffer to card
# 0's budget? Same flags as local/hy4-local.json, a 16,384-id pseudo-random prompt (/tmp/hy4-p16k.ids) and 100 greedy
# tokens, alternating runs as in docs/hy4/GGUF_NOTES.md. Reference (2026-11-20, before the budget charge): prompt
# 431.3 tok/s without the flag, 479.3/480.7 with it; card 0 free 1.10 GiB against 0.45 GiB with the staging.
set +e
cd "$(dirname "$0")/.."
BASE="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 --engram-place auto --mtp --fused-hc
      --dense-q8 --reserve-mib 1400 --fused-proj --fused-indexer --gather-attn --kv-q8 --ubatch 1024
      --temp 0 -n 100 --prompt-ids @/tmp/hy4-p16k.ids"
PAT='^hy4: (card [01]:|layers per card|placement|compute buffers|prefetch|[0-9.]+ GiB of (weights|CPU))|steps:|prompt [0-9]+ tokens'
i=0
run() {
    i=$((i + 1))
    echo "===== $1 (log /tmp/hy4-run-$i.log) ====="
    ALMOND_MMVF_ROWS=1 HY4_STEP_STATS=1 ./build-hip-fused/almond-engine $BASE $2 > "/tmp/hy4-run-$i.log" 2>&1
    echo "(engine exit $?)"
    grep -E "$PAT" "/tmp/hy4-run-$i.log"
    tail -3 "/tmp/hy4-run-$i.log" | grep -v "graph warmup"
}
run "256K, prefetch on"   "-c 262144 --prefetch-experts"
run "256K, prefetch off"  "-c 262144"
run "256K, prefetch on"   "-c 262144 --prefetch-experts"
run "256K, prefetch off"  "-c 262144"
run "64K, prefetch on"    "-c 65536 --prefetch-experts"

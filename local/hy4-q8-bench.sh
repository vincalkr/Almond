#!/usr/bin/env bash
# The dense Q8_0 mat-vec table, one block per kernel variant: build-hip-fused/bench_hy4_q8 runs as its own process per
# variant, because ggml reads ALMOND_MMVQ_ROWS / ALMOND_MMVQ_WPR once.  What it answers and why: docs/hy4/GGUF_NOTES.md,
# "The same for Q8_0 rows".  It runs on its own on the free card - no model load, nothing of the serve stack moves.
#   bash local/hy4-q8-bench.sh [device] [iters] [reps]        # device 1 while the engine holds card 0
set -u
cd "$(dirname "$0")/.."
DEV="${1:-1}"; N="${2:-4}"; REP="${3:-6}"
for v in "ALMOND_MMVQ_ROWS=0" "ALMOND_MMVQ_ROWS=1" "ALMOND_MMVQ_ROWS=1 ALMOND_MMVQ_WPR=1" \
         "ALMOND_MMVQ_ROWS=1 ALMOND_MMVQ_WPR=2" "ALMOND_MMVQ_ROWS=1 ALMOND_MMVQ_WPR=4"; do
    env $v ./build-hip-fused/bench_hy4_q8 "$DEV" "$N" "$REP" 2>&1 | grep -vE "graph warmup|ROCm devices|^  Device "
    echo
done

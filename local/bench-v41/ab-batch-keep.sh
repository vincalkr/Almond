#!/usr/bin/env bash
# ab-batch-keep.sh [OUT] - the cache loan (ALMOND_SERVE_BATCH_KEEP=512) against the configuration, through the server:
# a fresh 3-turn conversation per run (local/bench-v41/two-turns.py 1200 250 OFFSET: a ~2,760-token first turn, then two
# turns of ~470 new tokens, 48 tokens of answer each; another slice of long-docs.txt per run so the conversation disk
# serves none of it), arms alternated, 2 rounds.  Output: OUT/summary.txt (the turns' timings, the engine's lines).
cd "$(dirname "$0")/../.."
OUT=${1:-local/sweep-batch-keep}; mkdir -p $OUT
pgrep -x almond-engine >/dev/null && { echo "an engine is running" >&2; exit 3; }
off=${OFF0:-0}
run() {   # tag env...
    local tag=$1; shift
    local L=$OUT/$tag-serve.log
    local n0=$(wc -l < local/serve-engine.log)
    (env "$@" ./almond serve --port 8164 > $L 2>&1 &)
    for i in $(seq 300); do grep -q "^ready" $L && break; sleep 2; done
    grep -q "^ready" $L || { echo "$tag: no ready line" >&2; tail -3 $L; return; }
    echo "== $tag offset $off $(date +%T)" | tee -a $OUT/summary.txt
    .venv/bin/python local/bench-v41/two-turns.py 8164 1200 250 $off | tee $OUT/$tag-turns.txt |
        sed "s/'predicted_per_token_ms.*//" | tee -a $OUT/summary.txt
    P=$(pgrep -f "[s]erve_ds41.py"); kill -TERM $P; while pgrep -x almond-engine >/dev/null; do sleep 2; done
    tail -n +$((n0 + 1)) local/serve-engine.log > $OUT/$tag-engine.log
    grep -h "serve: prompt \|serve: VRAM\|vram cache\|tokens in .* tok/s\|ERR\|failed" $OUT/$tag-engine.log | cut -c1-200 | tee -a $OUT/summary.txt > $OUT/$tag-lines.txt
    off=$((off + ${STEP:-4000}))
    sleep 3
}
# ARMS="base keep" (default): the arms, alternated; KEEP=512: the loan's token limit
for r in 1 2; do
    for arm in ${ARMS:-base keep}; do
        case $arm in base) run base-r$r ALMOND_SERVE_BATCH_KEEP=0 ;; keep) run keep-r$r ALMOND_SERVE_BATCH_KEEP=${KEEP:-512} ;; esac
    done
done
echo DONE >> $OUT/summary.txt

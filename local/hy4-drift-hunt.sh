#!/usr/bin/env bash
# Where does the hy4 run-to-run drift come from?  Two runs of one configuration score KL 0.12-0.14 / top-1 88% over
# 8,190 teacher-forced rows, with graphs off and the expert prefetch off too (local/hy4-determinism-check.sh).  This
# bisects by taking one ingredient out at a time and comparing two runs of the rest: a pair that prints KL 0.00000
# names the ingredient, a pair at ~0.13 clears it.
#
# First suspect is the Engram: `--engram-place auto` chooses RAM or SSD from the RAM free at start, the table is
# 63.4 GB of F8_E4M3 (shard 2), and the first run of a battery takes 127 s where the next take 42 - that is the page
# cache, so `auto` does not decide the same way twice.  If the two placements read the rows differently, that is the
# drift.  Then the fused indexer (a pick that flips changes which experts run) and --fused-hc (the one ggml op in
# this tree with grid-wide atomics: dsv4-hc.cu:472).
#
#   bash local/hy4-drift-hunt.sh          # ~16 min: eight loads.  Cards through the router's unload, never ./stop.sh.
set -u
cd "$(dirname "$0")/.."
PPL=/tmp/hy4-p16k-text.ids
if [ ! -s "$PPL" ]; then echo "making $PPL"; .venv/bin/python local/hy4-text-ids.py 16384 "$PPL" || exit 1; fi
if [ -z "${HY4HM_FORCE:-}" ] && ps -eo args | grep -q "almond[-]-engine --serve"; then
    echo "an almond-engine --serve is running: unload it through the router first (or HY4HM_FORCE=1)."
    exit 1
fi
BASE="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 -c 8192 --mtp --dense-q8
      --reserve-mib 1400 --fused-proj --gather-attn --kv-q8 --ubatch 1024 --expert-layers-gpu 29 -t 12 --ppl $PPL"
STD="--fused-hc --fused-indexer --prefetch-experts"
PAT='^hy4: (against|ppl|engram|layers per card)|Error|error'
i=0
pair() {                              # $1 the name, $2 the flags for both runs
    for k in save ref; do
        i=$((i + 1))
        echo "----- $1 / $k (log /tmp/hy4dr-$i.log)"
        if [ "$k" = save ]; then F="--ppl-save /tmp/hy4dr-$1.bin"; else F="--ppl-ref /tmp/hy4dr-$1.bin"; fi
        env ALMOND_MMVF_ROWS=1 ./build-hip-fused/almond-engine $BASE $2 $F > "/tmp/hy4dr-$i.log" 2>&1
        echo "(engine exit $?)"
        grep -E "$PAT" "/tmp/hy4dr-$i.log"
    done
}
pair "engram-ram"   "$STD --engram-place ram"
pair "engram-ssd"   "$STD --engram-place ssd"
pair "no-indexer"   "--fused-hc --prefetch-experts --engram-place ram"
pair "no-fused-hc"  "--fused-indexer --prefetch-experts --engram-place ram"

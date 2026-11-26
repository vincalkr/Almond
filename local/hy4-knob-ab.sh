#!/usr/bin/env bash
# A/B one ggml environment knob on the hy4 engine: does it change the numbers, and what is it worth?
#
# Four runs at the configuration of record: a perplexity pass to take a reference, the same pass with the knob to get
# its KL (0.00000 = the same numbers, so the speed pair below compares the same token path), then the speed pair at
# 256K with a 16K real-text prompt.  Since 2026-11-22 the pick is sorted and a run repeats exactly (docs/hy4/SPEED_WORK.md,
# "The cause is the pick"), so a difference here is the knob's, not the machine's - the measurements of 2026-11-19 that
# this re-runs were taken under a +-2% of run-to-run noise.
#
#   bash local/hy4-knob-ab.sh ALMOND_MMVQ_ROWS=1        # ~16 min.  Cards through the router's unload, never ./stop.sh.
set -u
cd "$(dirname "$0")/.."
KNOB="${1:?usage: $0 NAME=value [-n N]}"
N="${2:-500}"
[ "$N" = -n ] && N="${3:-500}"
PPL=/tmp/hy4-p16k-text.ids
if [ ! -s "$PPL" ]; then echo "making $PPL"; .venv/bin/python local/hy4-text-ids.py 16384 "$PPL" || exit 1; fi
if [ -z "${HY4HM_FORCE:-}" ] && ps -eo args | grep -q "almond[-]-engine --serve"; then
    echo "an almond-engine --serve is running: unload it through the router first (or HY4HM_FORCE=1)."
    exit 1
fi
STD="--mtp --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj --fused-indexer --gather-attn --kv-q8 --prefetch-experts --engram-place ram"
MODEL="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 -t 12"
i=0
run() {                              # $1 tag, $2 extra env ("none" for the base arm), $3 the whole argument line
    i=$((i + 1))
    echo "----- $1 (log /tmp/hy4ab-$i.log)"
    if [ "$2" = none ]; then unset ALMOND_MMVQ_ROWS ALMOND_MMVF_WIDE; else export "$2"; fi
    env ALMOND_MMVF_ROWS=1 HY4_HM_STATS=1 HY4_STEP_STATS=1 ./build-hip-fused/almond-engine $3 \
        > "/tmp/hy4ab-$i.log" 2>&1
    echo "(engine exit $?)"
    grep -E "^hy4: (against|ppl|generated tokens|prompt |MTP:|steps:)|Error|error" "/tmp/hy4ab-$i.log"
}
run "base, ppl reference"        none   "$MODEL -c 8192 $STD --ubatch 1024 --expert-layers-gpu 29 --ppl $PPL --ppl-save /tmp/hy4ab-p.bin"
run "$KNOB, ppl against it"     "$KNOB" "$MODEL -c 8192 $STD --ubatch 1024 --expert-layers-gpu 29 --ppl $PPL --ppl-ref /tmp/hy4ab-p.bin"
run "base, speed"                none   "$MODEL -c 262144 $STD --ubatch 1024 --temp 0 -n $N --prompt-ids @$PPL"
run "$KNOB, speed"              "$KNOB" "$MODEL -c 262144 $STD --ubatch 1024 --temp 0 -n $N --prompt-ids @$PPL"

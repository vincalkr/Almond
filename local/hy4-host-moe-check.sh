#!/usr/bin/env bash
# local/hy4-host-moe-check.sh: is --host-moe worth it now that its two ends are split?
# The op used to publish and wait in one node, so the card stopped there and the host's experts never overlapped the
# card's own shared expert (docs/hy4/SPEED_WORK.md, "The decode pipeline"). Now ggml_host_moe_publish rings the bell
# where the expert input is ready and ggml_host_moe waits after the shared expert, so the two run beside each other.
# The ceiling is the same layer's card work: ~0,27 ms of shared expert against ~1,63 ms of host experts a RAM layer,
# 13 layers: +4,8% at best. The text printed must be the same in both arms - the host computes the same expert
# graphs, only earlier - so the md5 of the generated part is part of the result.
#   ids: /tmp/hy4-p16k-text.ids, 16,384 ids of REAL text (docs + hy4 sources + the router), made with
#   tools/almond_tokenizer.py; a pseudo-random prompt is a bad decode harness (it drives the MTP acceptance to
#   ~100%, see local/hy4-graph-stability-check.sh and "How to measure" in docs/hy4/SPEED_WORK.md).
# Needs the cards to itself: ./stop.sh first, ./almond serve --router after.
set +e
cd "$(dirname "$0")/.."
PROMPT=/tmp/hy4-p16k-text.ids
# Never grab the cards from a live serve stack (the CLI session dies with it).
# HY4HM_FORCE=1 to override.
if [ -z "${HY4HM_FORCE:-}" ] && ps -eo args | grep -q "almond[-]-engine --serve"; then
    echo "an almond-engine --serve is running: these runs would not fit and stopping"
    echo "the stack would kill the user's session. Stop it first, or HY4HM_FORCE=1."
    exit 1
fi
BASE="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 -c 262144 --engram-place auto --mtp
      --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj --fused-indexer --gather-attn --kv-q8 --ubatch 1024
      --prefetch-experts --temp 0 -n 1500 -t 12 --prompt-ids @$PROMPT"
PAT='^hy4: (layers per card|compute buffers|prefetch|warning|host-moe)|^hy4: (MTP|DFly2)|generated tokens|Error|error'
i=0
run() {                                  # $1 the name, $2 extra engine flags
    i=$((i + 1))
    echo "===== $1 (log /tmp/hy4hm-$i.log) ====="
    env ALMOND_MMVF_ROWS=1 HY4_HM_STATS=1 ./build-hip-fused/almond-engine $BASE $2 > "/tmp/hy4hm-$i.log" 2>&1
    echo "(engine exit $?)"
    grep -E "$PAT" "/tmp/hy4hm-$i.log"
    sed -n '/^hy4: generate/,/^hy4: \(MTP\|DFly2\)/p' "/tmp/hy4hm-$i.log" | md5sum | sed 's/^/  generated text md5: /'
}
# "pair" runs one of each (~10 min of cards); without it, alternate pairs as the docs' measurements do.
if [ "${1:-}" = pair ]; then
    run "as shipped, no --host-moe"           ""
    run "host-moe, publish early / wait late" "--host-moe"
else
    run "as shipped, no --host-moe"           ""
    run "host-moe, publish early / wait late" "--host-moe"
    run "host-moe, publish early / wait late" "--host-moe"
    run "as shipped, no --host-moe"           ""
fi

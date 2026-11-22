#!/usr/bin/env bash
# Why does the same hy4 configuration not repeat?  local/hy4-host-moe-quality-check.sh ran the reference twice and
# the second run scored KL 0.14530 / top-1 88.17% over 8,190 rows against the first - a KL floor that swamps any
# effect the papers in docs/hy4 quote at 0.01-0.1 level (and the same non-determinism is why two greedy runs with
# identical flags forked from their 11th token).  GGUF_NOTES' requant table quotes "the same run without it repeats
# exactly: KL 0.00000", so something changed since 2026-11-20.  This finds which ingredient: graphs, the expert
# prefetch, or neither.  Each pair saves its own reference and scores the second run against it.
#
#   bash local/hy4-determinism-check.sh          # ~10 min, five loads: take the cards through the router's unload
set -u
cd "$(dirname "$0")/.."
PPL=/tmp/hy4-p16k-text.ids
if [ ! -s "$PPL" ]; then echo "making $PPL"; .venv/bin/python local/hy4-text-ids.py 16384 "$PPL" || exit 1; fi
if [ -z "${HY4HM_FORCE:-}" ] && ps -eo args | grep -q "almond[-]-engine --serve"; then
    echo "an almond-engine --serve is running: unload it through the router first (or HY4HM_FORCE=1)."
    exit 1
fi
BASE="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 -c 8192 --engram-place auto --mtp
      --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj --fused-indexer --gather-attn --kv-q8 --ubatch 1024
      --expert-layers-gpu 29 -t 12 --ppl $PPL"
PAT='^hy4: (against|ppl|layers per card)|Error|error'
i=0
arm() {                                # $1 the name, $2 engine flags, $3 extra env
    i=$((i + 1))
    echo "===== $1 (log /tmp/hy4det-$i.log)"
    env ALMOND_MMVF_ROWS=1 ${3:-} ./build-hip-fused/almond-engine $BASE $2 > "/tmp/hy4det-$i.log" 2>&1
    echo "(engine exit $?)"
    grep -E "$PAT" "/tmp/hy4det-$i.log"
}
# graphs off: the whole batch runs node by node, no replay of a captured pointer set
arm "graphs off, prefetch on - save"   "--prefetch-experts --ppl-save /tmp/hy4det-nog.bin"
arm "graphs off, prefetch on - again"  "--prefetch-experts --ppl-ref /tmp/hy4det-nog.bin"
# prefetch off: the expert tensors go with ggml's own op offload instead of our staged copies
arm "graphs on, no prefetch - save"    "--ppl-save /tmp/hy4det-nopf.bin"
arm "graphs on, no prefetch - again"   "--ppl-ref /tmp/hy4det-nopf.bin"
# and does turning the graphs off move a run that already has no prefetch?
arm "graphs off, no prefetch"          "--ppl-ref /tmp/hy4det-nopf.bin" "GGML_CUDA_DISABLE_GRAPHS=1"

#!/usr/bin/env bash
# local/hy4-graph-stability-check.sh: is it worth stabilizing the HIP graph? The indexer's pool grows one group every
# idx_kpool (16) tokens, so nearly every compute comes with a new shape and ggml captures the graph again: with the
# two ceilings on, 11,774 of 12,288 computes run node by node (docs/hy4/SPEED_WORK.md, "The graph churn").
# HY4_POOL_PAD buckets n_pool to a multiple of 256: the rows past the count are scored and masked by inp_pool_mask,
# so they are read and never picked. The floor is no graphs at all (GGML_CUDA_DISABLE_GRAPHS=1).
# Same flags as local/hy4-local.json, the 16,384-id prompt of local/hy4-prefetch-check.sh, greedy, 1,500 tokens, 256K.
# Needs the cards to itself. The text the last two arms print must be the same: diff /tmp/hy4gs-1.txt against
# /tmp/hy4gs-2.txt (the generated part is stdout before the statistics).
set +e
cd "$(dirname "$0")/.."
BASE="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 -c 262144 --engram-place auto --mtp
      --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj --fused-indexer --gather-attn --kv-q8 --ubatch 1024
      --prefetch-experts --temp 0 -n 1500 -t 12 --prompt-ids @/tmp/hy4-p16k.ids"
PAT='^hy4: (layers per card|compute buffers|prefetch|warning|step )|^prompt |generated tokens|^hy4: MTP|cuda graph stats'
i=0
run() {
    i=$((i + 1))
    echo "===== $1 (log /tmp/hy4gs-$i.log) ====="
    env ALMOND_MMVF_ROWS=1 GGML_CUDA_GRAPH_STATS=1 \
        GGML_CUDA_GRAPH_RECAPTURE_MAX=16 GGML_CUDA_GRAPH_CAPTURE_MAX=128 $2 \
        ./build-hip-fused/almond-engine $BASE > "/tmp/hy4gs-$i.log" 2>&1
    echo "(engine exit $?)"
    grep -E "$PAT" "/tmp/hy4gs-$i.log"
    echo "  graphs switched off for churn: $(grep -c 'not capturing this graph' "/tmp/hy4gs-$i.log")"
    sed -n '1,/^$/p' "/tmp/hy4gs-$i.log" | head -c 400 > "/tmp/hy4gs-$i.txt"
}
run "pool as it grows (shipped)"  ""
run "pool bucketed to 256"        "HY4_POOL_PAD=256"
run "graphs off, node by node"    "GGML_CUDA_DISABLE_GRAPHS=1"
run "pool bucketed to 256"        "HY4_POOL_PAD=256"

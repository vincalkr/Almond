#!/usr/bin/env bash
# Does --indexer-argsort remove the run-to-run drift, and what does the sort cost?
# docs/hy4/SPEED_WORK.md, "Found it: --fused-indexer": ggml_top_k's CUDA/HIP kernel is a radix select that places the
# keys equal to the k-th with atomicAdd (top-k.cu:201,204), so when the boundary scores tie - and they do, every dead
# pool slot carries the same mask value - the picked groups change between two runs of the same input.
# --indexer-argsort keeps --fused-indexer's score kernel and pays a real sort for the picks instead.
#
#   bash local/hy4-indexer-sort-check.sh          # ~12 min: two ppl pairs and one speed pair.
#   bash local/hy4-indexer-sort-check.sh default   # ~6 min: the shipped default repeats, and what --ram-requant q3_k
#                                                  #   really costs against a reference that repeats.
#                                                  # Cards through the router's unload, never ./stop.sh.
set -u
cd "$(dirname "$0")/.."
PPL=/tmp/hy4-p16k-text.ids
if [ ! -s "$PPL" ]; then echo "making $PPL"; .venv/bin/python local/hy4-text-ids.py 16384 "$PPL" || exit 1; fi
if [ -z "${HY4HM_FORCE:-}" ] && ps -eo args | grep -q "almond[-]-engine --serve"; then
    echo "an almond-engine --serve is running: unload it through the router first (or HY4HM_FORCE=1)."
    exit 1
fi
STD="--mtp --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj --fused-indexer --gather-attn --kv-q8 --prefetch-experts --engram-place ram"
MODEL="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 -t 12"
i=0
run() {                              # $1 tag, $2 the whole argument line
    i=$((i + 1))
    echo "----- $1 (log /tmp/hy4is-$i.log)"
    env ALMOND_MMVF_ROWS=1 HY4_HM_STATS=1 ./build-hip-fused/almond-engine $2 > "/tmp/hy4is-$i.log" 2>&1
    echo "(engine exit $?)"
    grep -E "^hy4: (against|ppl|generated tokens|prompt |MTP:|layers per card)|Error|error" "/tmp/hy4is-$i.log"
}
if [ "${1:-all}" = default ]; then
    # The flipped default: the shipped flags, no indexer flag at all, twice.  This must print KL 0.00000 / 100.00%.
    run "shipped default, save" "$MODEL -c 8192 $STD --ubatch 1024 --expert-layers-gpu 29 --ppl $PPL --ppl-save /tmp/hy4is-d.bin"
    run "shipped default, ref"  "$MODEL -c 8192 $STD --ubatch 1024 --expert-layers-gpu 29 --ppl $PPL --ppl-ref /tmp/hy4is-d.bin"
    # The requant the docs left unproven, measured against a reference that repeats: its 0.012 sat under the drift floor.
    run "shipped default + --ram-requant q3_k" \
        "$MODEL -c 8192 $STD --ubatch 1024 --expert-layers-gpu 29 --ppl $PPL --ram-requant q3_k --ppl-ref /tmp/hy4is-d.bin"
    exit 0
fi
# 1. the picks deterministic?  (ppl, -c 8192 so --ppl scores the second half, 13 layers on the CPU)
for k in save ref; do
    f=$([ "$k" = save ] && echo "--ppl-save /tmp/hy4is-a.bin" || echo "--ppl-ref /tmp/hy4is-a.bin")
    run "fused + argsort, $k" "$MODEL -c 8192 $STD --ubatch 1024 --expert-layers-gpu 29 --ppl $PPL --indexer-argsort $f"
done
for k in save ref; do
    f=$([ "$k" = save ] && echo "--ppl-save /tmp/hy4is-b.bin" || echo "--ppl-ref /tmp/hy4is-b.bin")
    run "fused + top_k, $k" "$MODEL -c 8192 $STD --ubatch 1024 --expert-layers-gpu 29 --ppl $PPL $f"
done
# 2. what the sort costs at the configuration of record (750 tokens: the rate, not the tail)
run "config of record, top_k"         "$MODEL -c 262144 $STD --ubatch 1024 --temp 0 -n 750 --prompt-ids @$PPL"
run "config of record, indexer-argsort" "$MODEL -c 262144 $STD --ubatch 1024 --temp 0 -n 750 --prompt-ids @$PPL --indexer-argsort"

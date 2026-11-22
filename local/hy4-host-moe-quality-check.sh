#!/usr/bin/env bash
# The two edges local/hy4-host-moe-check.sh left open (docs/hy4/SPEED_WORK.md, "What has NOT been tested"):
# the KL of --host-moe, and whether the pinned ring slot a captured graph bakes in stays in sync with the host
# thread's sequence.  Teacher-forced scoring repeats exactly, so the second baseline arm is a control: it must
# print KL 0.00000.  If the ring desynchronizes under graph replay, arm 3 moves and arm 4 (graphs off) does not.
#
#   bash local/hy4-host-moe-quality-check.sh
#
# -c 8192 because --ppl scores the second half of each -c-token window (hy4_generate.cpp: `for (i = n_ctx/2; ...)`)
# and the engine keeps n_ctx x vocab logits in RAM (8192 x 120832 x 4 = 3,9 GB).  --expert-layers-gpu 29 puts the
# same 13 layers on the CPU as the 256K configuration does, which at -c 8192 alone would keep everything on the
# cards and make the measurement vacuous: check the "in VRAM for N of 42" line each arm prints.
# ~8 min: four loads, two 8,192-token windows each (8,190 scored rows).  Take the cards through the router's
# unload - never ./stop.sh, the serve stack is the user's session.
set -u
cd "$(dirname "$0")/.."
PPL=/tmp/hy4-p16k-text.ids                  # 16,384 real-text ids (local/hy4-text-ids.py)
REF=/tmp/hy4ppl-ref.bin
if [ ! -s "$PPL" ]; then echo "making $PPL"; .venv/bin/python local/hy4-text-ids.py 16384 "$PPL" || exit 1; fi
if [ -z "${HY4HM_FORCE:-}" ] && ps -eo args | grep -q "almond[-]-engine --serve"; then
    echo "an almond-engine --serve is running: unload it through the router first (or HY4HM_FORCE=1)."
    exit 1
fi
BASE="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 -c 8192 --engram-place auto --mtp
      --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj --fused-indexer --gather-attn --kv-q8 --ubatch 1024
      --expert-layers-gpu 29 --prefetch-experts -t 12 --ppl $PPL"
PAT='^hy4: (against|ppl|host-moe|layers per card|chunk)|Error|error'
i=0
arm() {                                # $1 the name, $2 engine flags, $3 extra env
    i=$((i + 1))
    echo "===== $1 (log /tmp/hy4q-$i.log)"
    env ALMOND_MMVF_ROWS=1 HY4_HM_STATS=1 ${3:-} \
        ./build-hip-fused/almond-engine $BASE $2 > "/tmp/hy4q-$i.log" 2>&1
    echo "(engine exit $?)"
    grep -E "$PAT" "/tmp/hy4q-$i.log"
}
arm "reference, save"                "--ppl-save $REF"
arm "the same again, control"        "--ppl-ref $REF"
arm "--host-moe, graphs as usual"    "--host-moe --ppl-ref $REF"
arm "--host-moe, graphs off"         "--host-moe --ppl-ref $REF" "GGML_CUDA_DISABLE_GRAPHS=1"

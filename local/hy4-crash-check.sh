#!/usr/bin/env bash
# local/hy4-crash-check.sh: the card-emptying deaths of 2026-11-21, reproduced and checked (docs/hy4/GGUF_NOTES.md,
# "The seven deaths of 2026-11-21, and the graphs that never settle").
#
# What was found, so these runs are for checking it again after a change, not for finding it:
#   - hy4's graph is unstable: 1,548 of the 1,874 trunk nodes change their data pointer between two computes, so ggml
#     re-captures it (~1,600 captures over 8,192 computes in one session).
#   - every ROCm capture + instantiation keeps ~2 MiB on the card, and hipGraphDestroy/hipGraphExecDestroy do not
#     give it back (tools/hy4/graph-vram.hip). The cards go to 0 MiB free and the runtime dies in libamdhip64.
#   - the ceilings stop it: per graph (GGML_CUDA_GRAPH_RECAPTURE_MAX) and for the whole process
#     (GGML_CUDA_GRAPH_CAPTURE_MAX). With them the free room stops moving (card 0 1628 / card 1 1834 MiB from step
#     512 to 1408) and 4,000-token runs finish.
#   - --min-free-mib 1024 is not reachable at 256K: the 1,024-token step leaves 465 MiB on the cards and no step
#     length reaches 1 GiB; the flag takes the largest step that fits and says so.
#
# Three runs at 256K, 16,384-token prompt and 4,000 greedy tokens, the cards sampled every 5 s (~5 min a run):
#   1  the disease:     no ceilings, --reserve-mib 1400  (what local/hy4-local.json had)
#   2  the ceilings:    16 / 128, --reserve-mib 1400
#   3  as configured:   16 / 128, --reserve-mib 2400
# Needs the prompt ids of local/hy4-prefetch-check.sh (/tmp/hy4-p16k.ids), and the cards to themselves.
set +e
cd "$(dirname "$0")/.."
PROMPT=/tmp/hy4-p16k.ids
[ -s "$PROMPT" ] || { echo "hy4-crash-check: $PROMPT missing (see local/hy4-prefetch-check.sh)" >&2; exit 1; }
BASE="--hy4 -m models/hy4-iq2ah/Hy4-flash-IQ2_AH-00001-of-00002.gguf --gpus 0,1 -c 262144 --engram-place auto --mtp
      --fused-hc --dense-q8 --fused-proj --fused-indexer --gather-attn --kv-q8 --ubatch 1024 --prefetch-experts
      --temp 0 -n 4000 --prompt-ids @$PROMPT"
PAT='^hy4: (layers per card|placement|compute buffers|prefetch|warning|prompt steps|step [0-9]+)|prompt [0-9]+ tokens|generated [0-9]+ tokens|cuda graph'
i=0

run() {
    local label="$1" extra="$2"; shift 2
    i=$((i + 1))
    echo "===== $label (log /tmp/hy4-crash-$i.log, cards /tmp/hy4-crash-$i.vram) ====="
    ( while sleep 5; do printf '%s used_MiB:' "$(date +%H:%M:%S)"
                     rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\]|GPU\[1\]/{printf " %.0f", $NF/1048576}'; echo
      done ) > "/tmp/hy4-crash-$i.vram" 2>&1 &
    local sampler=$!
    env ALMOND_MMVF_ROWS=1 HY4_VRAM_STATS=128 GGML_CUDA_GRAPH_STATS=1 \
        $extra ./build-hip-fused/almond-engine $BASE "$@" > "/tmp/hy4-crash-$i.log" 2>&1
    local rc=$?
    kill $sampler 2>/dev/null
    echo "(engine exit $rc)$([ $rc -ge 128 ] && echo '  <-- died on a signal: the death this script is about')"
    grep -E "$PAT" "/tmp/hy4-crash-$i.log" | head -40
    grep -ac "not capturing this" "/tmp/hy4-crash-$i.log" | sed 's/^/graphs the ceilings switched off: /'
    echo "--- the cards, 5 s apart (first / every 30 s / last) ---"
    awk 'NR==1 || NR%6==0' "/tmp/hy4-crash-$i.vram" | sed -n '1~4p;$p'
}

run "no ceilings, reserve 1400 (the disease)"  "GGML_CUDA_GRAPH_RECAPTURE_MAX=0 GGML_CUDA_GRAPH_CAPTURE_MAX=0" --reserve-mib 1400
run "ceilings 16/128, reserve 1400"            "GGML_CUDA_GRAPH_RECAPTURE_MAX=16 GGML_CUDA_GRAPH_CAPTURE_MAX=128" --reserve-mib 1400
run "ceilings 16/128, reserve 2400 (shipped)"  "GGML_CUDA_GRAPH_RECAPTURE_MAX=16 GGML_CUDA_GRAPH_CAPTURE_MAX=128" --reserve-mib 2400
echo "done: /tmp/hy4-crash-1..3.log"

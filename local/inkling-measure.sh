#!/usr/bin/env bash
# inkling-measure.sh - Inkling 2 Mini's quality and speed on this machine (docs/inkling/NOTES.md has the results).
#
#   ./local/inkling-measure.sh kl      PPL of Q8_0 (weights in RAM), UD-Q4_K_XL and IQ4_NL (card 0) on fixed documents, and
#                                      the KL of the 4-bit packs against Q8_0 (llama-perplexity, 2048-token chunks)
#   ./local/inkling-measure.sh speed   llama-bench of UD-Q4_K_XL on card 0: prefill 512/4096 and decode at depth 0/8K/32K
#
# The documents are this repository's own English docs at the commit (docs/MODELS.md, docs/ds4/V41_EXACT_FAST.md,
# docs/ds4/AMD_HIP_NOTES.md, HANDOVER.md), real prose with numbers and code, the same file for every run.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
B=$R/build-inkling/llama.cpp/build/bin
M=$R/models/inkling2-mini
OUT=$R/local/inkling-measure
mkdir -p "$OUT"
DOC=$OUT/docs.txt
cat "$R/docs/MODELS.md" "$R/docs/ds4/V41_EXACT_FAST.md" "$R/docs/ds4/AMD_HIP_NOTES.md" "$R/HANDOVER.md" > "$DOC"
case ${1:-} in
    kl)
        # the Q8_0 (27.8 GB) does not fit one card and llama.cpp cannot split this model's layers yet (its looped
        # layers share weights; ggml-org/llama.cpp#32751 is open): its weights stay in RAM (-ngl 0), card 0 computes
        # the batches
        [ -s "$OUT/q8-logits.bin" ] || HIP_VISIBLE_DEVICES=0 "$B/llama-perplexity" -m "$M/Inkling-2-Mini-Q8_0.gguf" \
            -f "$DOC" -c 2048 -b 2048 -ngl 0 -fa on --kl-divergence-base "$OUT/q8-logits.bin" > "$OUT/ppl-q8.txt" 2>&1
        [ -s "$OUT/q8-logits.bin" ] || { echo "no Q8_0 logits: $OUT/ppl-q8.txt" >&2; exit 1; }
        grep -a "Final estimate" "$OUT/ppl-q8.txt"
        for q in UD-Q4_K_XL IQ4_NL; do
            [ -s "$M/Inkling-2-Mini-$q.gguf" ] || continue
            HIP_VISIBLE_DEVICES=0 "$B/llama-perplexity" -m "$M/Inkling-2-Mini-$q.gguf" -c 2048 -b 2048 \
                -ngl 99 -fa on --kl-divergence-base "$OUT/q8-logits.bin" --kl-divergence > "$OUT/kl-$q.txt" 2>&1
            echo "== $q"; grep -aA40 "====== Perplexity statistics" "$OUT/kl-$q.txt" || tail -5 "$OUT/kl-$q.txt"
        done
        ;;
    speed)
        HIP_VISIBLE_DEVICES=0 "$B/llama-bench" -m "$M/Inkling-2-Mini-UD-Q4_K_XL.gguf" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
            -p 512,4096 -n 128 -d 0,8192,32768 -r 2 2>&1 | tee "$OUT/bench-ud-q4kxl.txt"
        ;;
    *) echo "usage: $0 kl|speed" >&2; exit 2;;
esac

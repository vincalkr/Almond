#!/usr/bin/env bash
# ab-skip-miss.sh - --skip-miss F (opt-in): in decode, a routed expert that is not in VRAM and weighs under F of the
# token's six weights is not computed (its share dropped) - exllamav3-rocm-moe's "tail skip" for CPU experts, measured
# here.  Arms against exact routing, each forced on the same reference stream (the first base run's) with the decode
# NLL (ALMOND_DEC_NLL): Promessi Sposi (300 tokens) and the 8K prompt (128 tokens), REPS (default 2) rounds; tok/s,
# the positions whose argmax differs and the forced text's decode PPL.  The chat server must be stopped.
# Results: local/sweep-skip-miss/summary.txt (+ nll.txt).
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=${OUT:-local/sweep-skip-miss}
if ss -ltn 2>/dev/null | grep -q ':8150 ' || pgrep -f '[a]lmond-engine.*--serve' >/dev/null; then
    echo "ab-skip-miss: a server is running: stop it first (./stop.sh)" >&2; exit 3
fi
mkdir -p $O; cp -n local/sweep-rco-rq/ref-sposi.ids local/sweep-rco-rq/ref-prompt8k.ids $O/ 2>/dev/null
export ARENA=80
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="ALMOND_DEC_NLL=1 ALMOND_NO_LARGEPAGES=1 ALMOND_PF_NO_DUMMY=1 ALMOND_VRAM_BG=1 ALMOND_DENSE_NOCONT=1 ALMOND_DENSE_ALIAS=1 ALMOND_DENSE_REQUANT=q6_K ALMOND_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 ALMOND_PEER_GIB=22 ALMOND_PEER_ADAPT=1 ALMOND_PEER_ADAPT_INFLIGHT=8 ALMOND_PEER_ADAPT_WIN=16 ALMOND_PEER_DISJOINT=1 ALMOND_ARENA_EVICT_PEER=1 ALMOND_SUM_ZEROCOPY=1"
run() {   # arm skip
    for p in "sposi 300" "prompt8k 128"; do
        set -- $p
        bash $S $O local/prompts/$1.i32 $2 $arm-r$r "$E" $R ${skip:+--skip-miss $skip} --ctx 12288
        printf '%-20s %-10s %s\n' "$arm-r$r" "$1" "$(grep '^decode nll:' $O/$1-$arm-r$r.out | tail -1)" | tee -a $O/nll.txt
    done
}
for r in $(seq 1 ${REPS:-2}); do
    for a in "base " "s05 0.05" "s10 0.10" "s15 0.15"; do
        set -- $a; arm=$1; skip=${2:-}; run
    done
done
echo ABSKIPDONE >> $O/summary.txt

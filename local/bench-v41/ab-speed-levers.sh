#!/usr/bin/env bash
# One-shot A/B of cheap DeepSeek levers against the chat configuration's flags, forced on the reference streams:
#   advise   ALMOND_ADVISE_AHEAD=2 (page-cache readahead of layer l+2's predicted file-tier experts)
#   arena90  --arena-gib 90 (fewer file-tier experts)
#   thr24    --threads 24 (the SMT siblings too)
cd /run/media/vincenzo/Dati/projects/other/almond
O=${1:-local/sweep-speed-levers}; mkdir -p $O; cp -n local/sweep-rco-rq/ref-sposi.ids local/sweep-rco-rq/ref-huge17k.ids $O/
pgrep -x almond-engine >/dev/null && { echo "an engine is running" >&2; exit 3; }
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="ALMOND_NO_LARGEPAGES=1 ALMOND_PF_NO_DUMMY=1 ALMOND_VRAM_BG=1 ALMOND_DENSE_NOCONT=1 ALMOND_DENSE_ALIAS=1 ALMOND_DENSE_REQUANT=q6_K ALMOND_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 ALMOND_PEER_GIB=22 ALMOND_PEER_ADAPT=1 ALMOND_PEER_ADAPT_INFLIGHT=8 ALMOND_PEER_ADAPT_WIN=16 ALMOND_PEER_DISJOINT=1 ALMOND_ARENA_EVICT_PEER=1 ALMOND_SUM_ZEROCOPY=1 ALMOND_DEC_PROF=1 DS4_TIER_PROF=1"
run() {   # tag extra-env arena threads [extra flags]
    ARENA=$3 THREADS=$4 bash local/bench-v41/sweep.sh $O local/prompts/huge17k.i32 128 $1-r$r "$E $2" $R --ctx 24576 "${@:5}"
    ARENA=$3 THREADS=$4 bash local/bench-v41/sweep.sh $O local/prompts/sposi.i32 300 $1-r$r "$E $2" $R "${@:5}"
}
for r in $(seq 1 ${REPS:-2}); do
    run base "" 80 12
    run advise "ALMOND_ADVISE_AHEAD=2" 80 12
    run arena88 "" 88 12
    run thr24 "" 80 24
    run skip21 "" 80 12 --skip-miss 0.15 --skip-miss-from 21
done
echo ABDONE >> $O/summary.txt

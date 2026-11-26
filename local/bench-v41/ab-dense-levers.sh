#!/usr/bin/env bash
# ab-dense-levers.sh - opt-in decode levers of the dense half against the chat configuration's flags (two cards, arena
# 80 GiB, zero-copy sum, q6_K requant, ALMOND_HC_HEAD=1 ALMOND_ROUTER_FUSE=1), arms alternated, REPS (default 3)
# rounds; Promessi Sposi (300 tokens) and the 17K prompt (128 tokens), forced on local/sweep-rco-rq's reference streams.
# ARMS: "name=ENV ..." words separated by ';' (default: base and ALMOND_SHEXP_LATE=1).
# The chat server must be stopped.  Results in $OUT/summary.txt (default local/sweep-dense-lev).
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=${OUT:-local/sweep-dense-lev}
if ss -ltn 2>/dev/null | grep -q ':8150 '; then echo "ab-dense-levers: the chat server is running: stop it first (./stop.sh)" >&2; exit 3; fi
mkdir -p $O; cp -n local/sweep-rco-rq/ref-sposi.ids local/sweep-rco-rq/ref-huge17k.ids $O/
export ARENA=80
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="ALMOND_NO_LARGEPAGES=1 ALMOND_PF_NO_DUMMY=1 ALMOND_VRAM_BG=1 ALMOND_DENSE_NOCONT=1 ALMOND_DENSE_ALIAS=1 ALMOND_DENSE_REQUANT=q6_K ALMOND_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 ALMOND_PEER_GIB=22 ALMOND_PEER_ADAPT=1 ALMOND_PEER_ADAPT_INFLIGHT=8 ALMOND_PEER_ADAPT_WIN=16 ALMOND_PEER_DISJOINT=1 ALMOND_ARENA_EVICT_PEER=1 ALMOND_SUM_ZEROCOPY=1 ALMOND_HC_HEAD=1 ALMOND_ROUTER_FUSE=1 ALMOND_ELASTIC_COLS=1"
IFS=';' read -ra A <<< "${ARMS:-base=;late=ALMOND_SHEXP_LATE=1}"
for r in $(seq 1 ${REPS:-3}); do
    for arm in "${A[@]}"; do
        n=${arm%%=*}; v=${arm#*=}
        bash $S $O local/prompts/huge17k.i32 128 $n-r$r "$E $v" $R --ctx 24576
        bash $S $O local/prompts/sposi.i32 300 $n-r$r "$E $v" $R
    done
done
echo ABDONE >> $O/summary.txt

#!/usr/bin/env bash
# ab-ds41e.sh - Almond's engine (the trunk behind MoeTrunk) against ds41e's (../ds41e, the same code before the
# interface), with ab-hip-spin.sh's chat flags, arms alternated, REPS (default 3) rounds; Promessi Sposi (300 tokens)
# and the 17K prompt (128 tokens, ctx 24576), forced on local/sweep-rco-rq's reference streams ("diff=" counts the
# tokens whose argmax differs).  The chat servers must be stopped.  Results in local/sweep-ab-ds41e/summary.txt.
cd "$(dirname "$0")/../.."
D=../ds41e
O=$PWD/${OUT:-local/sweep-ab-ds41e}
if ss -ltn 2>/dev/null | grep -q ':8150 ' || pgrep -f '[d]s41_generate.*--serve' >/dev/null; then
    echo "ab-ds41e: a chat server is running: stop it first (./stop.sh)" >&2; exit 3
fi
mkdir -p $O; cp -n $D/local/sweep-rco-rq/ref-sposi.ids $D/local/sweep-rco-rq/ref-huge17k.ids $O/
export ARENA=80 RANK=local/rank.bin
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="NO_LARGEPAGES=1 PF_NO_DUMMY=1 VRAM_BG=1 DENSE_NOCONT=1 DENSE_ALIAS=1 DENSE_REQUANT=q6_K PRESTAGE_GIB=1.5 PEER_GIB=22 PEER_ADAPT=1 PEER_ADAPT_INFLIGHT=8 PEER_ADAPT_WIN=16 PEER_DISJOINT=1 ARENA_EVICT_PEER=1 SUM_ZEROCOPY=1 ELASTIC_COLS=1"
envs() { echo "HIP_VISIBLE_DEVICES=0,1 $(for v in $E; do printf '%s_%s ' "$1" "$v"; done)"; }   # prefix
run() {   # arm dir prefix
    bash $2/local/bench-v41/sweep.sh $O local/prompts/huge17k.i32 128 $1-r$r "$(envs $3)" $R --ctx 24576
    bash $2/local/bench-v41/sweep.sh $O local/prompts/sposi.i32 300 $1-r$r "$(envs $3)" $R
}
for r in $(seq 1 ${REPS:-3}); do
    run ds41e $D STRATA
    run almond . ALMOND
done
echo ABDS41EDONE >> $O/summary.txt

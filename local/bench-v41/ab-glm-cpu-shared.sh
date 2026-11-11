#!/usr/bin/env bash
# A/B: GLM's CPU lane as one shared pool (ALMOND_GLM_CPU_SHARED=1) against the default (a 6-thread pool per card).
# decode tok/s through the server (local/glm-ref/speed.py, 2 rounds of 3 prompts), arms alternated, 2 rounds.
cd /run/media/vincenzo/Dati/projects/other/almond
OUT=${1:-local/sweep-speed-levers}; mkdir -p $OUT
pgrep -x almond-engine >/dev/null && { echo "an engine is running" >&2; exit 3; }
for r in 1 2; do
for arm in "base" "shared 1"; do
    set -- $arm
    if [ -n "${2:-}" ]; then export ALMOND_GLM_CPU_SHARED=$2; else unset ALMOND_GLM_CPU_SHARED; fi
    S=$OUT/$1-r$r.log
    echo "== $1 r$r $(date +%T)" | tee -a $OUT/summary.txt
    (./almond serve --model glm --port 8163 > $S 2>&1 &)
    for i in $(seq 200); do grep -q "^ready" $S && break; sleep 2; done
    grep -h "CPU lane:" local/serve-glm-engine.log | tail -2 | tee -a $OUT/summary.txt
    .venv/bin/python local/glm-ref/speed.py 8163 2 | tee -a $OUT/summary.txt
    grep -h "glm stat: decode" local/serve-glm-engine.log | tail -3 | cut -c1-220 | tee -a $OUT/summary.txt
    P=$(pgrep -f "[s]erve_glm.py"); kill -TERM $P; while pgrep -x almond-engine >/dev/null; do sleep 2; done
    sleep 5
done; done
echo DONE >> $OUT/summary.txt

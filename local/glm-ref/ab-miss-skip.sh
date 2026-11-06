#!/usr/bin/env bash
# ab-miss-skip.sh - GLM's ALMOND_GLM_MISS_SKIP_W (opt-in, changes the answers): a routed expert not in VRAM whose share
# of the route's weights is under W is left out, on layers from ALMOND_GLM_MISS_SKIP_FROM on.  Per arm: decode tok/s
# through the server (local/glm-ref/speed.py, 2 rounds of 3 prompts) and the PPL of 16 windows of 512 tokens of
# local/glm-ref/long-docs.txt (logp.py, 4,080 tokens scored).  No server may be running.  Output: stdout.
cd "$(dirname "$0")/../.."
pgrep -f '[a]lmond-engine' >/dev/null && { echo "ab-miss-skip: an engine is running" >&2; exit 3; }
S=${TMPDIR:-/tmp}/ab-miss-skip.log
for arm in "base" "w12f23 0.12 23" "w16f23 0.16 23" "w12f0 0.12 0"; do
    set -- $arm
    if [ -n "${2:-}" ]; then export ALMOND_GLM_MISS_SKIP_W=$2 ALMOND_GLM_MISS_SKIP_FROM=$3
    else unset ALMOND_GLM_MISS_SKIP_W ALMOND_GLM_MISS_SKIP_FROM; fi
    echo "== $1"
    (./almond serve --model glm --port 8163 > $S 2>&1 &)
    for i in $(seq 150); do grep -q "^ready" $S && break; sleep 2; done
    grep -m1 "light misses left out" local/serve-glm-engine.log >/dev/null 2>&1
    .venv/bin/python local/glm-ref/speed.py 8163 2
    P=$(pgrep -f "[s]erve_glm.py"); kill -TERM $P; while pgrep -f "[a]lmond-engine" >/dev/null; do sleep 2; done
    HIP_VISIBLE_DEVICES=0,1 .venv/bin/python local/glm-ref/logp.py local/glm-ref/long-docs.llama-ids.txt 512 1 local/glm-local.json 16 2>&1 | tail -1
done

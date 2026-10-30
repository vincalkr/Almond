#!/bin/sh
# DeepSeek-V4.1-Flash-1017 on RX 7900 XTX (gfx1100, ROCm/HIP).
#   ./run.sh                 two cards (0 and 1, card 1 as the expert tier), images on when the encoder is there
#   ./run.sh --gpu 1         one card (0 or 1)
#   ./run.sh --gpu 0,1       two cards, as the default
#   ./run.sh --router        every model in local/router.json behind one address (tools/router/almond_router.py): the
#                            request's "model" loads it (stopping the other); /v1/models lists them with their
#                            thinking levels.  The chat: ./almond (starts the router itself when nothing answers)
#   ./run.sh --model glm     GLM-5.3-Flash (GSQ-RCO 3.0bit) on Almond's engine (src/glm, build-hip-fused/almond-engine),
#                            the layers split over both cards: local/glm-local.json, tools/glm/serve/serve_glm.py
# --gpu picks the configuration: local/ds41-local-vision.json for two cards, local/ds41-local-1gpu-vision.json for
# one; without the vision encoder in models/vision-1017 (or the image routing bias local/rco-iq2xxs-meta/vl-bias.gguf)
# the text-only ones (local/ds41-local.json, local/ds41-local-1gpu.json).  The encoder is loaded on card 0 only for a
# request with images and freed after it; text requests never load it.  --config FILE uses that file instead.  Other arguments go to the server, for
# example:  ./run.sh --port 8151.  The settings and the measured numbers: docs/ds4/.
# The repository root is wherever this script lives, so the launcher works from a clone anywhere.
R=$(cd "$(dirname "$0")" && pwd) || exit 1
cd "$R" || exit 1

# ALMOND_HOME must be this repository: the shim imports serve/, tools/ and almond_tokenizer from it, and its own
# default is this directory anyway - set here so the launcher works from any working directory.
ALMOND_HOME="$R"
export ALMOND_HOME

# The server needs the standard library plus pillow, psutil, regex and jinja2 (almond_tokenizer imports regex;
# serve/server.py imports PIL, psutil and jinja2).  This repository's own venv, a venv/ next to it, or any python3
# that has them - no other checkout is involved; ALMOND_PY=/path/to/python forces one.
PY=""
for cand in "${ALMOND_PY:-}" "$R/.venv/bin/python" "$(command -v python3)" "$R/venv/bin/python"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    if "$cand" -c 'import PIL, psutil, regex, jinja2' 2>/dev/null; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
    echo "run.sh: no python with pillow, psutil, regex and jinja2 found." >&2
    echo "  make one:        python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2" >&2
    echo "  or point at one: ALMOND_PY=/path/to/python ./run.sh" >&2
    exit 1
fi
echo "run.sh: ALMOND_HOME=$ALMOND_HOME, python $PY"

# --gpu and --config are taken out of the arguments; the rest go to the server as they are.
GPU=""; CFG=""; MODEL=ds41; ROUTER=""
n=$#
while [ "$n" -gt 0 ]; do
    a=$1; shift; n=$((n - 1))
    case $a in
        --gpu|--config|--model)
            [ "$n" -gt 0 ] || { echo "run.sh: $a needs a value" >&2; exit 1; }
            case $a in --gpu) GPU=$1 ;; --config) CFG=$1 ;; *) MODEL=$1 ;; esac
            shift; n=$((n - 1)); continue ;;
        --model=*) MODEL=${a#--model=}; continue ;;
        --router) ROUTER=1; continue ;;
        --gpu=*) GPU=${a#--gpu=}; continue ;;
        --config=*) CFG=${a#--config=}; continue ;;
    esac
    set -- "$@" "$a"
done
case ${GPU:-0,1} in
    *[!0-9,]*|,*|*,|*,,*) echo "run.sh: --gpu takes card numbers: 0, 1 or 0,1 (got '$GPU')" >&2; exit 1 ;;
esac
# --describe (the router's listing: the model's name and thinking levels, nothing loaded) needs no power-tune
for x in "$@"; do [ "$x" = --describe ] && DS41_POWER_TUNE=0; done
if [ -n "$ROUTER" ]; then
    # the router starts each model through this script (power-tune with it), so it needs nothing else here
    exec "$PY" "$R/tools/router/almond_router.py" "$@"
fi
SHIM="$R/tools/v41/serve/serve_ds41.py"
case $MODEL in
    ds41|deepseek) ;;
    glm)
        SHIM="$R/tools/glm/serve/serve_glm.py"
        if [ -z "$CFG" ]; then
            CFG="$R/local/glm-local.json"
            # with the image encoder (local/fetch.sh glm-vision) and almond-vision built: images on, loaded per picture
            [ -s "$R/models/glm53-vision/mmproj-GLM-5.3-Flash-F16.gguf" ] && [ -x "$R/build-vision/build/bin/almond-vision" ] &&
                CFG="$R/local/glm-local-vision.json"
        fi
        [ -x "$R/build-hip-fused/almond-engine" ] || { echo "run.sh: build-hip-fused/almond-engine is not built (./build.sh builds it)" >&2; exit 1; }
        # the expert profile the engine starts its VRAM pools from and saves what it learns into: project-maya's
        # until this machine has its own
        [ -s "$R/local/glm-expert-profile.bin" ] || cp "$R/tools/glm/data/expert-profile-glm.bin" "$R/local/glm-expert-profile.bin" ;;
    *) echo "run.sh: --model takes ds41 or glm (got '$MODEL')" >&2; exit 1 ;;
esac
cards=$(echo "${GPU:-0,1}" | tr ',' '\n' | wc -l)
if [ -z "$CFG" ]; then
    [ -n "$GPU" ] || GPU=0,1
    case $cards in
        1) base=ds41-local-1gpu ;;
        2) base=ds41-local ;;
        *) echo "run.sh: one or two cards (--gpu 0, --gpu 1, --gpu 0,1), not '$GPU'" >&2; exit 1 ;;
    esac
    # the vision configurations need the tower and the image routing bias (the pack has none: DS41_VL_BIAS adds
    # local/rco-iq2xxs-meta/vl-bias.gguf, the checkpoint's bias_vl)
    if [ -s "$R/models/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf" ] && [ -s "$R/local/rco-iq2xxs-meta/vl-bias.gguf" ] &&
       [ -s "$R/local/$base-vision.json" ]; then
        CFG="$R/local/$base-vision.json"
    else
        CFG="$R/local/$base.json"
        echo "run.sh: no vision encoder + vision configuration for $base: text only"
    fi
fi
echo "run.sh: config $CFG, cards ${GPU:-as the config says}"
set -- --config "$CFG" ${GPU:+--gpu "$GPU"} "$@"

# The model files the configuration names (-m, --engram, its tokenizer) must be there.
missing=$("$PY" - "$CFG" "$R" <<'EOF'
import json, os, sys
cfg, root = sys.argv[1], sys.argv[2]
c = json.load(open(cfg))
a = c.get("args", [])
model = a[a.index("-m") + 1] if "-m" in a else ""
pack = a[a.index("--glm-pack") + 1] if "--glm-pack" in a else ""
engram = a[a.index("--engram") + 1].split(":") if "--engram" in a else []
meta = (c.get("env") or {}).get("DS41_META")          # the geometry overlay a foreign pack reads
mtp = (c.get("env") or {}).get("ALMOND_GLM_MTP_GGUF")   # GLM's draft block (local/glm-mtp-extract.py makes it)
need = ([model] if model else []) + ([pack] if pack else []) + engram + ([c["tokenizer"]] if c.get("tokenizer") else []) + ([meta] if meta else []) + ([mtp] if mtp else [])
def whole(p):
    q = p if os.path.isabs(p) else os.path.join(root, p)
    return os.path.exists(q) and not (os.path.isfile(q) and os.path.getsize(q) == 0)
# An --engram path this box does not have is not fatal when the pack's own Engram GGUF sits beside the pack: the
# engine falls back to that and prints that it does (ds41_engram.cpp). Everything else must be there.
soft = [p for p in engram if not whole(p)]
sib = os.path.join(os.path.dirname(os.path.abspath(os.path.join(root, model))),
                   "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf") if model else ""
fatal = [p for p in need if p not in soft and not whole(p)]
if soft and not (sib and whole(sib)): fatal += soft
for p in fatal: print(p)
if soft and sib and whole(sib):
    sys.stderr.write("run.sh: %s is not here - the engine reads the pack's own Engram table %s\n"
                     % (", ".join(soft), sib))
EOF
)
if [ -n "$missing" ]; then
    echo "run.sh: missing for $CFG:" >&2
    echo "$missing" | sed 's/^/  /' >&2
    echo "fetch the model first - see the README, '1. Get the model files'." >&2
    exit 1
fi

# local/power-tune.sh for as long as the server runs (+11% short answers, +21% at ~25K context, measured 2026-10-21):
# only when local/install-power-tune.sh has been run once with sudo (a root-owned copy and a sudo rule for it alone);
# otherwise nothing changes.  DS41_POWER_TUNE=0 leaves the machine's settings alone.
PT=/usr/local/sbin/ds41e-power-tune
tuned=0
if [ "${DS41_POWER_TUNE:-1}" != 0 ] && [ -x "$PT" ] && sudo -n "$PT" on >/dev/null 2>&1; then
    tuned=1
    echo "run.sh: power-tune on (NVMe APST off, CPU C3 off, performance governor); off again when the server ends"
elif [ "${DS41_POWER_TUNE:-1}" != 0 ] && [ ! -x "$PT" ]; then
    echo "run.sh: power-tune not installed (sudo local/install-power-tune.sh switches it on and off with the server)"
fi

if [ "$tuned" = 0 ]; then
    exec "$PY" "$SHIM" \
         "--engine" "almond" \
         "--port" "8150" "$@"
fi
# The server runs in the foreground (Ctrl+C reaches it as before); this shell waits and switches power-tune off
# however the server ends.  A no-op INT/TERM/HUP trap keeps the shell alive until then (an ignored signal would be
# inherited by the server).
trap ':' INT TERM HUP
rc=0
"$PY" "$SHIM" \
    "--engine" "almond" \
    "--port" "8150" "$@" || rc=$?
# Another ds41e server may have started meanwhile (a restart, an A/B that stops one and starts the next): it switched
# power-tune on and still needs it, so it stays on until the last server ends.
if pgrep -f "[t]ools/(v41/serve/serve_ds41|glm/serve/serve_glm).py" >/dev/null 2>&1; then
    echo "run.sh: another Almond server is running: power-tune stays on"
else
    sudo -n "$PT" off >/dev/null 2>&1 && echo "run.sh: power-tune off"
fi
exit "$rc"

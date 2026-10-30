#!/usr/bin/env bash
# setup.sh - a new installation, or a check of an existing one, in order:
#
#   1. the tools the build and the downloads need (ROCm, cmake, a C++ compiler, python3, aria2c, curl)
#   2. .venv from requirements.txt
#   3. the models: local/fetch.sh shows what is in models/ and asks what to download (and builds what derives from
#      the downloads: DeepSeek's geometry overlay and pack map, its FP8 Engram rows, GLM's pack, its MTP block)
#   4. ./build.sh: DeepSeek's engine, GLM's engine and GLM's image encoder
#   5. power-tune (optional, sudo once): NVMe and CPU idle states off while a server runs (local/install-power-tune.sh)
#   6. each model's context, written into its configurations
#
# Each step says what it finds and asks before it changes anything; a step with nothing to do is skipped.
#   ./setup.sh            all of them
#   ./setup.sh --check    only say what is there and what is missing
set -euo pipefail
R=$(cd "$(dirname "$0")" && pwd)
cd "$R"
CHECK=0; [ "${1:-}" = --check ] && CHECK=1
ROCM=${ROCM:-/opt/rocm}
ok()   { printf '  ok      %s\n' "$*"; }
miss() { printf '  MISSING %s\n' "$*"; }
ask()  { [ "$CHECK" = 1 ] && return 1; local a; read -r -p "$1 [y/N] " a; [[ $a =~ ^[YySs] ]]; }

echo "== 1. tools"
need=0
for t in cmake python3 curl aria2c git; do
    if command -v $t >/dev/null; then ok "$t"; else miss "$t"; need=1; fi
done
if [ -x "$ROCM/lib/llvm/bin/clang++" ]; then ok "ROCm ($ROCM)"; else miss "ROCm's clang++ in $ROCM (ROCM=... for another place)"; need=1; fi
[ $need = 0 ] || echo "  install what is missing (your distribution's packages; ROCm from AMD) and run setup.sh again"

echo "== 2. Python environment (.venv)"
if [ -x .venv/bin/python ] && .venv/bin/python -c "import numpy, yaml" 2>/dev/null; then ok ".venv"
else
    miss ".venv with requirements.txt"
    if ask "  create it now?"; then python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt && ok ".venv"; fi
fi

echo "== 3. models (models/)"
./local/fetch.sh --list
[ "$CHECK" = 1 ] || ./local/fetch.sh

echo "== 4. build"
built=1
for b in build-hip-fused/ds41_generate build-maya-hip/maya build-maya-vision/build/bin/maya-vision; do
    if [ -x "$b" ]; then ok "$b"; else miss "$b"; built=0; fi
done
if [ $built = 0 ] || [ "$CHECK" = 0 ]; then
    if ask "  run ./build.sh now (the first time ~30-60 minutes)?"; then ./build.sh; fi
fi

echo "== 5. power-tune (+11% short answers, +21% at ~25K context on DeepSeek, measured 2026-10-21)"
if [ -x /usr/local/sbin/ds41e-power-tune ]; then ok "installed: run.sh switches it on with a server and off after"
else
    miss "power-tune (optional)"
    if ask "  install it (asks for your sudo password once)?"; then sudo local/install-power-tune.sh; fi
fi

echo "== 6. context"
ctx_of() {   # the context a configuration sets (the flag after which it is)
    python3 -c "import json,sys;a=json.load(open('$1'))['args'];f='$2';print(a[a.index(f)+1] if f in a else '')"
}
set_ctx() {  # set_ctx FLAG VALUE CONFIG...
    local flag=$1 v=$2; shift 2
    for c in "$@"; do
        [ -f "$c" ] || continue
        python3 - "$c" "$flag" "$v" <<'P'
import json, sys
p, flag, v = sys.argv[1:]
c = json.load(open(p)); a = c["args"]
if flag in a: a[a.index(flag) + 1] = v
else: a += [flag, v]
open(p, "w").write(json.dumps(c, indent=2) + "\n")
P
        ok "$c: $flag $v"
    done
}
ram=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
echo "  this machine: ${ram} GB of RAM; $(rocm-smi --showmeminfo vram 2>/dev/null | grep -c 'Total Memory (B)' || echo '?') GPU(s)"
DS=(local/ds41-local.json local/ds41-local-1gpu.json local/ds41-local-vision.json local/ds41-local-1gpu-vision.json)
GL=(local/glm-local.json local/glm-local-vision.json)
cur=$(ctx_of local/ds41-local.json --ctx)
echo "  DeepSeek-V4.1-Flash: $cur tokens now (trained for 1,048,576; KV in q8: ~the context costs little VRAM)"
if [ "$CHECK" = 0 ]; then
    read -r -p "  new context for DeepSeek (empty = keep $cur): " v
    [ -n "$v" ] && set_ctx --ctx "$v" "${DS[@]}"
fi
cur=$(ctx_of local/glm-local.json --max-context)
echo "  GLM-5.3-Flash: $cur tokens now (the API's output cap is 131,072)"
if [ "$CHECK" = 0 ]; then
    read -r -p "  new context for GLM (empty = keep $cur): " v
    [ -n "$v" ] && set_ctx --max-context "$v" "${GL[@]}"
fi

echo
echo "done. ./run.sh starts DeepSeek, ./run.sh --model glm GLM, ./run.sh --router both on demand; ./almond is the CLI."

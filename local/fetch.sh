#!/usr/bin/env bash
# fetch.sh [ITEM ...] - what Almond can download, what is already in models/, and the download of the items chosen.
#
#   ./local/fetch.sh            the list with each item's state, then asks which to download (numbers or names)
#   ./local/fetch.sh glm glm-mtp   those items, no question
#   ./local/fetch.sh --list     the list only
#
# Every file goes to models/ with aria2c (CONN=4 connections, resumable); a file counts as present when its size is
# the Hugging Face API's.  DeepSeek's pack and its FP8 Engram rows keep their own scripts (fetch-rco-1017.sh,
# fetch-model.sh), which this one calls.  Env: CONN, HF_TOKEN, DRY_RUN=1 (print the plan, change nothing).
set -euo pipefail
export LC_ALL=C LANG=C
R=$(cd "$(dirname "$0")/.." && pwd)
M=$R/models
PY=${ALMOND_PY:-$R/.venv/bin/python}
CONN=${CONN:-4}
DRY_RUN=${DRY_RUN:-0}
TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

GLM_REPO=pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF
MAYA_REPO=peasantsmith/GLM-5.3-Flash-Maya-GGUF
DS_SHARDS=$(for i in 1 2 3 4; do echo "$M/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000$i-of-00004.gguf"; done)

ITEMS=(deepseek deepseek-vision deepseek-engram-fp8 glm glm-mtp glm-vision)
desc() {
    case $1 in
        deepseek)            echo "DeepSeek-V4.1-Flash-1017, GSQ-RCO IQ2_XXS pack + Engram IQ4_NL (ISTA-DASLab)   ~249 GB";;
        deepseek-vision)     echo "DeepSeek image encoder (built from the 1017 checkpoint's vision tower)          0.9 GB";;
        deepseek-engram-fp8) echo "DeepSeek Engram, the checkpoint's FP8 rows (optional; ~570 GB while extracting) 203 GB";;
        glm)                 echo "GLM-5.3-Flash GSQ-RCO 3.0bit (pfeifferj) + its pack                           117 GB";;
        glm-mtp)             echo "GLM MTP draft block (from Maya-M's shard 3, 20.5 GB download, kept 3.6 GB)      3.6 GB";;
        glm-vision)          echo "GLM image encoder (mmproj F16 + vocab, from project-maya)                      1.1 GB";;
    esac
}
files() {   # the files that make the item present
    case $1 in
        deepseek)            echo "$DS_SHARDS"; echo "$M/rco-iq2xxs/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";;
        deepseek-vision)     echo "$M/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf";;
        deepseek-engram-fp8) echo "$M/engram-1017/layers1/engram-L1.safetensors"; echo "$M/engram-1017/layers14/engram-L14.safetensors";;
        glm)                 echo "$M/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"; echo "$M/glm53-gsq-rco/pack";;
        glm-mtp)             echo "$M/glm53-gsq-rco/GLM-5.3-Flash-Maya-M-NextN.gguf";;
        glm-vision)          echo "$M/glm53-vision/mmproj-GLM-5.3-Flash-F16.gguf"; echo "$M/glm53-vision/GLM-5.3-Flash-vocab.gguf";;
    esac
}
state() {   # present / partial / missing (by the files being there; sizes are checked at download)
    local n=0 have=0 f
    while read -r f; do n=$((n+1)); [ -e "$f" ] && [ -n "$(ls -A "$f" 2>/dev/null)" ] && have=$((have+1)); done < <(files "$1")
    if [ $have = $n ]; then echo present; elif [ $have = 0 ]; then echo missing; else echo partial; fi
}
size_of() {   # the API's size of REPO's PATH
    local d; d=$(dirname "$2"); [ "$d" = . ] && d=
    curl -sf "${AUTH[@]}" "https://huggingface.co/api/models/$1/tree/main${d:+/$d}" |
        python3 -c "import json,sys
for x in json.load(sys.stdin):
    if x['path']=='$2': print(x['size']); break"
}
get() {   # get REPO PATH DEST: aria2c, resumable, size-checked
    local want cur
    want=$(size_of "$1" "$2"); [ -n "$want" ] || { echo "fetch: the API has no $1/$2" >&2; exit 1; }
    cur=$(stat -c%s "$3" 2>/dev/null || echo 0)
    if [ "$cur" = "$want" ]; then echo "have   $(basename "$3")"; return; fi
    echo "get    $1/$2 ($((want / 1000000)) MB, $CONN connections)"
    [ "$DRY_RUN" = 1 ] && return
    mkdir -p "$(dirname "$3")"
    aria2c -c -x"$CONN" -s"$CONN" -k4M --file-allocation=none --console-log-level=warn --summary-interval=60 \
           --retry-wait=5 -m8 ${TOKEN:+--header="Authorization: Bearer $TOKEN"} \
           -d "$(dirname "$3")" -o "$(basename "$3")" "https://huggingface.co/$1/resolve/main/$2?download=true"
    [ "$(stat -c%s "$3")" = "$want" ] || { echo "fetch: $3 is short - run it again to resume" >&2; exit 1; }
}
fetch() {
    echo "== $1"
    case $1 in
        deepseek)            DRY_RUN=$DRY_RUN VISION=0 CONN=$CONN "$R/local/fetch-rco-1017.sh";;
        deepseek-vision)     [ "$DRY_RUN" = 1 ] || { mkdir -p "$M/vision-1017"
                             "$PY" "$R/tools/v41/vision/make_vision_gguf.py" --out "$M/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf"; };;
        deepseek-engram-fp8) DRY_RUN=$DRY_RUN VISION=0 CONN=$CONN "$R/local/fetch-model.sh";;
        glm)                 get $GLM_REPO GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf "$M/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"
                             [ "$DRY_RUN" = 1 ] || [ -n "$(ls -A "$M/glm53-gsq-rco/pack" 2>/dev/null)" ] ||
                                 sh "$R/local/glm-pack.sh";;
        glm-mtp)             [ -s "$M/glm53-gsq-rco/GLM-5.3-Flash-Maya-M-NextN.gguf" ] && { echo "have   GLM-5.3-Flash-Maya-M-NextN.gguf"; return; }
                             local src=$M/glm53-gsq-rco/src/GLM-5.3-Flash-Maya-M-IQ2_S-00003-of-00003.gguf
                             get $MAYA_REPO Maya-M/$(basename "$src") "$src"
                             [ "$DRY_RUN" = 1 ] && return
                             GGUF_ARCH=glm5-next ALMOND_GGUF_PY="$R/third_party/llama.cpp/gguf-py" "$PY" "$R/local/glm-mtp-extract.py" \
                                 "$src" "$M/glm53-gsq-rco/GLM-5.3-Flash-Maya-M-NextN.gguf"
                             gio trash "$M/glm53-gsq-rco/src" && echo "the 20.5 GB source went to the trash";;
        glm-vision)          get $MAYA_REPO vision/mmproj-GLM-5.3-Flash-F16.gguf "$M/glm53-vision/mmproj-GLM-5.3-Flash-F16.gguf"
                             get $MAYA_REPO vision/GLM-5.3-Flash-vocab.gguf "$M/glm53-vision/GLM-5.3-Flash-vocab.gguf";;
        *) echo "fetch: no item '$1' (${ITEMS[*]})" >&2; exit 2;;
    esac
}
show() {
    echo "Models in $M (free: $(df -h --output=avail "$M/" | tail -1 | tr -d ' ')):"
    local i=1 it st
    for it in "${ITEMS[@]}"; do
        st=$(state "$it")
        printf "  %d) %-20s %-8s %s\n" $i "$it" "$st" "$(desc "$it")"
        i=$((i+1))
    done
}

[ "${1:-}" = --list ] && { show; exit 0; }
if [ $# = 0 ]; then
    show
    echo
    read -r -p "Which to download? (numbers or names, separated by spaces; empty = none): " -a pick
    set -- "${pick[@]}"
    [ $# = 0 ] && { echo "nothing to do"; exit 0; }
fi
for p in "$@"; do
    [[ $p =~ ^[0-9]+$ ]] && { [ "$p" -ge 1 ] && [ "$p" -le ${#ITEMS[@]} ] || { echo "fetch: no item $p" >&2; exit 2; }; p=${ITEMS[$((p-1))]}; }
    fetch "$p"
done
echo; show

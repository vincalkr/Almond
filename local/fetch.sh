#!/usr/bin/env bash
# fetch.sh [ITEM ...] - what Almond can download, what is already in models/, and the download of the items chosen.
#
#   ./local/fetch.sh            the list with each item's state, then asks which to download (numbers or names)
#   ./local/fetch.sh glm glm-mtp   those items, no question
#   ./local/fetch.sh qwen qwen-mtp  Qwen3.8-Flash-Next and its draft block
#   ./local/fetch.sh qwen27-q8   the dense Qwen3.8-27B at Q8_0 (the quantization is chosen here: qwen27 or qwen27-q8)
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
GLM_SRC_REPO=peasantsmith/GLM-5.3-Flash-Maya-GGUF
QWEN_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
QWEN_SHARDS=$(for i in 1 2; do echo "$M/qwen38-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000$i-of-00002.gguf"; done)
Q27_REPO=ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
Q27_Q8_REPO=unsloth/Qwen3.8-27B-GGUF
Q27=$M/qwen38-27b
DS_SHARDS=$(for i in 1 2 3 4; do echo "$M/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000$i-of-00004.gguf"; done)

ITEMS=(deepseek deepseek-vision deepseek-engram-fp8 glm glm-3.0 glm-mtp glm-vision qwen qwen-mtp qwen-vision
       qwen27 qwen27-q8 qwen27-dspine qwen27-vision)
desc() {
    case $1 in
        deepseek)            echo "DeepSeek-V4.1-Flash-1017, GSQ-RCO IQ2_XXS pack + Engram IQ4_NL (ISTA-DASLab)   ~249 GB";;
        deepseek-vision)     echo "DeepSeek image encoder (built from the 1017 checkpoint's vision tower)          0.9 GB";;
        deepseek-engram-fp8) echo "DeepSeek Engram, the checkpoint's FP8 rows (optional; ~570 GB while extracting) 203 GB";;
        glm)                 echo "GLM-5.3-Flash GSQ-RCO 3.5bit (pfeifferj) + its pack                           137 GB";;
        glm-3.0)             echo "GLM-5.3-Flash GSQ-RCO 3.0bit + its pack: +16% decode, PPL 14.0 vs 13.4         118 GB";;
        glm-mtp)             echo "GLM MTP draft block (from an IQ2_S quant's shard 3, 20.5 GB download, kept 3.6 GB)";;
        glm-vision)          echo "GLM image encoder (mmproj F16 + vocab)                                               1.1 GB";;
        qwen)                echo "Qwen3.8-Flash-Next GSQ-RCO IQ3_S (ISTA-DASLab) + its pack                       85 GB";;
        qwen-mtp)            echo "Qwen MTP draft block (the BF16 checkpoint's mtp.* tensors, 5.2 GB read, kept 0.8 GB)";;
        qwen-vision)         echo "Qwen image encoder (mmproj BF16)                                                     0.9 GB";;
        qwen27)              echo "Qwen3.8-27B dense, GSQ-RCO IQ3_S + MTP block (ISTA-DASLab), one card           12.1 GB";;
        qwen27-q8)           echo "Qwen3.8-27B dense, Q8_0 + MTP block (unsloth), two cards                       29.0 GB";;
        qwen27-dspine)       echo "Qwen3.8-27B DSpine drafter Q4_K_M (aj9o9 of tencent): drafts inside code         1.3 GB";;
        qwen27-vision)       echo "Qwen3.8-27B image encoder (mmproj BF16)                                          0.9 GB";;
    esac
}
files() {   # the files that make the item present
    case $1 in
        deepseek)            echo "$DS_SHARDS"; echo "$M/rco-iq2xxs/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";;
        deepseek-vision)     echo "$M/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf";;
        deepseek-engram-fp8) echo "$M/engram-1017/layers1/engram-L1.safetensors"; echo "$M/engram-1017/layers14/engram-L14.safetensors";;
        glm)                 echo "$M/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf"; echo "$M/glm53-gsq-rco/pack";;
        glm-3.0)             echo "$M/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"; echo "$M/glm53-gsq-rco/pack-3.0";;
        glm-mtp)             echo "$M/glm53-gsq-rco/GLM-5.3-Flash-NextN.gguf";;
        glm-vision)          echo "$M/glm53-vision/mmproj-GLM-5.3-Flash-F16.gguf"; echo "$M/glm53-vision/GLM-5.3-Flash-vocab.gguf";;
        qwen)                echo "$QWEN_SHARDS"; echo "$M/qwen38-iq3s-pack/native_experts.txt";;
        qwen-mtp)            echo "$M/qwen38-mtp/experts.bin"; echo "$M/qwen38-mtp/draft_vocab.bin";;
        qwen-vision)         echo "$M/qwen38-iq3s/mmproj-Qwen3.8-Flash-Next-BF16.gguf";;
        qwen27)              echo "$Q27/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"; echo "$Q27/tokenizer";;
        qwen27-q8)           echo "$Q27/Qwen3.8-27B-Q8_0.gguf"; echo "$Q27/tokenizer";;
        qwen27-dspine)       echo "$Q27/Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf";;
        qwen27-vision)       echo "$Q27/mmproj-Qwen3.8-27B-BF16.gguf";;
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
    [ -L "$3" ] && [ -s "$3" ] && { echo "have   $(basename "$3") (a link)"; return; }
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
        glm)                 get $GLM_REPO GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf "$M/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf"
                             [ "$DRY_RUN" = 1 ] || [ -n "$(ls -A "$M/glm53-gsq-rco/pack" 2>/dev/null)" ] ||
                                 sh "$R/local/glm-pack.sh";;
        glm-3.0)             get $GLM_REPO GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf "$M/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"
                             [ "$DRY_RUN" = 1 ] || [ -n "$(ls -A "$M/glm53-gsq-rco/pack-3.0" 2>/dev/null)" ] ||
                                 sh "$R/local/glm-pack.sh" 3.0;;
        glm-mtp)             [ -s "$M/glm53-gsq-rco/GLM-5.3-Flash-NextN.gguf" ] && { echo "have   GLM-5.3-Flash-NextN.gguf"; return; }
                             local src=$M/glm53-gsq-rco/src/GLM-5.3-Flash-Maya-M-IQ2_S-00003-of-00003.gguf
                             get $GLM_SRC_REPO Maya-M/$(basename "$src") "$src"
                             [ "$DRY_RUN" = 1 ] && return
                             GGUF_ARCH=glm5-next ALMOND_GGUF_PY="$R/third_party/llama.cpp/gguf-py" "$PY" "$R/local/glm-mtp-extract.py" \
                                 "$src" "$M/glm53-gsq-rco/GLM-5.3-Flash-NextN.gguf"
                             gio trash "$M/glm53-gsq-rco/src" && echo "the 20.5 GB source went to the trash";;
        glm-vision)          get $GLM_SRC_REPO vision/mmproj-GLM-5.3-Flash-F16.gguf "$M/glm53-vision/mmproj-GLM-5.3-Flash-F16.gguf"
                             get $GLM_SRC_REPO vision/GLM-5.3-Flash-vocab.gguf "$M/glm53-vision/GLM-5.3-Flash-vocab.gguf";;
        qwen)                for f in $QWEN_SHARDS; do get $QWEN_REPO IQ3_S/$(basename "$f") "$f"; done
                             [ "$DRY_RUN" = 1 ] || [ -s "$M/qwen38-iq3s-pack/native_experts.txt" ] ||
                                 "$PY" "$R/tools/qwen/iq_pack.py" --gguf "$M/qwen38-iq3s/$(basename "$(echo "$QWEN_SHARDS" | head -1)")" \
                                     --out "$M/qwen38-iq3s-pack";;
        qwen-mtp)            [ -s "$M/qwen38-mtp/experts.bin" ] && { echo "have   qwen38-mtp"; return; }
                             [ "$DRY_RUN" = 1 ] && { echo "get    Qwen/Qwen3.8-Flash-Next mtp.* tensors (5.2 GB of range reads)"; return; }
                             # the checkpoint's 31 mtp.* tensors (HTTP ranges of 28 shards, SHA256-checked), packed with
                             # Q2_0 experts, laid out for the engine, and the draft head's token subset beside them
                             "$PY" "$R/tools/qwen/mtp_fetch.py" fetch --out "$M/qwen38-mtp-bf16"
                             "$PY" "$R/tools/qwen/mtp_pack.py" --src "$M/qwen38-mtp-bf16" --experts q2_0 --out "$M/qwen38-mtp-bf16/mtp-q2_0.gguf"
                             "$PY" "$R/tools/qwen/mtp_rt.py" --gguf "$M/qwen38-mtp-bf16/mtp-q2_0.gguf" --out "$M/qwen38-mtp"
                             cp "$R/tools/qwen/data/draft_vocab.bin" "$M/qwen38-mtp/draft_vocab.bin"
                             gio trash "$M/qwen38-mtp-bf16" && echo "the 5.2 GB of BF16 tensors went to the trash";;
        qwen-vision)         get $QWEN_REPO mmproj-Qwen3.8-Flash-Next-BF16.gguf "$M/qwen38-iq3s/mmproj-Qwen3.8-Flash-Next-BF16.gguf";;
        qwen27)              get $Q27_REPO Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf "$Q27/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
                             tok27 "$Q27/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf";;
        qwen27-q8)           get $Q27_Q8_REPO Qwen3.8-27B-Q8_0.gguf "$Q27/Qwen3.8-27B-Q8_0.gguf"
                             tok27 "$Q27/Qwen3.8-27B-Q8_0.gguf";;
        qwen27-dspine)       get aj9o9/Qwen3.8-27B-DSpine-b16-GGUF Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf "$Q27/Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf";;
        qwen27-vision)       get $Q27_REPO mmproj-Qwen3.8-27B-BF16.gguf "$Q27/mmproj-Qwen3.8-27B-BF16.gguf";;
        *) echo "fetch: no item '$1' (${ITEMS[*]})" >&2; exit 2;;
    esac
}
tok27() {   # the 27B's tokenizer, exported from the GGUF; the chat template is the official one (unsloth's Q8_0
            # carries a copy that lacks a `| safe`), so the folder is the same whichever quantization it came from
    [ "$DRY_RUN" = 1 ] && return
    [ -s "$Q27/tokenizer/vocab.json" ] && return
    PYTHONPATH="$R/tools/qwen" "$PY" "$R/tools/almond_tokenizer.py" --gguf "$1" --out "$Q27"
    cp "$R/tools/qwen/data/qwen27_chat_template.jinja" "$Q27/tokenizer/chat_template.jinja"
}
# this machine: the cards with 16 GiB or more (an integrated GPU does not count), the RAM, the free disk
CARDS=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: d={}
print(sum(1 for v in d.values() if int(v.get('VRAM Total Memory (B)',0)) >= 16*2**30))" 2>/dev/null || echo 0)
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
fit() {   # why an item does not fit this machine, or nothing
    local cards=1 ram=0
    case $1 in
        glm|glm-3.0)         cards=2; ram=100;;   # the layers over both cards, the experts in pinned RAM
        deepseek|qwen)       ram=100;;            # the 80 GiB pinned arena (README, Requirements: 128 GB class)
        qwen27-q8)           cards=2;;            # 29 GB of weights
    esac
    [ "$CARDS" -lt $cards ] && { echo "needs $cards cards of 24 GB"; return; }
    [ "$RAM_GB" -lt $ram ] && { echo "needs ~128 GB of RAM"; return; }
    return 0
}
show() {
    echo "This machine: $CARDS card(s) of 16 GB or more, $RAM_GB GB of RAM; free in $M: $(df -h --output=avail "$M/" | tail -1 | tr -d ' ')"
    echo "Models (the quantizations of a model are separate items: choose here; each has its own configuration):"
    local i=1 it st why
    for it in "${ITEMS[@]}"; do
        st=$(state "$it"); why=$(fit "$it")
        printf "  %2d) %-20s %-8s %s%s\n" $i "$it" "$st" "$(desc "$it")" "${why:+   [$why]}"
        i=$((i+1))
    done
    echo
    echo "Suggested for this machine:"
    if [ "$CARDS" -ge 2 ]; then
        echo "  Qwen3.8-27B: qwen27 (IQ3_S, one card, the fastest, 192K context) or qwen27-q8 (Q8_0, the closest to the"
        echo "               original weights, both cards, 256K); qwen27-dspine: DSpine drafts inside code blocks and tool calls (+31%
               on code with qwen27), MTP elsewhere"
        echo "  GLM-5.3-Flash: glm (3.5 bit, better answers) or glm-3.0 (3.0 bit, +16% decode, 20 GB less on disk)"
    elif [ "$CARDS" = 1 ]; then
        echo "  Qwen3.8-27B: qwen27 (IQ3_S, fits one card with its MTP block); qwen27-q8 needs two cards"
        echo "  GLM-5.3-Flash needs two cards; DeepSeek and Qwen3.8-Flash-Next run on one"
    else
        echo "  no 24 GB card found (rocm-smi): Almond needs at least one RX 7900 XT / XTX"
    fi
    [ "$RAM_GB" -lt 100 ] && echo "  with $RAM_GB GB of RAM: the dense Qwen3.8-27B; the MoE models want ~128 GB"
    return 0
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

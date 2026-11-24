#!/usr/bin/env bash
# build.sh - builds ggml (third_party/llama.cpp/build-hip) and the engine (build-hip-fused/almond-engine: DeepSeek, src/ds41,
# and GLM, src/glm), HIP only.
# The same commands as the README's "2. Build".  ggml is built once (tens of minutes: many HIP kernels); a second run
# only rebuilds what changed.  Environment:
#   ROCM=/opt/rocm      the ROCm install
#   GFX=gfx1100         the GPU architecture (RX 7900 XT / XTX); others are not tested
#   JOBS=$(nproc)       parallel compile jobs
#   REBUILD_GGML=1      configure and build ggml again even if build-hip exists
set -euo pipefail
cd "$(dirname "$0")"
ROCM=${ROCM:-/opt/rocm}
GFX=${GFX:-gfx1100}
JOBS=${JOBS:-$(nproc)}
CXX_HIP="$ROCM/lib/llvm/bin/clang++"
for need in cmake gcc "$CXX_HIP" "$ROCM/bin/hipcc"; do
    command -v "$need" >/dev/null 2>&1 || [ -x "$need" ] || { echo "build.sh: $need not found (ROCm with hipcc and clang++, cmake >= 3.24, gcc)" >&2; exit 1; }
done
COMMON=(-DCMAKE_PREFIX_PATH="$ROCM" -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES="$GFX" -DCMAKE_HIP_COMPILER="$CXX_HIP")

# a build-hip configured from another source tree (llama.cpp's own CMakeLists, before third_party/ggml-build) cannot
# regenerate here: it is moved aside and built again
B=third_party/llama.cpp/build-hip
if [ -f "$B/CMakeCache.txt" ] && ! grep -q "^CMAKE_HOME_DIRECTORY:INTERNAL=$PWD/third_party/ggml-build$" "$B/CMakeCache.txt"; then
    echo "build.sh: $B was configured from another source tree; moving it to $B.old and building ggml again"
    [ -e "$B.old" ] && gio trash "$B.old"; mv "$B" "$B.old"
fi
if [ ! -f "$B/ggml/src/libggml-base.a" ] || [ -n "${REBUILD_GGML:-}" ]; then
    echo "=== ggml (third_party/llama.cpp/ggml -> third_party/llama.cpp/build-hip)"
    cmake -S third_party/ggml-build -B third_party/llama.cpp/build-hip "${COMMON[@]}" \
        -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON   # static archives linked into PIE executables (Ubuntu's default, issue #4)
fi
cmake --build third_party/llama.cpp/build-hip -j "$JOBS"

echo "=== engine (build-hip-fused)"
cmake -S . -B build-hip-fused "${COMMON[@]}" \
    -DALMOND_GGML_DIR="$PWD/third_party/llama.cpp" -DALMOND_GGML_PREBUILT="$PWD/third_party/llama.cpp/build-hip" \
    -DALMOND_DSV41_HC_MIX=ON -DALMOND_DSV41_FQ_OP=ON
cmake --build build-hip-fused -j "$JOBS"
echo "=== image encoder for GLM and Qwen (tools/glm/vision: llama.cpp's mtmd at the engine's pinned commit + GLM-5.3's tower)"
# the pinned llama.cpp the GLM engine's build fetched, copied and patched (tools/vision/glm5next_patch.py, idempotent)
VL=build-vision/llama.cpp
# (llama.cpp's mtmd at the commit GLM's encoder was written against: a shallow fetch, once)
if [ ! -d "$VL" ]; then
    mkdir -p "$VL" && git -C "$VL" init -q && git -C "$VL" remote add origin https://github.com/ggml-org/llama.cpp &&
        git -C "$VL" fetch -q --depth 1 origin 3cf03257f219afbe7334045ff7c6a06ac68c627d && git -C "$VL" checkout -q FETCH_HEAD
fi
python3 tools/glm/vision/glm5next_patch.py "$VL"
cmake -S tools/glm/vision -B build-vision/build "${COMMON[@]}" -DLLAMA_DIR="$PWD/$VL" \
    -DALMOND_VISION_HIP=ON -DGGML_HIP_NO_VMM=ON
cmake --build build-vision/build --target almond-vision -j "$JOBS"
echo "=== image encoder for Hy4-flash (the same tools/glm/vision against llama.cpp PR #32517's mtmd: its hy_v4_flash projector)"
# llama.cpp PR #32517 at the commit the engine was checked against (a shallow fetch of the PR's ref, once)
VH=build-vision-hy4/llama.cpp
HY4_PR_COMMIT=0a054d0b02454c998e068381d15f3f6615bdd987
if [ ! -d "$VH" ]; then
    mkdir -p "$VH" && git -C "$VH" init -q && git -C "$VH" remote add origin https://github.com/ggml-org/llama.cpp &&
        { git -C "$VH" fetch -q --depth 1 origin "$HY4_PR_COMMIT" || git -C "$VH" fetch -q origin pull/32517/head; } &&
        git -C "$VH" checkout -q "$HY4_PR_COMMIT"
fi
cmake -S tools/glm/vision -B build-vision-hy4 -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$PWD/$VH"
cmake --build build-vision-hy4 --target almond-vision -j "$JOBS"
echo "=== done: build-hip-fused/almond-engine, build-vision/build/bin/almond-vision, build-vision-hy4/bin/almond-vision"
# Inkling 2 Mini (docs/inkling/): llama.cpp's inkling2_loop graph at a pinned master commit with its own ggml, and
# almond-inkling (src/inkling) linked to it. Built when its model is downloaded (local/fetch.sh inkling) or INKLING=1.
if [ -d models/inkling2-mini ] || [ -n "${INKLING:-}" ]; then
    VI=build-inkling/llama.cpp
    INKLING_COMMIT=30ec35ae2e27f1eca486af02b9d3d53fa7a26950
    echo "=== Inkling 2 Mini (build-inkling: llama.cpp $INKLING_COMMIT + almond-inkling)"
    if [ ! -d "$VI" ]; then
        mkdir -p "$VI" && git -C "$VI" init -q && git -C "$VI" remote add origin https://github.com/ggml-org/llama.cpp &&
            git -C "$VI" fetch -q --depth 1 origin "$INKLING_COMMIT" && git -C "$VI" checkout -q FETCH_HEAD
    fi
    cmake -S "$VI" -B "$VI/build" "${COMMON[@]}" -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF
    cmake --build "$VI/build" -j "$JOBS" --target llama llama-perplexity llama-bench llama-mtmd-cli
    g++ -O2 -std=c++17 -DALMOND_VERSION="\"inkling-$(git rev-parse --short HEAD)\"" -I"$VI/include" -I"$VI/ggml/include" \
        src/inkling/inkling_serve.cpp -L"$VI/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/$VI/build/bin" \
        -pthread -o build-inkling/almond-inkling
    echo "=== done: build-inkling/almond-inkling"
    # almond-inkling-dspark (opt-in, docs/inkling/NOTES.md): the same source with -DALMOND_DSPARK on llama.cpp
    # PR #32218 (Inkling 2 DSpark, not on master yet), in its own tree; built when the drafter is downloaded
    # (local/fetch.sh inkling-dspark) or INKLING_DSPARK=1
    if [ -f models/inkling2-mini/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf ] || [ -n "${INKLING_DSPARK:-}" ]; then
        VD=build-inkling/llama.cpp-dspark
        DSPARK_COMMIT=e4b1cb5318f1ac55d148be228c6347c6a94cc4c6
        if [ ! -d "$VD" ]; then
            mkdir -p "$VD" && git -C "$VD" init -q && git -C "$VD" remote add origin https://github.com/ggml-org/llama.cpp &&
                git -C "$VD" fetch -q --depth 1 origin "$DSPARK_COMMIT" && git -C "$VD" checkout -q FETCH_HEAD
        fi
        cmake -S "$VD" -B "$VD/build" "${COMMON[@]}" -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF
        cmake --build "$VD/build" -j "$JOBS" --target llama llama-common llama-server
        g++ -O2 -std=c++17 -DALMOND_DSPARK -DALMOND_VERSION="\"inkling-dspark-$(git rev-parse --short HEAD)\"" \
            -I"$VD/include" -I"$VD/ggml/include" -I"$VD/common" -I"$VD/vendor" src/inkling/inkling_serve.cpp \
            -L"$VD/build/bin" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/$VD/build/bin" \
            -pthread -o build-inkling/almond-inkling-dspark
        echo "=== done: build-inkling/almond-inkling-dspark"
    fi
    # almond-inkling-native: Almond's own Inkling engine (src/inkling/native, docs/inkling/ENGINE.md), HIP only
    "$CXX_HIP" -O2 -std=c++17 --offload-arch="$GFX" src/inkling/native/ink_native.hip -o build-inkling/almond-inkling-native
    echo "=== done: build-inkling/almond-inkling-native"
    # almond-vision against this llama.cpp's mtmd (its inkling2v projector), on the GPU in memory the engine lends it:
    # images for local/inkling-local-vision.json
    if [ -f build-inkling/llama.cpp/tools/mtmd/models/inkling2v.cpp ]; then
        cmake -S tools/glm/vision -B build-vision-inkling "${COMMON[@]}" -DLLAMA_DIR="$PWD/build-inkling/llama.cpp" \
            -DALMOND_VISION_HIP=ON -DGGML_HIP_NO_VMM=ON
        cmake --build build-vision-inkling --target almond-vision -j "$JOBS"
        echo "=== done: build-vision-inkling/bin/almond-vision"
    fi
fi

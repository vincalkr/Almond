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
echo "=== done: build-hip-fused/almond-engine, build-vision/build/bin/almond-vision"

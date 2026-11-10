#!/usr/bin/env bash
# bench-expert-hotcold.sh - builds and runs bench-expert-hotcold.cpp on card 0: native_expert_grouped on real expert
# blobs of the GSQ-RCO pack, 6 experts and one token per call, the weights hot (the same 6 experts, inside the 96 MB
# Infinity Cache) or cold (48 rotated, ~430 MB, from VRAM). REV=1 runs the cold arm first. Needs a built
# build-hip-fused (./build.sh); the binary goes to the scratch path $OUT (default /tmp/bench-expert-hotcold).
set -e
cd "$(dirname "$0")/../.."
R=$PWD; B=$R/build-hip-fused; G=$R/third_party/llama.cpp/build-hip/ggml/src; OUT=${OUT-/tmp/bench-expert-hotcold}
ROCM_CORE=$(ls -d /opt/rocm/core-* 2>/dev/null | head -1)
c++ -w -O2 -std=c++20 -DALMOND_USE_HIP=1 -DALMOND_NATIVE_EXPERTS=1 -D__HIP_PLATFORM_AMD__=1 \
    -I$R/include -I$R/src/ds41 -I$R/include/almond/hip_compat -I$R/third_party/llama.cpp/ggml/include \
    -isystem /opt/rocm/include -isystem $ROCM_CORE/include -include $R/include/almond/hip_compat/cuda_runtime.h \
    -fopenmp local/bench-v41/bench-expert-hotcold.cpp -o $OUT -Wl,-rpath,$ROCM_CORE/lib:/opt/rocm/lib \
    -Wl,--start-group $B/libds41_moe_hip.a $B/libds41_moe_cpu.a $B/libds41_transcode_hip.a $B/libalmond_kernels.a \
    $B/libalmond_kernels_cpu.a $B/libalmond_core.a $B/libalmond_engine.a $B/libalmond_mmq.a $G/libggml.a \
    $G/libggml-cpu.a $G/libggml-base.a $G/ggml-hip/libggml-hip.a $B/lib_almond_ggml_prebuilt_stubs.a -Wl,--end-group \
    -L$ROCM_CORE/lib -L/opt/rocm/lib -lhipblas -lamdhip64 -lgomp -lpthread
DS41_META=${DS41_META-local/rco-iq2xxs-meta/meta.gguf} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-0} \
    $OUT models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf

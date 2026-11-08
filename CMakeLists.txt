# Almond - a multi-model engine for MoE and dense models (DeepSeek-V4.1-Flash in src/ds41, GLM in src/glm, Qwen3.8-Flash-Next
# in src/qwen, the dense Qwen3.8-27B in src/qwen35) and the few libraries it links.
#
# Of the libraries this tree started from, exactly the objects ds41_generate links remain (traced with -Wl,--trace on
# 2026-10-18): the VRAM expert cache and VMM, pinned host memory, the CPU expert kernels, the HIP MMVQ / IQ kernels and
# ggml's MMQ instances for the prompt path.  The flags below are the ones those files were built with in the full
# tree, so the engine binary is the same.
#
# Build (AMD, ROCm; ggml comes prebuilt from third_party/llama.cpp/build-hip, see docs/ds4/AMD_HIP_NOTES.md):
#   cmake -S . -B build-hip-fused -DCMAKE_BUILD_TYPE=Release -DALMOND_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
#     -DALMOND_GGML_DIR=$PWD/third_party/llama.cpp -DALMOND_GGML_PREBUILT=$PWD/third_party/llama.cpp/build-hip \
#     -DALMOND_DSV41_HC_MIX=ON -DALMOND_DSV41_FQ_OP=ON
#   cmake --build build-hip-fused --target ds41_generate -j   (the executable: almond-engine)
cmake_minimum_required(VERSION 3.24)
project(almond VERSION 0.1.41 LANGUAGES CXX)
add_compile_definitions(ALMOND_VERSION="${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(ALMOND_ENABLE_HIP "Build for wave32 AMD GPUs (gfx1100 validated)" ON)
option(ALMOND_PREFILL_MMQ "ggml's MMQ kernels for the prompt path's experts" ON)
option(ALMOND_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
option(ALMOND_WERROR "Treat compiler warnings as errors" OFF)
set(ALMOND_GGML_DIR "" CACHE PATH "the llama.cpp checkout ggml comes from (third_party/llama.cpp, with tools/v41/ggml patches)")
set(ALMOND_GGML_PREBUILT "" CACHE PATH "an existing ggml build directory of ALMOND_GGML_DIR: link its ggml")
if(NOT ALMOND_ENABLE_HIP)
  message(FATAL_ERROR "Almond builds the HIP engine only (no CUDA or CPU-only path)")
endif()
if(NOT ALMOND_GGML_DIR OR NOT ALMOND_GGML_PREBUILT)
  message(FATAL_ERROR "set ALMOND_GGML_DIR and ALMOND_GGML_PREBUILT (third_party/llama.cpp and its build-hip)")
endif()

include(cmake/hip_backend.cmake)
set(_almond_gpu_runtime_target almond_hip_runtime)
set(_almond_gpu_include_directories ${ALMOND_HIP_COMPAT_INCLUDE_DIR})

# ggml's MMQ host API spelling (see src/prefill/moe_mmq.cu: ALMOND_MMQ_PREC_SRC1)
function(almond_mmq_api_version target mmq_dir)
  if(EXISTS "${mmq_dir}/mmq.cuh")
    file(READ "${mmq_dir}/mmq.cuh" _almond_mmq_h)
    if(_almond_mmq_h MATCHES "const ggml_prec prec_src1")
      target_compile_definitions(${target} PRIVATE ALMOND_MMQ_PREC_SRC1=1)
    endif()
  endif()
endfunction()

# ggml_backend_cuda_context has CUDA-graph members when ggml-hip was built with GGML_HIP_GRAPHS, and in almond-engine
# its destructor is ggml-hip's: every MMQ library that creates one (DeepSeek's, GLM's, Qwen's) must give it the same
# layout (without it the object was 4,336 bytes where ggml-hip's destructor reads 4,400)
function(almond_ggml_layout target)
  file(STRINGS "${ALMOND_GGML_PREBUILT}/CMakeCache.txt" _almond_hip_graphs REGEX "^GGML_HIP_GRAPHS:BOOL=ON$")
  if(_almond_hip_graphs)
    target_compile_definitions(${target} PRIVATE GGML_HIP_GRAPHS=1)
  endif()
endfunction()
add_library(almond_warnings INTERFACE)
target_compile_options(almond_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall> $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(ALMOND_WERROR)
  target_compile_options(almond_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror> $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# the GGUF reader and scalar dequantizers (header-only)
add_library(almond_artifact INTERFACE)
target_include_directories(almond_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(almond_artifact INTERFACE almond_warnings)
add_library(almond_plan INTERFACE)
target_include_directories(almond_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(almond_plan INTERFACE almond_warnings)

# ---- ggml (prebuilt from third_party/llama.cpp) ----
enable_language(C)
find_package(Threads REQUIRED)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/cmake/ggml_prebuilt.cmake)

# ---- almond_core: pinned host memory, the memory probe ----
add_library(almond_core STATIC src/core/pinned.cu src/platform/memory.cpp)
target_include_directories(almond_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(almond_core PUBLIC almond_plan almond_warnings ${_almond_gpu_runtime_target})

# ---- almond_kernels: the HIP MMVQ and IQ expert kernels ----
add_library(almond_kernels STATIC src/kernels/cuda/iq_kernels.cu src/kernels/cuda/native_mmvq.cu)
target_include_directories(almond_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(almond_kernels PRIVATE third_party/ggml)
target_link_libraries(almond_kernels PUBLIC almond_artifact almond_core almond_warnings ${_almond_gpu_runtime_target})

# ---- almond_kernels_cpu: the CPU expert kernels and their thread pool ----
add_library(almond_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp src/kernels/cpu/expert_layout.cpp
            src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
            src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/ptq1_0.cpp)
target_include_directories(almond_kernels_cpu PUBLIC include PRIVATE third_party/ggml)
set(_almond_avx512 "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mbmi2;-mf16c;-mfma")
set_source_files_properties(src/kernels/cpu/expert.cpp src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${_almond_avx512}")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp
                            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
# AVX-VNNI copies of the AVX-2 kernels (target("avxvnni") per function, taken where cpu_avxvnni_ok())
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
check_cxx_source_compiles([=[
#include <immintrin.h>
__attribute__((target("avxvnni")))
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] ALMOND_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(ALMOND_HAVE_AVXVNNI_INTRINSICS)
  set_property(SOURCE src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ALMOND_AVXVNNI=1)
endif()
target_link_libraries(almond_kernels_cpu PUBLIC ggml-cpu ggml-base)
target_compile_definitions(almond_kernels_cpu PUBLIC ALMOND_NATIVE_EXPERTS=1)

# ---- almond_engine: the VRAM expert cache and VMM ----
add_library(almond_engine STATIC src/core/expert_cache.cpp src/core/vmm.cpp)
target_include_directories(almond_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(almond_engine PUBLIC almond_core almond_kernels almond_kernels_cpu ${_almond_gpu_runtime_target})

# ---- almond_mmq: ggml's MMQ kernels for the prompt path's experts (pinned ggml tree, compiled as HIP) ----
if(ALMOND_PREFILL_MMQ)
  set(_almond_gg "${ALMOND_GGML_DIR}/ggml")
  set(_almond_mmq "${_almond_gg}/src/ggml-cuda")
  set(_almond_mmq_sources src/prefill/moe_mmq.cu src/prefill/ggml_cuda_host.cu "${_almond_mmq}/quantize.cu")
  foreach(_t IN ITEMS q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
    list(APPEND _almond_mmq_sources "${_almond_mmq}/template-instances/mmq-instance-${_t}.cu")
  endforeach()
  set_source_files_properties(${_almond_mmq_sources} PROPERTIES LANGUAGE HIP)
  add_library(almond_mmq STATIC ${_almond_mmq_sources})
  target_include_directories(almond_mmq PRIVATE "${_almond_gg}/include" "${_almond_gg}/src" "${_almond_mmq}"
                                        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  almond_mmq_api_version(almond_mmq "${_almond_mmq}")
  almond_ggml_layout(almond_mmq)
  target_compile_definitions(almond_mmq PRIVATE GGML_USE_HIP=1)
  target_link_libraries(almond_mmq PUBLIC ggml-base hip::host roc::hipblas)
endif()

# ---- the DeepSeek-V4.1 engine ----
include(${CMAKE_CURRENT_SOURCE_DIR}/src/ds41/v41_engine.cmake)

# GLM-5.3-Flash (src/glm: docs/ENGINE_PLAN.md) -> almond-glm
add_subdirectory(src/glm)
add_subdirectory(src/qwen)

# One executable (docs/ENGINE_PLAN.md, step 3): DeepSeek's engine target with GLM's path linked in (glm_program, its
# usage requirements kept out: each tree force-includes its own hip_compat/cuda_runtime.h), src/engine_main.cpp picks
# the path from the arguments
if(TARGET ds41_generate)
  add_library(glm_program STATIC src/glm/program/generate.cpp)
  target_include_directories(glm_program PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/glm/include/almond/glm/hip_compat)
  target_compile_definitions(glm_program PRIVATE ALMOND_VERSION="${PROJECT_VERSION}")
  target_link_libraries(glm_program PUBLIC glm_engine glm_prefill glm_spec glm_hip_runtime)
  target_sources(ds41_generate PRIVATE src/engine_main.cpp)
  target_link_libraries(ds41_generate PRIVATE "$<LINK_ONLY:glm_program>")
  # Qwen3.8-Flash-Next's path (src/qwen), the same way: its own forced hip_compat/cuda_runtime.h, linked in only
  add_library(qwen_program STATIC src/qwen/program/generate.cpp)
  target_include_directories(qwen_program PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen/include/almond/qwen/hip_compat)
  target_compile_definitions(qwen_program PRIVATE ALMOND_VERSION="${PROJECT_VERSION}")
  target_link_libraries(qwen_program PUBLIC qwen_engine qwen_prefill qwen_spec qwen_hip_runtime)
  target_link_libraries(ds41_generate PRIVATE "$<LINK_ONLY:qwen_program>")
  # the dense Qwen3.8-27B path (src/qwen35, GGUF architecture qwen35): ggml graphs on the prebuilt ggml / ggml-hip
  add_library(qwen35_program STATIC src/qwen35/qwen35.cpp)
  target_compile_definitions(qwen35_program PRIVATE ALMOND_VERSION="${PROJECT_VERSION}")
  target_link_libraries(qwen35_program PUBLIC ggml ggml-base)
  target_link_libraries(ds41_generate PRIVATE "$<LINK_ONLY:qwen35_program>")
  set_target_properties(ds41_generate PROPERTIES OUTPUT_NAME almond-engine)
endif()

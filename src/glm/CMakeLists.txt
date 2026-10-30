# src/glm - the GLM-5.3-Flash path (from project-maya v1.1.37, MIT: LICENSE), built inside Almond's project
# (add_subdirectory from the top CMakeLists.txt) on Almond's ggml (the top level's prebuilt ggml-base / ggml-cpu and
# third_party/llama.cpp/ggml's MMQ sources).  HIP only.  Its targets keep project-maya's names until the rename
# (docs/ENGINE_PLAN.md, step 4); the executable is `almond-engine` (the top level's).
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
include(cmake/hip_backend.cmake)
set(_strata_gpu_runtime_target strata_hip_runtime)
set(_strata_gpu_blas_target roc::hipblas)
set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
set(_glm_ggml_common ${PROJECT_SOURCE_DIR}/third_party/ggml)   # llama.cpp's block layouts and codebook grids

add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall> $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_library(strata_core STATIC core/device.cu core/pinned.cu platform/memory.cpp core/graph.cpp core/weights.cpp
            core/layout.cpp)
target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

add_library(strata_kernels STATIC
  kernels/cuda/dequant_s2.cu kernels/cuda/s2_gemv.cu kernels/cuda/s_gemv.cu kernels/cuda/s2_gemv_quads.cu
  kernels/cuda/s2_gemv_fast.cu kernels/cuda/router_top10.cu kernels/cuda/router_sigmoid.cu kernels/cuda/quantize_act.cu
  kernels/cuda/rope.cu kernels/cuda/sampler.cu kernels/cuda/s2_gemv_q8.cu kernels/cuda/s2_expert_grouped.cu
  kernels/cuda/shared_expert.cu kernels/cuda/gr.cu kernels/cuda/gdn.cu kernels/cuda/glm_hc.cu kernels/cuda/glm_kda.cu
  kernels/cuda/glm_ffn.cu kernels/cuda/glm_dsa.cu kernels/cuda/qsa.cu kernels/cuda/elementwise.cu
  kernels/cuda/bf16_gemv.cu kernels/cuda/native_mmvq.cu kernels/cuda/native_bf16.cu kernels/cuda/native_gr_norm.cu
  kernels/cuda/native_moe.cu kernels/cuda/native_gdn.cu kernels/cuda/native_gr_postops.cu kernels/cuda/native_router.cu
  kernels/cuda/native_gdn_preprocess.cu kernels/cuda/native_qsa.cu kernels/cuda/native_rope.cu
  kernels/cuda/native_ple_postops.cu kernels/cuda/native_flash_attn.cu kernels/cuda/native_qsa_indexer.cu
  kernels/cuda/native_qsa_score.cu kernels/cuda/ple.cu kernels/cuda/kv_q8.cu kernels/cuda/kv_stream.cu
  kernels/cuda/kv_q4.cu kernels/cuda/fused_gr.cu kernels/cuda/cvec.cu kernels/cuda/qsa_decode_attn.cu
  kernels/cuda/qsa_prompt_attn.cu kernels/cuda/fused_gdn.cu kernels/cuda/dequant_bf16.cu kernels/cuda/qsa_select.cu
  kernels/cuda/verify_kernels.cu kernels/cuda/iq_kernels.cu kernels/cuda/glm_fast.cu kernels/cuda/glm_batch.cu
  kernels/cuda/exl3.cu kernels/cuda/exl3_legacy.cu kernels/cuda/glm_kv_stream.cu
  kernels/ngram.cpp ngram/ple_reader.cpp platform/direct_file.cpp)
target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
target_include_directories(strata_kernels PRIVATE ${_glm_ggml_common})
# Q8_K's multiply and magic-bias rounding stay separate FP32 operations (an FMA changes tie results); EXL3's
# multiply-adds fused where the source writes them, not after inlining
set_source_files_properties(kernels/cuda/elementwise.cu kernels/cuda/quantize_act.cu
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(kernels/cuda/exl3.cu PROPERTIES COMPILE_OPTIONS "-ffp-contract=on")

set(_glm_cpu_sources kernels/cpu/pool.cpp kernels/cpu/expert_layout.cpp kernels/cpu/exl3_ref.cpp kernels/cpu/expert.cpp
    kernels/cpu/q2_avx2.cpp kernels/cpu/exl3_cpu.cpp kernels/cpu/native_expert.cpp kernels/cpu/iq_avx512.cpp
    kernels/cpu/iq_avx2.cpp kernels/cpu/kq_avx512.cpp kernels/cpu/iq_avx512_one.cpp)
add_library(strata_kernels_cpu STATIC ${_glm_cpu_sources})
target_include_directories(strata_kernels_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${_glm_ggml_common})
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma")
check_cxx_source_compiles([=[
#include <immintrin.h>
__m256i f(__m256i a, __m256i b, __m256i c) { return _mm256_dpbusd_epi32(a, b, c); }
__m512i g(__m512i a, __m512i b, __m512i c) { return _mm512_dpbusd_epi32(a, b, c); }
int main() { return 0; }
]=] GLM_CPU_VNNI_ASM)
unset(CMAKE_REQUIRED_FLAGS)
if(GLM_CPU_VNNI_ASM)
  set_source_files_properties(kernels/cpu/expert.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
else()
  set_source_files_properties(kernels/cpu/expert.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vbmi;-mf16c;-mfma"
               COMPILE_DEFINITIONS "STRATA_CPU_DPBUSD_BW=1")
endif()
set_source_files_properties(kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma;-ffp-contract=off")
set_source_files_properties(kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
set_source_files_properties(kernels/cpu/iq_avx512_one.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vbmi;-mbmi2;-mf16c;-mfma;-ffp-contract=off")
set_source_files_properties(kernels/cpu/kq_avx512.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma")
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads ggml-cpu ggml-base)
target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)

add_library(strata_engine STATIC core/layer.cpp core/session.cpp core/expert_source.cpp core/remote_experts.cpp
  core/expert_cache.cpp core/native_head.cpp core/native_dense.cpp core/verify.cpp core/mtp.cpp
  core/glm_model.cu core/glm_fast_path.cu core/glm_prefill.cu core/glm_multi.cu core/glm_rows.cu
  core/glm_window.cu core/glm_dflash.cu core/glm_mtp.cu core/glm_mtp_rows.cu)
target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
target_compile_definitions(strata_engine PRIVATE STRATA_HIP_PREFILL_FUSED=1)

add_library(strata_prefill STATIC prefill/gemm.cu prefill/kernels.cu prefill/prefill.cpp)
target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
if(STRATA_HIPBLASLT_AVAILABLE)
  target_compile_definitions(strata_prefill PRIVATE STRATA_HIPBLASLT_AVAILABLE=1)
  target_link_libraries(strata_prefill PRIVATE roc::hipblaslt)
  add_executable(tune_hipblaslt ${PROJECT_SOURCE_DIR}/tools/glm/hip/tune_hipblaslt.cpp)
  target_link_libraries(tune_hipblaslt PRIVATE hip::host roc::hipblas roc::hipblaslt)
endif()

# ggml's MMQ for the prompt path's experts, from Almond's ggml source
set(_glm_mmq "${ALMOND_GGML_DIR}/ggml/src/ggml-cuda")
set(_glm_mmq_sources prefill/moe_mmq.cu prefill/moe_fused_rdna4.cu prefill/ggml_cuda_host.cu
    "${_glm_mmq}/quantize.cu" "${_glm_mmq}/convert.cu")
foreach(_t q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0 iq1_s q2_k q3_k q4_k q5_k q6_k)
  list(APPEND _glm_mmq_sources "${_glm_mmq}/template-instances/mmq-instance-${_t}.cu")
endforeach()
set_source_files_properties(${_glm_mmq_sources} PROPERTIES LANGUAGE HIP)
add_library(strata_mmq STATIC ${_glm_mmq_sources})
target_include_directories(strata_mmq PRIVATE "${ALMOND_GGML_DIR}/ggml/include" "${ALMOND_GGML_DIR}/ggml/src" "${_glm_mmq}"
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1)
target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
target_link_libraries(strata_prefill PUBLIC strata_mmq)
target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)
target_link_libraries(strata_engine PUBLIC strata_mmq ${_strata_gpu_blas_target})

add_library(strata_spec STATIC spec/suffix_drafter.cpp spec/controller.cpp spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC include)

# (the executable: almond-engine, the top CMakeLists.txt - src/engine_main.cpp)

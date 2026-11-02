# Qwen3.8-Flash-Next's path of almond-engine (src/qwen, namespace almond::qwen): the model, its expert pools and RAM
# tier, the prompt path, MTP drafting and the conversation cache.  Built on the top level's prebuilt ggml
# (third_party/llama.cpp); the top CMakeLists.txt links qwen_program into almond-engine, which runs this path when its
# arguments name --qwen (src/engine_main.cpp).
include(cmake/hip_backend.cmake)
set(_qwen_gpu_runtime_target qwen_hip_runtime)
set(_qwen_ggml_common ${PROJECT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/ggml)   # llama.cpp's block layouts and codebook grids, and the trellis types

add_library(qwen_warnings INTERFACE)
target_compile_options(qwen_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall> $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
add_library(qwen_artifact INTERFACE)
target_include_directories(qwen_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qwen_artifact INTERFACE qwen_warnings)
add_library(qwen_plan INTERFACE)
target_include_directories(qwen_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qwen_plan INTERFACE qwen_warnings)

add_library(qwen_core STATIC core/device.cu core/pinned.cu platform/memory.cpp core/graph.cpp core/weights.cpp
            core/layout.cpp)
target_include_directories(qwen_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# the rows of a verify / batch window (16 on HIP) and the Q6_K gate/up expert kernels (UD-Q6_K_XL, UD-Q5_K_XL): PUBLIC,
# so every source that includes the headers - qwen_program's generate.cpp, compiled at the top level, too - sees the
# same struct layouts
target_compile_definitions(qwen_core PUBLIC ALMOND_HAVE_DFLASH2=1 ALMOND_VERIFY_MAX_T=16 ALMOND_Q6K_EXPERTS=1)
target_link_libraries(qwen_core PUBLIC qwen_plan qwen_warnings ${_qwen_gpu_runtime_target})

add_library(qwen_kernels STATIC
  kernels/cuda/dequant_s2.cu kernels/cuda/s2_gemv.cu kernels/cuda/s_gemv.cu kernels/cuda/s_gemv_pair.cu
  kernels/cuda/s2_gemv_quads.cu kernels/cuda/s2_gemv_fast.cu kernels/cuda/router_top10.cu kernels/cuda/quantize_act.cu
  kernels/cuda/rope.cu kernels/cuda/sampler.cu kernels/cuda/s2_gemv_q8.cu kernels/cuda/s2_expert_grouped.cu
  kernels/cuda/shared_expert.cu kernels/cuda/gr.cu kernels/cuda/gdn.cu kernels/cuda/qsa.cu kernels/cuda/elementwise.cu
  kernels/cuda/bf16_gemv.cu kernels/cuda/native_mmvq.cu kernels/cuda/native_bf16.cu kernels/cuda/native_gr_norm.cu
  kernels/cuda/native_moe.cu kernels/cuda/native_gdn.cu kernels/cuda/native_gr_postops.cu kernels/cuda/native_router.cu
  kernels/cuda/route_prior.cu kernels/cuda/native_gdn_preprocess.cu kernels/cuda/native_qsa.cu kernels/cuda/native_rope.cu
  kernels/cuda/native_ple_postops.cu kernels/cuda/native_flash_attn.cu kernels/cuda/native_qsa_indexer.cu
  kernels/cuda/native_qsa_score.cu kernels/cuda/ple.cu kernels/cuda/kv_q8.cu kernels/cuda/kv_stream.cu
  kernels/cuda/kv_q4.cu kernels/cuda/fused_gr.cu kernels/cuda/cvec.cu kernels/cuda/qsa_decode_attn.cu
  kernels/cuda/qsa_prompt_attn.cu kernels/cuda/fused_gdn.cu kernels/cuda/dequant_bf16.cu kernels/cuda/qsa_select.cu
  kernels/cuda/qsa_pattn.cu kernels/cuda/verify_kernels.cu kernels/cuda/dflash2.cu kernels/cuda/iq_kernels.cu
  kernels/ngram.cpp ngram/ple_reader.cpp platform/direct_file.cpp)
target_link_libraries(qwen_kernels PUBLIC qwen_core qwen_artifact)
target_include_directories(qwen_kernels PRIVATE ${_qwen_ggml_common})

# the CPU expert kernels: each ISA's flags on its own source, so the scalar paths stay what they are
add_library(qwen_kernels_cpu STATIC kernels/cpu/expert.cpp kernels/cpu/pool.cpp kernels/cpu/expert_layout.cpp
  kernels/cpu/q2_avx2.cpp kernels/cpu/hadamard.cpp kernels/cpu/native_expert.cpp kernels/cpu/iq_avx512.cpp
  kernels/cpu/iq_avx2.cpp kernels/cpu/kq_avx2.cpp kernels/cpu/kq_avx1.cpp kernels/cpu/q2_avx1.cpp kernels/cpu/iq_avx1.cpp
  kernels/cpu/nvfp4_avx512.cpp kernels/cpu/nvfp4_avx2.cpp kernels/cpu/tq_cpu.cpp kernels/cpu/tq_avx2.cpp
  kernels/cpu/tq_avx512.cpp)
target_include_directories(qwen_kernels_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${_qwen_ggml_common})
set_source_files_properties(kernels/cpu/expert.cpp
  PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(kernels/cpu/iq_avx512.cpp kernels/cpu/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mavx512vpopcntdq;-mbmi2;-mf16c;-mfma")
set_source_files_properties(kernels/cpu/q2_avx2.cpp kernels/cpu/iq_avx2.cpp kernels/cpu/kq_avx2.cpp
  kernels/cpu/nvfp4_avx2.cpp kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mavx512vbmi;-mavx2;-mfma;-mf16c")
set_source_files_properties(kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
set_source_files_properties(kernels/cpu/q2_avx1.cpp kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS
  "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
if(ALMOND_HAVE_AVXVNNI_INTRINSICS)
  set_property(SOURCE kernels/cpu/q2_avx2.cpp kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ALMOND_AVXVNNI=1)
else()
  set_property(SOURCE kernels/cpu/q2_avx2.cpp kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ALMOND_AVXVNNI=0)
endif()
target_link_libraries(qwen_kernels_cpu PUBLIC Threads::Threads ggml-cpu ggml-base)
target_compile_definitions(qwen_kernels_cpu PUBLIC ALMOND_NATIVE_EXPERTS=1)

# the composed layer, the expert tiers, verification, MTP / DFlash2 and the conversation cache
add_library(qwen_engine STATIC core/layer.cpp core/session.cpp core/expert_source.cpp core/remote_experts.cpp
  core/remote_expert_opt.cu core/expert_cache.cpp core/native_head.cpp core/native_dense.cpp core/verify.cpp core/mtp.cpp
  core/dflash2.cpp core/conversation_snapshot.cpp core/conversation_state.cpp core/conversation_memory.cpp
  core/peer_experts.cpp core/arch_defaults.cpp core/nv_defaults.cpp core/vmm.cpp core/conversation_file.cpp)
target_include_directories(qwen_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qwen_engine PUBLIC qwen_core qwen_kernels qwen_kernels_cpu ${_qwen_gpu_runtime_target})

add_library(qwen_prefill STATIC prefill/gemm.cu prefill/kernels.cu prefill/prefill.cpp prefill/wmma_gemm.cu)
target_include_directories(qwen_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${QWEN_HIP_COMPAT_INCLUDE_DIR})
target_link_libraries(qwen_prefill PUBLIC qwen_engine ${QWEN_HIP_BLAS_TARGETS} ${_qwen_gpu_runtime_target})
if(QWEN_HIPBLASLT_AVAILABLE)
  target_compile_definitions(qwen_prefill PRIVATE ALMOND_HIPBLASLT_AVAILABLE=1)
  target_link_libraries(qwen_prefill PRIVATE roc::hipblaslt)
endif()
if(QWEN_ROCBLAS_AVAILABLE)
  target_compile_definitions(qwen_prefill PRIVATE ALMOND_ROCBLAS_AVAILABLE=1)
  target_link_libraries(qwen_prefill PRIVATE roc::rocblas)
endif()

# the prompt path's experts on ggml's MMQ kernels (llama.cpp's ggml-cuda sources, compiled as HIP)
set(_qwen_gg "${ALMOND_GGML_DIR}/ggml")
set(_qwen_mmq "${_qwen_gg}/src/ggml-cuda")
set(_qwen_mmq_sources prefill/moe_mmq.cu prefill/moe_fused.cu prefill/moe_fused_iq.cu prefill/moe_f16_kq.cu
    prefill/ggml_cuda_host.cu "${_qwen_mmq}/quantize.cu")
foreach(_t IN ITEMS q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0 q4_k q5_k q5_1 q6_k)
  list(APPEND _qwen_mmq_sources "${_qwen_mmq}/template-instances/mmq-instance-${_t}.cu")
endforeach()
set_source_files_properties(${_qwen_mmq_sources} PROPERTIES LANGUAGE HIP)
set_source_files_properties(prefill/moe_fused.cu prefill/moe_fused_iq.cu prefill/moe_f16_kq.cu
                            PROPERTIES INCLUDE_DIRECTORIES "${QWEN_HIP_COMPAT_INCLUDE_DIR}")
add_library(qwen_mmq STATIC ${_qwen_mmq_sources})
target_include_directories(qwen_mmq PRIVATE "${_qwen_gg}/include" "${_qwen_gg}/src" "${_qwen_mmq}"
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
almond_mmq_api_version(qwen_mmq "${_qwen_mmq}")
target_compile_definitions(qwen_mmq PRIVATE GGML_USE_HIP=1 ALMOND_MMQ_KQUANTS=1)
almond_ggml_layout(qwen_mmq)   # ggml-hip's layout of ggml_backend_cuda_context (the top CMakeLists.txt)
target_link_libraries(qwen_mmq PUBLIC ggml-base hip::host roc::hipblas)
target_link_libraries(qwen_prefill PUBLIC qwen_mmq)
target_compile_definitions(qwen_prefill PUBLIC ALMOND_PREFILL_MMQ=1 ALMOND_PREFILL_FUSED=1)

add_library(qwen_spec STATIC spec/suffix_drafter.cpp spec/controller.cpp spec/draft_policy.cpp spec/draft_source.cpp)
target_include_directories(qwen_spec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

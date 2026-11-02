# Qwen3.8-Flash-Next's path on HIP: the gfx targets, the ROCm packages and the CUDA-to-HIP shim force-included into
# every source of src/qwen (include/almond/qwen/hip_compat), as src/glm/cmake/hip_backend.cmake does for GLM's.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "HIP target architecture(s), e.g. gfx1100 or gfx1100;gfx1201")
endif()
string(REPLACE " " ";" _qwen_hip_norm "${CMAKE_HIP_ARCHITECTURES}")
set(_qwen_hip_arch_list "")
foreach(_arch IN LISTS _qwen_hip_norm)
  if(_arch STREQUAL "")
    continue()
  endif()
  string(REGEX REPLACE ":.*$" "" _base "${_arch}")      # gfx1100:xnack- -> gfx1100
  if(NOT _base MATCHES "^gfx1[12]0[01]$")
    message(FATAL_ERROR "the Qwen engine builds for wave32 gfx1100, gfx1101, gfx1200 and gfx1201; "
                        "CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
  endif()
  list(APPEND _qwen_hip_arch_list "${_base}")
endforeach()
list(REMOVE_DUPLICATES _qwen_hip_arch_list)
string(REPLACE ";" "," _qwen_hip_archs "${_qwen_hip_arch_list}")

enable_language(HIP)
find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
set(QWEN_HIP_BLAS_TARGETS roc::hipblas)
if(hipblas_VERSION VERSION_LESS "1.0")
  find_package(rocblas CONFIG REQUIRED)
  list(APPEND QWEN_HIP_BLAS_TARGETS roc::rocblas)
else()
  find_package(rocblas CONFIG QUIET)
endif()
find_package(hipblaslt CONFIG QUIET)
set(QWEN_HIPBLASLT_AVAILABLE OFF)
if(TARGET roc::hipblaslt)
  set(QWEN_HIPBLASLT_AVAILABLE ON)
endif()
set(QWEN_ROCBLAS_AVAILABLE OFF)
if(TARGET roc::rocblas)
  set(QWEN_ROCBLAS_AVAILABLE ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(QWEN_HIP_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/almond/qwen/hip_compat")
add_library(qwen_hip_runtime INTERFACE)
target_include_directories(qwen_hip_runtime BEFORE INTERFACE
  "${QWEN_HIP_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(qwen_hip_runtime INTERFACE ALMOND_USE_HIP=1 "ALMOND_HIP_ARCHS=\"${_qwen_hip_archs}\"")
# the RDNA3 WMMA GEMM's intrinsics compile for gfx11 targets only and the host pass does not define __gfx11xx__
if(CMAKE_HIP_ARCHITECTURES MATCHES "gfx11")
  target_compile_definitions(qwen_hip_runtime INTERFACE ALMOND_WMMA_GFX11=1)
endif()
target_link_libraries(qwen_hip_runtime INTERFACE hip::host)
foreach(_language IN ITEMS CXX HIP)
  target_compile_options(qwen_hip_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:${_language}>:-include>"
    "$<$<COMPILE_LANGUAGE:${_language}>:${QWEN_HIP_COMPAT_INCLUDE_DIR}/cuda_runtime.h>")
endforeach()

# CMake does not infer HIP from the CUDA-shaped .cu suffixes
file(GLOB_RECURSE _qwen_hip_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
if(_qwen_hip_sources)
  set_source_files_properties(${_qwen_hip_sources} PROPERTIES LANGUAGE HIP)
endif()
message(STATUS "Almond: the Qwen engine on HIP, arch ${_qwen_hip_archs}")

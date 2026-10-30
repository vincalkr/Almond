# tools/vision - almond-vision, the image encoder for the multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native]
#   cmake --build build-vision --target almond-vision
cmake_minimum_required(VERSION 3.21)
project(glm_vision C CXX)

# the GLM engine's build options by the names its docs give them: -DMAYA_X=... sets ALMOND_X, the name this file reads
get_cmake_property(_maya_cache CACHE_VARIABLES)
foreach(_v IN LISTS _maya_cache)
  if(_v MATCHES "^(MAYA|ALMOND)_(.+)$")   # (Almond: -DALMOND_X as well)
    set(ALMOND_${CMAKE_MATCH_2} "${${_v}}")
  endif()
endforeach()

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(ALMOND_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(ALMOND_VISION_HIP "run the vision encoder on an AMD GPU (builds ggml-hip; Almond)" OFF)
option(ALMOND_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(ALMOND_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
  if(ALMOND_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
else()
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  if(NOT GGML_CPU_ARM_ARCH)
    set(GGML_CPU_ARM_ARCH "armv8.6-a+dotprod+i8mm+fp16" CACHE STRING "ggml-cpu ARM arch flags" FORCE)
  endif()
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${ALMOND_VISION_CUDA} CACHE BOOL "" FORCE)
set(GGML_HIP ${ALMOND_VISION_HIP} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(almond-vision vision.cpp)
target_link_libraries(almond-vision PRIVATE mtmd llama)
target_compile_features(almond-vision PRIVATE cxx_std_17)
set_target_properties(almond-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

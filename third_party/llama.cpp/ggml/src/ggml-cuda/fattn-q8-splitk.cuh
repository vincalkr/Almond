#pragma once
#include "common.cuh"

// true when it ran the op (GGML_CUDA_FA_Q8_SPLITK=1, q8_0 K and V, D 256, a long context): fattn-q8-splitk.cu
bool ggml_cuda_fattn_q8_splitk_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

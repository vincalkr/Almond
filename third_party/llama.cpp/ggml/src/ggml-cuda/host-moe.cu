// Almond: ggml_host_moe - routed experts the host computes while the GPU graph waits on a mapped flag, with no
// return to the host's launch loop (the doorbell of src/glm, measured there at 0.05 ms a handoff).
#include "host-moe.cuh"
#include "ggml-cuda.h"

static __global__ void host_moe_publish(const float * __restrict__ x, const int32_t * __restrict__ ids,
                                        const float * __restrict__ w, int nx, int nk, int tag, int nt,
                                        float * hx, int32_t * hids, float * hw, int32_t * meta, uint32_t * seq) {
    for (int i = threadIdx.x; i < nx; i += blockDim.x) hx[i] = x[i];
    for (int i = threadIdx.x; i < nk; i += blockDim.x) { hids[i] = ids[i]; hw[i] = w[i]; }
    if (threadIdx.x == 0) { meta[0] = tag; meta[1] = nt; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile uint32_t *) seq = *(volatile uint32_t *) seq + 1u;
        __threadfence_system();
    }
}

static __global__ void host_moe_wait_copy(const volatile uint32_t * flag, const volatile uint32_t * seq,
                                          const volatile float * out, float * dst, int n) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint32_t want = *seq;
        while (*flag != want) __builtin_amdgcn_s_sleep(2);
        __threadfence_system();
        ok = 1;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = out[i];
}

void ggml_cuda_op_host_moe_publish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1], * w = dst->src[2];
    const ggml_host_moe_bell * bell;
    memcpy(&bell, dst->op_params, sizeof(bell));
    const int tag = ((const int32_t *) dst->op_params)[2];
    const int nx = (int) ggml_nelements(x), nk = (int) ggml_nelements(ids), nt = (int) x->ne[1];
    host_moe_publish<<<1, 1024, 0, ctx.stream()>>>((const float *) x->data, (const int32_t *) ids->data, (const float *) w->data,
                                                   nx, nk, tag, nt, bell->x, bell->ids, bell->w, bell->meta, bell->seq);
}

void ggml_cuda_op_host_moe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[1];                                  // for its size: the answer is in bell->out
    const ggml_host_moe_bell * bell;
    memcpy(&bell, dst->op_params, sizeof(bell));
    const int nx = (int) ggml_nelements(x);
    host_moe_wait_copy<<<1, 1024, 0, ctx.stream()>>>(bell->flag, bell->seq, bell->out, (float *) dst->data, nx);
}

void * ggml_backend_cuda_alloc_mapped(size_t size) {
    void * p = nullptr;
    if (cudaHostAlloc(&p, size, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return nullptr;
    memset(p, 0, size);
    return p;
}

void ggml_backend_cuda_free_mapped(void * ptr) {
    if (ptr) cudaFreeHost(ptr);
}

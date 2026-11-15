#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}


// ---- Strata-DS4: GGML_OP_DSV41_FAKE_QUANT - DeepSeek-V4.1's activation quantize-dequantize in one kernel.
// tools/v41/engine/ds41_ops.hpp builds it from ~45 stock ops per call (abs, pool max, clamp, div, scale, step, ...);
// this repeats that composite step for step under the same -use_fast_math flags, so the values are bit-identical:
// a stock binary op is one plain op (__fadd_rn / __fsub_rn / __fmul_rn: .rn.ftz, never contracted, = binbcast's
// a op b), division is '/' (div.approx.ftz, = binbcast's op_div), ggml_scale is scale_f32's `scale * x + bias` = one
// FMA (__fmaf_rn), abs = fabsf, step = x > 0, clamp = fminf(fmaxf), the pool's max = fmaxf over the block.

static __device__ __forceinline__ float dsv41_fq_scale(float x, float s, float b) { return __fmaf_rn(s, x, b); }
static __device__ __forceinline__ float dsv41_fq_step(float x) { return x > 0.0f ? 1.0f : 0.0f; }
static __device__ __forceinline__ float dsv41_fq_clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

static __device__ __forceinline__ float dsv41_fq_pow2_ceil(float a) {
    const float g  = dsv41_fq_scale(a, 8388609.0f, 0.0f);                 // a * (2^23 + 1)
    const float hi = __fadd_rn(g, __fsub_rn(a, g));                        // a rounded to 1 significant bit
    return __fmul_rn(hi, dsv41_fq_scale(dsv41_fq_step(__fsub_rn(a, hi)), 1.0f, 1.0f));
}

static __device__ __forceinline__ float dsv41_fq_pow2_floor(float a) {
    const float f = dsv41_fq_pow2_ceil(a);
    return __fmul_rn(f, dsv41_fq_scale(dsv41_fq_step(__fsub_rn(f, a)), -0.5f, 1.0f));
}

static __device__ __forceinline__ float dsv41_fq_round_fmt(float v, int mbits, float min_normal) {
    const float p = dsv41_fq_clamp(dsv41_fq_pow2_floor(fabsf(v)), min_normal, INFINITY);
    const float k = (float) (1 << mbits);
    const float C = 12582912.0f;                                           // 1.5 * 2^23: round half to even
    const float y = dsv41_fq_scale(v / p, k, 0.0f);
    const float r = dsv41_fq_scale(dsv41_fq_scale(y, 1.0f, C), 1.0f, -C);
    return __fmul_rn(dsv41_fq_scale(r, 1.0f / k, 0.0f), p);
}

// one thread per element; a block of BS elements = BS consecutive lanes (n % BS == 0, so a lane group is wholly in or
// wholly out of range and every lane takes part in the shuffles)
template <int BS>
static __global__ void dsv41_fake_quant_f32(const float * __restrict__ x, float * __restrict__ y, const int64_t n,
                                            const int mode, const float maxv, const float floor_) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const float xi = i < n ? x[i] : 0.0f;
    float a = i < n ? fabsf(xi) : -FLT_MAX;
#pragma unroll
    for (int o = BS / 2; o > 0; o >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o, BS));
    }
    if (i >= n) {
        return;
    }
    const float t = dsv41_fq_clamp(a, floor_, INFINITY) / maxv;
    const float s = mode == 1 ? dsv41_fq_round_fmt(t, 3, 0x1p-6f) : dsv41_fq_pow2_ceil(t);
    const float v = dsv41_fq_clamp(xi / s, -maxv, maxv);
    const float q = mode == 0 ? dsv41_fq_round_fmt(v, 3, 0x1p-6f) : dsv41_fq_round_fmt(v, 1, 1.0f);
    y[i] = __fmul_rn(q, s);
}

void ggml_cuda_op_dsv41_fake_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(src0));
    const int mode = ggml_get_op_params_i32(dst, 0);
    const int64_t n = ggml_nelements(src0);
    const float maxv = mode == 0 ? 448.0f : 6.0f;
    const float floor_ = mode == 0 ? 1e-4f : mode == 1 ? 6.0f / 512.0f : 6.0f * 0x1p-126f;
    const int64_t blocks = (n + 255) / 256;
    cudaStream_t stream = ctx.stream();
    if (mode == 1) {
        GGML_ASSERT(n % 16 == 0);
        dsv41_fake_quant_f32<16><<<blocks, 256, 0, stream>>>((const float *) src0->data, (float *) dst->data, n, mode, maxv, floor_);
    } else {
        GGML_ASSERT(n % 32 == 0);
        dsv41_fake_quant_f32<32><<<blocks, 256, 0, stream>>>((const float *) src0->data, (float *) dst->data, n, mode, maxv, floor_);
    }
    CUDA_CHECK(cudaGetLastError());
}


// ---- Strata-DS4: GGML_OP_DSV41_HC_MIXES - the two HC mixes of one model.py Block.hc_mixes in one kernel.
// tools/v41/engine/ds41_dense.cpp hc_mixes builds them from mul + add + sigmoid + ggml_scale(_bias); this repeats that
// composite step for step under the same -use_fast_math flags, so the values are bit-identical:
// a stock binary op is one plain op (__fmul_rn / __fadd_rn: .rn.ftz, never contracted, = binbcast's a op b), ggml_scale
// is scale_f32's `scale * x + bias` = one FMA (__fmaf_rn), sigmoid is unary_op_kernel's op_sigmoid.
static __device__ __forceinline__ float dsv41_hc_mix_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// one thread per output element: dst [hc, 2*nt] = [ pre | post ]
static __global__ void dsv41_hc_mixes_f32(
        const float * __restrict__ mixes,
        const int64_t nrow,                                // mixes->ne[0] = 2*hc (+ hc*hc comb rows the op does not read)
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float * __restrict__ dst,
        const int64_t hc,
        const int64_t nt,
        const float eps) {
    const int64_t n = 2*hc*nt;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t h    = i % hc;
    const int64_t col  = i / hc;                       // 0..nt-1 = the pre mix, nt..2*nt-1 = the post mix
    const int64_t post = col >= nt;
    const int64_t t    = post ? col - nt : col;
    const int64_t r    = post ? hc + h : h;
    const float   v    = __fadd_rn(__fmul_rn(mixes[r + nrow*t], scale[post]), base[r]);
    const float   s    = dsv41_hc_mix_sigmoid(v);
    dst[i] = post ? __fmaf_rn(2.0f, s, 0.0f) : __fmaf_rn(1.0f, s, eps);
}

// ---- Almond: ggml_dsv41_hc_head - one token's Block.hc_mixes in one kernel (rms_norm, the 24-row product, the
// pre / post mixes, the comb's Sinkhorn). Each block sums its slice of the row (24 dot products and the sum of squares);
// the last block to finish adds the partials and does the 24 small outputs. The sums run in another order than
// rms_norm + mul_mat, so the last bits differ.
static constexpr int DSV41_HEAD_NC = 80;               // column chunks
static constexpr int DSV41_HEAD_NG = 1;                // row groups (4 x 6 rows measured slower)
static constexpr int DSV41_HEAD_NB = DSV41_HEAD_NC * DSV41_HEAD_NG;
static constexpr int DSV41_HEAD_NT = 256;
static __device__ float    dsv41_head_part[DSV41_HEAD_NB * 25];
static __device__ unsigned dsv41_head_count = 0;

static __global__ void __launch_bounds__(DSV41_HEAD_NT) dsv41_hc_head_f32(
        const float * __restrict__ x, const float * __restrict__ fn, const float * __restrict__ scale,
        const float * __restrict__ base, float * __restrict__ dst, const int K, const float rms_eps,
        const float hc_eps, const int n_iter) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = DSV41_HEAD_NT / warp_size;
    __shared__ float red[25][nwarps];
    __shared__ bool last;
    float acc[25];
#pragma unroll
    for (int j = 0; j < 25; ++j) acc[j] = 0.0f;
    const int cb = blockIdx.x % DSV41_HEAD_NC, rg = blockIdx.x / DSV41_HEAD_NC;
    for (int c = cb * DSV41_HEAD_NT + threadIdx.x; c < K; c += DSV41_HEAD_NC * DSV41_HEAD_NT) {
        const float xv = x[c];
#pragma unroll
        for (int r = 0; r < 24; ++r) {
            if (r * DSV41_HEAD_NG / 24 == rg) acc[r] += fn[(int64_t) r * K + c] * xv;
        }
        if (rg == 0) acc[24] += xv * xv;
    }
    const int lane = threadIdx.x % warp_size, w = threadIdx.x / warp_size;
#pragma unroll
    for (int j = 0; j < 25; ++j) {
        const float v = warp_reduce_sum<warp_size>(acc[j]);
        if (lane == 0) red[j][w] = v;
    }
    __syncthreads();
    if (threadIdx.x < 25) {
        float v = 0.0f;
        for (int i = 0; i < nwarps; ++i) v += red[threadIdx.x][i];
        dsv41_head_part[blockIdx.x * 25 + threadIdx.x] = v;
        __threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(&dsv41_head_count, 1u) == DSV41_HEAD_NB - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    // the partials in parallel: thread t < 200 sums blocks t/25, t/25 + 8, ... of value t%25 (10 independent loads)
    __shared__ float tot[25];
    __shared__ float sub[8][25];
    static_assert(DSV41_HEAD_NB % 8 == 0, "");
    if (threadIdx.x < 200) {
        const int j = threadIdx.x % 25, g = threadIdx.x / 25;
        float v[DSV41_HEAD_NB / 8];
#pragma unroll
        for (int i = 0; i < DSV41_HEAD_NB / 8; ++i) v[i] = __builtin_nontemporal_load(&dsv41_head_part[(g + 8 * i) * 25 + j]);
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < DSV41_HEAD_NB / 8; ++i) t += v[i];
        sub[g][j] = t;
    }
    __syncthreads();
    if (threadIdx.x < 25) {
        float t = 0.0f;
        for (int g = 0; g < 8; ++g) t += sub[g][threadIdx.x];
        tot[threadIdx.x] = t;
    }
    __syncthreads();
    if (threadIdx.x >= warp_size) return;
    // warp 0: lanes 0-15 the comb (lane = idst + 4*isrc), lanes 16-23 the pre / post mixes
    const int l = threadIdx.x;
    if (l == 0) dsv41_head_count = 0;
    const float rs = rsqrtf(tot[24] / (float) K + rms_eps);
    if (l >= 16 && l < 24) {
        const int h = l - 16;
        const float v = tot[h] * rs * scale[h < DSV4_HC ? 0 : 1] + base[h];
        const float sg = 1.0f / (1.0f + expf(-v));
        dst[h] = h < DSV4_HC ? sg + hc_eps : 2.0f * sg;
    }
    const int idx = l & 15;
    float v = tot[2 * DSV4_HC + idx] * rs * scale[2] + base[2 * DSV4_HC + idx];
    // softmax over idst (lanes xor 1, 2), then + eps
    float mx = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1, warp_size)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2, warp_size));
    v = expf(v - mx);
    float sm = v + __shfl_xor_sync(0xffffffff, v, 1, warp_size); sm += __shfl_xor_sync(0xffffffff, sm, 2, warp_size);
    v = v * (1.0f / sm) + hc_eps;
    auto norm_cols = [&](float x) {   // sum over isrc: lanes xor 4, 8
        float t = x + __shfl_xor_sync(0xffffffff, x, 4, warp_size); t += __shfl_xor_sync(0xffffffff, t, 8, warp_size);
        return x * (1.0f / (t + hc_eps));
    };
    auto norm_rows = [&](float x) {   // sum over idst: lanes xor 1, 2
        float t = x + __shfl_xor_sync(0xffffffff, x, 1, warp_size); t += __shfl_xor_sync(0xffffffff, t, 2, warp_size);
        return x * (1.0f / (t + hc_eps));
    };
    v = norm_cols(v);
    for (int i = 1; i < n_iter; ++i) {
        v = norm_rows(v);
        v = norm_cols(v);
    }
    if (l < 16) dst[2 * DSV4_HC + l] = v;
}

void ggml_cuda_op_dsv41_hc_mixes(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 1) == 1) {
        const ggml_tensor * x  = dst->src[0];
        const ggml_tensor * fn = dst->src[3];
        GGML_ASSERT(x->type == GGML_TYPE_F32 && fn->type == GGML_TYPE_F32 && fn->ne[1] == 24 && fn->ne[0] == x->ne[0]);
        dsv41_hc_head_f32<<<DSV41_HEAD_NB, DSV41_HEAD_NT, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) fn->data, (const float *) dst->src[1]->data,
            (const float *) dst->src[2]->data, (float *) dst->data, (int) x->ne[0], ggml_get_op_params_f32(dst, 2),
            ggml_get_op_params_f32(dst, 0), ggml_get_op_params_i32(dst, 3));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];
    GGML_ASSERT(mixes->type == GGML_TYPE_F32 && ggml_is_contiguous(mixes));
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && ggml_nelements(scale) == 2);
    GGML_ASSERT(base->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[1] % 2 == 0 && mixes->ne[0] >= dst->ne[0] * 2);
    const int64_t hc = base->ne[0] / 2;
    const int64_t nt = dst->ne[1] / 2;
    const float   eps = ggml_get_op_params_f32(dst, 0);
    const int64_t n = 2*hc*nt;
    const int64_t blocks = (n + 255) / 256;
    cudaStream_t stream = ctx.stream();
    dsv41_hc_mixes_f32<<<blocks, 256, 0, stream>>>((const float *) mixes->data, mixes->ne[0], (const float *) scale->data,
                                                   (const float *) base->data, (float *) dst->data, hc, nt, eps);
    CUDA_CHECK(cudaGetLastError());
}

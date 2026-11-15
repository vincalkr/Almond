// Decode attention over a q8_0 KV cache, split over the context (opt-in: GGML_CUDA_FA_Q8_SPLITK=1).
//
// For a few query tokens (decode, a speculative verify window) with grouped-query attention the tile kernel first
// converts the whole K and V views to f16 and then reads them once per query head: at 123K tokens of q8_0 KV (D 256,
// 4 KV heads) that was 2.3 ms per layer on an RX 7900 XTX against ~0.3 ms to read the q8_0 bytes once. Here one block
// takes one KV head and CH context positions and computes all R = n_tokens * gqa query rows of that head from the
// q8_0 bytes (an online softmax over sub-chunks of SUB positions); a second kernel merges the chunks.
#include "common.cuh"
#include "fattn-q8-splitk.cuh"

#include <cstdlib>

namespace {

constexpr int FQ_D   = 256;   // head size (the only one supported)
constexpr int FQ_SUB = 64;    // positions per sub-chunk (FQ_NT / FQ_SUB threads per position for the scores)
constexpr int FQ_CH  = 256;   // positions per block
constexpr int FQ_NT  = 256;   // threads per block (one per output dimension for P*V)

template <int R>
__launch_bounds__(FQ_NT, 1)
__global__ void fattn_q8_splitk(const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
                                const char * __restrict__ mask, float * __restrict__ part_o, float2 * __restrict__ part_ml,
                                const int n_kv, const int gqa, const float scale,
                                const int64_t q_nb1, const int64_t q_nb2,
                                const int64_t k_nb1, const int64_t k_nb2, const int64_t v_nb1, const int64_t v_nb2,
                                const int64_t m_nb1) {
    constexpr int RP   = (R + 3) / 4 * 4;              // rows padded to 4: one 16-byte LDS read gives 4 rows
    constexpr int ROWB = FQ_D / QK8_0 * sizeof(block_q8_0);   // 272 bytes: one head's row of q8_0
    constexpr int ROW4 = ROWB / 16;                     // 17 16-byte words
    const int chunk = blockIdx.x, kvh = blockIdx.y, nchunk = gridDim.x;
    const int tid = threadIdx.x;

    __shared__ __align__(16) half2 Qs[FQ_D / 2][RP];
    __shared__ __align__(16) float S[FQ_SUB][RP];
    __shared__ __align__(16) uint4 Ks[FQ_SUB * ROW4];
    __shared__ __align__(16) uint4 Vs[FQ_SUB * ROW4];
    __shared__ float m_s[R], l_s[R], a_s[R];

    for (int i = tid; i < RP * FQ_D / 2; i += FQ_NT) {
        const int r = i / (FQ_D / 2), d2 = i % (FQ_D / 2);
        float2 q = make_float2(0.0f, 0.0f);
        if (r < R) {
            const int j = r / gqa, h = kvh * gqa + r % gqa;
            q = ((const float2 *) (Q + j * q_nb1 + h * q_nb2))[d2];
        }
        // the 4 lanes of a position read 4 dimension ranges 256 bytes apart in LDS: rotate the rows per range so
        // their 16-byte reads fall on different banks
        Qs[d2][(r + 4 * (d2 / (QK8_0 / 2))) % RP] = __floats2half2_rn(q.x * scale, q.y * scale);
    }
    if (tid < R) { m_s[tid] = -INFINITY; l_s[tid] = 0.0f; }

    float acc[R], acc2[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { acc[r] = 0.0f; acc2[r] = 0.0f; }

    const int p_begin = chunk * FQ_CH, p_end = min(n_kv, p_begin + FQ_CH);
    for (int p0 = p_begin; p0 < p_end; p0 += FQ_SUB) {
        const int pn = min(FQ_SUB, p_end - p0);
        __syncthreads();
        // the sub-chunk's K and V rows of this head into LDS, 16 bytes a thread
        for (int i = tid; i < FQ_SUB * ROW4; i += FQ_NT) {
            const int c = i / ROW4, w = i % ROW4;
            if (c < pn) {
                Ks[i] = ((const uint4 *) (K + (int64_t) (p0 + c) * k_nb1 + kvh * k_nb2))[w];
                Vs[i] = ((const uint4 *) (V + (int64_t) (p0 + c) * v_nb1 + kvh * v_nb2))[w];
            }
        }
        __syncthreads();
        // scores: 8 lanes (one q8_0 block of dimensions each) per pair of positions c and c + FQ_SUB/2: each Q read
        // from LDS feeds both
        {
            const int c = tid / 8, part = tid % 8;
            float s0[R], s1[R];
#pragma unroll
            for (int r = 0; r < R; ++r) { s0[r] = 0.0f; s1[r] = 0.0f; }
            const block_q8_0 * kb0 = (const block_q8_0 *) ((const char *) Ks + c * ROWB) + part;
            const block_q8_0 * kb1 = (const block_q8_0 *) ((const char *) Ks + (c + FQ_SUB / 2) * ROWB) + part;
            const float dk0 = __half2float(kb0->d), dk1 = __half2float(kb1->d);
            const unsigned short * w0 = (const unsigned short *) kb0->qs;
            const unsigned short * w1 = (const unsigned short *) kb1->qs;
            const int d2b = part * (QK8_0 / 2);
#pragma unroll 1
            for (int e = 0; e < QK8_0 / 2; ++e) {
                const unsigned short x0 = w0[e], x1 = w1[e];
                const float a0 = dk0 * (float) (signed char) (x0 & 0xff), a1 = dk0 * (float) (signed char) (x0 >> 8);
                const float b0 = dk1 * (float) (signed char) (x1 & 0xff), b1 = dk1 * (float) (signed char) (x1 >> 8);
#pragma unroll
                for (int r4 = 0; r4 < R; r4 += 4) {
                    const float4 raw = *(const float4 *) &Qs[d2b + e][(r4 + 4 * part) % RP];
                    const half2 * h4 = (const half2 *) &raw;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        if (r4 + k < R) {
                            const float2 q2 = __half22float2(h4[k]);
                            s0[r4 + k] += q2.x * a0 + q2.y * a1;
                            s1[r4 + k] += q2.x * b0 + q2.y * b1;
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int o = 1; o < 8; o *= 2) {
                    s0[r] += __shfl_xor(s0[r], o, 8);
                    s1[r] += __shfl_xor(s1[r], o, 8);
                }
            }
            if (part < 2) {
                const int cc = c + part * (FQ_SUB / 2);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float v = -INFINITY;
                    if (cc < pn) {
                        v = part ? s1[r] : s0[r];
                        if (mask) v += __half2float(((const half *) (mask + (r / gqa) * m_nb1))[p0 + cc]);
                    }
                    S[cc][r] = v;
                }
            }
        }
        __syncthreads();
        if (tid < R) {
            float mx = m_s[tid];
            for (int i = 0; i < FQ_SUB; ++i) mx = fmaxf(mx, S[i][tid]);
            a_s[tid] = mx == -INFINITY ? 1.0f : __expf(m_s[tid] - mx);
            m_s[tid] = mx;
        }
        __syncthreads();
        for (int i = tid; i < R * FQ_SUB; i += FQ_NT) {
            const int c = i / R, r = i % R;
            const float mx = m_s[r];
            S[c][r] = mx == -INFINITY ? 0.0f : __expf(S[c][r] - mx);
        }
        __syncthreads();
        if (tid < R) {
            float sum = 0.0f;
            for (int i = 0; i < FQ_SUB; ++i) sum += S[i][tid];
            l_s[tid] = l_s[tid] * a_s[tid] + sum;
        }
        // P*V: thread tid owns dimensions 2*(tid%128) and +1 over the positions of parity tid/128
#pragma unroll
        for (int r = 0; r < R; ++r) { acc[r] *= a_s[r]; acc2[r] *= a_s[r]; }
        {
            const int dd = 2 * (tid % (FQ_NT / 2)), bi = dd / QK8_0, ei = dd % QK8_0;
#pragma unroll 2
            for (int c = tid / (FQ_NT / 2); c < pn; c += 2) {
                const block_q8_0 * vb = (const block_q8_0 *) ((const char *) Vs + c * ROWB) + bi;
                const float dv = __half2float(vb->d);
                const unsigned short x = *(const unsigned short *) &vb->qs[ei];
                const float v0 = dv * (float) (signed char) (x & 0xff), v1 = dv * (float) (signed char) (x >> 8);
#pragma unroll
                for (int r4 = 0; r4 < R; r4 += 4) {
                    const float4 p4 = *(const float4 *) &S[c][r4];
                    const float pp[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        if (r4 + k < R) { acc[r4 + k] += pp[k] * v0; acc2[r4 + k] += pp[k] * v1; }
                    }
                }
            }
        }
    }
    __syncthreads();
    // the two parities: the odd half adds its sums through LDS (the K/V staging is free now)
    float * red = (float *) Ks;   // R * FQ_D floats
    if (tid >= FQ_NT / 2) {
        const int dd = 2 * (tid % (FQ_NT / 2));
#pragma unroll
        for (int r = 0; r < R; ++r) { red[r * FQ_D + dd] = acc[r]; red[r * FQ_D + dd + 1] = acc2[r]; }
    }
    __syncthreads();
    const int64_t base = ((int64_t) kvh * nchunk + chunk) * R;
    if (tid < FQ_NT / 2) {
        const int dd = 2 * tid;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float2 o = make_float2(acc[r] + red[r * FQ_D + dd], acc2[r] + red[r * FQ_D + dd + 1]);
            *(float2 *) &part_o[(base + r) * FQ_D + dd] = o;
        }
    }
    if (tid < R) part_ml[base + tid] = make_float2(m_s[tid], l_s[tid]);
}

__global__ void fattn_q8_splitk_merge(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
                                      float * __restrict__ dst, const int R, const int nchunk, const int gqa,
                                      const int64_t d_nb1, const int64_t d_nb2) {
    const int r = blockIdx.x, kvh = blockIdx.y, d = threadIdx.x;
    float M = -INFINITY;
    for (int c = 0; c < nchunk; ++c) M = fmaxf(M, part_ml[((int64_t) kvh * nchunk + c) * R + r].x);
    float L = 0.0f, O = 0.0f;
    if (M != -INFINITY) {
        for (int c = 0; c < nchunk; ++c) {
            const int64_t i = ((int64_t) kvh * nchunk + c) * R + r;
            const float2 ml = part_ml[i];
            if (ml.x == -INFINITY) continue;
            const float w = __expf(ml.x - M);
            L += ml.y * w;
            O += part_o[i * FQ_D + d] * w;
        }
    }
    const int j = r / gqa, h = kvh * gqa + r % gqa;
    ((float *) ((char *) dst + j * d_nb2 + h * d_nb1))[d] = L > 0.0f ? O / L : 0.0f;
}

template <int R>
void launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int n_kv = K->ne[1], n_head_kv = K->ne[2], gqa = Q->ne[2] / K->ne[2];
    const int nchunk = (n_kv + FQ_CH - 1) / FQ_CH;
    cudaStream_t st = ctx.stream();
    ggml_cuda_pool_alloc<float>  part_o(ctx.pool(), (size_t) n_head_kv * nchunk * R * FQ_D);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), (size_t) n_head_kv * nchunk * R);
    fattn_q8_splitk<R><<<dim3(nchunk, n_head_kv), FQ_NT, 0, st>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        mask ? (const char *) mask->data : nullptr, part_o.get(), part_ml.get(), n_kv, gqa, scale,
        Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask ? mask->nb[1] : 0);
    fattn_q8_splitk_merge<<<dim3(R, n_head_kv), FQ_D, 0, st>>>(part_o.get(), part_ml.get(), (float *) dst->data, R,
                                                                nchunk, gqa, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

bool ggml_cuda_fattn_q8_splitk_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_FA_Q8_SPLITK"); return e && atoi(e) != 0; }();
    if (!on) return false;
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 || Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return false;
    if (Q->ne[0] != FQ_D || K->ne[0] != FQ_D || V->ne[0] != FQ_D || Q->ne[3] != 1 || K->ne[3] != 1) return false;
    if (Q->nb[0] != sizeof(float) || K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) return false;
    if (dst->src[4] != nullptr) return false;   // sinks
    if (mask && mask->type != GGML_TYPE_F16) return false;
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) return false;
    if (K->ne[1] < 16384) return false;   // under 16K the existing kernels (even at 8K, measured)
    const int R = Q->ne[1] * (Q->ne[2] / K->ne[2]);
    switch (R) {
        case 6:  launch<6>(ctx, dst);  return true;
        case 12: launch<12>(ctx, dst); return true;
        case 18: launch<18>(ctx, dst); return true;
        case 24: launch<24>(ctx, dst); return true;
        default: return false;
    }
}

// src/inkling/native/ink_native.hip - almond-inkling-native: Almond's own engine for Inkling 2 Mini (GGUF
// architecture inkling2_loop), HIP for the RX 7900 XTX, no ggml. docs/inkling/ENGINE.md has the design and the math;
// the reference is llama.cpp's src/models/inkling2*.cpp, which this follows op for op.
//
// Step 1 (this file): plain kernels, one op at a time, F32 activations, one token at a time - written to be checked
// against llama.cpp, not to be fast.
//
//   almond-inkling-native -m MODEL.gguf [--gpu N] [-c N] --score IDS.txt [--dump-logits FILE N]
//
// --score: the token ids of a text (whitespace separated); every token is fed in order, and the mean negative
// log-likelihood of the second half's next tokens gives a PPL (the same rule as almond-inkling --score).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <tuple>
#include <map>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {

// ---------------------------------------------------------------- GGUF
enum : uint32_t { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_BF16 = 30, T_NVFP4 = 40,
                  T_Q8P = 1008 };   // ours: Q8_0 repacked at load, a row = its K int8 quants, then its K/32 F16 scales

struct Tensor {
    std::string name;
    uint32_t type = 0;
    std::vector<int64_t> ne;
    uint64_t off = 0;           // from the data section
    const uint8_t * host = nullptr;
    size_t bytes = 0;
    void * dev = nullptr;
};

size_t row_bytes(uint32_t type, int64_t n) {
    switch (type) {
        case T_F32: return 4 * n;
        case T_F16: case T_BF16: return 2 * n;
        case T_Q8_0: return n / 32 * 34;
        case T_Q8P: return n + n / 32 * 2;
        case T_NVFP4: return n / 64 * 36;
    }
    std::fprintf(stderr, "unknown tensor type %u\n", type);
    std::exit(1);
}

struct Gguf {
    const uint8_t * map = nullptr;
    size_t size = 0;
    std::map<std::string, Tensor> t;
    std::map<std::string, double> num;          // numeric keys (scalars)
    std::map<std::string, std::vector<double>> arr;

    void open(const std::string & path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { std::perror(path.c_str()); std::exit(1); }
        struct stat st{};
        fstat(fd, &st);
        size = (size_t) st.st_size;
        map = (const uint8_t *) mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
        const uint8_t * p = map;
        auto rd = [&](void * d, size_t n) { std::memcpy(d, p, n); p += n; };
        auto u32 = [&]() { uint32_t v; rd(&v, 4); return v; };
        auto u64 = [&]() { uint64_t v; rd(&v, 8); return v; };
        auto str = [&]() { uint64_t n = u64(); std::string s((const char *) p, n); p += n; return s; };
        if (u32() != 0x46554747u) { std::fprintf(stderr, "%s: not a GGUF\n", path.c_str()); std::exit(1); }
        u32();
        const uint64_t n_t = u64(), n_kv = u64();
        uint64_t align = 32;
        std::function<double(uint32_t)> scalar;
        auto skip_or_num = [&](uint32_t ty, double * out) {
            double v = 0;
            switch (ty) {
                case 0: { uint8_t x; rd(&x, 1); v = x; } break;
                case 1: { int8_t x; rd(&x, 1); v = x; } break;
                case 2: { uint16_t x; rd(&x, 2); v = x; } break;
                case 3: { int16_t x; rd(&x, 2); v = x; } break;
                case 4: { uint32_t x; rd(&x, 4); v = x; } break;
                case 5: { int32_t x; rd(&x, 4); v = x; } break;
                case 6: { float x; rd(&x, 4); v = x; } break;
                case 7: { uint8_t x; rd(&x, 1); v = x; } break;
                case 8: { str(); } return false;
                case 10: { uint64_t x; rd(&x, 8); v = (double) x; } break;
                case 11: { int64_t x; rd(&x, 8); v = (double) x; } break;
                case 12: { double x; rd(&x, 8); v = x; } break;
                default: std::fprintf(stderr, "gguf: value type %u\n", ty); std::exit(1);
            }
            if (out) *out = v;
            return true;
        };
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string k = str();
            const uint32_t ty = u32();
            if (ty == 9) {
                const uint32_t et = u32();
                const uint64_t n = u64();
                std::vector<double> v;
                for (uint64_t j = 0; j < n; ++j) { double x = 0; if (skip_or_num(et, &x)) v.push_back(x); }
                if (et != 8) arr[k] = v;
            } else {
                double x = 0;
                if (skip_or_num(ty, &x)) num[k] = x;
            }
            if (k == "general.alignment") align = (uint64_t) num[k];
        }
        std::vector<Tensor> list;
        for (uint64_t i = 0; i < n_t; ++i) {
            Tensor x;
            x.name = str();
            const uint32_t nd = u32();
            for (uint32_t d = 0; d < nd; ++d) x.ne.push_back((int64_t) u64());
            x.type = u32();
            x.off = u64();
            list.push_back(x);
        }
        size_t data = (size_t) (p - map);
        data = (data + align - 1) / align * align;
        for (auto & x : list) {
            int64_t rows = 1;
            for (size_t d = 1; d < x.ne.size(); ++d) rows *= x.ne[d];
            x.bytes = row_bytes(x.type, x.ne[0]) * rows;
            x.host = map + data + x.off;
            t[x.name] = x;
        }
    }
    double get(const std::string & k) const {
        auto it = num.find(k);
        if (it == num.end()) { std::fprintf(stderr, "gguf: no key %s\n", k.c_str()); std::exit(1); }
        return it->second;
    }
    Tensor * find(const std::string & n) { auto it = t.find(n); return it == t.end() ? nullptr : &it->second; }
    // upload a tensor; Q8_0 is repacked into T_Q8P (the same values: aligned 8-byte quant loads) unless raw
    Tensor & need(const std::string & n, bool raw = false) {
        Tensor * x = find(n);
        if (!x) { std::fprintf(stderr, "gguf: no tensor %s\n", n.c_str()); std::exit(1); }
        if (!x->dev) {
            if (x->type == T_Q8_0 && !raw) {
                const int64_t K = x->ne[0], rows = (int64_t) (x->bytes / row_bytes(T_Q8_0, K));
                const size_t rb = row_bytes(T_Q8P, K);
                std::vector<uint8_t> out((size_t) rows * rb);
                for (int64_t r = 0; r < rows; ++r) {
                    const uint8_t * src = x->host + (size_t) r * row_bytes(T_Q8_0, K);
                    uint8_t * q = out.data() + (size_t) r * rb;
                    uint8_t * d = q + K;
                    for (int64_t b = 0; b < K / 32; ++b) {
                        std::memcpy(d + b * 2, src + b * 34, 2);
                        std::memcpy(q + b * 32, src + b * 34 + 2, 32);
                    }
                }
                x->type = T_Q8P;
                x->bytes = out.size();
                CK(hipMalloc(&x->dev, x->bytes));
                CK(hipMemcpy(x->dev, out.data(), x->bytes, hipMemcpyHostToDevice));
            } else {
                CK(hipMalloc(&x->dev, x->bytes));
                CK(hipMemcpy(x->dev, x->host, x->bytes, hipMemcpyHostToDevice));
            }
        }
        return *x;
    }
};

// ---------------------------------------------------------------- kernels
constexpr int WS = 32;   // wave32 on gfx11

__device__ __forceinline__ float wsum(float v) {
    for (int o = WS / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
    for (int o = WS / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
    return v;
}
// block reduction (blockDim a multiple of 32, at most 1024)
__device__ float bsum(float v) {
    __shared__ float s[32];
    v = wsum(v);
    const int w = threadIdx.x / WS, l = threadIdx.x % WS;
    __syncthreads();
    if (l == 0) s[w] = v;
    __syncthreads();
    const int nw = blockDim.x / WS;
    v = threadIdx.x < nw ? s[threadIdx.x] : 0.0f;
    if (w == 0) v = wsum(v);
    __shared__ float r;
    if (threadIdx.x == 0) r = v;
    __syncthreads();
    return r;
}

__device__ __forceinline__ float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 0x7;
    const float raw = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;
}
// E2M1, doubled as ggml's kvalues_fp4 (0 1 2 3 4 6 8 12, then negated); the kernels that read NVFP4 copy it to LDS
__constant__ float kFp4c[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
__shared__ float sFp4[16];
__device__ float gUe4m3[256];   // NVFP4's sub-block scales, decoded once at start (k_init_tables)
__global__ void k_init_tables() { gUe4m3[threadIdx.x] = ue4m3((uint8_t) threadIdx.x); }
__device__ __forceinline__ void fp4_init() {
    if (threadIdx.x < 16) sFp4[threadIdx.x] = kFp4c[threadIdx.x];
    __syncthreads();
}
__device__ __forceinline__ float fp4(uint32_t v) { return sFp4[v]; }

// one row of a matrix in its stored type, dotted with x (F32), by one wave. Quantized rows: 4 lanes a block, each
// lane 8 (Q8_0) or 16 (NVFP4) values with vector loads; the products are F32, as in the plain version.
__device__ __forceinline__ float dot8_i8(const uint8_t * q, const float * x) {
    const uint16_t * h = (const uint16_t *) q;    // the quants of a Q8_0 block start 2 bytes in: 16-bit loads
    const float4 x0 = *(const float4 *) x, x1 = *(const float4 *) (x + 4);
    const uint16_t a = h[0], b = h[1], c = h[2], d = h[3];
    return (float) (int8_t) (a & 0xFF) * x0.x + (float) (int8_t) (a >> 8) * x0.y + (float) (int8_t) (b & 0xFF) * x0.z +
           (float) (int8_t) (b >> 8) * x0.w + (float) (int8_t) (c & 0xFF) * x1.x + (float) (int8_t) (c >> 8) * x1.y +
           (float) (int8_t) (d & 0xFF) * x1.z + (float) (int8_t) (d >> 8) * x1.w;
}
__device__ float row_dot(const uint8_t * row, uint32_t type, const float * x, int K) {
    const int l = threadIdx.x % WS;
    float acc = 0.0f;
    if (type == T_F32) {
        const float4 * w = (const float4 *) row;
        const float4 * xv = (const float4 *) x;
        for (int k = l; k < K / 4; k += WS) {
            const float4 a = w[k], b = xv[k];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else if (type == T_BF16) {
        const uint16_t * w = (const uint16_t *) row;
        for (int k = l; k < K; k += WS) acc += __uint_as_float((uint32_t) w[k] << 16) * x[k];
    } else if (type == T_Q8P) {
        const __half * dsc = (const __half *) (row + K);
        for (int c = l; c < K / 8; c += WS) {
            const uint2 qq = *(const uint2 *) (row + (size_t) c * 8);
            const float4 x0 = *(const float4 *) (x + c * 8), x1 = *(const float4 *) (x + c * 8 + 4);
            const float dot = (float) (int8_t) (qq.x & 0xFF) * x0.x + (float) (int8_t) ((qq.x >> 8) & 0xFF) * x0.y +
                              (float) (int8_t) ((qq.x >> 16) & 0xFF) * x0.z + (float) (int8_t) (qq.x >> 24) * x0.w +
                              (float) (int8_t) (qq.y & 0xFF) * x1.x + (float) (int8_t) ((qq.y >> 8) & 0xFF) * x1.y +
                              (float) (int8_t) ((qq.y >> 16) & 0xFF) * x1.z + (float) (int8_t) (qq.y >> 24) * x1.w;
            acc += __half2float(dsc[c >> 2]) * dot;
        }
    } else if (type == T_Q8_0) {
        const int sub = l & 3;
        for (int b = l >> 2; b < K / 32; b += WS / 4) {
            const uint8_t * blk = row + (size_t) b * 34;
            const float d = __half2float(*(const __half *) blk);
            acc += d * dot8_i8(blk + 2 + sub * 8, x + b * 32 + sub * 8);
        }
    } else if (type == T_NVFP4) {
        const int sub = l & 3;
        for (int b = l >> 2; b < K / 64; b += WS / 4) {
            const uint8_t * blk = row + (size_t) b * 36;
            const float d = gUe4m3[blk[sub]];
            const uint32_t * qv = (const uint32_t *) (blk + 4 + sub * 8);
            const uint32_t q0 = qv[0], q1 = qv[1];
            const float * xb = x + b * 64 + sub * 16;
            const float4 xa = *(const float4 *) xb, xc = *(const float4 *) (xb + 4);
            const float4 xd = *(const float4 *) (xb + 8), xe = *(const float4 *) (xb + 12);
            const float lo[8] = {xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w};
            const float hi[8] = {xd.x, xd.y, xd.z, xd.w, xe.x, xe.y, xe.z, xe.w};
            float a = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t v = ((j < 4 ? q0 : q1) >> (8 * (j & 3))) & 0xFF;
                a += fp4(v & 0xF) * lo[j] + fp4(v >> 4) * hi[j];
            }
            acc += d * a;
        }
    }
    return wsum(acc);
}

// y[n] = W[n] . x, rows of K; 8 waves a block
__global__ void k_matvec(const uint8_t * W, uint32_t type, const float * x, float * y, int N, int K, size_t rb) {
    fp4_init();
    const int n = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (n >= N) return;
    const float v = row_dot(W + (size_t) n * rb, type, x, K);
    if (threadIdx.x % WS == 0) y[n] = v;
}

// y = rmsnorm(x) * w (w may be null), n channels, one block
__global__ void k_rms(const float * x, const float * w, float * y, int n, float eps) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    s = bsum(s);
    const float r = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * r * (w ? w[i] : 1.0f);
}

// mix: states[ns][C] (rms-normalised inside), queries[nq][C] -> out[nq][C]; one block per query
__global__ void k_mix(const float * const * states, int ns, const float * queries, float * out, int C) {
    const float * q = queries + (size_t) blockIdx.x * C;
    __shared__ float sc[16], rr[16];
    for (int s = 0; s < ns; ++s) {
        const float * x = states[s];
        float a = 0.0f, b = 0.0f;
        for (int i = threadIdx.x; i < C; i += blockDim.x) { a += x[i] * x[i]; b += x[i] * q[i]; }
        a = bsum(a);
        b = bsum(b);
        if (threadIdx.x == 0) { rr[s] = rsqrtf(a / C + 1e-6f); sc[s] = b * rr[s] / sqrtf((float) C); }
        __syncthreads();
    }
    __shared__ float p[16];
    if (threadIdx.x == 0) {
        float m = -INFINITY, z = 0.0f;
        for (int s = 0; s < ns; ++s) m = fmaxf(m, sc[s]);
        for (int s = 0; s < ns; ++s) { p[s] = expf(sc[s] - m); z += p[s]; }
        for (int s = 0; s < ns; ++s) p[s] /= z;
    }
    __syncthreads();
    float * o = out + (size_t) blockIdx.x * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < ns; ++s) v += p[s] * rr[s] * states[s][i];
        o[i] = v;
    }
}

// causal 4-tap conv with residual and optional per-head dynamic taps; hist[3][C] (oldest first) is updated
__global__ void k_conv(float * x, const float * w, float * hist, int C, const float * dyn, int head_dim, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c], v = x[c];
    float o = w[c * 4 + 0] * h0 + w[c * 4 + 1] * h1 + w[c * 4 + 2] * h2 + w[c * 4 + 3] * v;
    if (dyn) {
        const float * t = dyn + (c / head_dim) * 4;
        o += t[0] * v + t[1] * h2 + t[2] * h1 + t[3] * h0;
    }
    hist[c] = h1;
    hist[C + c] = h2;
    hist[2 * C + c] = v;
    x[c] = (v + o) * scale;
}

// x[h][D] *= s[h][D] (s may be null), then scaled to an L2 norm of at most th, then times tau; one block per head
__device__ __forceinline__ float tau_at(int pos, float alpha, float floor_) {
    return alpha > 0.0f ? 1.0f + alpha * logf(fmaxf(1.0f, (float) (pos + 1) / floor_)) : 1.0f;
}
__device__ __forceinline__ float tau_of(const int * pos, float alpha, float floor_) { return tau_at(*pos, alpha, floor_); }
// alpha 0: no log scale
__global__ void k_clip(float * x, const float * s, int D, float th, const int * pos, float alpha, float floor_) {
    const float tau = tau_of(pos, alpha, floor_);
    float * v = x + (size_t) blockIdx.x * D;
    float a = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        if (s) v[i] *= s[(size_t) blockIdx.x * D + i];
        a += v[i] * v[i];
    }
    a = bsum(a);
    const float f = fminf(fmaxf(th / (sqrtf(a) + 1e-6f), 0.0f), 1.0f) * tau;
    for (int i = threadIdx.x; i < D; i += blockDim.x) v[i] *= f;
}

// attention for one query token: block per query head. K/V caches [cap][Hkv][D] (F32), slot = pos % cap; keys at
// positions max(0, pos-span+1) .. pos. bias(d) = R[h] . table[d] - sink[h] for d < ext, else -sink[h]
__global__ void k_attn(const float * q, const float * K, const float * V, int cap, const int * dpos, int span, int Hkv, int D,
                       const float * R, const float * table, int nrel, int ext, const float * sink, float * out,
                       float * scratch, const float * gate) {
    const int h = blockIdx.x, kvh = h / (gridDim.x / Hkv), pos = *dpos;
    const float * qh = q + (size_t) h * D;
    const int first = max(0, pos - span + 1), n = pos - first + 1;
    float * sc = scratch + (size_t) h * span;
    const float sk = sink[h];
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int p = first + j, d = pos - p;
        const float * k = K + ((size_t) (p % cap) * Hkv + kvh) * D;
        float s = 0.0f;
        for (int i = 0; i < D; ++i) s += qh[i] * k[i];
        s /= sqrtf((float) D);
        float b = -sk;
        if (d < ext) {
            const float * tr = table + (size_t) d * nrel;
            const float * rh = R + (size_t) h * nrel;
            for (int i = 0; i < nrel; ++i) b += rh[i] * tr[i];
        }
        sc[j] = s + b;
    }
    __syncthreads();
    float m = 0.0f;   // the zero sink logit
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, sc[j]);
    __shared__ float smax[32];
    m = wmax(m);
    if (threadIdx.x % WS == 0) smax[threadIdx.x / WS] = m;
    __syncthreads();
    if (threadIdx.x == 0) { float x = smax[0]; for (int i = 1; i < (int) blockDim.x / WS; ++i) x = fmaxf(x, smax[i]); smax[0] = x; }
    __syncthreads();
    m = smax[0];
    float z = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) { sc[j] = expf(sc[j] - m); z += sc[j]; }
    z = bsum(z) + expf(-m);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float a = 0.0f;
        for (int j = 0; j < n; ++j) a += sc[j] * V[((size_t) ((first + j) % cap) * Hkv + kvh) * D + i];
        const float gv = gate ? 1.0f / (1.0f + expf(-gate[(size_t) h * D + i])) : 1.0f;
        out[(size_t) h * D + i] = a / z * gv;
    }
}

__global__ void k_store_kv(const float * k, const float * v, float * K, float * V, const int * pos, int cap, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, slot = *pos % cap;
    if (i < n) { K[(size_t) slot * n + i] = k[i]; V[(size_t) slot * n + i] = v[i]; }
}
__global__ void k_gate(float * x, const float * g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-g[i]));
}
__global__ void k_add(float * y, const float * x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
// control: y = x - tanh(f)*x + tanh(g)*m
__global__ void k_control(float * x, const float * m, const float * f, const float * g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] - tanhf(f[i]) * x[i] + tanhf(g[i]) * m[i];
}
// router: sigmoid, top-k of sigmoid + bias, weights = sigmoids normalised times scale; one block of n_exp threads
// (a multiple of 32, at most 256). Ties go to the lower index, as a stable sort would.
__global__ void k_route(const float * logits, const float * bias, int n_exp, int k, float scale, int * ids, float * w) {
    logits += (size_t) blockIdx.x * n_exp; ids += (size_t) blockIdx.x * k; w += (size_t) blockIdx.x * k;
    __shared__ float p[256];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ int pick[16];
    const int i = threadIdx.x, nw = blockDim.x / WS;
    float sel = -INFINITY;
    if (i < n_exp) { p[i] = 1.0f / (1.0f + expf(-logits[i])); sel = p[i] + (bias ? bias[i] : 0.0f); }
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float v = sel;
        int idx = i;
        for (int o = WS / 2; o > 0; o >>= 1) {
            const float v2 = __shfl_xor(v, o);
            const int i2 = __shfl_xor(idx, o);
            if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
        }
        if (i % WS == 0) { bv[i / WS] = v; bi[i / WS] = idx; }
        __syncthreads();
        if (i == 0) {
            float bvv = bv[0];
            int bii = bi[0];
            for (int q = 1; q < nw; ++q) if (bv[q] > bvv || (bv[q] == bvv && bi[q] < bii)) { bvv = bv[q]; bii = bi[q]; }
            pick[j] = bii;
        }
        __syncthreads();
        if (i == pick[j]) sel = -INFINITY;
        __syncthreads();
    }
    if (i == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) sum += p[pick[j]];
        sum = fmaxf(sum, 1e-20f);
        for (int j = 0; j < k; ++j) { ids[j] = pick[j]; w[j] = p[pick[j]] / sum * scale; }
    }
}
// experts gate_up: out[j][2*F] = (W[id_j] . x) * scale[id_j]; grid (rows/8, k)
__global__ void k_exp_gu(const uint8_t * W, uint32_t type, size_t ebytes, size_t rb, const float * scale, const int * ids,
                         const float * x, float * out, int N, int K) {
    fp4_init();
    const int j = blockIdx.y, e = ids[j];
    const int n = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (n >= N) return;
    const float v = row_dot(W + (size_t) e * ebytes + (size_t) n * rb, type, x, K);
    if (threadIdx.x % WS == 0) out[(size_t) j * N + n] = v * (scale ? scale[e] : 1.0f);
}
// SITU: h = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25); gu[j][2F] -> h[j][F]
__global__ void k_situ(const float * gu, float * h, int F, int k, float beta, float lb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F * k) return;
    const int j = i / F, c = i % F;
    const float g = gu[(size_t) j * 2 * F + c], u = gu[(size_t) j * 2 * F + F + c];
    h[i] = beta * tanhf(g / beta) / (1.0f + expf(-g)) * lb * tanhf(u / lb);
}
// experts down, weighted and summed: y[n] = sum_j w_j * scale[id_j] * (Wd[id_j][n] . h_j)
__global__ void k_exp_down(const uint8_t * W, uint32_t type, size_t ebytes, size_t rb, const float * scale, const int * ids,
                           const float * wts, const float * h, float * y, int N, int K, int k) {
    fp4_init();
    const int n = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (n >= N) return;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        const int e = ids[j];
        const float v = row_dot(W + (size_t) e * ebytes + (size_t) n * rb, type, h + (size_t) j * K, K);
        acc += v * (scale ? scale[e] : 1.0f) * wts[j];
    }
    if (threadIdx.x % WS == 0) y[n] = acc;
}
__global__ void k_embed(const uint8_t * E, const int * dtok, float * x, int C) {
    const int tok = dtok[blockIdx.x];
    x += (size_t) blockIdx.x * C;
    const uint8_t * row = E + (size_t) tok * (C / 32 * 34);
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const uint8_t * blk = row + (i / 32) * 34;
        x[i] = __half2float(*(const __half *) blk) * (float) ((const int8_t *) (blk + 2))[i % 32];
    }
}
__global__ void k_softcap(float * x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}


// ---------------------------------------------------------------- fused kernels (step 2)
// several matvecs in one launch: segment j has its own matrix, input and output; 8 rows a block
struct Seg { const uint8_t * w; const float * x; float * y; size_t rb; int N, K; uint32_t type; int xs = 0, ys = 0; };
// (xs, ys: the per-token strides of x and y for the batched GEMM; 0 = K and N)
struct Segs { Seg s[12]; int start[13]; int n; };
__global__ void k_mmv(Segs S) {
    fp4_init();
    const int row = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (row >= S.start[S.n]) return;
    int j = 0;
    while (row >= S.start[j + 1]) ++j;
    const Seg & g = S.s[j];
    const int n = row - S.start[j];
    const float v = row_dot(g.w + (size_t) n * g.rb, g.type, g.x, g.K);
    if (threadIdx.x % WS == 0) g.y[n] = v;
}
// up to 4 RMS norms (weight may be null), one block each; in place allowed
struct Norms { const float * in[4]; const float * w[4]; float * out[4]; };
__global__ void k_rms_n(Norms R, int n, float eps) {
    const float * x = R.in[blockIdx.x] + (size_t) blockIdx.y * n;
    const float * w = R.w[blockIdx.x];
    float * y = R.out[blockIdx.x] + (size_t) blockIdx.y * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    s = bsum(s);
    const float r = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * r * (w ? w[i] : 1.0f);
}

// F32 rows with few rows and long K (the loop controllers' down projections, the router): W waves share a row
template <int W>
__global__ void k_mmv_f32_splitk(Segs S) {
    constexpr int RPB = 8 / W;
    const int wv = threadIdx.x / WS, lane = threadIdx.x % WS;
    const int row = blockIdx.x * RPB + wv / W, part = wv % W;
    __shared__ float red[8];
    float acc = 0.0f;
    int j = 0, n = 0;
    const bool live = row < S.start[S.n];
    if (live) {
        while (row >= S.start[j + 1]) ++j;
        n = row - S.start[j];
        const Seg & g = S.s[j];
        const float4 * w = (const float4 *) (g.w + (size_t) n * g.rb);
        const float4 * xv = (const float4 *) g.x;
        const int K4 = g.K / 4;
        for (int k = part * WS + lane; k < K4; k += W * WS) {
            const float4 a = w[k], b = xv[k];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    acc = wsum(acc);
    if (lane == 0) red[wv] = acc;
    __syncthreads();
    if (live && part == 0 && lane == 0) {
        float t = 0.0f;
        for (int q = 0; q < W; ++q) t += red[wv + q];
        S.s[j].y[n] = t;
    }
}

// mix with an optional RMS norm of each output (its own weight per query)
struct NormW { const float * w[4]; };
__global__ void k_mix_n(const float * const * states, int ns, const float * queries, float * out, int C, NormW nw,
                        float eps) {
    const float * q = queries + (size_t) blockIdx.x * C;
    const size_t toff = (size_t) blockIdx.y * C;
    __shared__ float sc[16], rr[16], p[16];
    __shared__ float red[32][16];
    // every state's sum of squares and dot with the query, in one pass and one reduction (at most 8 states)
    float a[8], b[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) { a[s] = 0.0f; b[s] = 0.0f; }
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float qi = q[i];
#pragma unroll
        for (int s = 0; s < 8; ++s) if (s < ns) { const float v = states[s][toff + i]; a[s] += v * v; b[s] += v * qi; }
    }
    const int wi = threadIdx.x / WS, ln = threadIdx.x % WS, nwv = blockDim.x / WS;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s >= ns) break;
        const float ra = wsum(a[s]), rb = wsum(b[s]);
        if (ln == 0) { red[wi][2 * s] = ra; red[wi][2 * s + 1] = rb; }
    }
    __syncthreads();
    if (threadIdx.x < ns) {
        float ra = 0.0f, rb = 0.0f;
        for (int k = 0; k < nwv; ++k) { ra += red[k][2 * threadIdx.x]; rb += red[k][2 * threadIdx.x + 1]; }
        rr[threadIdx.x] = rsqrtf(ra / C + 1e-6f);
        sc[threadIdx.x] = rb * rr[threadIdx.x] / sqrtf((float) C);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -INFINITY, z = 0.0f;
        for (int s = 0; s < ns; ++s) mx = fmaxf(mx, sc[s]);
        for (int s = 0; s < ns; ++s) { p[s] = expf(sc[s] - mx); z += p[s]; }
        for (int s = 0; s < ns; ++s) p[s] = p[s] / z * rr[s];
    }
    __syncthreads();
    float * o = out + ((size_t) blockIdx.y * gridDim.x + blockIdx.x) * C;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < ns; ++s) v += p[s] * states[s][toff + i];
        o[i] = v;
        ss += v * v;
    }
    const float * w = nw.w[blockIdx.x];
    if (w) {
        ss = bsum(ss);
        const float r = rsqrtf(ss / C + eps);
        for (int i = threadIdx.x; i < C; i += blockDim.x) o[i] = o[i] * r * w[i];
    }
}
// loop controllers, up to 4 at once: cat_j = [rms(x_j); rms(m_j)]
// per token t: x[j] + t*xs, m[j] + t*ms, out[j] + t*C, cat[j] + t*8C, f[j] and g[j] + t*4C
struct Ctl { const float * x[4]; const float * m[4]; float * cat[4]; const float * f[4]; const float * g[4]; float * out[4];
             int xs, ms; };
__global__ void k_ctrl_cat(Ctl P, int C) {
    const int j = blockIdx.x, t = blockIdx.y;
    for (int h = 0; h < 2; ++h) {
        const float * x = h ? P.m[j] + (size_t) t * P.ms : P.x[j] + (size_t) t * P.xs;
        float s = 0.0f;
        for (int i = threadIdx.x; i < C; i += blockDim.x) s += x[i] * x[i];
        s = bsum(s);
        const float r = rsqrtf(s / C + 1e-6f);
        for (int i = threadIdx.x; i < C; i += blockDim.x) P.cat[j][(size_t) t * 8 * C + h * C + i] = x[i] * r;
    }
}
__global__ void k_ctrl_apply(Ctl P, int C) {
    const int j = blockIdx.y, t = blockIdx.z, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    const float x = P.x[j][(size_t) t * P.xs + i];
    const size_t o4 = (size_t) t * 4 * C + i;
    P.out[j][(size_t) t * C + i] = x - tanhf(P.f[j][o4]) * x + tanhf(P.g[j][o4]) * P.m[j][(size_t) t * P.ms + i];
}
// K and V convolutions in one launch (blockIdx.y)
struct Conv2 { float * x[2]; const float * w[2]; float * hist[2]; const float * dyn[2]; };
__global__ void k_conv2(Conv2 P, int C, int head_dim, float scale, int T, int dyn_stride) {
    const int j = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float * hist = P.hist[j];
    const float * w = P.w[j];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c];
    for (int t = 0; t < T; ++t) {
        float * xp = P.x[j] + (size_t) t * C + c;
        const float v = *xp;
        float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * v;
        if (P.dyn[j]) {
            const float * tp = P.dyn[j] + (size_t) t * dyn_stride + (c / head_dim) * 4;
            o += tp[0] * v + tp[1] * h2 + tp[2] * h1 + tp[3] * h0;
        }
        *xp = (v + o) * scale;
        h0 = h1; h1 = h2; h2 = v;
    }
    hist[c] = h0; hist[C + c] = h1; hist[2 * C + c] = h2;
}
// conv of x (no dynamic taps) plus an optional residual, into dst
__global__ void k_conv_res(const float * x, const float * w, float * hist, int C, const float * res, float * dst, float scale,
                           int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c];
    for (int t = 0; t < T; ++t) {
        const size_t i = (size_t) t * C + c;
        const float v = x[i];
        const float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * v;
        dst[i] = (v + o) * scale + (res ? res[i] : 0.0f);
        h0 = h1; h1 = h2; h2 = v;
    }
    hist[c] = h0; hist[C + c] = h1; hist[2 * C + c] = h2;
}
// clip q and r (times tau on full layers), clip k and RMS-norm v, and write k and v to the cache slot; one block a head
__global__ void k_qkvr(float * q, const float * k, const float * v, float * r, const float * qs, const float * rs, int H,
                       int Hkv, int D, int nrel, float th, float rth, const int * pos, float alpha, float floor_, float eps,
                       float * Kc, float * Vc, int cap) {
    int b = blockIdx.x;
    const int tk = blockIdx.y, p = *pos + tk;
    q += (size_t) tk * H * D; k += (size_t) tk * Hkv * D; v += (size_t) tk * Hkv * D; r += (size_t) tk * H * nrel;
    const float tau = tau_at(p, alpha, floor_);
    float * x;
    const float * sc = nullptr;
    int n = D;
    float t = th, f2 = 1.0f;
    float * dst = nullptr;
    bool vnorm = false;
    if (b < H) { x = q + (size_t) b * D; sc = qs + (size_t) b * D; f2 = tau; }
    else if ((b -= H) < Hkv) { x = (float *) k + (size_t) b * D; dst = Kc + ((size_t) (p % cap) * Hkv + b) * D; }
    else if ((b -= Hkv) < Hkv) { x = (float *) v + (size_t) b * D; dst = Vc + ((size_t) (p % cap) * Hkv + b) * D; vnorm = true; }
    else { b -= Hkv; x = r + (size_t) b * nrel; sc = rs + (size_t) b * nrel; n = nrel; t = rth; f2 = tau; }
    float a = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float y = x[i] * (sc ? sc[i] : 1.0f); a += y * y; }
    a = bsum(a);
    const float f = vnorm ? rsqrtf(a / n + eps) : fminf(fmaxf(t / (sqrtf(a) + 1e-6f), 0.0f), 1.0f) * f2;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = x[i] * (sc ? sc[i] : 1.0f) * f;
        if (dst) dst[i] = y; else x[i] = y;
    }
}


// relative bias for this token, per query head and distance d < ext: R[h] . table[d] - sink[h]; grid H
__global__ void k_relbias(const float * R, const float * table, int nrel, int ext, const float * sink, float * bias) {
    const int h = blockIdx.x, H = gridDim.x, tk = blockIdx.y;
    R += (size_t) tk * H * nrel; sink += (size_t) tk * H; bias += (size_t) tk * H * ext;
    const float * rh = R + (size_t) h * nrel;
    for (int d = threadIdx.x; d < ext; d += blockDim.x) {
        const float * t = table + (size_t) d * nrel;
        float b = 0.0f;
        for (int i = 0; i < nrel; ++i) b += rh[i] * t[i];
        bias[(size_t) h * ext + d] = b - sink[h];
    }
}
// flash-decoding, D = 128, G query heads per KV head: grid (Hkv, nsplit), 4 waves; lane l holds dims 4l..4l+3.
// part[(h * nsplit + s) * (D + 2)]: m, l, acc[D]
template <int G>
__global__ void k_attn_split(const float * q, const float * Kc, const float * Vc, int cap, const int * dpos, int span,
                             int Hkv, const float * bias, int ext, const float * sink, float * part, int splitsz,
                             int nsplit) {
    constexpr int D = 128;
    const int kvh = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x % WS, wv = threadIdx.x / WS, tk = blockIdx.z;
    const int pos = *dpos + tk, first = max(0, pos - span + 1);
    q += (size_t) tk * Hkv * G * D; bias += (size_t) tk * Hkv * G * ext; sink += (size_t) tk * Hkv * G;
    part += (size_t) tk * Hkv * G * nsplit * (D + 2);
    const int j0 = first + sp * splitsz, j1 = min(pos, j0 + splitsz - 1);
    const float qs = rsqrtf((float) D);
    float4 qv[G];
    float m[G], ls[G];
    float4 acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float4 t = *(const float4 *) (q + (size_t) (kvh * G + g) * D + lane * 4);
        qv[g] = make_float4(t.x * qs, t.y * qs, t.z * qs, t.w * qs);
        m[g] = -INFINITY; ls[g] = 0.0f; acc[g] = make_float4(0, 0, 0, 0);
    }
    for (int j = j0 + wv; j <= j1; j += 4) {
        const size_t row = ((size_t) (j % cap) * Hkv + kvh) * D + lane * 4;
        const float4 kk = *(const float4 *) (Kc + row), vv = *(const float4 *) (Vc + row);
        const int d = pos - j;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const int h = kvh * G + g;
            float sc = wsum(qv[g].x * kk.x + qv[g].y * kk.y + qv[g].z * kk.z + qv[g].w * kk.w);
            sc += d < ext ? bias[(size_t) h * ext + d] : -sink[h];
            const float mn = fmaxf(m[g], sc), f = expf(m[g] - mn), p = expf(sc - mn);
            ls[g] = ls[g] * f + p;
            acc[g] = make_float4(acc[g].x * f + p * vv.x, acc[g].y * f + p * vv.y, acc[g].z * f + p * vv.z,
                                 acc[g].w * f + p * vv.w);
            m[g] = mn;
        }
    }
    // merge the 4 waves
    __shared__ float sm[4][G], sl[4][G];
    __shared__ float4 sa[4][G][WS];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        if (lane == 0) { sm[wv][g] = m[g]; sl[wv][g] = ls[g]; }
        sa[wv][g][lane] = acc[g];
    }
    __syncthreads();
    if (wv == 0) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float M = -INFINITY;
            for (int w = 0; w < 4; ++w) M = fmaxf(M, sm[w][g]);
            float L = 0.0f;
            float4 A = make_float4(0, 0, 0, 0);
            for (int w = 0; w < 4; ++w) {
                const float f = sm[w][g] == -INFINITY ? 0.0f : expf(sm[w][g] - M);
                L += sl[w][g] * f;
                const float4 t = sa[w][g][lane];
                A = make_float4(A.x + t.x * f, A.y + t.y * f, A.z + t.z * f, A.w + t.w * f);
            }
            float * o = part + ((size_t) (kvh * G + g) * nsplit + sp) * (D + 2);
            if (lane == 0) { o[0] = M; o[1] = L; }
            *(float4 *) (o + 2 + lane * 4) = A;
        }
    }
}
// merge the splits with the zero sink logit, times the gate's sigmoid; block per query head, D threads
__global__ void k_attn_combine(const float * part, int nsplit, const float * gate, float * out, int D) {
    const int h = blockIdx.x, i = threadIdx.x, H = gridDim.x, tk = blockIdx.y;
    part += (size_t) tk * H * nsplit * (D + 2); gate += (size_t) tk * H * D; out += (size_t) tk * H * D;
    const float * ph = part + (size_t) h * nsplit * (D + 2);
    float M = 0.0f;   // the sink
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, ph[(size_t) s * (D + 2)]);
    float L = expf(-M), A = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float ms = ph[(size_t) s * (D + 2)];
        if (ms == -INFINITY) continue;
        const float f = expf(ms - M);
        L += ph[(size_t) s * (D + 2) + 1] * f;
        A += ph[(size_t) s * (D + 2) + 2 + i] * f;
    }
    const float gv = 1.0f / (1.0f + expf(-gate[(size_t) h * D + i]));
    out[(size_t) h * D + i] = A / L * gv;
}


// ---------------------------------------------------------------- batched kernels (prompts)
// 8 values of a row from column k (a multiple of 8), dequantized
__device__ __forceinline__ void load8(const uint8_t * row, uint32_t type, int K, int k, float * o) {
    if (type == T_Q8P) {
        const uint2 qq = *(const uint2 *) (row + k);
        const float d = __half2float(((const __half *) (row + K))[k >> 5]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o[i] = (float) (int8_t) ((qq.x >> (8 * i)) & 0xFF) * d;
            o[4 + i] = (float) (int8_t) ((qq.y >> (8 * i)) & 0xFF) * d;
        }
    } else if (type == T_NVFP4) {
        const uint8_t * blk = row + (size_t) (k >> 6) * 36;
        const int v = k & 63, sb = v >> 4, j0 = v & 15;
        const float d = gUe4m3[blk[sb]];
        const uint8_t * qs = blk + 4 + sb * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = sFp4[j0 ? (qs[i] >> 4) : (qs[i] & 0xF)] * d;
    } else if (type == T_F32) {
        const float4 a = *(const float4 *) (row + (size_t) k * 4), b = *(const float4 *) (row + (size_t) k * 4 + 16);
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
    } else {   // BF16
        const uint16_t * w = (const uint16_t *) row + k;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __uint_as_float((uint32_t) w[i] << 16);
    }
}
// Y[t][n] = W[n] . X[t] for several segments; tiles of 64 rows x 32 tokens, K steps of 32 through LDS.
// gx: optional gather (the X row of tile token i is gx[i]), gy: scatter of the Y row; with an expert table.
struct GSegs { Seg s[12]; int tstart[13]; int n; };
template <int BT>   // tokens a tile (32 or 16); 64 rows, 256 threads: each thread one row, BT/4 tokens
__device__ __forceinline__ void gemm_tile(const Seg & g, int n0, int T, int t0, const int * xrow, const int * yrow,
                                          float yscale_fixed, const float * yscale_row) {
    constexpr int TPT = BT / 4;
    __shared__ float Ws[64][33], Xs[BT][33];
    const int tid = threadIdx.x, rn = tid % 64, tg = tid / 64;
    const int xs = g.xs ? g.xs : g.K, ys = g.ys ? g.ys : g.N;
    float acc[TPT];
#pragma unroll
    for (int i = 0; i < TPT; ++i) acc[i] = 0.0f;
    for (int k0 = 0; k0 < g.K; k0 += 32) {
        {   // W tile: thread -> row tid/4, 8 columns
            const int r = tid / 4, kp = (tid % 4) * 8, n = n0 + r;
            float o[8];
            if (n < g.N && k0 + kp < g.K) load8(g.w + (size_t) n * g.rb, g.type, g.K, k0 + kp, o);
            else for (int i = 0; i < 8; ++i) o[i] = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) Ws[r][kp + i] = o[i];
        }
        if (tid < BT * 8) {   // X tile: thread -> token tid/8, 4 columns
            const int tt = tid / 8, kp = (tid % 8) * 4, t = t0 + tt;
            float4 v = make_float4(0, 0, 0, 0);
            if (t < T && k0 + kp < g.K) v = *(const float4 *) (g.x + (size_t) (xrow ? xrow[t] : t) * xs + k0 + kp);
            Xs[tt][kp] = v.x; Xs[tt][kp + 1] = v.y; Xs[tt][kp + 2] = v.z; Xs[tt][kp + 3] = v.w;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; ++kk) {
            const float w = Ws[rn][kk];
#pragma unroll
            for (int i = 0; i < TPT; ++i) acc[i] += w * Xs[tg * TPT + i][kk];
        }
        __syncthreads();
    }
    const int n = n0 + rn;
    if (n >= g.N) return;
#pragma unroll
    for (int i = 0; i < TPT; ++i) {
        const int t = t0 + tg * TPT + i;
        if (t >= T) break;
        const int yr = yrow ? yrow[t] : t;
        g.y[(size_t) yr * ys + n] = acc[i] * yscale_fixed * (yscale_row ? yscale_row[yr] : 1.0f);
    }
}
__global__ void k_gemm(GSegs S, int T) {
    fp4_init();
    const int tile = blockIdx.x;
    int j = 0;
    while (tile >= S.tstart[j + 1]) ++j;
    gemm_tile<32>(S.s[j], (tile - S.tstart[j]) * 64, T, blockIdx.y * 32, nullptr, nullptr, 1.0f, nullptr);
}
// group the (token, slot) entries by expert: counts, offsets and the entry list (one block)
__global__ void k_group(const int * ids, int n, int n_exp, int * offs, int * list) {
    __shared__ int cnt[256], cur[256];
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&cnt[ids[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0;
        for (int e = 0; e < n_exp; ++e) { offs[e] = a; cur[e] = a; a += cnt[e]; }
        offs[n_exp] = a;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) list[atomicAdd(&cur[ids[i]], 1)] = i;
}
// one expert's rows for its entries: grid (row tiles, experts, entry tiles of 32). x row of entry i = i / xdiv
// (gate_up: the token; down: the entry), y row = the entry, times the expert's scale (and the entry's weight)
__global__ void k_gemm_exp(const uint8_t * W, uint32_t type, size_t ebytes, size_t rb, int N, int K, const float * x,
                           float * y, const int * offs, const int * list, int xdiv, const float * scale,
                           const float * wts) {
    fp4_init();
    const int e = blockIdx.y, c0 = offs[e], cnt = offs[e + 1] - c0, t0 = blockIdx.z * 16;
    if (t0 >= cnt) return;
    __shared__ int xr[16], yr[16];
    if (threadIdx.x < 16) {
        const int i = t0 + threadIdx.x < cnt ? list[c0 + t0 + threadIdx.x] : 0;
        xr[threadIdx.x] = i / xdiv;
        yr[threadIdx.x] = i;
    }
    __syncthreads();
    Seg g{W + (size_t) e * ebytes, x, y, rb, N, K, type};
    // the tile's tokens are local 0..31: shift the row tables so token t0 + i reads xr[i]
    gemm_tile<16>(g, blockIdx.x * 64, cnt, t0, xr - t0, yr - t0, scale ? scale[e] : 1.0f, wts);
}
// y[t][c] = sum over the token's k entries of yd[t*k + j][c]
__global__ void k_exp_sum(const float * yd, float * y, int C, int k) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float a = 0.0f;
    for (int j = 0; j < k; ++j) a += yd[((size_t) t * k + j) * C + c];
    y[(size_t) t * C + c] = a;
}

// ---------------------------------------------------------------- model
struct Mat { const uint8_t * w; uint32_t type; int N, K; size_t rb; };

struct Layer {
    bool swa = true;
    Mat wq, wk, wv, wo, gate, rel, sink, kproj, vproj, router;
    const float *q_clip, *rel_clip, *rel_bias, *kconv, *vconv, *post_conv, *ffn_conv, *attn_norm, *qn, *kn, *vn,
        *res_q = nullptr, *ffn_res_q, *ffn_norm, *probs_b = nullptr, *gu_scale = nullptr, *down_scale = nullptr;
    const uint8_t *gu, *down;
    uint32_t gu_type, down_type;
    size_t gu_ebytes, gu_rb, down_ebytes, down_rb;
    int n_res_q = 0;
    // loop controllers (core layers)
    const float * msg_q = nullptr;
    Mat cdown[4], cgain[4], cforget[4];
};

struct Model {
    Gguf g;
    int C = 2304, H = 24, Hkv = 8, D = 128, n_phys = 24, n_exp = 128, k_exp = 8, F = 1152, nrel = 16, swa = 1024,
        ext = 1024, block = 6, core0 = 6, core1 = 18, passes = 3, rank = 64, vocab = 200058, vocab_pad = 201024;
    float eps = 1e-6f, qk_clip = 22.6274f, rel_clip = 8.0f, alpha = 0.1f, floor_ = 8192.0f, wscale = 8.0f, situ_b = 4,
          situ_lb = 25, softcap = 30;
    std::vector<Layer> L;
    const uint8_t * embd = nullptr;
    const float *embd_norm = nullptr, *out_norm = nullptr, *out_res_q = nullptr;
    Mat head;

    const float * f32(const std::string & n) { return (const float *) g.need(n).dev; }
    const float * f32_opt(const std::string & n) { return g.find(n) ? f32(n) : nullptr; }
    Mat mat(const std::string & n) {
        Tensor & t = g.need(n);
        return Mat{(const uint8_t *) t.dev, t.type, (int) t.ne[1], (int) t.ne[0], row_bytes(t.type, t.ne[0])};
    }
    void load(const std::string & path) {
        g.open(path);
        const std::string a = "inkling2_loop.";
        C = (int) g.get(a + "embedding_length");
        n_phys = (int) g.get(a + "block_count");
        n_exp = (int) g.get(a + "expert_count");
        k_exp = (int) g.get(a + "expert_used_count");
        F = (int) g.get(a + "expert_feed_forward_length");
        D = (int) g.get(a + "attention.key_length");
        H = (int) g.arr[a + "attention.head_count"][0];
        Hkv = (int) g.arr[a + "attention.head_count_kv"][0];
        eps = (float) g.get(a + "attention.layer_norm_rms_epsilon");
        qk_clip = (float) g.get(a + "attention.qk_clip");
        rel_clip = (float) g.get(a + "attention.relative_clip");
        nrel = (int) g.get(a + "attention.relative_length");
        ext = (int) g.get(a + "attention.relative_extent");
        swa = (int) g.get(a + "attention.sliding_window");
        alpha = (float) g.get(a + "attention.log_scale_alpha");
        floor_ = (float) g.get(a + "attention.log_scale_floor");
        block = (int) g.get(a + "attn_res.block_size");
        core0 = (int) g.get(a + "loop.start");
        core1 = (int) g.get(a + "loop.end");
        passes = (int) g.get(a + "loop.count");
        rank = (int) g.get(a + "loop.gate_rank");
        wscale = (float) g.get(a + "expert_weights_scale");
        situ_b = (float) g.get(a + "activation.situ_beta");
        situ_lb = (float) g.get(a + "activation.situ_linear_beta");
        softcap = (float) g.get(a + "final_logit_softcapping");
        vocab = (int) g.get(a + "vocab_size_unpadded");
        vocab_pad = (int) g.get(a + "vocab_size");
        const auto & pat = g.arr[a + "attention.sliding_window_pattern"];
        embd = (const uint8_t *) g.need("token_embd.weight", true).dev;
        embd_norm = f32("token_embd_norm.weight");
        out_norm = f32("output_norm.weight");
        out_res_q = f32("output_res_query.weight");
        head = mat("output.weight");
        L.resize(n_phys);
        for (int i = 0; i < n_phys; ++i) {
            Layer & l = L[i];
            const std::string b = "blk." + std::to_string(i) + ".";
            l.swa = pat[i] != 0.0;
            l.wq = mat(b + "attn_q.weight"); l.wk = mat(b + "attn_k.weight"); l.wv = mat(b + "attn_v.weight");
            l.wo = mat(b + "attn_output.weight"); l.gate = mat(b + "attn_gate.weight"); l.rel = mat(b + "attn_rel.weight");
            l.sink = mat(b + "attn_sink_proj.weight");
            l.q_clip = f32(b + "attn_q_clip.weight"); l.rel_clip = f32(b + "attn_rel_clip.weight");
            l.rel_bias = f32(b + "attn_rel_bias.weight");
            l.kconv = f32(b + "attn_k_conv.weight"); l.vconv = f32(b + "attn_v_conv.weight");
            l.post_conv = f32(b + "attn_post_conv.weight"); l.ffn_conv = f32(b + "ffn_post_conv.weight");
            if (l.swa) l.attn_norm = f32(b + "attn_norm.weight");
            else {
                l.qn = f32(b + "attn_q_inp_norm.weight"); l.kn = f32(b + "attn_k_inp_norm.weight");
                l.vn = f32(b + "attn_v_inp_norm.weight");
                l.kproj = mat(b + "attn_k_conv_proj.weight"); l.vproj = mat(b + "attn_v_conv_proj.weight");
            }
            if (g.find(b + "attn_res_query.weight")) {
                Tensor & t = g.need(b + "attn_res_query.weight");
                l.res_q = (const float *) t.dev;
                l.n_res_q = t.ne.size() > 1 ? (int) t.ne[1] : 1;
            }
            l.ffn_res_q = f32(b + "ffn_res_query.weight");
            l.ffn_norm = f32(b + "ffn_norm.weight");
            l.router = mat(b + "ffn_gate_inp.weight");
            l.probs_b = f32_opt(b + "exp_probs_b.bias");
            Tensor & gu = g.need(b + "ffn_gate_up_exps.weight");
            Tensor & dn = g.need(b + "ffn_down_exps.weight");
            l.gu = (const uint8_t *) gu.dev; l.gu_type = gu.type;
            l.gu_rb = row_bytes(gu.type, gu.ne[0]); l.gu_ebytes = l.gu_rb * gu.ne[1];
            l.down = (const uint8_t *) dn.dev; l.down_type = dn.type;
            l.down_rb = row_bytes(dn.type, dn.ne[0]); l.down_ebytes = l.down_rb * dn.ne[1];
            l.gu_scale = f32_opt(b + "ffn_gate_up_exps.scale");
            l.down_scale = f32_opt(b + "ffn_down_exps.scale");
            if (i >= core0 && i < core1) {
                l.msg_q = f32(b + "loop_msg_query.weight");
                const char * nm[4] = {"q", "k", "v", "ffn"};
                for (int c = 0; c < 4; ++c) {
                    const std::string p = b + "loop_" + nm[c] + "_ctrl_";
                    l.cdown[c] = mat(p + "down.weight"); l.cgain[c] = mat(p + "gain.weight");
                    l.cforget[c] = mat(p + "forget.weight");
                }
            }
        }
    }
};

// ---------------------------------------------------------------- one token at a time
struct Slot {     // a layer visit's cache
    float *K = nullptr, *V = nullptr, *hist = nullptr;   // hist: [3][Ck], [3][Ck], [3][C], [3][C] for k, v, attn, ffn
    int cap = 0;
};

struct Runner {
    static constexpr int TM = 128;     // the largest prompt chunk
    Model & m;
    int ctx;
    std::vector<Slot> slots;
    // per-token buffers, [TM][...]
    float *x, *t0, *t1, *t2, *inp[3], *q, *k, *v, *r, *sinkv, *att, *g, *dk, *dv, *scr, *logits, *msgs, *ctrl_g,
        *ctrl_a, *ctrl_b, *cat, *gu, *hh, *wts, *rl, *yd;
    int *ids, *offs, *list;
    float *relb, *part;          // relative bias [T][H][ext]; the attention's split partials
    std::vector<float *> pool;   // the layer outputs and block states, [TM][C] each
    int pos = 0;
    hipStream_t st;
    int *dpos, *dtok, *hpos;          // hpos: pinned host [pos, tok...]
    std::vector<const float **> marr; // one device array of state pointers per mix call of a token
    size_t mix_i = 0;
    bool eager_done = false;
    hipGraphExec_t gexec = nullptr;
    int nsplit_max = 64;

    float * dalloc(size_t n) { float * p; CK(hipMalloc(&p, n * 4)); CK(hipMemset(p, 0, n * 4)); return p; }

    Runner(Model & m_, int ctx_) : m(m_), ctx(ctx_) {
        const int C = m.C, Ck = m.Hkv * m.D;
        const size_t T = TM;
        const int n_slots = m.n_phys + (m.passes - 1) * (m.core1 - m.core0);
        slots.resize(n_slots);
        for (int s = 0; s < n_slots; ++s) {
            const int ip = s < m.n_phys ? s : m.core0 + (s - m.n_phys) % (m.core1 - m.core0);
            Slot & sl = slots[s];
            // a window layer keeps its window plus a chunk: a chunk's first token still sees its whole window
            sl.cap = m.L[ip].swa ? std::min(ctx, m.swa + TM) : ctx;
            sl.K = dalloc((size_t) sl.cap * Ck);
            sl.V = dalloc((size_t) sl.cap * Ck);
            sl.hist = dalloc((size_t) 3 * (2 * Ck + 2 * C));
        }
        x = dalloc(T * C); t0 = dalloc(T * C); t1 = dalloc(T * C); t2 = dalloc(T * C);
        for (auto & p : inp) p = dalloc(T * C);
        q = dalloc(T * m.H * m.D); k = dalloc(T * Ck); v = dalloc(T * Ck); r = dalloc(T * m.H * m.nrel);
        sinkv = dalloc(T * m.H); att = dalloc(T * m.H * m.D); g = dalloc(T * m.H * m.D);
        dk = dalloc(T * 4 * m.Hkv); dv = dalloc(T * 4 * m.Hkv);
        scr = dalloc(T * 3 * C); logits = dalloc(T * m.vocab_pad); msgs = dalloc(T * 4 * C);
        ctrl_g = dalloc(T * 4 * m.rank); ctrl_a = dalloc(T * 4 * C); ctrl_b = dalloc(T * 4 * C); cat = dalloc(T * 8 * C);
        gu = dalloc(T * m.k_exp * 2 * m.F); hh = dalloc(T * m.k_exp * m.F); wts = dalloc(T * m.k_exp);
        yd = dalloc(T * m.k_exp * C);
        rl = dalloc(T * m.n_exp);
        relb = dalloc(T * m.H * m.ext);
        part = dalloc(T * m.H * nsplit_max * (m.D + 2));
        CK(hipMalloc(&ids, T * m.k_exp * 4));
        CK(hipMalloc(&offs, (m.n_exp + 1) * 4));
        CK(hipMalloc(&list, T * m.k_exp * 4));
        CK(hipStreamCreate(&st));
        CK(hipMalloc(&dpos, 4)); CK(hipMalloc(&dtok, T * 4)); CK(hipHostMalloc(&hpos, (T + 1) * 4));
        for (int i = 0; i < 128; ++i) pool.push_back(dalloc(T * C));   // 2 a visit (48 visits) and a few more
    }

    int ew(int n) { return (n + 255) / 256; }
    void mv(const Mat & a, const float * in, float * out) {
        hipLaunchKernelGGL(k_matvec, dim3((a.N + 7) / 8), dim3(256), 0, st, a.w, a.type, in, out, a.N, a.K, a.rb);
    }
    // several matvecs (T = 1) or GEMMs (T > 1) in one launch; xs: the per-token stride of each input (0 = K)
    using MV = std::tuple<const Mat *, const float *, float *, int>;
    void mmv(int T, std::initializer_list<MV> list) {
        if (T == 1) {
            Segs S{};
            S.n = 0; S.start[0] = 0;
            for (const auto & e : list) {
                const Mat & a = *std::get<0>(e);
                S.s[S.n] = Seg{a.w, std::get<1>(e), std::get<2>(e), a.rb, a.N, a.K, a.type};
                S.start[S.n + 1] = S.start[S.n] + a.N;
                ++S.n;
            }
            hipLaunchKernelGGL(k_mmv, dim3((S.start[S.n] + 7) / 8), dim3(256), 0, st, S);
            return;
        }
        GSegs S{};
        S.n = 0; S.tstart[0] = 0;
        for (const auto & e : list) {
            const Mat & a = *std::get<0>(e);
            S.s[S.n] = Seg{a.w, std::get<1>(e), std::get<2>(e), a.rb, a.N, a.K, a.type, std::get<3>(e), 0};
            S.tstart[S.n + 1] = S.tstart[S.n] + (a.N + 63) / 64;
            ++S.n;
        }
        hipLaunchKernelGGL(k_gemm, dim3(S.tstart[S.n], (T + 31) / 32), dim3(256), 0, st, S, T);
    }
    // mix(blocks, cur, queries) -> out[T][nq][C], each output RMS-normed with nw.w[j] when given
    void mix(int T, const std::vector<float *> & blocks, const float * cur, const float * queries, int nq, float * out,
             NormW nw = NormW{{nullptr, nullptr, nullptr, nullptr}}) {
        // the pointer sets are the same every token (the pool is handed out in the same order): written once
        std::vector<const float *> sv(blocks.begin(), blocks.end());
        sv.push_back(cur);
        if (mix_i == marr.size()) {
            const float ** dd;
            CK(hipMalloc(&dd, 16 * sizeof(float *)));
            CK(hipMemcpy(dd, sv.data(), sv.size() * sizeof(float *), hipMemcpyHostToDevice));
            marr.push_back(dd);
        }
        hipLaunchKernelGGL(k_mix_n, dim3(nq, T), dim3(1024), 0, st, marr[mix_i++], (int) sv.size(), queries, out, m.C,
                           nw, m.eps);
    }
    // n loop controllers at once: out[j] = control(x[j], msg[j]) with the layer's controller c0 + j
    void controls(int T, const Layer & l, int c0, int n, const float * const * xs, int xstride, const float * const * ms,
                  int mstride, float * const * outs) {
        const int C = m.C;
        Ctl P{};
        P.xs = xstride; P.ms = mstride;
        for (int j = 0; j < n; ++j) {
            P.x[j] = xs[j]; P.m[j] = ms[j]; P.cat[j] = cat + (size_t) j * 2 * C;
            P.f[j] = ctrl_b + (size_t) j * C; P.g[j] = ctrl_a + (size_t) j * C; P.out[j] = outs[j];
        }
        hipLaunchKernelGGL(k_ctrl_cat, dim3(n, T), dim3(256), 0, st, P, C);
        if (T == 1) {
            Segs S{};
            S.n = 0; S.start[0] = 0;
            auto add = [&](const Mat & a, const float * in, float * out) {
                S.s[S.n] = Seg{a.w, in, out, a.rb, a.N, a.K, a.type};
                S.start[S.n + 1] = S.start[S.n] + a.N;
                ++S.n;
            };
            for (int j = 0; j < n; ++j) add(l.cdown[c0 + j], P.cat[j], ctrl_g + (size_t) j * m.rank);
            hipLaunchKernelGGL(k_mmv_f32_splitk<8>, dim3(S.start[S.n]), dim3(256), 0, st, S);
            S.n = 0;
            for (int j = 0; j < n; ++j) {
                add(l.cgain[c0 + j], ctrl_g + (size_t) j * m.rank, ctrl_a + (size_t) j * C);
                add(l.cforget[c0 + j], ctrl_g + (size_t) j * m.rank, ctrl_b + (size_t) j * C);
            }
            hipLaunchKernelGGL(k_mmv, dim3((S.start[S.n] + 7) / 8), dim3(256), 0, st, S);
        } else {
            // per token the layouts are cat [8C], ctrl_g [4 rank], ctrl_a / ctrl_b [4C]
            GSegs S{};
            S.n = 0; S.tstart[0] = 0;
            auto add = [&](const Mat & a, const float * in, float * out, int xs_, int ys_) {
                S.s[S.n] = Seg{a.w, in, out, a.rb, a.N, a.K, a.type, xs_, ys_};
                S.tstart[S.n + 1] = S.tstart[S.n] + (a.N + 63) / 64;
                ++S.n;
            };
            for (int j = 0; j < n; ++j) add(l.cdown[c0 + j], P.cat[j], ctrl_g + (size_t) j * m.rank, 8 * C, 4 * m.rank);
            hipLaunchKernelGGL(k_gemm, dim3(S.tstart[S.n], (T + 31) / 32), dim3(256), 0, st, S, T);
            S.n = 0;
            for (int j = 0; j < n; ++j) {
                add(l.cgain[c0 + j], ctrl_g + (size_t) j * m.rank, ctrl_a + (size_t) j * C, 4 * m.rank, 4 * C);
                add(l.cforget[c0 + j], ctrl_g + (size_t) j * m.rank, ctrl_b + (size_t) j * C, 4 * m.rank, 4 * C);
            }
            hipLaunchKernelGGL(k_gemm, dim3(S.tstart[S.n], (T + 31) / 32), dim3(256), 0, st, S, T);
        }
        hipLaunchKernelGGL(k_ctrl_apply, dim3(ew(C), n, T), dim3(256), 0, st, P, C);
    }

    // one layer visit for T tokens: cur is replaced by the visit's output (a new pool buffer); the input stays as it
    // was, so it can be a block state
    void layer(int T, int slot, int ip, float *& cur, std::vector<float *> & blocks, const float * messages, int & pool_i) {
        const Layer & l = m.L[ip];
        const int C = m.C, Ck = m.Hkv * m.D;
        Slot & sl = slots[slot];
        float * inpSA = cur;
        const float * in3[3];
        int xs3 = C;   // the per-token stride of the attention inputs
        if (!messages) {
            if (l.swa) {
                if (l.res_q) mix(T, blocks, cur, l.res_q, 1, inp[0], NormW{{l.attn_norm, nullptr, nullptr, nullptr}});
                else hipLaunchKernelGGL(k_rms_n, dim3(1, T), dim3(256), 0, st, Norms{{cur}, {l.attn_norm}, {inp[0]}}, C, m.eps);
                in3[0] = in3[1] = in3[2] = inp[0];
            } else {
                mix(T, blocks, cur, l.res_q, 3, scr, NormW{{l.qn, l.kn, l.vn, nullptr}});
                for (int j = 0; j < 3; ++j) in3[j] = scr + (size_t) j * C;
                xs3 = 3 * C;
            }
        } else {
            // loop pass: mix, then the controllers with the q, k, v messages, then the norms
            mix(T, blocks, cur, l.res_q, l.n_res_q, scr);
            const float * xs[3], * ms[3];
            for (int j = 0; j < 3; ++j) { xs[j] = scr + (size_t) (l.swa ? 0 : j) * C; ms[j] = messages + (size_t) j * C; }
            controls(T, l, 0, 3, xs, l.n_res_q * C, ms, 4 * C, inp);
            Norms R{};
            for (int j = 0; j < 3; ++j) {
                R.in[j] = inp[j]; R.out[j] = inp[j];
                R.w[j] = l.swa ? l.attn_norm : (j == 0 ? l.qn : j == 1 ? l.kn : l.vn);
            }
            hipLaunchKernelGGL(k_rms_n, dim3(3, T), dim3(256), 0, st, R, C, m.eps);
            for (int j = 0; j < 3; ++j) in3[j] = inp[j];
        }
        // projections, one launch
        if (l.swa)
            mmv(T, {MV{&l.wq, in3[0], q, xs3}, MV{&l.wk, in3[1], k, xs3}, MV{&l.wv, in3[2], v, xs3}, MV{&l.rel, in3[0], r, xs3},
                    MV{&l.sink, in3[0], sinkv, xs3}, MV{&l.gate, in3[0], g, xs3}});
        else
            mmv(T, {MV{&l.wq, in3[0], q, xs3}, MV{&l.wk, in3[1], k, xs3}, MV{&l.wv, in3[2], v, xs3}, MV{&l.rel, in3[0], r, xs3},
                    MV{&l.sink, in3[0], sinkv, xs3}, MV{&l.gate, in3[0], g, xs3}, MV{&l.kproj, in3[1], dk, xs3},
                    MV{&l.vproj, in3[2], dv, xs3}});
        const float cs = 1.0f / sqrtf((float) ((l.swa ? 1 : 2) * 4 + 1));
        hipLaunchKernelGGL(k_conv2, dim3(ew(Ck), 2), dim3(256), 0, st,
                           Conv2{{k, v}, {l.kconv, l.vconv}, {sl.hist, sl.hist + 3 * Ck},
                                 {l.swa ? nullptr : dk, l.swa ? nullptr : dv}}, Ck, m.D, cs, T, 4 * m.Hkv);
        hipLaunchKernelGGL(k_qkvr, dim3(2 * m.H + 2 * m.Hkv, T), dim3(128), 0, st, q, k, v, r, l.q_clip, l.rel_clip, m.H,
                           m.Hkv, m.D, m.nrel, m.qk_clip, m.rel_clip, dpos, l.swa ? 0.0f : m.alpha, m.floor_, m.eps,
                           sl.K, sl.V, sl.cap);
        const int span = l.swa ? std::min(m.swa, ctx) : ctx;
        hipLaunchKernelGGL(k_relbias, dim3(m.H, T), dim3(256), 0, st, r, l.rel_bias, m.nrel, m.ext, sinkv, relb);
        const int splitsz = std::max(64, (span + nsplit_max - 1) / nsplit_max), nsplit = (span + splitsz - 1) / splitsz;
        hipLaunchKernelGGL(k_attn_split<3>, dim3(m.Hkv, nsplit, T), dim3(128), 0, st, q, sl.K, sl.V, sl.cap, dpos, span,
                           m.Hkv, relb, m.ext, sinkv, part, splitsz, nsplit);
        hipLaunchKernelGGL(k_attn_combine, dim3(m.H, T), dim3(m.D), 0, st, part, nsplit, g, att, m.D);
        mmv(T, {MV{&l.wo, att, t0, 0}});
        float * h1 = pool[pool_i++];
        const bool starts = ip % m.block == 0;
        if (starts) blocks.push_back(inpSA);
        hipLaunchKernelGGL(k_conv_res, dim3(ew(C)), dim3(256), 0, st, t0, l.post_conv, sl.hist + 6 * Ck, C,
                           starts ? (const float *) nullptr : inpSA, h1, 1.0f / sqrtf(5.0f), T);
        // MoE
        if (!messages) mix(T, blocks, h1, l.ffn_res_q, 1, t1, NormW{{l.ffn_norm, nullptr, nullptr, nullptr}});
        else {
            mix(T, blocks, h1, l.ffn_res_q, 1, t2);
            const float * xs[1] = {t2}, * ms[1] = {messages + 3 * (size_t) C};
            float * os[1] = {t1};
            controls(T, l, 3, 1, xs, C, ms, 4 * C, os);
            hipLaunchKernelGGL(k_rms_n, dim3(1, T), dim3(256), 0, st, Norms{{t1}, {l.ffn_norm}, {t1}}, C, m.eps);
        }
        if (T == 1) {
            Segs S{};
            S.n = 1; S.start[0] = 0; S.start[1] = l.router.N;
            S.s[0] = Seg{l.router.w, t1, rl, l.router.rb, l.router.N, l.router.K, l.router.type};
            hipLaunchKernelGGL(k_mmv_f32_splitk<4>, dim3((l.router.N + 1) / 2), dim3(256), 0, st, S);
        } else {
            mmv(T, {MV{&l.router, t1, rl, 0}});
        }
        hipLaunchKernelGGL(k_route, dim3(T), dim3(m.n_exp), 0, st, rl, l.probs_b, m.n_exp, m.k_exp, m.wscale, ids, wts);
        float * h2 = pool[pool_i++];
        if (T == 1) {
            hipLaunchKernelGGL(k_exp_gu, dim3((2 * m.F + 7) / 8, m.k_exp), dim3(256), 0, st, l.gu, l.gu_type, l.gu_ebytes,
                               l.gu_rb, l.gu_scale, ids, t1, gu, 2 * m.F, C);
            hipLaunchKernelGGL(k_situ, dim3(ew(m.F * m.k_exp)), dim3(256), 0, st, gu, hh, m.F, m.k_exp, m.situ_b, m.situ_lb);
            hipLaunchKernelGGL(k_exp_down, dim3((C + 7) / 8), dim3(256), 0, st, l.down, l.down_type, l.down_ebytes,
                               l.down_rb, l.down_scale, ids, wts, hh, t2, C, m.F, m.k_exp);
        } else {
            const int ne = T * m.k_exp;
            hipLaunchKernelGGL(k_group, dim3(1), dim3(1024), 0, st, ids, ne, m.n_exp, offs, list);
            hipLaunchKernelGGL(k_gemm_exp, dim3((2 * m.F + 63) / 64, m.n_exp, (T + 15) / 16), dim3(256), 0, st, l.gu,
                               l.gu_type, l.gu_ebytes, l.gu_rb, 2 * m.F, C, t1, gu, offs, list, m.k_exp, l.gu_scale,
                               (const float *) nullptr);
            hipLaunchKernelGGL(k_situ, dim3(ew(m.F * ne)), dim3(256), 0, st, gu, hh, m.F, ne, m.situ_b, m.situ_lb);
            hipLaunchKernelGGL(k_gemm_exp, dim3((C + 63) / 64, m.n_exp, (T + 15) / 16), dim3(256), 0, st, l.down,
                               l.down_type, l.down_ebytes, l.down_rb, C, m.F, hh, yd, offs, list, 1, l.down_scale, wts);
            hipLaunchKernelGGL(k_exp_sum, dim3(ew(C), T), dim3(256), 0, st, yd, t2, C, m.k_exp);
        }
        hipLaunchKernelGGL(k_conv_res, dim3(ew(C)), dim3(256), 0, st, t2, l.ffn_conv, sl.hist + 6 * Ck + 3 * C, C,
                           h1, h2, 1.0f / sqrtf(5.0f), T);
        cur = h2;
    }

    // feed one token: the first one runs eagerly (it writes the mix pointer sets), the second is captured as a graph,
    // every later one replays it
    void step(int tok, std::vector<float> * out) {
        hpos[0] = pos;
        hpos[1] = tok;
        CK(hipMemcpyAsync(dpos, hpos, 4, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(dtok, hpos + 1, 4, hipMemcpyHostToDevice, st));
        if (!eager_done) {
            body(1, false);
            eager_done = true;
        } else {
            if (!gexec) {
                hipGraph_t gr;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
                body(1, false);
                CK(hipStreamEndCapture(st, &gr));
                CK(hipGraphInstantiate(&gexec, gr, nullptr, nullptr, 0));
            }
            CK(hipGraphLaunch(gexec, st));
        }
        ++pos;
        if (out) {
            out->resize(m.vocab);
            CK(hipMemcpyAsync(out->data(), logits, (size_t) m.vocab * 4, hipMemcpyDeviceToHost, st));
        }
        CK(hipStreamSynchronize(st));
    }
    // feed up to TM tokens at once (a prompt chunk); all_logits: every token's logits (else only the last one's, in
    // logits[0]); out gets them when given
    void chunk(const int * toks, int T, bool all_logits, std::vector<float> * out) {
        hpos[0] = pos;
        std::memcpy(hpos + 1, toks, (size_t) T * 4);
        CK(hipMemcpyAsync(dpos, hpos, 4, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(dtok, hpos + 1, (size_t) T * 4, hipMemcpyHostToDevice, st));
        body(T, all_logits);
        pos += T;
        if (out) {
            const int nl = all_logits ? T : 1;
            out->resize((size_t) nl * m.vocab);
            for (int i = 0; i < nl; ++i)
                CK(hipMemcpyAsync(out->data() + (size_t) i * m.vocab, logits + (size_t) i * m.vocab_pad,
                                  (size_t) m.vocab * 4, hipMemcpyDeviceToHost, st));
        }
        CK(hipStreamSynchronize(st));
    }
    void body(int T, bool all_logits) {
        const int C = m.C;
        int pool_i = 0;
        mix_i = 0;
        hipLaunchKernelGGL(k_embed, dim3(T), dim3(256), 0, st, m.embd, dtok, x, C);
        hipLaunchKernelGGL(k_rms_n, dim3(1, T), dim3(256), 0, st, Norms{{x}, {m.embd_norm}, {x}}, C, m.eps);
        float * cur = x;
        std::vector<float *> blocks;
        for (int il = 0; il < m.core0; ++il) layer(T, il, il, cur, blocks, nullptr, pool_i);
        float * prelude = cur;
        const std::vector<float *> prelude_blocks = blocks;
        const int ncore = m.core1 - m.core0;
        for (int p = 0; p < m.passes; ++p) {
            std::vector<float *> bank;
            float * bank_out = nullptr;
            if (p > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (int ip = m.core0; ip < m.core1; ++ip) {
                const int slot = p == 0 ? ip : m.n_phys + (p - 1) * ncore + (ip - m.core0);
                if (p == 0) { layer(T, slot, ip, cur, blocks, nullptr, pool_i); continue; }
                mix(T, bank, bank_out, m.L[ip].msg_q, 4, msgs);
                layer(T, slot, ip, cur, blocks, msgs, pool_i);
            }
        }
        for (int il = m.core1; il < m.n_phys; ++il) layer(T, il, il, cur, blocks, nullptr, pool_i);
        mix(T, blocks, cur, m.out_res_q, 1, t1, NormW{{m.out_norm, nullptr, nullptr, nullptr}});
        if (T == 1) mv(m.head, t1, logits);
        else if (all_logits) {
            GSegs S{};
            S.n = 1; S.tstart[0] = 0; S.tstart[1] = (m.head.N + 63) / 64;
            S.s[0] = Seg{m.head.w, t1, logits, m.head.rb, m.head.N, m.head.K, m.head.type, 0, m.vocab_pad};
            hipLaunchKernelGGL(k_gemm, dim3(S.tstart[1], (T + 31) / 32), dim3(256), 0, st, S, T);
        } else mv(m.head, t1 + (size_t) (T - 1) * C, logits);
        const int nl = all_logits ? T : 1;
        hipLaunchKernelGGL(k_softcap, dim3(ew(nl * m.vocab_pad)), dim3(256), 0, st, logits, nl * m.vocab_pad, m.softcap);
    }
};

}  // namespace

int main(int argc, char ** argv) {
    std::string path, score, dump;
    int gpu = 0, ctx = 4096, n_dump = 0, n_bench = 0, chunk = 1;
    int n_pf = 0;   // --bench-prefill N: a fresh N-token prompt (the text repeated) in chunks, only the last logits   // --chunk N: the text in chunks of N   // --bench N: N decode steps after the text, timed
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") path = next();
        else if (a == "--gpu") gpu = std::atoi(next());
        else if (a == "-c") ctx = std::atoi(next());
        else if (a == "--score") score = next();
        else if (a == "--dump-logits") { dump = next(); n_dump = std::atoi(next()); }
        else if (a == "--bench") n_bench = std::atoi(next());
        else if (a == "--chunk") chunk = std::atoi(next());
        else if (a == "--bench-prefill") n_pf = std::atoi(next());
        else { std::fprintf(stderr, "almond-inkling-native: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (path.empty() || score.empty()) {
        std::fprintf(stderr, "usage: almond-inkling-native -m MODEL.gguf [--gpu N] [-c N] --score IDS.txt [--dump-logits FILE N]\n");
        return 2;
    }
    CK(hipSetDevice(gpu));
    hipLaunchKernelGGL(k_init_tables, dim3(1), dim3(256), 0, 0);
    CK(hipDeviceSynchronize());
    Model m;
    const auto t0 = std::chrono::steady_clock::now();
    m.load(path);
    std::fprintf(stderr, "loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::vector<int> ids;
    { std::ifstream f(score); int v; while (f >> v) ids.push_back(v); }
    if ((int) ids.size() > ctx) ids.resize(ctx);
    if (n_pf > 0) {
        Runner pr(m, ctx);
        std::vector<int> t;
        while ((int) t.size() < n_pf) t.insert(t.end(), ids.begin(), ids.end());
        t.resize((size_t) n_pf);
        const int c = std::max(2, std::min(chunk, Runner::TM));
        pr.chunk(t.data(), c, false, nullptr);   // warm-up chunk (pointer sets, kernels)
        pr.pos = 0;
        for (auto & sl : pr.slots) CK(hipMemset(sl.hist, 0, (size_t) 3 * (2 * m.Hkv * m.D + 2 * m.C) * 4));
        const auto tp = std::chrono::steady_clock::now();
        for (int i = 0; i < n_pf; i += c) pr.chunk(t.data() + i, std::min(c, n_pf - i), false, nullptr);
        const double ps = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
        std::printf("BENCH prefill %d tokens in chunks of %d: %.1f tok/s (%.2f s)\n", n_pf, c, n_pf / ps, ps);
        return 0;
    }
    Runner run(m, ctx);
    std::vector<float> lg;
    FILE * df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    double nll = 0;
    int n_scored = 0, dumped = 0;
    std::vector<int> top1;
    const int first = (int) ids.size() / 2;
    const auto t1 = std::chrono::steady_clock::now();
    auto score_one = [&](size_t i, const float * l) {
        const float mx = *std::max_element(l, l + m.vocab);
        double z = 0;
        for (int j = 0; j < m.vocab; ++j) z += std::exp((double) l[j] - mx);
        const double lp = (double) l[ids[i + 1]] - mx - std::log(z);
        top1.push_back((int) (std::max_element(l, l + m.vocab) - l));
        if ((int) i >= first) { nll -= lp; ++n_scored; }
        if (df && (int) i >= first && dumped < n_dump) { std::fwrite(l, 4, (size_t) m.vocab, df); ++dumped; }
    };
    chunk = std::max(1, std::min(chunk, Runner::TM));
    for (size_t i = 0; i + 1 < ids.size();) {
        const int T = (int) std::min<size_t>((size_t) chunk, ids.size() - 1 - i);
        if (T == 1) { run.step(ids[i], &lg); score_one(i, lg.data()); }
        else {
            run.chunk(ids.data() + i, T, true, &lg);
            for (int t = 0; t < T; ++t) score_one(i + t, lg.data() + (size_t) t * m.vocab);
        }
        i += (size_t) T;
    }
    if (df) std::fclose(df);
    if (n_bench > 0) {
        // decode alone: the next token is fed without reading the logits back (the speed the serve loop will see,
        // less its sampling)
        const auto tb = std::chrono::steady_clock::now();
        for (int i = 0; i < n_bench; ++i) run.step(ids.back(), nullptr);
        const double bs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
        std::printf("BENCH decode %d tokens at depth %zu: %.2f tok/s\n", n_bench, ids.size(), n_bench / bs);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("SCORE ppl=%.4f scored=%d tokens=%zu tok/s=%.2f\n", std::exp(nll / std::max(1, n_scored)), n_scored,
                ids.size(), (double) (ids.size() - 1) / secs);
    std::printf("TOP1");
    for (int t : top1) std::printf(" %d", t);
    std::printf("\n");
    return 0;
}

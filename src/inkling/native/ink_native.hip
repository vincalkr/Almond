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
#include <map>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace {

// ---------------------------------------------------------------- GGUF
enum : uint32_t { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_BF16 = 30, T_NVFP4 = 40 };

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
    Tensor & need(const std::string & n) {
        Tensor * x = find(n);
        if (!x) { std::fprintf(stderr, "gguf: no tensor %s\n", n.c_str()); std::exit(1); }
        if (!x->dev) {
            CK(hipMalloc(&x->dev, x->bytes));
            CK(hipMemcpy(x->dev, x->host, x->bytes, hipMemcpyHostToDevice));
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
__constant__ int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// one row of a matrix in its stored type, dotted with x (F32), by one wave
__device__ float row_dot(const uint8_t * row, uint32_t type, const float * x, int K) {
    const int l = threadIdx.x % WS;
    float acc = 0.0f;
    if (type == T_F32) {
        const float * w = (const float *) row;
        for (int k = l; k < K; k += WS) acc += w[k] * x[k];
    } else if (type == T_BF16) {
        const uint16_t * w = (const uint16_t *) row;
        for (int k = l; k < K; k += WS) acc += __uint_as_float((uint32_t) w[k] << 16) * x[k];
    } else if (type == T_Q8_0) {
        for (int b = l; b < K / 32; b += WS) {
            const uint8_t * blk = row + (size_t) b * 34;
            const float d = __half2float(*(const __half *) blk);
            const int8_t * q = (const int8_t *) (blk + 2);
            float s = 0.0f;
            for (int j = 0; j < 32; ++j) s += (float) q[j] * x[b * 32 + j];
            acc += d * s;
        }
    } else if (type == T_NVFP4) {
        for (int b = l; b < K / 64; b += WS) {
            const uint8_t * blk = row + (size_t) b * 36;
            const uint8_t * qs = blk + 4;
            for (int s = 0; s < 4; ++s) {
                const float d = ue4m3(blk[s]);
                const float * xb = x + b * 64 + s * 16;
                float a = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t v = qs[s * 8 + j];
                    a += (float) kFp4[v & 0xF] * xb[j] + (float) kFp4[v >> 4] * xb[j + 8];
                }
                acc += d * a;
            }
        }
    }
    return wsum(acc);
}

// y[n] = W[n] . x, rows of K; 8 waves a block
__global__ void k_matvec(const uint8_t * W, uint32_t type, const float * x, float * y, int N, int K, size_t rb) {
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
__global__ void k_clip(float * x, const float * s, int D, float th, float tau) {
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
__global__ void k_attn(const float * q, const float * K, const float * V, int cap, int pos, int span, int Hkv, int D,
                       const float * R, const float * table, int nrel, int ext, const float * sink, float * out,
                       float * scratch) {
    const int h = blockIdx.x, kvh = h / (gridDim.x / Hkv);
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
        out[(size_t) h * D + i] = a / z;
    }
}

__global__ void k_store_kv(const float * k, const float * v, float * K, float * V, int slot, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void k_route(const float * logits, const float * bias, int n_exp, int k, float scale, int * ids, float * w) {
    __shared__ float p[256], sel[256];
    const int i = threadIdx.x;
    if (i < n_exp) { p[i] = 1.0f / (1.0f + expf(-logits[i])); sel[i] = p[i] + (bias ? bias[i] : 0.0f); }
    __syncthreads();
    if (i == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            int best = 0;
            for (int e = 1; e < n_exp; ++e) if (sel[e] > sel[best]) best = e;
            ids[j] = best;
            w[j] = p[best];
            sum += p[best];
            sel[best] = -INFINITY;
        }
        sum = fmaxf(sum, 1e-20f);
        for (int j = 0; j < k; ++j) w[j] = w[j] / sum * scale;
    }
}
// experts gate_up: out[j][2*F] = (W[id_j] . x) * scale[id_j]; grid (rows/8, k)
__global__ void k_exp_gu(const uint8_t * W, uint32_t type, size_t ebytes, size_t rb, const float * scale, const int * ids,
                         const float * x, float * out, int N, int K) {
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
__global__ void k_embed(const uint8_t * E, int tok, float * x, int C) {
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
        embd = (const uint8_t *) g.need("token_embd.weight").dev;
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
    Model & m;
    int ctx;
    std::vector<Slot> slots;
    float *x, *t0, *t1, *t2, *inp[3], *q, *k, *v, *r, *sinkv, *att, *g, *dk, *dv, *scr, *logits, *msgs, *ctrl_g,
        *ctrl_a, *ctrl_b, *cat, *gu, *hh, *wts, *rl;
    int * ids;
    const float ** dstates;    // device array of state pointers for k_mix
    std::vector<float *> pool; // the block states
    int pos = 0;

    float * dalloc(size_t n) { float * p; CK(hipMalloc(&p, n * 4)); CK(hipMemset(p, 0, n * 4)); return p; }

    Runner(Model & m_, int ctx_) : m(m_), ctx(ctx_) {
        const int C = m.C, Ck = m.Hkv * m.D;
        const int n_slots = m.n_phys + (m.passes - 1) * (m.core1 - m.core0);
        slots.resize(n_slots);
        for (int s = 0; s < n_slots; ++s) {
            const int ip = s < m.n_phys ? s : m.core0 + (s - m.n_phys) % (m.core1 - m.core0);
            Slot & sl = slots[s];
            sl.cap = m.L[ip].swa ? std::min(ctx, m.swa) : ctx;
            sl.K = dalloc((size_t) sl.cap * Ck);
            sl.V = dalloc((size_t) sl.cap * Ck);
            sl.hist = dalloc((size_t) 3 * (2 * Ck + 2 * C));
        }
        x = dalloc(C); t0 = dalloc(C); t1 = dalloc(C); t2 = dalloc(C);
        for (auto & p : inp) p = dalloc(C);
        q = dalloc(m.H * m.D); k = dalloc(Ck); v = dalloc(Ck); r = dalloc(m.H * m.nrel); sinkv = dalloc(m.H);
        att = dalloc(m.H * m.D); g = dalloc(m.H * m.D); dk = dalloc(4 * m.Hkv); dv = dalloc(4 * m.Hkv);
        scr = dalloc((size_t) m.H * ctx); logits = dalloc(m.vocab_pad); msgs = dalloc(4 * C);
        ctrl_g = dalloc(m.rank); ctrl_a = dalloc(C); ctrl_b = dalloc(C); cat = dalloc(2 * C);
        gu = dalloc((size_t) m.k_exp * 2 * m.F); hh = dalloc((size_t) m.k_exp * m.F); wts = dalloc(m.k_exp);
        rl = dalloc(m.n_exp);
        CK(hipMalloc(&ids, m.k_exp * 4));
        CK(hipMalloc(&dstates, 32 * sizeof(float *)));
        for (int i = 0; i < 128; ++i) pool.push_back(dalloc(C));   // 2 a visit (48 visits) and a few more
    }

    void mv(const Mat & a, const float * in, float * out) {
        hipLaunchKernelGGL(k_matvec, dim3((a.N + 7) / 8), dim3(256), 0, 0, a.w, a.type, in, out, a.N, a.K, a.rb);
    }
    void rms(const float * in, const float * w, float * out, int n) {
        hipLaunchKernelGGL(k_rms, dim3(1), dim3(256), 0, 0, in, w, out, n, m.eps);
    }
    // mix(blocks, cur, queries) -> out[nq][C]
    void mix(const std::vector<float *> & blocks, const float * cur, const float * queries, int nq, float * out) {
        std::vector<const float *> st(blocks.begin(), blocks.end());
        st.push_back(cur);
        CK(hipMemcpy(dstates, st.data(), st.size() * sizeof(float *), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_mix, dim3(nq), dim3(256), 0, 0, dstates, (int) st.size(), queries, out, m.C);
    }
    int ew(int n) { return (n + 255) / 256; }
    void control(float * xi, const float * msg, const Layer & l, int c) {
        const int C = m.C;
        rms(xi, nullptr, cat, C);
        rms(msg, nullptr, cat + C, C);
        mv(l.cdown[c], cat, ctrl_g);
        mv(l.cgain[c], ctrl_g, ctrl_a);
        mv(l.cforget[c], ctrl_g, ctrl_b);
        hipLaunchKernelGGL(k_control, dim3(ew(C)), dim3(256), 0, 0, xi, msg, ctrl_b, ctrl_a, C);
    }

    // one layer visit; cur in/out (x); blocks list of device pointers (owned by pool)
    void layer(int slot, int ip, float * cur, std::vector<float *> & blocks, const float * messages, int & pool_i) {
        const Layer & l = m.L[ip];
        const int C = m.C, Ck = m.Hkv * m.D;
        Slot & sl = slots[slot];
        float * inpSA = pool[pool_i++];
        CK(hipMemcpy(inpSA, cur, C * 4, hipMemcpyDeviceToDevice));
        // attention inputs (the mix goes through scr, which the attention reuses later)
        if (l.res_q) {
            float * tmp = scr;   // H*ctx floats, large enough for 3*C
            mix(blocks, cur, l.res_q, l.n_res_q, tmp);
            for (int j = 0; j < 3; ++j)
                CK(hipMemcpy(inp[j], tmp + (size_t) (l.swa ? 0 : j) * C, C * 4, hipMemcpyDeviceToDevice));
        } else {
            for (int j = 0; j < 3; ++j) CK(hipMemcpy(inp[j], cur, C * 4, hipMemcpyDeviceToDevice));
        }
        if (messages) for (int j = 0; j < 3; ++j) control(inp[j], messages + (size_t) j * C, l, j);
        if (l.swa && messages) for (int j = 0; j < 3; ++j) rms(inp[j], l.attn_norm, inp[j], C);
        else if (l.swa) {
            rms(inp[0], l.attn_norm, inp[0], C);
            CK(hipMemcpy(inp[1], inp[0], C * 4, hipMemcpyDeviceToDevice));
            CK(hipMemcpy(inp[2], inp[0], C * 4, hipMemcpyDeviceToDevice));
        } else {
            rms(inp[0], l.qn, inp[0], C); rms(inp[1], l.kn, inp[1], C); rms(inp[2], l.vn, inp[2], C);
        }
        // projections
        mv(l.wq, inp[0], q);
        mv(l.wk, inp[1], k);
        mv(l.wv, inp[2], v);
        mv(l.rel, inp[0], r);
        mv(l.sink, inp[0], sinkv);
        mv(l.gate, inp[0], g);
        const float * dkp = nullptr, * dvp = nullptr;
        if (!l.swa) { mv(l.kproj, inp[1], dk); mv(l.vproj, inp[2], dv); dkp = dk; dvp = dv; }
        const float cs = 1.0f / sqrtf((float) ((l.swa ? 1 : 2) * 4 + 1));
        hipLaunchKernelGGL(k_conv, dim3(ew(Ck)), dim3(256), 0, 0, k, l.kconv, sl.hist, Ck, dkp, m.D, cs);
        hipLaunchKernelGGL(k_conv, dim3(ew(Ck)), dim3(256), 0, 0, v, l.vconv, sl.hist + 3 * Ck, Ck, dvp, m.D, cs);
        const float tau = l.swa ? 1.0f : 1.0f + m.alpha * logf(std::max(1.0f, (float) (pos + 1) / m.floor_));
        hipLaunchKernelGGL(k_clip, dim3(m.H), dim3(128), 0, 0, q, l.q_clip, m.D, m.qk_clip, tau);
        hipLaunchKernelGGL(k_clip, dim3(m.Hkv), dim3(128), 0, 0, k, (const float *) nullptr, m.D, m.qk_clip, 1.0f);
        hipLaunchKernelGGL(k_clip, dim3(m.H), dim3(32), 0, 0, r, l.rel_clip, m.nrel, m.rel_clip, tau);
        for (int h = 0; h < m.Hkv; ++h)
            hipLaunchKernelGGL(k_rms, dim3(1), dim3(128), 0, 0, v + h * m.D, (const float *) nullptr, v + h * m.D, m.D, m.eps);
        hipLaunchKernelGGL(k_store_kv, dim3(ew(Ck)), dim3(256), 0, 0, k, v, sl.K, sl.V, pos % sl.cap, Ck);
        const int span = l.swa ? std::min(m.swa, ctx) : ctx;
        hipLaunchKernelGGL(k_attn, dim3(m.H), dim3(256), 0, 0, q, sl.K, sl.V, sl.cap, pos, span, m.Hkv, m.D, r,
                           l.rel_bias, m.nrel, m.ext, sinkv, att, scr);
        hipLaunchKernelGGL(k_gate, dim3(ew(m.H * m.D)), dim3(256), 0, 0, att, g, m.H * m.D);
        mv(l.wo, att, t0);
        hipLaunchKernelGGL(k_conv, dim3(ew(C)), dim3(256), 0, 0, t0, l.post_conv, sl.hist + 6 * Ck, C,
                           (const float *) nullptr, 1, 1.0f / sqrtf(5.0f));
        if (ip % m.block == 0) {
            blocks.push_back(inpSA);
            CK(hipMemcpy(cur, t0, C * 4, hipMemcpyDeviceToDevice));
        } else {
            hipLaunchKernelGGL(k_add, dim3(ew(C)), dim3(256), 0, 0, t0, inpSA, C);
            CK(hipMemcpy(cur, t0, C * 4, hipMemcpyDeviceToDevice));
        }
        float * ffn_inp = pool[pool_i++];
        CK(hipMemcpy(ffn_inp, cur, C * 4, hipMemcpyDeviceToDevice));
        // MoE
        mix(blocks, cur, l.ffn_res_q, 1, t1);
        if (messages) control(t1, messages + 3 * (size_t) C, l, 3);
        rms(t1, l.ffn_norm, t1, C);
        mv(l.router, t1, rl);
        hipLaunchKernelGGL(k_route, dim3(1), dim3(m.n_exp), 0, 0, rl, l.probs_b, m.n_exp, m.k_exp, m.wscale, ids, wts);
        hipLaunchKernelGGL(k_exp_gu, dim3((2 * m.F + 7) / 8, m.k_exp), dim3(256), 0, 0, l.gu, l.gu_type, l.gu_ebytes,
                           l.gu_rb, l.gu_scale, ids, t1, gu, 2 * m.F, C);
        hipLaunchKernelGGL(k_situ, dim3(ew(m.F * m.k_exp)), dim3(256), 0, 0, gu, hh, m.F, m.k_exp, m.situ_b, m.situ_lb);
        hipLaunchKernelGGL(k_exp_down, dim3((C + 7) / 8), dim3(256), 0, 0, l.down, l.down_type, l.down_ebytes,
                           l.down_rb, l.down_scale, ids, wts, hh, t2, C, m.F, m.k_exp);
        hipLaunchKernelGGL(k_conv, dim3(ew(C)), dim3(256), 0, 0, t2, l.ffn_conv, sl.hist + 6 * Ck + 3 * C, C,
                           (const float *) nullptr, 1, 1.0f / sqrtf(5.0f));
        hipLaunchKernelGGL(k_add, dim3(ew(C)), dim3(256), 0, 0, t2, ffn_inp, C);
        CK(hipMemcpy(cur, t2, C * 4, hipMemcpyDeviceToDevice));
    }

    // feed one token; the logits of the next one land in `logits` (host copy in out when given)
    void step(int tok, std::vector<float> * out) {
        const int C = m.C;
        int pool_i = 0;
        hipLaunchKernelGGL(k_embed, dim3(1), dim3(256), 0, 0, m.embd, tok, x, C);
        rms(x, m.embd_norm, x, C);
        std::vector<float *> blocks;
        for (int il = 0; il < m.core0; ++il) layer(il, il, x, blocks, nullptr, pool_i);
        float * prelude = pool[pool_i++];
        CK(hipMemcpy(prelude, x, C * 4, hipMemcpyDeviceToDevice));
        const std::vector<float *> prelude_blocks = blocks;
        const int ncore = m.core1 - m.core0;
        for (int p = 0; p < m.passes; ++p) {
            std::vector<float *> bank;
            float * bank_out = nullptr;
            if (p > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = pool[pool_i++];
                CK(hipMemcpy(bank_out, x, C * 4, hipMemcpyDeviceToDevice));
                CK(hipMemcpy(x, prelude, C * 4, hipMemcpyDeviceToDevice));
                blocks = prelude_blocks;
            }
            for (int ip = m.core0; ip < m.core1; ++ip) {
                const int slot = p == 0 ? ip : m.n_phys + (p - 1) * ncore + (ip - m.core0);
                if (p == 0) { layer(slot, ip, x, blocks, nullptr, pool_i); continue; }
                mix(bank, bank_out, m.L[ip].msg_q, 4, msgs);
                layer(slot, ip, x, blocks, msgs, pool_i);
            }
        }
        for (int il = m.core1; il < m.n_phys; ++il) layer(il, il, x, blocks, nullptr, pool_i);
        mix(blocks, x, m.out_res_q, 1, t1);
        rms(t1, m.out_norm, t1, C);
        mv(m.head, t1, logits);
        hipLaunchKernelGGL(k_softcap, dim3(ew(m.vocab)), dim3(256), 0, 0, logits, m.vocab, m.softcap);
        ++pos;
        if (out) {
            out->resize(m.vocab);
            CK(hipMemcpy(out->data(), logits, (size_t) m.vocab * 4, hipMemcpyDeviceToHost));
        }
    }
};

}  // namespace

int main(int argc, char ** argv) {
    std::string path, score, dump;
    int gpu = 0, ctx = 4096, n_dump = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") path = next();
        else if (a == "--gpu") gpu = std::atoi(next());
        else if (a == "-c") ctx = std::atoi(next());
        else if (a == "--score") score = next();
        else if (a == "--dump-logits") { dump = next(); n_dump = std::atoi(next()); }
        else { std::fprintf(stderr, "almond-inkling-native: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (path.empty() || score.empty()) {
        std::fprintf(stderr, "usage: almond-inkling-native -m MODEL.gguf [--gpu N] [-c N] --score IDS.txt [--dump-logits FILE N]\n");
        return 2;
    }
    CK(hipSetDevice(gpu));
    Model m;
    const auto t0 = std::chrono::steady_clock::now();
    m.load(path);
    std::fprintf(stderr, "loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::vector<int> ids;
    { std::ifstream f(score); int v; while (f >> v) ids.push_back(v); }
    if ((int) ids.size() > ctx) ids.resize(ctx);
    Runner run(m, ctx);
    std::vector<float> lg;
    FILE * df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    double nll = 0;
    int n_scored = 0, dumped = 0;
    std::vector<int> top1;
    const int first = (int) ids.size() / 2;
    const auto t1 = std::chrono::steady_clock::now();
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
        run.step(ids[i], &lg);
        const float mx = *std::max_element(lg.begin(), lg.end());
        double z = 0;
        for (float v : lg) z += std::exp((double) v - mx);
        const double lp = (double) lg[ids[i + 1]] - mx - std::log(z);
        top1.push_back((int) (std::max_element(lg.begin(), lg.end()) - lg.begin()));
        if ((int) i >= first) { nll -= lp; ++n_scored; }
        if (df && (int) i >= first && dumped < n_dump) { std::fwrite(lg.data(), 4, lg.size(), df); ++dumped; }
        if (i % 256 == 0) std::fprintf(stderr, "%zu / %zu\n", i, ids.size());
    }
    if (df) std::fclose(df);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("SCORE ppl=%.4f scored=%d tokens=%zu tok/s=%.2f\n", std::exp(nll / std::max(1, n_scored)), n_scored,
                ids.size(), (double) (ids.size() - 1) / secs);
    std::printf("TOP1");
    for (int t : top1) std::printf(" %d", t);
    std::printf("\n");
    return 0;
}

// src/inkling/native/ink_native.hip - almond-inkling-native: Almond's own engine for Inkling 2 Mini (GGUF
// architecture inkling2_loop), HIP for the RX 7900 XTX, no ggml. docs/inkling/ENGINE.md has the design and the math;
// the reference is llama.cpp's src/models/inkling2*.cpp, which this follows op for op.
//
// Step 1 (this file): plain kernels, one op at a time, F32 activations, one token at a time - written to be checked
// against llama.cpp, not to be fast.
//
//   almond-inkling-native -m MODEL.gguf [--gpu N] [-c N] [--kv-f16 | --kv-q8] [--ubatch N] --serve [--eos ID]...
//   almond-inkling-native -m MODEL.gguf [--gpu N] [-c N] --score IDS.txt [--chunk N] [--dump-logits FILE N]
//
// --serve: almond-inkling's line protocol (src/inkling/inkling_serve.cpp), what ./almond -m inkling runs.
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
#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
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

#ifndef ALMOND_VERSION
#define ALMOND_VERSION "dev"
#endif

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

// ---- the experts' memory lent to an on-demand image encoder (VLEND / VRECLAIM, as the GLM engine does). The
// expert tensors sit in reserved virtual address ranges with their own physical memory: lending unmaps and releases
// the last ones' memory (their addresses stay reserved, so the captured graphs stay valid), reclaiming maps new
// memory at the same addresses and copies the tensors back from the mapped GGUF - the model is bit for bit what it was
struct Lendable { void * va; size_t size; const uint8_t * host; size_t bytes; hipMemGenericAllocationHandle_t h; bool mapped; };
std::vector<Lendable> g_lend;
hipMemAllocationProp lend_prop() {
    hipMemAllocationProp pr{};
    pr.type = hipMemAllocationTypePinned;
    pr.location.type = hipMemLocationTypeDevice;
    int dev = 0;
    (void) hipGetDevice(&dev);
    pr.location.id = dev;
    return pr;
}
bool lend_map(Lendable & L) {
    hipMemAllocationProp pr = lend_prop();
    if (hipMemCreate(&L.h, L.size, &pr, 0) != hipSuccess) { (void) hipGetLastError(); return false; }
    if (hipMemMap(L.va, L.size, 0, L.h, 0) != hipSuccess) { (void) hipMemRelease(L.h); (void) hipGetLastError(); return false; }
    hipMemAccessDesc ad{};
    ad.location = pr.location;
    ad.flags = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess(L.va, L.size, &ad, 1) != hipSuccess) {
        (void) hipMemUnmap(L.va, L.size); (void) hipMemRelease(L.h); (void) hipGetLastError(); return false;
    }
    L.mapped = true;
    return true;
}
// a lendable device copy of host bytes (null when the device cannot: then a plain allocation)
void * lend_alloc(const uint8_t * host, size_t bytes) {
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, v = 0;
        (void) hipGetDevice(&dev);
        ok = hipDeviceGetAttribute(&v, hipDeviceAttributeVirtualMemoryManagementSupported, dev) == hipSuccess && v &&
             !std::getenv("INK_NO_LEND");
    }
    if (!ok) return nullptr;
    hipMemAllocationProp pr = lend_prop();
    size_t gran = 0;
    if (hipMemGetAllocationGranularity(&gran, &pr, hipMemAllocationGranularityRecommended) != hipSuccess || !gran) gran = 2 << 20;
    Lendable L{nullptr, (bytes + gran - 1) / gran * gran, host, bytes, {}, false};
    if (hipMemAddressReserve(&L.va, L.size, 0, nullptr, 0) != hipSuccess) { (void) hipGetLastError(); return nullptr; }
    if (!lend_map(L)) { (void) hipMemAddressFree(L.va, L.size); return nullptr; }
    if (hipMemcpy(L.va, host, bytes, hipMemcpyHostToDevice) != hipSuccess) { std::fprintf(stderr, "lend: upload failed\n"); std::exit(1); }
    g_lend.push_back(L);
    return L.va;
}
// frees the last expert tensors' memory until `want` bytes are free on the device (the encoder's measured need);
// returns the bytes released
size_t lend_out(size_t want) {
    (void) hipDeviceSynchronize();
    size_t fr = 0, tot = 0;
    (void) hipMemGetInfo(&fr, &tot);
    size_t got = 0;
    for (size_t i = g_lend.size(); i-- > 0 && fr + got < want;) {
        Lendable & L = g_lend[i];
        if (!L.mapped) continue;
        (void) hipMemUnmap(L.va, L.size);
        (void) hipMemRelease(L.h);
        L.mapped = false;
        got += L.size;
    }
    (void) hipDeviceSynchronize();
    return got;
}
size_t lend_owed() { size_t n = 0; for (const Lendable & L : g_lend) n += L.mapped ? 0 : 1; return n; }
// maps and refills what was lent; false while the memory is still taken (then nothing more is mapped)
bool lend_back() {
    for (Lendable & L : g_lend) {
        if (L.mapped) continue;
        if (!lend_map(L)) return false;
        if (hipMemcpy(L.va, L.host, L.bytes, hipMemcpyHostToDevice) != hipSuccess) {
            std::fprintf(stderr, "lend: the copy back failed\n");
            std::exit(1);
        }
    }
    return true;
}

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
            } else if (n.size() > 12 && n.compare(n.size() - 12, 12, "_exps.weight") == 0 &&
                       (x->dev = lend_alloc(x->host, x->bytes))) {
                // (the experts: lendable to an image encoder, see lend_alloc)
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
                         const float * x, float * out, int N, int K, int kx = 1 << 30) {
    fp4_init();
    const int j = blockIdx.y, e = ids[j];   // entry j (token j / kx)
    const int n = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (n >= N) return;
    const float v = row_dot(W + (size_t) e * ebytes + (size_t) n * rb, type, x + (size_t) (j / kx) * K, K);
    if (threadIdx.x % WS == 0) out[(size_t) j * N + n] = v * (scale ? scale[e] : 1.0f);
}
// NVFP4 decode experts (one token's entries): one block of 64 values, lane part sub (16 values), x already loaded
__device__ __forceinline__ float nv_blk(const uint8_t * blk, int sub, const float * xv) {
    const float d = gUe4m3[blk[sub]];
    const uint32_t * qv = (const uint32_t *) (blk + 4 + sub * 8);
    const uint32_t q0 = qv[0], q1 = qv[1];
    float a = 0.0f;
    const uint32_t qs[4] = {q0 & 0x0F0F0F0Fu, q1 & 0x0F0F0F0Fu, (q0 >> 4) & 0x0F0F0F0Fu, (q1 >> 4) & 0x0F0F0F0Fu};
#pragma unroll
    for (int g = 0; g < 4; ++g) {   // 4 nibbles -> 4 signed int8 (the doubled E2M1 values) with two byte permutes
        const uint32_t sel = qs[g] & 0x07070707u, m = ((qs[g] >> 3) & 0x01010101u) * 0xFFu;
        const uint32_t v = (__builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, sel) & m) |
                           (__builtin_amdgcn_perm(0x0C080604u, 0x03020100u, sel) & ~m);
        const float * xg = xv + (g & 1) * 4 + (g >> 1) * 8;
        a += (float) (int8_t) v * xg[0] + (float) (int8_t) (v >> 8) * xg[1] + (float) (int8_t) (v >> 16) * xg[2] +
             (float) (int8_t) (v >> 24) * xg[3];
    }
    return d * a;
}
// gate/up: R rows a wave that share each x load
template <int R>
__global__ void k_exp_gu_nv(const uint8_t * W, size_t ebytes, size_t rb, const float * scale, const int * ids,
                            const float * x, float * out, int N, int K, int kx) {
    const int j = blockIdx.y, e = ids[j];
    const int n0 = (blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS) * R;
    if (n0 >= N) return;
    const int l = threadIdx.x % WS, sub = l & 3;
    const uint8_t * w = W + (size_t) e * ebytes + (size_t) n0 * rb;
    x += (size_t) (j / kx) * K;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int b = l >> 2; b < K / 64; b += WS / 4) {
        const float * xb = x + b * 64 + sub * 16;
        float xv[16];
        *(float4 *) xv = *(const float4 *) xb; *(float4 *) (xv + 4) = *(const float4 *) (xb + 4);
        *(float4 *) (xv + 8) = *(const float4 *) (xb + 8); *(float4 *) (xv + 12) = *(const float4 *) (xb + 12);
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += nv_blk(w + (size_t) r * rb + (size_t) b * 36, sub, xv);
    }
    const float s = scale ? scale[e] : 1.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = wsum(acc[r]);
        if (l == 0 && n0 + r < N) out[(size_t) j * N + n0 + r] = v * s;
    }
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
    const int n = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS, tk = blockIdx.y;   // token tk's k entries
    if (n >= N) return;
    ids += tk * k; wts += tk * k; h += (size_t) tk * k * K; y += (size_t) tk * N;
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
// a prompt's picture rows take their tokens' place, as the encoder gave them (llama.cpp does not normalise raw
// embeddings): x[t] = rows[t] where mask[t]
__global__ void k_img_put(float * x, const float * rows, const int * mask, int C) {
    if (!mask[blockIdx.x]) return;
    for (int i = threadIdx.x; i < C; i += blockDim.x) x[(size_t) blockIdx.x * C + i] = rows[(size_t) blockIdx.x * C + i];
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
// k_ctrl_apply then the RMS norm of each output times w[j] (k_rms_n's arithmetic: 256 threads, same sums); grid (n, T)
__global__ void k_ctrl_apply_n(Ctl P, int C, Norms W, float eps) {
    const int j = blockIdx.x, t = blockIdx.y;
    float * out = P.out[j] + (size_t) t * C;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float x = P.x[j][(size_t) t * P.xs + i];
        const size_t o4 = (size_t) t * 4 * C + i;
        const float y = x - tanhf(P.f[j][o4]) * x + tanhf(P.g[j][o4]) * P.m[j][(size_t) t * P.ms + i];
        out[i] = y;
        ss += y * y;
    }
    ss = bsum(ss);
    const float r = rsqrtf(ss / C + eps);
    const float * w = W.w[j];
    for (int i = threadIdx.x; i < C; i += blockDim.x) out[i] = out[i] * r * (w ? w[i] : 1.0f);
}
__global__ void k_ctrl_apply(Ctl P, int C) {
    const int j = blockIdx.y, t = blockIdx.z, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    const float x = P.x[j][(size_t) t * P.xs + i];
    const size_t o4 = (size_t) t * 4 * C + i;
    P.out[j][(size_t) t * C + i] = x - tanhf(P.f[j][o4]) * x + tanhf(P.g[j][o4]) * P.m[j][(size_t) t * P.ms + i];
}
// K and V convolutions in one launch (blockIdx.y)
struct Conv2 { float * x[2]; const float * w[2]; float * hist[2]; const float * dyn[2]; float * sv[2] = {nullptr, nullptr}; };
__global__ void k_conv2(Conv2 P, int C, int head_dim, float scale, int T, int dyn_stride) {
    const int j = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float * hist = P.hist[j];
    const float * w = P.w[j];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c];
    float * sv = P.sv[j];   // speculative verification: the state before the batch, then every input
    if (sv) { sv[c] = h0; sv[C + c] = h1; sv[2 * C + c] = h2; }
    for (int t = 0; t < T; ++t) {
        float * xp = P.x[j] + (size_t) t * C + c;
        const float v = *xp;
        if (sv) sv[(size_t) (3 + t) * C + c] = v;
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
                           int T, float * sv = nullptr) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c];
    if (sv) { sv[c] = h0; sv[C + c] = h1; sv[2 * C + c] = h2; }
    for (int t = 0; t < T; ++t) {
        const size_t i = (size_t) t * C + c;
        const float v = x[i];
        if (sv) sv[(size_t) (3 + t) * C + c] = v;
        const float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * v;
        dst[i] = (v + o) * scale + (res ? res[i] : 0.0f);
        h0 = h1; h1 = h2; h2 = v;
    }
    hist[c] = h0; hist[C + c] = h1; hist[2 * C + c] = h2;
}
// clip q and r (times tau on full layers), clip k and RMS-norm v, and write k and v to the cache slot; one block a head
// KV cache rows (one position, one KV head, D = 128): F32, or q8_0 (--kv-q8: 128 int8 then 4 F16 scales, 136 bytes)
constexpr int KVQ8_ROW = 136;
// kvt: 0 F32, 1 q8_0, 2 F16
__device__ __forceinline__ float4 kv_load(const void * C, size_t row, int l4, int q8) {
    if (q8 == 2) {
        const uint2 u = *(const uint2 *) ((const uint8_t *) C + row * 256 + 8 * l4);
        const __half2 a = __builtin_bit_cast(__half2, u.x), b = __builtin_bit_cast(__half2, u.y);
        return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
    }
    if (q8) {
        const uint8_t * b = (const uint8_t *) C + row * KVQ8_ROW;
        const uint32_t u = *(const uint32_t *) (b + 4 * l4);
        const float d = __half2float(((const __half *) (b + 128))[l4 >> 3]);
        return make_float4((float) (int8_t) (u & 0xFF) * d, (float) (int8_t) ((u >> 8) & 0xFF) * d,
                           (float) (int8_t) ((u >> 16) & 0xFF) * d, (float) (int8_t) (u >> 24) * d);
    }
    return *(const float4 *) ((const float *) C + row * 128 + l4 * 4);
}
// fused into k_qkvr (decode): the K / V short convolution of one token (k_conv2's arithmetic, T = 1), and the
// relative bias of every query head (k_relbias's arithmetic) once its R is clipped
struct QFuse {
    const float * cw[2]; float * hist[2]; const float * dyn[2]; float cscale; int dyn_stride; int conv;
    const float * table; int ext; const float * sink; float * bias;
};
__device__ __forceinline__ bool sc_is_q(int b, int H, int Hkv) { return b < H + 2 * Hkv; }   // not an R head
__global__ void k_qkvr(float * q, const float * k, const float * v, float * r, const float * qs, const float * rs, int H,
                       int Hkv, int D, int nrel, float th, float rth, const int * pos, float alpha, float floor_, float eps,
                       void * Kc, void * Vc, int cap, int q8, QFuse fz = QFuse{}) {
    int b = blockIdx.x;
    const int tk = blockIdx.y, p = *pos + tk;
    q += (size_t) tk * H * D; k += (size_t) tk * Hkv * D; v += (size_t) tk * Hkv * D; r += (size_t) tk * H * nrel;
    const float tau = tau_at(p, alpha, floor_);
    float * x;
    const float * sc = nullptr;
    int n = D;
    float t = th, f2 = 1.0f;
    void * dst = nullptr;
    size_t drow = 0;
    bool vnorm = false;
    if (b < H) { x = q + (size_t) b * D; sc = qs + (size_t) b * D; f2 = tau; }
    else if ((b -= H) < Hkv) { x = (float *) k + (size_t) b * D; dst = Kc; drow = (size_t) (p % cap) * Hkv + b; }
    else if ((b -= Hkv) < Hkv) { x = (float *) v + (size_t) b * D; dst = Vc; drow = (size_t) (p % cap) * Hkv + b; vnorm = true; }
    else { b -= Hkv; x = r + (size_t) b * nrel; sc = rs + (size_t) b * nrel; n = nrel; t = rth; f2 = tau; }
    if (fz.conv && dst) {   // one token: the convolution of this head's channels first (thread = channel)
        const int j = vnorm ? 1 : 0, i = threadIdx.x, c = b * D + i, Ck = Hkv * D;
        float * hist = fz.hist[j];
        const float * w = fz.cw[j];
        const float h0 = hist[c], h1 = hist[Ck + c], h2 = hist[2 * Ck + c], vin = x[i];
        float o = w[c * 4 + 0] * h0 + w[c * 4 + 1] * h1 + w[c * 4 + 2] * h2 + w[c * 4 + 3] * vin;
        if (fz.dyn[j]) {
            const float * tp = fz.dyn[j] + (c / D) * 4;
            o += tp[0] * vin + tp[1] * h2 + tp[2] * h1 + tp[3] * h0;
        }
        x[i] = (vin + o) * fz.cscale;
        hist[c] = h1; hist[Ck + c] = h2; hist[2 * Ck + c] = vin;
    }
    float a = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float y = x[i] * (sc ? sc[i] : 1.0f); a += y * y; }
    a = bsum(a);
    const float f = vnorm ? rsqrtf(a / n + eps) : fminf(fmaxf(t / (sqrtf(a) + 1e-6f), 0.0f), 1.0f) * f2;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = x[i] * (sc ? sc[i] : 1.0f) * f;
        if (!dst) x[i] = y;
        else if (!q8) ((float *) dst)[drow * D + i] = y;
        else if (q8 == 2) ((__half *) dst)[drow * D + i] = __float2half(y);
        else {   // q8_0 as llama.cpp rounds it: d = amax / 127 per 32 values (one wave: blockDim = D = 128)
            float am = fabsf(y);
#pragma unroll
            for (int o = WS / 2; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor(am, o, WS));
            const float d = am / 127.0f, id = d ? 1.0f / d : 0.0f;
            uint8_t * rb = (uint8_t *) dst + drow * KVQ8_ROW;
            rb[i] = (uint8_t) (int8_t) roundf(y * id);
            if ((i & 31) == 0) ((__half *) (rb + 128))[i >> 5] = __float2half(d);
        }
    }
    if (fz.bias && n == nrel && !sc_is_q(blockIdx.x, H, Hkv)) {   // an R head: its relative bias, as k_relbias
        __syncthreads();
        const int h = b, ext = fz.ext;
        const float sk = fz.sink[(size_t) tk * H + h];
        float * bo = fz.bias + ((size_t) tk * H + h) * ext;
        for (int d = threadIdx.x; d < ext; d += blockDim.x) {
            const float * tb = fz.table + (size_t) d * nrel;
            float bb = 0.0f;
            for (int i = 0; i < nrel; ++i) bb += x[i] * tb[i];
            bo[d] = bb - sk;
        }
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
__global__ void k_attn_split(const float * q, const void * Kc, const void * Vc, int cap, const int * dpos, int span,
                             int Hkv, const float * bias, int ext, const float * sink, float * part, int splitsz,
                             int nsplit, int q8) {
    constexpr int D = 128;
    const int kvh = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x % WS, wv = threadIdx.x / WS, tk = blockIdx.z;
    const int pos = *dpos + tk, first = max(0, pos - span + 1);
    q += (size_t) tk * Hkv * G * D; bias += (size_t) tk * Hkv * G * ext; sink += (size_t) tk * Hkv * G;
    part += (size_t) tk * Hkv * G * nsplit * (D + 2);
    // the splits follow the keys there are now (the position is read on the device, so a captured graph adapts):
    // at least splitsz keys a split
    const int nk = pos - first + 1, ssz = max(splitsz, (nk + nsplit - 1) / nsplit);
    const int j0 = first + sp * ssz, j1 = min(pos, j0 + ssz - 1);
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
        const size_t row = (size_t) (j % cap) * Hkv + kvh;
        const float4 kk = kv_load(Kc, row, lane, q8), vv = kv_load(Vc, row, lane, q8);
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
// prompt attention: a block takes 16 tokens x the 3 query heads of one KV head (48 rows, 6 a wave) against one split
// of the keys, the keys and values staged in LDS 32 at a time and shared by all 48 rows (the decode kernel reads
// them again for every token). F32 throughout, same online softmax; writes part[] like k_attn_split.
// grid (Hkv, ceil(T / 16), nsplit), 256 threads
__global__ void k_attn_pf(const float * q, const void * Kc, const void * Vc, int cap, const int * dpos, int span,
                          int Hkv, const float * bias, int ext, const float * sink, float * part, int splitsz,
                          int nsplit, int T, int q8) {
    constexpr int D = 128, G = 3, KT = 32, R = 6;
    __shared__ float4 Ks[KT][D / 4 + 1];
    __shared__ float Vs[KT][D];
    __shared__ float Qs[16 * G][D];
    const int kvh = blockIdx.x, tt0 = blockIdx.y * 16, sp = blockIdx.z;
    const int lane = threadIdx.x % WS, wv = threadIdx.x / WS;
    const int pos0 = *dpos, H = Hkv * G;
    const int lastT = min(T, tt0 + 16) - 1;
    const int kfirst = max(0, pos0 + tt0 - span + 1), kend = pos0 + lastT;   // the tile's keys
    const int j0 = kfirst + sp * splitsz, j1 = min(kend, j0 + splitsz - 1);
    const float qsc = rsqrtf((float) D);
    for (int i = threadIdx.x; i < 16 * G * D; i += blockDim.x) {
        const int rr = i / D, d = i % D, tk = tt0 + rr / G;
        Qs[rr][d] = tk < T ? q[((size_t) tk * H + kvh * G + rr % G) * D + d] * qsc : 0.0f;
    }
    // wave wv: rows 6wv..6wv+5 = tokens tt0 + 2wv (+1), heads g = row % 3
    float m[R], ls[R];
    float4 acc[R];
    int qpos[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { m[r] = -INFINITY; ls[r] = 0.0f; acc[r] = make_float4(0, 0, 0, 0); qpos[r] = pos0 + tt0 + (R * wv + r) / G; }
    for (int jb = j0; jb <= j1; jb += KT) {
        __syncthreads();
        for (int i = threadIdx.x; i < KT * D / 4; i += blockDim.x) {
            const int kk = i / (D / 4), d4 = (i % (D / 4)) * 4, j = jb + kk;
            float4 a = make_float4(0, 0, 0, 0), b = a;
            if (j <= j1) {
                const size_t row = (size_t) (j % cap) * Hkv + kvh;
                a = kv_load(Kc, row, d4 / 4, q8); b = kv_load(Vc, row, d4 / 4, q8);
            }
            Ks[kk][d4 / 4] = a;
            *(float4 *) &Vs[kk][d4] = b;
        }
        __syncthreads();
        // lane = key jb + lane: its score for the wave's 6 rows
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) s[r] = 0.0f;
        for (int d = 0; d < D / 4; ++d) {
            const float4 kv = Ks[lane][d];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float4 qv = *(const float4 *) &Qs[R * wv + r][4 * d];
                s[r] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            }
        }
        const int j = jb + lane;
        float pr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int tk = tt0 + (R * wv + r) / G, h = kvh * G + (R * wv + r) % G, dd = qpos[r] - j;
            const bool ok = tk < T && j <= j1 && dd >= 0 && dd < span;
            float sc = -INFINITY;
            if (ok) sc = s[r] + (dd < ext ? bias[((size_t) tk * H + h) * ext + dd] : -sink[(size_t) tk * H + h]);
            float mx = sc;
#pragma unroll
            for (int o = WS / 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, WS));
            const float mn = fmaxf(m[r], mx);
            if (mn == -INFINITY) { pr[r] = 0.0f; continue; }   // nothing visible yet for this row
            const float f = expf(m[r] - mn);
            pr[r] = ok ? expf(sc - mn) : 0.0f;
            ls[r] = ls[r] * f + wsum(pr[r]);
            acc[r] = make_float4(acc[r].x * f, acc[r].y * f, acc[r].z * f, acc[r].w * f);
            m[r] = mn;
        }
        // P . V: every value row read once for the wave's 6 rows
        for (int kk = 0; kk < KT; ++kk) {
            const float4 vv = *(const float4 *) &Vs[kk][lane * 4];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float pk = __shfl(pr[r], kk, WS);
                acc[r].x += pk * vv.x; acc[r].y += pk * vv.y; acc[r].z += pk * vv.z; acc[r].w += pk * vv.w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int tk = tt0 + (R * wv + r) / G, h = kvh * G + (R * wv + r) % G;
        if (tk >= T) continue;
        float * o = part + (((size_t) tk * H + h) * nsplit + sp) * (D + 2);
        if (lane == 0) { o[0] = m[r]; o[1] = ls[r]; }
        *(float4 *) (o + 2 + lane * 4) = acc[r];
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


// speculative rollback: every slot's convolution state after the first `keep` tokens of the verified batch, from the
// state before it and its inputs (k_conv2 / k_conv_res with sv). grid (ceil((2Ck + 2C) / 256), slots)
constexpr int SPEC_T = 16;   // the longest verification batch
__global__ void k_spec_roll(float * const * hists, float * const * specs, int keep, int Ck, int C) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= 2 * Ck + 2 * C) return;
    float * h = hists[blockIdx.y];
    const float * sp = specs[blockIdx.y];
    int W, c, hoff, soff;
    constexpr int R = 3 + SPEC_T;
    if (e < Ck)                { W = Ck; c = e;              hoff = 0;               soff = 0; }
    else if (e < 2 * Ck)       { W = Ck; c = e - Ck;         hoff = 3 * Ck;          soff = R * Ck; }
    else if (e < 2 * Ck + C)   { W = C;  c = e - 2 * Ck;     hoff = 6 * Ck;          soff = 2 * R * Ck; }
    else                       { W = C;  c = e - 2 * Ck - C; hoff = 6 * Ck + 3 * C;  soff = 2 * R * Ck + R * C; }
#pragma unroll
    for (int i = 0; i < 3; ++i) h[hoff + i * W + c] = sp[soff + (size_t) (keep + i) * W + c];
}

// ---------------------------------------------------------------- batched kernels (prompts)
// 8 values of a row from column k (a multiple of 8), dequantized
__device__ __forceinline__ void load8(const uint8_t * row, uint32_t type, int K, int k, float * o);
// a few tokens (T <= SPEC_T: verification batches, the drafter's block): one wave a row, the row read once for all
// T tokens. grid (rows / 8), 256 threads; Seg xs / ys: the per-token strides
constexpr int MMV_T = 16;
__global__ void k_mmvT(Segs S, int T) {
    fp4_init();
    const int row = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS, lane = threadIdx.x % WS;
    if (row >= S.start[S.n]) return;
    int j = 0;
    while (row >= S.start[j + 1]) ++j;
    const Seg & g = S.s[j];
    const int n = row - S.start[j], xs = g.xs ? g.xs : g.K, ys = g.ys ? g.ys : g.N;
    const uint8_t * w = g.w + (size_t) n * g.rb;
    float acc[MMV_T];
#pragma unroll
    for (int t = 0; t < MMV_T; ++t) acc[t] = 0.0f;
    for (int c = lane; c < g.K / 8; c += WS) {
        float wv[8];
        load8(w, g.type, g.K, c * 8, wv);
#pragma unroll
        for (int t = 0; t < MMV_T; ++t) {
            if (t >= T) break;
            const float * xp = g.x + (size_t) t * xs + c * 8;
            const float4 a = *(const float4 *) xp, b = *(const float4 *) (xp + 4);
            acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z +
                      wv[7] * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < MMV_T; ++t) {
        if (t >= T) break;
        const float v = wsum(acc[t]);
        if (lane == 0) g.y[(size_t) t * ys + n] = v;
    }
}
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
constexpr int EXP_BT = 32;   // entries a tile in the matrix-core expert GEMM
// group the (token, slot) entries by expert: counts, offsets and the entry list (one block)
__global__ void k_group(const int * ids, int n, int n_exp, int * offs, int * list) {
    __shared__ int cnt[256], cur[256];
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&cnt[ids[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0;
        int nt = 0;   // and the list of (expert, first entry) tiles of EXP_BT after offs[n_exp + 1]
        for (int e = 0; e < n_exp; ++e) {
            offs[e] = a; cur[e] = a; a += cnt[e];
            for (int t = 0; t < cnt[e]; t += EXP_BT) { offs[n_exp + 2 + 2 * nt] = e; offs[n_exp + 3 + 2 * nt] = t; ++nt; }
        }
        offs[n_exp] = a;
        offs[n_exp + 1] = nt;
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
// ---- matrix cores (gfx11 WMMA, 16x16x16 F16 -> F32). The weights go in as their quantized integers (Q8_0: the int8,
// NVFP4: the doubled E2M1 values), exact in F16; the activations as two F16 parts (hi + lo, about 22 bits of
// mantissa); each 16-wide K step is scaled by its block scale in F32. Tiles of 64 rows x BT tokens (BT/16 x 4 waves).
typedef _Float16 h16v __attribute__((ext_vector_type(16)));
// the low part is kept times 2^12 (an exact scale) so small values do not fall into F16's subnormals, and is summed in
// its own accumulator then scaled back
constexpr float LO_UP = 4096.0f, LO_DN = 1.0f / 4096.0f;
typedef float f8v __attribute__((ext_vector_type(8)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
__device__ __forceinline__ bool wmma_ok(const Seg & g) { return (g.type == T_Q8P || g.type == T_NVFP4) && g.K % 64 == 0; }
template <int BT, int NB, bool FOLD = false>   // NB: 16-token tiles a wave (the decoded weights reused NB times)
__device__ __forceinline__ void wmma_tile(const Seg & g, int n0, int T, int t0, const int * xrow, const int * yrow,
                                          float yscale_fixed, const float * yscale_row) {
    __shared__ _Float16 Xh[BT][64 + 8], Xl[BT][64 + 8];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    constexpr int NT = 128 * BT / (16 * NB);
    const int rw = (w & 3) * 16, tw = (w >> 2) * 16 * NB;
    const int xs = g.xs ? g.xs : g.K, ys = g.ys ? g.ys : g.N;
    const int m = lane & 15, n = n0 + rw + m;
    const bool rok = n < g.N;
    const uint8_t * row = g.w + (size_t) (rok ? n : 0) * g.rb;
    f8v acc[NB] = {};
    const int hf = lane >> 4;
    // the next step's weights and activations are loaded into registers while this one computes
    auto loadW = [&](int k0, uint32_t * pw, uint32_t & pws) {
        if (g.type == T_Q8P) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint2 qq = *(const uint2 *) (row + k0 + 16 * s + 8 * hf);
                pw[2 * s] = qq.x; pw[2 * s + 1] = qq.y;
            }
            pws = *(const uint32_t *) (row + g.K + (k0 >> 5) * 2);
        } else {
            const uint8_t * blk = row + (size_t) (k0 >> 6) * 36;
            pws = *(const uint32_t *) blk;
#pragma unroll
            for (int i = 0; i < 8; ++i) pw[i] = *(const uint32_t *) (blk + 4 + 4 * i);
        }
    };
    auto loadX = [&](int k0, float (*v)[8]) {
#pragma unroll
        for (int e = 0; e < NB; ++e) {
            const int idx = tid + e * NT, tt = idx / 8, kp = (idx % 8) * 8, t = t0 + tt;
            float4 a = make_float4(0, 0, 0, 0), b = a;
            if (t < T) {
                const float * src = g.x + (size_t) (xrow ? xrow[t] : t) * xs + k0 + kp;
                a = *(const float4 *) src; b = *(const float4 *) (src + 4);
            }
            v[e][0] = a.x; v[e][1] = a.y; v[e][2] = a.z; v[e][3] = a.w;
            v[e][4] = b.x; v[e][5] = b.y; v[e][6] = b.z; v[e][7] = b.w;
        }
    };
    uint32_t wr[8], sw;
    float xv[NB][8];
    loadW(0, wr, sw);
    loadX(0, xv);
    for (int k0 = 0; k0 < g.K; k0 += 64) {
#pragma unroll
        for (int e = 0; e < NB; ++e) {   // X: thread -> token idx/8, 8 columns
            const int idx = tid + e * NT, tt = idx / 8, kp = (idx % 8) * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const _Float16 h = (_Float16) xv[e][i];
                Xh[tt][kp + i] = h;
                Xl[tt][kp + i] = (_Float16) ((xv[e][i] - (float) h) * LO_UP);
            }
        }
        __syncthreads();
        uint32_t cw[8], cs = sw;
#pragma unroll
        for (int i = 0; i < 8; ++i) cw[i] = wr[i];
        if (k0 + 64 < g.K) { loadW(k0 + 64, wr, sw); loadX(k0 + 64, xv); }
        f8v zf[NB] = {}, zfl[NB] = {};   // FOLD: 64 values on the matrix cores, then added in F32
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            h16v a;
            float sc;
            // each half of the wave decodes 8 of the row's 16 values and swaps them with the other half
            _Float16 own[8];
            if (g.type == T_Q8P) {
#pragma unroll
                for (int i = 0; i < 8; ++i) own[i] = (_Float16) (int8_t) ((cw[2 * s + (i >> 2)] >> (8 * (i & 3))) & 0xFF);
                sc = __half2float(__builtin_bit_cast(__half, (uint16_t) (s < 2 ? (cs & 0xFFFF) : (cs >> 16))));
            } else if (FOLD) {
                // NVFP4: the doubled E2M1 value times its E4M3 scale has at most 6 significant bits: exact in F16, so
                // the scale goes into the weights and the matrix cores accumulate the whole row
                const float d = rok ? gUe4m3[(cs >> (8 * s)) & 0xFF] : 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t bb = (cw[2 * s + (i >> 2)] >> (8 * (i & 3))) & 0xFF;
                    own[i] = (_Float16) (sFp4[hf ? (bb >> 4) : (bb & 0xF)] * d);
                }
                sc = 1.0f;
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t bb = (cw[2 * s + (i >> 2)] >> (8 * (i & 3))) & 0xFF;
                    own[i] = (_Float16) sFp4[hf ? (bb >> 4) : (bb & 0xF)];
                }
                sc = gUe4m3[(cs >> (8 * s)) & 0xFF];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t pk = (uint32_t) __builtin_bit_cast(uint16_t, own[2 * i]) |
                                    ((uint32_t) __builtin_bit_cast(uint16_t, own[2 * i + 1]) << 16);
                const uint32_t ot = __shfl_xor(pk, 16, 32);
                const _Float16 o0 = __builtin_bit_cast(_Float16, (uint16_t) (ot & 0xFFFF));
                const _Float16 o1 = __builtin_bit_cast(_Float16, (uint16_t) (ot >> 16));
                a[2 * i] = hf ? o0 : own[2 * i];
                a[2 * i + 1] = hf ? o1 : own[2 * i + 1];
                a[8 + 2 * i] = hf ? own[2 * i] : o0;
                a[8 + 2 * i + 1] = hf ? own[2 * i + 1] : o1;
            }
            if (FOLD) {
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    if (b > 0 && t0 + tw + 16 * b >= T) break;
                    h16v bh, bl;
                    const int tb = tw + 16 * b + m;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const h8v vh = *(const h8v *) &Xh[tb][s * 16 + 8 * i], vl = *(const h8v *) &Xl[tb][s * 16 + 8 * i];
#pragma unroll
                        for (int j = 0; j < 8; ++j) { bh[8 * i + j] = vh[j]; bl[8 * i + j] = vl[j]; }
                    }
                    zf[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bh, zf[b]);
                    zfl[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bl, zfl[b]);
                }
                continue;
            }
            if (!rok) sc = 0.0f;
            float scr[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) scr[i] = __shfl(sc, 2 * i + (lane >> 4), 32);
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                if (b > 0 && t0 + tw + 16 * b >= T) break;   // wave-uniform: no tokens left
                h16v bh, bl;
                const int tb = tw + 16 * b + m;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const h8v vh = *(const h8v *) &Xh[tb][s * 16 + 8 * i], vl = *(const h8v *) &Xl[tb][s * 16 + 8 * i];
#pragma unroll
                    for (int j = 0; j < 8; ++j) { bh[8 * i + j] = vh[j]; bl[8 * i + j] = vl[j]; }
                }
                f8v z = {}, zl = {};
                z = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bh, z);
                zl = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bl, zl);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[b][i] += (z[i] + zl[i] * LO_DN) * scr[i];
            }
        }
        if (FOLD)
#pragma unroll
            for (int b = 0; b < NB; ++b)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[b][i] += zf[b][i] + zfl[b][i] * LO_DN;
        __syncthreads();
    }
    // lane holds rows 2i + lane/16 of the wave's 16, token lane%16
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const int t = t0 + tw + 16 * b + m;
        if (t >= T) break;
        const int yr = yrow ? yrow[t] : t;
        const float f = yscale_fixed * (yscale_row ? yscale_row[yr] : 1.0f);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nn = n0 + rw + 2 * i + (lane >> 4);
            if (nn < g.N) g.y[(size_t) yr * ys + nn] = acc[b][i] * f;
        }
    }
}
// k_gemm on the matrix cores where a segment's type allows (the others as before); grid (row tiles, T / 64), 256 threads
__global__ void k_gemm_w(GSegs S, int T) {
    fp4_init();
    const int tile = blockIdx.x;
    int j = 0;
    while (tile >= S.tstart[j + 1]) ++j;
    __syncthreads();
    if (wmma_ok(S.s[j])) wmma_tile<64, 2>(S.s[j], (tile - S.tstart[j]) * 64, T, blockIdx.y * 64, nullptr, nullptr, 1.0f, nullptr);
    else
        for (int h = 0; h < 2; ++h)
            gemm_tile<32>(S.s[j], (tile - S.tstart[j]) * 64, T, blockIdx.y * 64 + 32 * h, nullptr, nullptr, 1.0f, nullptr);
}
// k_gemm_exp on the matrix cores: 128 threads, entry tiles of EXP_BT (each wave its 16 rows x all EXP_BT entries)
__global__ void k_gemm_exp_w(const uint8_t * W, uint32_t type, size_t ebytes, size_t rb, int N, int K, const float * x,
                             float * y, const int * offs, const int * list, int xdiv, const float * scale,
                             const float * wts, int n_exp) {
    fp4_init();
    // grid (row tiles, an upper bound of the tiles)
    if ((int) blockIdx.y >= offs[n_exp + 1]) return;
    const int e = offs[n_exp + 2 + 2 * blockIdx.y], c0 = offs[e], cnt = offs[e + 1] - c0,
              t0 = offs[n_exp + 3 + 2 * blockIdx.y];
    __shared__ int xr[EXP_BT], yr[EXP_BT];
    if (threadIdx.x < EXP_BT) {
        const int i = t0 + threadIdx.x < cnt ? list[c0 + t0 + threadIdx.x] : 0;
        xr[threadIdx.x] = i / xdiv;
        yr[threadIdx.x] = i;
    }
    __syncthreads();
    Seg g{W + (size_t) e * ebytes, x, y, rb, N, K, type};
    if (type == T_NVFP4) wmma_tile<EXP_BT, EXP_BT / 16, true>(g, blockIdx.x * 64, cnt, t0, xr - t0, yr - t0, scale ? scale[e] : 1.0f, wts);
    else wmma_tile<EXP_BT, EXP_BT / 16>(g, blockIdx.x * 64, cnt, t0, xr - t0, yr - t0, scale ? scale[e] : 1.0f, wts);
}
// 16 halves from LDS as two 16-byte reads
__device__ __forceinline__ h16v ld16(const _Float16 * p) {
    const h8v a = *(const h8v *) p, b = *(const h8v *) (p + 8);
    h16v r;
#pragma unroll
    for (int e = 0; e < 8; ++e) { r[e] = a[e]; r[8 + e] = b[e]; }
    return r;
}
// prompt attention on the matrix cores, for the F16 KV cache (--kv-f16): the keys and values are exact in F16; Q and
// the probabilities go in as two F16 parts (hi + lo), so the scores and P . V keep about F32 precision. One wave a
// block: 16 rows (token, query head) of one KV head against one split of the keys, 16 keys at a time.
// grid (Hkv, ceil(3T / 16), nsplit), 32 threads; writes part[] like k_attn_pf.
__global__ void k_attn_pf_w(const float * q, const void * Kc, const void * Vc, int cap, const int * dpos, int span,
                            int Hkv, const float * bias, int ext, const float * sink, float * part, int splitsz,
                            int nsplit, int T) {
    constexpr int D = 128, G = 3;
    __shared__ __attribute__((aligned(16))) _Float16 Qh[16][D + 8], Ql[16][D + 8], Vt[D][16 + 8], Ph[16][16 + 8],
        Pl[16][16 + 8];
    const int kvh = blockIdx.x, r0 = blockIdx.y * 16, sp = blockIdx.z, lane = threadIdx.x, hf = lane >> 4, c = lane & 15;
    const int pos0 = *dpos, H = Hkv * G, nrows = T * G;
    const int tkf = r0 / G, tkl = min(nrows - 1, r0 + 15) / G;
    const int kfirst = max(0, pos0 + tkf - span + 1), kend = pos0 + tkl;
    const int j0 = kfirst + sp * splitsz, j1 = min(kend, j0 + splitsz - 1);
    const float qsc = rsqrtf((float) D);
    for (int i = lane; i < 16 * D; i += WS) {
        const int rr = i / D, d = i % D, row = r0 + rr;
        float v = 0.0f;
        if (row < nrows) v = q[((size_t) (row / G) * H + kvh * G + row % G) * D + d] * qsc;
        const _Float16 h = (_Float16) v;
        Qh[rr][d] = h; Ql[rr][d] = (_Float16) ((v - (float) h) * LO_UP);
    }
    // this lane's 8 rows in the D layout: 2i + hf
    float m[8], ls[8];
    int qpos[8], brow[8];
    bool rok[8];
    f8v acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = r0 + 2 * i + hf;
        rok[i] = row < nrows;
        qpos[i] = pos0 + row / G;
        brow[i] = rok[i] ? (row / G) * H + kvh * G + row % G : 0;
        m[i] = -INFINITY; ls[i] = 0.0f;
        acc[i] = f8v{};
    }
    __syncthreads();
    const _Float16 * K16 = (const _Float16 *) Kc;
    const _Float16 * V16 = (const _Float16 *) Vc;
    for (int jb = j0; jb <= j1; jb += 16) {
        // values of the 16 keys into LDS (lane: half a key row)
        {
            const int kk = lane >> 1, j = jb + kk, d0 = (lane & 1) * 64;
            const uint4 * src = (const uint4 *) (V16 + ((size_t) (j % cap) * Hkv + kvh) * D + d0);
#pragma unroll
            for (int u = 0; u < 8; ++u) {   // transposed: Vt[dim][key], so a B operand is one row
                const uint4 v = j <= j1 ? src[u] : make_uint4(0, 0, 0, 0);
                const h8v hv = __builtin_bit_cast(h8v, v);
#pragma unroll
                for (int e = 0; e < 8; ++e) Vt[d0 + 8 * u + e][kk] = hv[e];
            }
        }
        // S = Q K^T: lane = key column c, its K row straight from the cache
        f8v S = {}, Sl = {};
        {
            const int j = min(jb + c, j1);
            const _Float16 * kr = K16 + ((size_t) (j % cap) * Hkv + kvh) * D;
#pragma unroll
            for (int dc = 0; dc < D / 16; ++dc) {
                const h16v b = *(const h16v *) (kr + dc * 16);
                const h16v ah = ld16(&Qh[c][dc * 16]), al = ld16(&Ql[c][dc * 16]);
                S = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, S);
                Sl = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, Sl);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S[i] += Sl[i] * LO_DN;
        }
        const int j = jb + c;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int dd = qpos[i] - j;
            const bool ok = rok[i] && j <= j1 && dd >= 0 && dd < span;
            float sc = -INFINITY;
            if (ok) sc = S[i] + (dd < ext ? bias[(size_t) brow[i] * ext + dd] : -sink[brow[i]]);
            float mx = sc;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, WS));
            const float mn = fmaxf(m[i], mx);
            float p = 0.0f;
            if (mn != -INFINITY) {
                const float f = expf(m[i] - mn);
                p = ok ? expf(sc - mn) : 0.0f;
                float ps = p;
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) ps += __shfl_xor(ps, o, WS);
                ls[i] = ls[i] * f + ps;
#pragma unroll
                for (int dt = 0; dt < 8; ++dt) acc[dt][i] *= f;
                m[i] = mn;
            }
            const _Float16 h = (_Float16) p;
            Ph[2 * i + hf][c] = h; Pl[2 * i + hf][c] = (_Float16) ((p - (float) h) * LO_UP);
        }
        __syncthreads();
        // O += P V: A = P rows (lane: row c), B = V columns (lane: dim c of the tile)
        const h16v ph = ld16(&Ph[c][0]), pl = ld16(&Pl[c][0]);
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            const h16v b = ld16(&Vt[dt * 16 + c][0]);
            f8v zl = {};
            zl = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pl, b, zl);
            acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ph, b, acc[dt]);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[dt][i] += zl[i] * LO_DN;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (!rok[i]) continue;
        float * o = part + ((size_t) brow[i] * nsplit + sp) * (D + 2);
        if (c == 0) { o[0] = m[i]; o[1] = ls[i]; }
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) o[2 + dt * 16 + c] = acc[dt][i];
    }
}
// y[t][c] = sum over the token's k entries of yd[t*k + j][c]
__global__ void k_exp_sum(const float * yd, float * y, int C, int k) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float a = 0.0f;
    for (int j = 0; j < k; ++j) a += yd[((size_t) t * k + j) * C + c];
    y[(size_t) t * C + c] = a;
}

// ---------------------------------------------------------------- DSpark drafter kernels (Unsloth's Inkling 2 Mini
// drafter, llama.cpp PR #32218's dflash/dspark graphs: 3 dense layers over context K/V made from the target's
// features, a block [last token, 7 masks] read without a causal mask, a recurrent Markov head that biases the draft
// logits and picks the tokens one after another). Its numbers only change which tokens are proposed; the target
// checks every one with the request's own sampler.
// out[o][t] = rms(sum_k mix_o[k] * f_k[t]) * norm_o, for the 4 mixes (the 3 layers' context and the Markov state)
struct DMix { const float * f[4]; const float * mix[4]; const float * norm[4]; float * out[4]; };
__global__ void k_dmix(DMix P, int C, float eps) {
    const int t = blockIdx.x, o = blockIdx.y;
    const float * w = P.mix[o];
    float * y = P.out[o] + (size_t) t * C;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) v += w[k] * P.f[k][(size_t) t * C + i];
        y[i] = v;
        ss += v * v;
    }
    const float r = rsqrtf(bsum(ss) / C + eps);
    for (int i = threadIdx.x; i < C; i += blockDim.x) y[i] = y[i] * r * P.norm[o][i];
}
// RMS norm of one head (128 values, blockDim 128) times w, then NeoX rope at position pos (pairs i, i + 64)
__device__ __forceinline__ float head_norm_rope(float v, const float * w, int pos, float theta, float eps, bool rope) {
    __shared__ float sh[128];
    const int i = threadIdx.x;
    const float r = rsqrtf(bsum(v * v) / 128.0f + eps);
    v = v * r * w[i];
    if (!rope) return v;
    sh[i] = v;
    __syncthreads();
    const int j = i & 63;
    const float ang = (float) pos * powf(theta, -2.0f * (float) j / 128.0f);
    const float cs = cosf(ang), sn = sinf(ang);
    const float a = sh[j], b = sh[j + 64];
    __syncthreads();
    return i < 64 ? a * cs - b * sn : b * cs + a * sn;
}
// context K / V of T committed tokens at positions p0..: K normed (k_ctx_norm) and roped, both stored as F16.
// grid (Hkv, T, layers), 128 threads
struct DKV { const float * k[3]; const float * v[3]; const float * kn[3]; __half * Kc[3]; __half * Vc[3]; int cap[3]; };
__global__ void k_dkv(DKV P, int Hkv, int p0, float theta, float eps) {
    const int h = blockIdx.x, t = blockIdx.y, l = blockIdx.z, i = threadIdx.x, p = p0 + t;
    const size_t src = ((size_t) t * Hkv + h) * 128 + i, dst = ((size_t) (p % P.cap[l]) * Hkv + h) * 128 + i;
    const float kv = head_norm_rope(P.k[l][src], P.kn[l], p, theta, eps, true);
    P.Kc[l][dst] = __float2half(kv);
    P.Vc[l][dst] = __float2half(P.v[l][src]);
}
// the block's Q (normed, roped, in place) and K / V (normed and roped K, raw V, into the cache) at positions n0 + t.
// grid (H + Hkv, B), 128 threads
__global__ void k_dqkv(float * q, const float * k, const float * v, const float * qn, const float * kn, int H, int Hkv,
                       const int * dn0, float theta, float eps, __half * Kc, __half * Vc, int cap) {
    const int b = blockIdx.x, t = blockIdx.y, i = threadIdx.x, p = *dn0 + t;
    if (b < H) {
        float * x = q + ((size_t) t * H + b) * 128;
        x[i] = head_norm_rope(x[i], qn, p, theta, eps, true);
    } else {
        const int h = b - H;
        const size_t src = ((size_t) t * Hkv + h) * 128 + i, dst = ((size_t) (p % cap) * Hkv + h) * 128 + i;
        Kc[dst] = __float2half(head_norm_rope(k[src], kn, p, theta, eps, true));
        Vc[dst] = __float2half(v[src]);
    }
}
// the block's attention, non-causal: every row sees the keys [first, n0 + B - 1], first = n0 - win + 1 on a window
// layer (the anchor's mask row for the whole block), else 0. grid (H, B, nsplit), 128 threads; part like k_attn_split
__global__ void k_dattn(const float * q, const __half * Kc, const __half * Vc, int cap, const int * dn0, int B, int win,
                        int Hkv, float * part, int nsplit) {
    constexpr int D = 128;
    const int h = blockIdx.x, t = blockIdx.y, sp = blockIdx.z, H = gridDim.x, kvh = h / (H / Hkv);
    const int lane = threadIdx.x % WS, wv = threadIdx.x / WS;
    const int n0 = *dn0, last = n0 + B - 1, first = win > 0 ? max(0, n0 - win + 1) : 0;
    const int nk = last - first + 1, ssz = (nk + nsplit - 1) / nsplit;
    const int j0 = first + sp * ssz, j1 = min(last, j0 + ssz - 1);
    const float4 qv0 = *(const float4 *) (q + ((size_t) t * H + h) * D + lane * 4);
    const float qs = rsqrtf((float) D);
    const float4 qv = make_float4(qv0.x * qs, qv0.y * qs, qv0.z * qs, qv0.w * qs);
    float m = -INFINITY, ls = 0.0f;
    float4 acc = make_float4(0, 0, 0, 0);
    for (int j = j0 + wv; j <= j1; j += 4) {
        const size_t row = (size_t) (j % cap) * Hkv + kvh;
        const float4 kk = kv_load(Kc, row, lane, 2), vv = kv_load(Vc, row, lane, 2);
        const float sc = wsum(qv.x * kk.x + qv.y * kk.y + qv.z * kk.z + qv.w * kk.w);
        const float mn = fmaxf(m, sc), f = expf(m - mn), p = expf(sc - mn);
        ls = ls * f + p;
        acc = make_float4(acc.x * f + p * vv.x, acc.y * f + p * vv.y, acc.z * f + p * vv.z, acc.w * f + p * vv.w);
        m = mn;
    }
    __shared__ float sm[4], sl[4];
    __shared__ float4 sa[4][WS];
    if (lane == 0) { sm[wv] = m; sl[wv] = ls; }
    sa[wv][lane] = acc;
    __syncthreads();
    if (wv == 0) {
        float M = -INFINITY;
        for (int w = 0; w < 4; ++w) M = fmaxf(M, sm[w]);
        float L = 0.0f;
        float4 A = make_float4(0, 0, 0, 0);
        for (int w = 0; w < 4; ++w) {
            const float f = sm[w] == -INFINITY ? 0.0f : expf(sm[w] - M);
            L += sl[w] * f;
            const float4 x = sa[w][lane];
            A = make_float4(A.x + x.x * f, A.y + x.y * f, A.z + x.z * f, A.w + x.w * f);
        }
        float * o = part + (((size_t) t * H + h) * nsplit + sp) * (D + 2);
        if (lane == 0) { o[0] = M; o[1] = L; }
        *(float4 *) (o + 2 + lane * 4) = A;
    }
}
// merge the splits (no sink): grid (H, B), D threads
__global__ void k_dcomb(const float * part, int nsplit, float * out, int D) {
    const int h = blockIdx.x, H = gridDim.x, t = blockIdx.y, i = threadIdx.x;
    const float * ph = part + ((size_t) t * H + h) * nsplit * (D + 2);
    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, ph[(size_t) s * (D + 2)]);
    float L = 0.0f, A = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float ms = ph[(size_t) s * (D + 2)];
        if (ms == -INFINITY) continue;
        const float f = expf(ms - M);
        L += ph[(size_t) s * (D + 2) + 1] * f;
        A += ph[(size_t) s * (D + 2) + 2 + i] * f;
    }
    out[((size_t) t * H + h) * D + i] = A / L;
}
__global__ void k_silu_mul(const float * g, const float * u, float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float a = g[i]; y[i] = a / (1.0f + expf(-a)) * u[i]; }
}
// block input: row 0 the anchor's embedding (normed by the caller), rows 1.. the mask rows
__global__ void k_dmask(const float * mask, float * x, int C, int B) {
    const int t = blockIdx.y + 1, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < B && i < C) x[(size_t) t * C + i] = mask[(size_t) (t - 1) * C + i];
}
// Markov step input [state, w1[prev], hidden_i, f_rnn] (5,120 values), w1 BF16 rows of R
__global__ void k_mk_joint(const float * state, const uint16_t * w1, const int * prev, const float * hid,
                           const float * frnn, float * joint, int R, int C) {
    const uint16_t * wr = w1 + (size_t) *prev * R;
    for (int i = threadIdx.x; i < 2 * R + 2 * C; i += blockDim.x) {
        float v;
        if (i < R) v = state[i];
        else if (i < 2 * R) v = __uint_as_float((uint32_t) wr[i - R] << 16);
        else if (i < 2 * R + C) v = hid[i - 2 * R];
        else v = frnn[i - 2 * R - C];
        joint[i] = v;
    }
}
// gate = sigmoid, cand = tanh, state = cand + gate * (state - cand), markov = tanh (the three chunks of jo + bias)
__global__ void k_mk_upd(const float * jo, const float * jb, float * state, float * mk, int R) {
    const int i = threadIdx.x;
    if (i >= R) return;
    const float g = 1.0f / (1.0f + expf(-(jo[i] + jb[i]))), c = tanhf(jo[R + i] + jb[R + i]);
    state[i] = c + g * (state[i] - c);
    mk[i] = tanhf(jo[2 * R + i] + jb[2 * R + i]);
}
// the draft token: argmax of base + bias + b2 (the softcap after the bias does not move the argmax; ties: the lower
// id, as ggml_argmax). One block of 1024 threads
__global__ void k_mk_pick(const float * base, const float * bias, const float * b2, int V, int * out, int * prev) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float x = base[i] + bias[i] + b2[i];
        if (x > v) { v = x; idx = i; }
    }
    for (int o = WS / 2; o > 0; o >>= 1) {
        const float v2 = __shfl_xor(v, o);
        const int i2 = __shfl_xor(idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if (threadIdx.x % WS == 0) { bv[threadIdx.x / WS] = v; bi[threadIdx.x / WS] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) blockDim.x / WS; ++w)
            if (bv[w] > v || (bv[w] == v && bi[w] < idx)) { v = bv[w]; idx = bi[w]; }
        *out = idx;
        *prev = idx;
    }
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
    void *K = nullptr, *V = nullptr;
    float * hist = nullptr;   // hist: [3][Ck], [3][Ck], [3][C], [3][C] for k, v, attn, ffn
    int cap = 0;
    size_t kvbytes = 0;   // each of K and V
    bool far = false;     // --kv-gpu: a full attention slot whose K/V and attention live on the second card
    float * spec = nullptr;   // speculative verification: the convolutions' state before the batch and its inputs
};

struct Runner {
    int TM = 1024;    // the largest prompt chunk
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
    float * dimg = nullptr; int * dimgm = nullptr; bool img_on = false;   // a chunk's picture rows (GENI)
    std::vector<const float **> marr; // one device array of state pointers per mix call of a token
    size_t mix_i = 0;
    bool eager_done = false;
    hipGraphExec_t gexec = nullptr;
    int nsplit_max = 64;
    // --kv-gpu N (opt-in, long contexts): the full attention slots' K/V on card N, their attention computed there;
    // q, the gate, the relative bias and the sinks cross over, the gated output comes back
    int dev0 = 0, kvdev = -1;
    hipStream_t st1 = nullptr;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    float *q1 = nullptr, *g1 = nullptr, *relb1 = nullptr, *sinkv1 = nullptr, *att1 = nullptr, *part1 = nullptr;
    int * dpos1 = nullptr;
    int nsplit_far = 256;   // decode splits over a far slot (its keys spread over more blocks)

    float * dalloc(size_t n) { float * p; CK(hipMalloc(&p, n * 4)); CK(hipMemset(p, 0, n * 4)); return p; }

    int kvt = 0;   // the KV cache: 0 F32, 1 q8_0, 2 F16
    // speculative decoding (DSpark): verification batches record the convolutions' inputs (spec_rec) and take the
    // decode attention (vmode); the target features of the last body() (feat: the inputs of visits fvisit[], then
    // the final norm's output)
    bool spec_rec = false, vmode = false;
    int fvisit[3] = {-1, -1, -1};
    const float * feat[4] = {nullptr, nullptr, nullptr, nullptr};
    std::map<int, hipGraphExec_t> vgraph;
    std::map<int, bool> vwarm;
    float * hlogT = nullptr;               // pinned, SPEC_T rows of logits
    float **dhists = nullptr, **dspecs = nullptr;
    void enable_spec() {
        const int Ck = m.Hkv * m.D;
        std::vector<float *> hs, ss;
        for (auto & sl : slots) {
            sl.spec = dalloc((size_t) (3 + SPEC_T) * (2 * Ck + 2 * m.C));
            hs.push_back(sl.hist); ss.push_back(sl.spec);
        }
        CK(hipMalloc(&dhists, hs.size() * sizeof(float *)));
        CK(hipMalloc(&dspecs, ss.size() * sizeof(float *)));
        CK(hipMemcpy(dhists, hs.data(), hs.size() * sizeof(float *), hipMemcpyHostToDevice));
        CK(hipMemcpy(dspecs, ss.data(), ss.size() * sizeof(float *), hipMemcpyHostToDevice));
        CK(hipHostMalloc(&hlogT, (size_t) SPEC_T * m.vocab * 4));
    }
    // feed T <= SPEC_T tokens and keep every one's logits (in hlogT); the batch can then be cut back with rollback()
    const float * verify(const int * toks, int T) {
        hpos[0] = pos;
        std::memcpy(hpos + 1, toks, (size_t) T * 4);
        CK(hipMemcpyAsync(dpos, hpos, 4, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(dtok, hpos + 1, (size_t) T * 4, hipMemcpyHostToDevice, st));
        spec_rec = vmode = true;
        if (!vwarm[T] || (kvdev >= 0 && !far_graph)) { body(T, true); vwarm[T] = true; }
        else {
            hipGraphExec_t & ge = vgraph[T];
            if (!ge) {
                hipGraph_t gr;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
                body(T, true);
                CK(hipStreamEndCapture(st, &gr));
                CK(hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0));
            }
            CK(hipGraphLaunch(ge, st));
        }
        spec_rec = vmode = false;
        pos += T;
        for (int i = 0; i < T; ++i)
            CK(hipMemcpyAsync(hlogT + (size_t) i * m.vocab, logits + (size_t) i * m.vocab_pad, (size_t) m.vocab * 4,
                              hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        return hlogT;
    }
    // keep the first `keep` tokens of the last verify(T)
    void rollback(int T, int keep) {
        if (keep == T) return;
        hipLaunchKernelGGL(k_spec_roll, dim3(ew(2 * m.Hkv * m.D + 2 * m.C), (int) slots.size()), dim3(256), 0, st,
                           dhists, dspecs, keep, m.Hkv * m.D, m.C);
        pos -= T - keep;
    }
    bool all_cap = false;   // room for every token's logits (scoring)
    bool far_graph = std::getenv("INK_FAR_GRAPH") != nullptr;   // capture decode graphs across the two cards
    Runner(Model & m_, int ctx_, int kvt_ = 0, int tm_ = 1024, bool all_cap_ = false, int kvdev_ = -1)
        : m(m_), ctx(ctx_), kvt(kvt_), TM(std::max(tm_, SPEC_T)), all_cap(all_cap_) {
        const int C = m.C, Ck = m.Hkv * m.D;
        const size_t T = TM;
        CK(hipGetDevice(&dev0));
        if (kvdev_ >= 0 && kvdev_ != dev0) {
            kvdev = kvdev_;
            int a = 0, b = 0;
            CK(hipDeviceCanAccessPeer(&a, dev0, kvdev)); CK(hipDeviceCanAccessPeer(&b, kvdev, dev0));
            if (!a || !b) { std::fprintf(stderr, "almond-inkling-native: cards %d and %d cannot reach each other\n", dev0, kvdev); std::exit(1); }
            hipError_t e = hipDeviceEnablePeerAccess(kvdev, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CK(e);
            CK(hipSetDevice(kvdev));
            e = hipDeviceEnablePeerAccess(dev0, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CK(e);
            CK(hipStreamCreateWithFlags(&st1, hipStreamNonBlocking));
            CK(hipEventCreateWithFlags(&ev1, hipEventDisableTiming));
            q1 = dalloc(T * m.H * m.D); g1 = dalloc(T * m.H * m.D); att1 = dalloc(T * m.H * m.D);
            relb1 = dalloc(T * m.H * m.ext); sinkv1 = dalloc(T * m.H);
            part1 = dalloc((size_t) std::max<size_t>(nsplit_far, 16 * 4096 / m.Hkv) * m.H * (m.D + 2));
            CK(hipMalloc(&dpos1, 4));
            CK(hipSetDevice(dev0));
            CK(hipEventCreateWithFlags(&ev0, hipEventDisableTiming));
        }
        const int n_slots = m.n_phys + (m.passes - 1) * (m.core1 - m.core0);
        slots.resize(n_slots);
        for (int s = 0; s < n_slots; ++s) {
            const int ip = s < m.n_phys ? s : m.core0 + (s - m.n_phys) % (m.core1 - m.core0);
            Slot & sl = slots[s];
            // a window layer keeps its window plus a chunk: a chunk's first token still sees its whole window
            sl.cap = m.L[ip].swa ? std::min(ctx, m.swa + TM) : ctx;
            sl.far = kvdev >= 0 && !m.L[ip].swa;
            const size_t kvb = (size_t) sl.cap * m.Hkv * (kvt == 1 ? KVQ8_ROW : kvt == 2 ? m.D * 2 : m.D * 4);
            if (sl.far) CK(hipSetDevice(kvdev));
            CK(hipMalloc(&sl.K, kvb)); CK(hipMemset(sl.K, 0, kvb));
            CK(hipMalloc(&sl.V, kvb)); CK(hipMemset(sl.V, 0, kvb));
            if (sl.far) { CK(hipDeviceSynchronize()); CK(hipSetDevice(dev0)); }
            sl.kvbytes = kvb;
            sl.hist = dalloc((size_t) 3 * (2 * Ck + 2 * C));
        }
        x = dalloc(T * C); t0 = dalloc(T * C); t1 = dalloc(T * C); t2 = dalloc(T * C);
        for (auto & p : inp) p = dalloc(T * C);
        q = dalloc(T * m.H * m.D); k = dalloc(T * Ck); v = dalloc(T * Ck); r = dalloc(T * m.H * m.nrel);
        sinkv = dalloc(T * m.H); att = dalloc(T * m.H * m.D); g = dalloc(T * m.H * m.D);
        dk = dalloc(T * 4 * m.Hkv); dv = dalloc(T * 4 * m.Hkv);
        scr = dalloc(T * 3 * C); logits = dalloc((all_cap ? std::max<size_t>(T, SPEC_T) : SPEC_T) * m.vocab_pad); msgs = dalloc(T * 4 * C);
        ctrl_g = dalloc(T * 4 * m.rank); ctrl_a = dalloc(T * 4 * C); ctrl_b = dalloc(T * 4 * C); cat = dalloc(T * 8 * C);
        gu = dalloc(T * m.k_exp * 2 * m.F); hh = dalloc(T * m.k_exp * m.F); wts = dalloc(T * m.k_exp);
        yd = dalloc(T * m.k_exp * C);
        rl = dalloc(T * m.n_exp);
        relb = dalloc(T * m.H * m.ext);
        part = dalloc((size_t) std::max<size_t>(nsplit_max, 16 * 4096 / m.Hkv) * m.H * (m.D + 2));   // T x nsplit <= 16 x 4096 / Hkv
        CK(hipMalloc(&ids, T * m.k_exp * 4));
        CK(hipMalloc(&offs, (m.n_exp + 2 + 2 * ((T * m.k_exp + EXP_BT - 1) / EXP_BT + m.n_exp)) * 4));
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
        if (T <= MMV_T) {
            Segs S{};
            S.n = 0; S.start[0] = 0;
            for (const auto & e : list) {
                const Mat & a = *std::get<0>(e);
                S.s[S.n] = Seg{a.w, std::get<1>(e), std::get<2>(e), a.rb, a.N, a.K, a.type, std::get<3>(e), 0};
                S.start[S.n + 1] = S.start[S.n] + a.N;
                ++S.n;
            }
            hipLaunchKernelGGL(k_mmvT, dim3((S.start[S.n] + 7) / 8), dim3(256), 0, st, S, T);
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
        hipLaunchKernelGGL(k_gemm_w, dim3(S.tstart[S.n], (T + 63) / 64), dim3(256), 0, st, S, T);
    }
    // the F32 GEMM of several segments; a few tokens go through k_mmvT
    void gemm_f32(const GSegs & G, int T) {
        if (T <= MMV_T) {
            Segs S{};
            S.n = G.n; S.start[0] = 0;
            for (int i = 0; i < G.n; ++i) { S.s[i] = G.s[i]; S.start[i + 1] = S.start[i] + G.s[i].N; }
            hipLaunchKernelGGL(k_mmvT, dim3((S.start[S.n] + 7) / 8), dim3(256), 0, st, S, T);
            return;
        }
        hipLaunchKernelGGL(k_gemm, dim3(G.tstart[G.n], (T + 31) / 32), dim3(256), 0, st, G, T);
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
                  int mstride, float * const * outs, const float * const * nw) {
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
            gemm_f32(S, T);
            S.n = 0;
            for (int j = 0; j < n; ++j) {
                add(l.cgain[c0 + j], ctrl_g + (size_t) j * m.rank, ctrl_a + (size_t) j * C, 4 * m.rank, 4 * C);
                add(l.cforget[c0 + j], ctrl_g + (size_t) j * m.rank, ctrl_b + (size_t) j * C, 4 * m.rank, 4 * C);
            }
            gemm_f32(S, T);
        }
        Norms W{};
        for (int j = 0; j < n; ++j) W.w[j] = nw[j];
        hipLaunchKernelGGL(k_ctrl_apply_n, dim3(n, T), dim3(256), 0, st, P, C, W, m.eps);   // with the norms
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
            const float * nw[3];
            for (int j = 0; j < 3; ++j) nw[j] = l.swa ? l.attn_norm : (j == 0 ? l.qn : j == 1 ? l.kn : l.vn);
            controls(T, l, 0, 3, xs, l.n_res_q * C, ms, 4 * C, inp, nw);
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
        // one token: the convolution goes into k_qkvr; the relative bias always does
        QFuse fz{};
        fz.table = l.rel_bias; fz.ext = m.ext; fz.sink = sinkv; fz.bias = relb;
        if (T == 1) {
            fz.conv = 1;
            fz.cw[0] = l.kconv; fz.cw[1] = l.vconv; fz.hist[0] = sl.hist; fz.hist[1] = sl.hist + 3 * Ck;
            fz.dyn[0] = l.swa ? nullptr : dk; fz.dyn[1] = l.swa ? nullptr : dv; fz.cscale = cs;
        } else
            hipLaunchKernelGGL(k_conv2, dim3(ew(Ck), 2), dim3(256), 0, st,
                               Conv2{{k, v}, {l.kconv, l.vconv}, {sl.hist, sl.hist + 3 * Ck},
                                     {l.swa ? nullptr : dk, l.swa ? nullptr : dv},
                                     {spec_rec ? sl.spec : nullptr, spec_rec ? sl.spec + (3 + SPEC_T) * Ck : nullptr}},
                               Ck, m.D, cs, T, 4 * m.Hkv);
        hipLaunchKernelGGL(k_qkvr, dim3(2 * m.H + 2 * m.Hkv, T), dim3(128), 0, st, q, k, v, r, l.q_clip, l.rel_clip, m.H,
                           m.Hkv, m.D, m.nrel, m.qk_clip, m.rel_clip, dpos, l.swa ? 0.0f : m.alpha, m.floor_, m.eps,
                           sl.K, sl.V, sl.cap, kvt, fz);
        // the attention: here, or on the second card for a far slot (its inputs copied over first)
        const float *aq = q, *ag = g, *arb = relb, *ask = sinkv;
        const int * adp = dpos;
        float *apt = part, *aat = att;
        hipStream_t ast = st;
        if (sl.far) {
            CK(hipMemcpyAsync(q1, q, (size_t) T * m.H * m.D * 4, hipMemcpyDeviceToDevice, st));
            CK(hipMemcpyAsync(g1, g, (size_t) T * m.H * m.D * 4, hipMemcpyDeviceToDevice, st));
            CK(hipMemcpyAsync(relb1, relb, (size_t) T * m.H * m.ext * 4, hipMemcpyDeviceToDevice, st));
            CK(hipMemcpyAsync(sinkv1, sinkv, (size_t) T * m.H * 4, hipMemcpyDeviceToDevice, st));
            CK(hipMemcpyAsync(dpos1, dpos, 4, hipMemcpyDeviceToDevice, st));
            CK(hipEventRecord(ev0, st));
            CK(hipSetDevice(kvdev));
            CK(hipStreamWaitEvent(st1, ev0, 0));
            aq = q1; ag = g1; arb = relb1; ask = sinkv1; adp = dpos1; apt = part1; aat = att1; ast = st1;
        }
        const int span = l.swa ? std::min(m.swa, ctx) : ctx;
        int nsplit;
        if (T == 1 || vmode) {
            const int splitsz = 32;   // the least a split; the kernel spreads the keys present over nsplit
            nsplit = std::min(sl.far ? nsplit_far : nsplit_max, (span + splitsz - 1) / splitsz);
            hipLaunchKernelGGL(k_attn_split<3>, dim3(m.Hkv, nsplit, T), dim3(128), 0, ast, aq, sl.K, sl.V, sl.cap, adp,
                               span, m.Hkv, arb, m.ext, ask, apt, splitsz, nsplit, kvt);
        } else {
            // the keys a 16-token tile can see: at most span + 15; split so the card has enough blocks
            const int nk = std::min(span + 15, pos + T), tiles = (T + 15) / 16;
            nsplit = std::min(nsplit_max, std::max(1, (nk + 511) / 512));
            while (nsplit > 1 && m.Hkv * tiles * nsplit > 4096) --nsplit;
            const int splitsz = ((nk + nsplit - 1) / nsplit + 31) / 32 * 32;
            nsplit = (nk + splitsz - 1) / splitsz;
            if (kvt == 2) {   // F16 cache: the matrix cores, 16 (token, head) rows a block
                const int sz16 = (splitsz + 15) / 16 * 16;
                const int ns16 = (nk + sz16 - 1) / sz16;
                hipLaunchKernelGGL(k_attn_pf_w, dim3(m.Hkv, (3 * T + 15) / 16, ns16), dim3(WS), 0, ast, aq, sl.K, sl.V,
                                   sl.cap, adp, span, m.Hkv, arb, m.ext, ask, apt, sz16, ns16, T);
                nsplit = ns16;
            } else
                hipLaunchKernelGGL(k_attn_pf, dim3(m.Hkv, tiles, nsplit), dim3(256), 0, ast, aq, sl.K, sl.V, sl.cap, adp,
                                   span, m.Hkv, arb, m.ext, ask, apt, splitsz, nsplit, T, kvt);
        }
        hipLaunchKernelGGL(k_attn_combine, dim3(m.H, T), dim3(m.D), 0, ast, apt, nsplit, ag, aat, m.D);
        if (sl.far) {
            CK(hipEventRecord(ev1, st1));
            CK(hipSetDevice(dev0));
            CK(hipStreamWaitEvent(st, ev1, 0));
            CK(hipMemcpyAsync(att, att1, (size_t) T * m.H * m.D * 4, hipMemcpyDeviceToDevice, st));
        }
        mmv(T, {MV{&l.wo, att, t0, 0}});
        float * h1 = pool[pool_i++];
        const bool starts = ip % m.block == 0;
        if (starts) blocks.push_back(inpSA);
        hipLaunchKernelGGL(k_conv_res, dim3(ew(C)), dim3(256), 0, st, t0, l.post_conv, sl.hist + 6 * Ck, C,
                           starts ? (const float *) nullptr : inpSA, h1, 1.0f / sqrtf(5.0f), T,
                           spec_rec ? sl.spec + 2 * (3 + SPEC_T) * Ck : nullptr);
        // MoE
        if (!messages) mix(T, blocks, h1, l.ffn_res_q, 1, t1, NormW{{l.ffn_norm, nullptr, nullptr, nullptr}});
        else {
            mix(T, blocks, h1, l.ffn_res_q, 1, t2);
            const float * xs[1] = {t2}, * ms[1] = {messages + 3 * (size_t) C};
            float * os[1] = {t1};
            const float * nw[1] = {l.ffn_norm};
            controls(T, l, 3, 1, xs, C, ms, 4 * C, os, nw);
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
        if (T <= MMV_T) {   // one token, or a few (each entry its own expert row reads, as the decode)
            if (l.gu_type == T_NVFP4 && C % 64 == 0)   // 4 rows a wave, each x load used for all four
                hipLaunchKernelGGL(k_exp_gu_nv<4>, dim3((2 * m.F + 31) / 32, m.k_exp * T), dim3(256), 0, st, l.gu,
                                   l.gu_ebytes, l.gu_rb, l.gu_scale, ids, t1, gu, 2 * m.F, C, m.k_exp);
            else
            hipLaunchKernelGGL(k_exp_gu, dim3((2 * m.F + 7) / 8, m.k_exp * T), dim3(256), 0, st, l.gu, l.gu_type,
                               l.gu_ebytes, l.gu_rb, l.gu_scale, ids, t1, gu, 2 * m.F, C, m.k_exp);
            hipLaunchKernelGGL(k_situ, dim3(ew(m.F * m.k_exp * T)), dim3(256), 0, st, gu, hh, m.F, m.k_exp * T, m.situ_b,
                               m.situ_lb);
            hipLaunchKernelGGL(k_exp_down, dim3((C + 7) / 8, T), dim3(256), 0, st, l.down, l.down_type, l.down_ebytes,
                               l.down_rb, l.down_scale, ids, wts, hh, t2, C, m.F, m.k_exp);
        } else {
            const int ne = T * m.k_exp;
            hipLaunchKernelGGL(k_group, dim3(1), dim3(1024), 0, st, ids, ne, m.n_exp, offs, list);
            hipLaunchKernelGGL(k_gemm_exp_w, dim3((2 * m.F + 63) / 64, (T * m.k_exp + EXP_BT - 1) / EXP_BT + m.n_exp), dim3(128), 0, st, l.gu,
                               l.gu_type, l.gu_ebytes, l.gu_rb, 2 * m.F, C, t1, gu, offs, list, m.k_exp, l.gu_scale,
                               (const float *) nullptr, m.n_exp);
            hipLaunchKernelGGL(k_situ, dim3(ew(m.F * ne)), dim3(256), 0, st, gu, hh, m.F, ne, m.situ_b, m.situ_lb);
            hipLaunchKernelGGL(k_gemm_exp_w, dim3((C + 63) / 64, (T * m.k_exp + EXP_BT - 1) / EXP_BT + m.n_exp), dim3(128), 0, st, l.down,
                               l.down_type, l.down_ebytes, l.down_rb, C, m.F, hh, yd, offs, list, 1, l.down_scale, wts, m.n_exp);
            hipLaunchKernelGGL(k_exp_sum, dim3(ew(C), T), dim3(256), 0, st, yd, t2, C, m.k_exp);
        }
        hipLaunchKernelGGL(k_conv_res, dim3(ew(C)), dim3(256), 0, st, t2, l.ffn_conv, sl.hist + 6 * Ck + 3 * C, C,
                           h1, h2, 1.0f / sqrtf(5.0f), T,
                           spec_rec ? sl.spec + 2 * (3 + SPEC_T) * Ck + (3 + SPEC_T) * C : nullptr);
        cur = h2;
    }

    // ---- the state that cannot be cut back (serve): the convolution histories and the window layers' rings. A full
    // attention layer's rows below n are never written again, so a checkpoint at n needs none of them.
    size_t hist_bytes() const { return (size_t) 3 * (2 * m.Hkv * m.D + 2 * m.C) * 4; }
    bool ring(const Slot & sl) const { return sl.cap < ctx; }
    void clear() {
        pos = 0;
        for (auto & sl : slots) CK(hipMemsetAsync(sl.hist, 0, hist_bytes(), st));
        CK(hipStreamSynchronize(st));
    }
    struct Ckpt { int n; std::vector<uint8_t> data; };
    void save_state(std::vector<uint8_t> & out) {
        size_t tot = 0;
        for (auto & sl : slots) tot += hist_bytes() + (ring(sl) ? 2 * sl.kvbytes : 0);
        out.resize(tot);
        uint8_t * p = out.data();
        for (auto & sl : slots) {
            CK(hipMemcpyAsync(p, sl.hist, hist_bytes(), hipMemcpyDeviceToHost, st)); p += hist_bytes();
            if (ring(sl)) {
                CK(hipMemcpyAsync(p, sl.K, sl.kvbytes, hipMemcpyDeviceToHost, st)); p += sl.kvbytes;
                CK(hipMemcpyAsync(p, sl.V, sl.kvbytes, hipMemcpyDeviceToHost, st)); p += sl.kvbytes;
            }
        }
        CK(hipStreamSynchronize(st));
    }
    void load_state(const std::vector<uint8_t> & in, int n) {
        const uint8_t * p = in.data();
        for (auto & sl : slots) {
            CK(hipMemcpyAsync(sl.hist, p, hist_bytes(), hipMemcpyHostToDevice, st)); p += hist_bytes();
            if (ring(sl)) {
                CK(hipMemcpyAsync(sl.K, p, sl.kvbytes, hipMemcpyHostToDevice, st)); p += sl.kvbytes;
                CK(hipMemcpyAsync(sl.V, p, sl.kvbytes, hipMemcpyHostToDevice, st)); p += sl.kvbytes;
            }
        }
        CK(hipStreamSynchronize(st));
        pos = n;
    }
    // the whole cache to a file (SAVE): the state above, then the full layers' rows below pos
    bool save_file(const std::string & path, const std::vector<int> & toks, size_t & bytes) {
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        std::vector<uint8_t> st_;
        save_state(st_);
        const uint32_t magic = 0x4B4E4931, n = (uint32_t) toks.size(), kt = (uint32_t) kvt;
        bool ok = std::fwrite(&magic, 4, 1, f) == 1 && std::fwrite(&n, 4, 1, f) == 1 && std::fwrite(&kt, 4, 1, f) == 1 &&
                  std::fwrite(toks.data(), 4, n, f) == n && std::fwrite(st_.data(), 1, st_.size(), f) == st_.size();
        std::vector<uint8_t> buf;
        for (auto & sl : slots) {
            if (ring(sl) || !ok) continue;
            const size_t b = sl.kvbytes / (size_t) sl.cap * n;
            buf.resize(b);
            for (void * src : {sl.K, sl.V}) {
                CK(hipMemcpy(buf.data(), src, b, hipMemcpyDeviceToHost));
                ok = ok && std::fwrite(buf.data(), 1, b, f) == b;
            }
        }
        bytes = (size_t) std::ftell(f);
        return std::fclose(f) == 0 && ok;
    }
    bool load_file(const std::string & path, std::vector<int> & toks, size_t & bytes) {
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        uint32_t magic = 0, n = 0, kt = 0;
        bool ok = std::fread(&magic, 4, 1, f) == 1 && std::fread(&n, 4, 1, f) == 1 && std::fread(&kt, 4, 1, f) == 1 &&
                  magic == 0x4B4E4931 && kt == (uint32_t) kvt && (int) n < ctx;
        if (ok) {
            toks.resize(n);
            ok = std::fread(toks.data(), 4, n, f) == n;
        }
        std::vector<uint8_t> st_;
        if (ok) {
            size_t tot = 0;
            for (auto & sl : slots) tot += hist_bytes() + (ring(sl) ? 2 * sl.kvbytes : 0);
            st_.resize(tot);
            ok = std::fread(st_.data(), 1, tot, f) == tot;
        }
        std::vector<uint8_t> buf;
        for (auto & sl : slots) {
            if (ring(sl) || !ok) continue;
            const size_t b = sl.kvbytes / (size_t) sl.cap * n;
            buf.resize(b);
            for (void * dst : {sl.K, sl.V}) {
                ok = ok && std::fread(buf.data(), 1, b, f) == b;
                if (ok) CK(hipMemcpy(dst, buf.data(), b, hipMemcpyHostToDevice));
            }
        }
        bytes = (size_t) std::ftell(f);
        std::fclose(f);
        if (ok) load_state(st_, (int) n);
        return ok;
    }
    // the last token's logits into pinned host memory (serve): one copy, no staging
    float * hlog = nullptr;
    const float * fetch_logits() {
        if (!hlog) CK(hipHostMalloc(&hlog, (size_t) m.vocab * 4));
        CK(hipMemcpyAsync(hlog, logits, (size_t) m.vocab * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        return hlog;
    }
    // feed one token: the first one runs eagerly (it writes the mix pointer sets), the second is captured as a graph,
    // every later one replays it
    void step(int tok, std::vector<float> * out) {
        hpos[0] = pos;
        hpos[1] = tok;
        CK(hipMemcpyAsync(dpos, hpos, 4, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(dtok, hpos + 1, 4, hipMemcpyHostToDevice, st));
        if (!eager_done || (kvdev >= 0 && !far_graph)) {
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
    // ---- DSpark drafter (opt-in, --dspark FILE)
    struct DLayer {
        Mat wq, wk, wv, wo, wk_ctx, wv_ctx, gate, up, down;
        const float *attn_norm, *qn, *kn, *kn_ctx, *ctx_mix, *ctx_norm, *ffn_norm;
        __half *Kc = nullptr, *Vc = nullptr;
        int cap = 0, win = 0;   // win: the sliding window (0: full)
    };
    struct Draft {
        Gguf g;
        int B = 8, n_layer = 3, R = 256, F = 9216, Hd = 24, Hkvd = 8;
        float theta = 1e6f, eps = 1e-6f;
        std::vector<DLayer> L;
        const float *tok_norm, *mask, *out_norm, *mk_mix, *mk_norm, *jb, *b2;
        const uint16_t * w1;
        Mat jp, w2;
        // buffers
        float *mixo[4], *kc, *vc, *x, *h, *q, *k, *v, *att, *t0, *gg, *uu, *hn, *dlog, *part, *state, *joint, *jo, *mk,
            *bias;
        int *dn0, *dtoks, *dprev, *hn0;
        const float * frnn = nullptr;   // the Markov context state of the last injected token
        int frnn_pos = -1;
        int TI = 0;                     // the largest injection
    };
    Draft * dr = nullptr;
    Mat dmat(Gguf & g, const std::string & n) {
        Tensor & t = g.need(n);
        return Mat{(const uint8_t *) t.dev, t.type, (int) t.ne[1], (int) t.ne[0], row_bytes(t.type, t.ne[0])};
    }
    void dspark_load(const std::string & path) {
        dr = new Draft;
        Draft & d = *dr;
        d.g.open(path);
        Gguf & g = d.g;
        const std::string a = "dflash.";
        d.n_layer = (int) g.get(a + "block_count");
        d.B = (int) g.get(a + "block_size");
        d.F = (int) g.get(a + "feed_forward_length");
        d.Hd = (int) g.get(a + "attention.head_count");
        d.Hkvd = (int) g.get(a + "attention.head_count_kv");
        d.theta = (float) g.get(a + "rope.freq_base");
        d.eps = (float) g.get(a + "attention.layer_norm_rms_epsilon");
        const int win = (int) g.get(a + "attention.sliding_window");
        const auto & pat = g.arr[a + "attention.sliding_window_pattern"];
        const auto & tl = g.arr[a + "target_layers"];
        for (int k = 0; k < 3; ++k) fvisit[k] = (int) tl[k];   // the 4th is the final norm's output
        if (d.B > SPEC_T || d.Hd != m.H || d.Hkvd != m.Hkv || tl.size() != 4 || (int) tl[3] != m.n_phys + (m.passes - 1) * (m.core1 - m.core0)) {
            std::fprintf(stderr, "dspark: unexpected drafter shape\n");
            std::exit(1);
        }
        auto f = [&](const std::string & n) { return (const float *) g.need(n).dev; };
        d.tok_norm = f("token_embd_norm.weight");
        d.mask = f("mask_embd.weight");
        d.out_norm = f("output_norm.weight");
        d.mk_mix = f("markov_ctx_mix.weight");
        d.mk_norm = f("markov_ctx_norm.weight");
        d.jb = f("joint_proj.bias");
        d.b2 = f("markov_w2.bias");
        d.w1 = (const uint16_t *) g.need("markov_w1.weight").dev;
        d.R = (int) g.find("markov_w1.weight")->ne[0];
        d.jp = dmat(g, "joint_proj.weight");
        d.w2 = dmat(g, "markov_w2.weight");
        const int Ck = m.Hkv * m.D;
        for (int il = 0; il < d.n_layer; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            DLayer l;
            l.wq = dmat(g, b + "attn_q.weight"); l.wk = dmat(g, b + "attn_k.weight"); l.wv = dmat(g, b + "attn_v.weight");
            l.wo = dmat(g, b + "attn_output.weight");
            l.wk_ctx = dmat(g, b + "attn_k_ctx.weight"); l.wv_ctx = dmat(g, b + "attn_v_ctx.weight");
            l.gate = dmat(g, b + "ffn_gate.weight"); l.up = dmat(g, b + "ffn_up.weight"); l.down = dmat(g, b + "ffn_down.weight");
            l.attn_norm = f(b + "attn_norm.weight"); l.qn = f(b + "attn_q_norm.weight"); l.kn = f(b + "attn_k_norm.weight");
            l.kn_ctx = f(b + "attn_k_ctx_norm.weight"); l.ctx_mix = f(b + "ctx_mix.weight"); l.ctx_norm = f(b + "ctx_norm.weight");
            l.ffn_norm = f(b + "ffn_norm.weight");
            l.win = il < (int) pat.size() && pat[il] != 0 ? win : 0;
            // a window layer keeps a ring: the window, the block and an injected chunk
            l.cap = l.win ? win + d.B + TM : ctx + d.B;
            const size_t kvb = (size_t) l.cap * Ck * 2;
            CK(hipMalloc(&l.Kc, kvb)); CK(hipMemset(l.Kc, 0, kvb));
            CK(hipMalloc(&l.Vc, kvb)); CK(hipMemset(l.Vc, 0, kvb));
            d.L.push_back(l);
        }
        d.TI = TM;
        const size_t T = (size_t) std::max(TM, d.B);
        for (auto & o : d.mixo) o = dalloc(T * m.C);
        d.kc = dalloc(T * 3 * Ck); d.vc = dalloc(T * 3 * Ck);
        d.x = dalloc(d.B * m.C); d.h = dalloc(d.B * m.C); d.q = dalloc(d.B * m.H * m.D); d.k = dalloc(d.B * Ck);
        d.v = dalloc(d.B * Ck); d.att = dalloc(d.B * m.H * m.D); d.t0 = dalloc(d.B * m.C);
        d.gg = dalloc((size_t) d.B * d.F); d.uu = dalloc((size_t) d.B * d.F); d.hn = dalloc(d.B * m.C);
        d.dlog = dalloc((size_t) d.B * m.vocab_pad);
        d.part = dalloc((size_t) d.B * m.H * 64 * (m.D + 2));
        d.state = dalloc(d.R); d.joint = dalloc(2 * d.R + 2 * m.C); d.jo = dalloc(3 * d.R); d.mk = dalloc(d.R);
        d.bias = dalloc(m.vocab_pad);
        CK(hipMalloc(&d.dn0, 4)); CK(hipMalloc(&d.dtoks, (d.B + 1) * 4)); CK(hipMalloc(&d.dprev, 4)); CK(hipMalloc(&d.hn0, 4));
        enable_spec();
    }
    // after a body(T) that fed positions p0..p0 + T - 1: the first n rows' context K/V into the drafter's cache
    void dspark_inject(int n, int p0) {
        if (!dr || n <= 0) return;
        Draft & d = *dr;
        const int Ck = m.Hkv * m.D;
        for (int at = 0; at < n; at += d.TI) {
            const int T = std::min(d.TI, n - at);
            DMix P{};
            for (int k = 0; k < 4; ++k) P.f[k] = feat[k] + (size_t) at * m.C;
            for (int o = 0; o < 3; ++o) { P.mix[o] = d.L[o].ctx_mix; P.norm[o] = d.L[o].ctx_norm; P.out[o] = d.mixo[o]; }
            P.mix[3] = d.mk_mix; P.norm[3] = d.mk_norm; P.out[3] = d.mixo[3];
            hipLaunchKernelGGL(k_dmix, dim3(T, 4), dim3(256), 0, st, P, m.C, d.eps);
            for (int o = 0; o < 3; ++o)
                mmv(T, {MV{&d.L[o].wk_ctx, d.mixo[o], d.kc + (size_t) o * T * Ck, 0},
                        MV{&d.L[o].wv_ctx, d.mixo[o], d.vc + (size_t) o * T * Ck, 0}});
            DKV K{};
            for (int o = 0; o < 3; ++o) {
                K.k[o] = d.kc + (size_t) o * T * Ck; K.v[o] = d.vc + (size_t) o * T * Ck; K.kn[o] = d.L[o].kn_ctx;
                K.Kc[o] = d.L[o].Kc; K.Vc[o] = d.L[o].Vc; K.cap[o] = d.L[o].cap;
            }
            hipLaunchKernelGGL(k_dkv, dim3(m.Hkv, T, 3), dim3(128), 0, st, K, m.Hkv, p0 + at, d.theta, d.eps);
            d.frnn = d.mixo[3] + (size_t) (T - 1) * m.C;
            d.frnn_pos = p0 + at + T - 1;
        }
    }
    // greedy draft of nd <= B - 1 tokens after `anchor` at position n0 (the context must hold n0 - 1)
    bool dspark_draft(int anchor, int n0, int nd, int * out) {
        Draft & d = *dr;
        if (d.frnn_pos != n0 - 1 || n0 + d.B > ctx) return false;
        const int C = m.C, Ck = m.Hkv * m.D, B = d.B;
        hpos[0] = n0; hpos[1] = anchor;
        CK(hipMemcpyAsync(d.dn0, hpos, 4, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(d.dprev, hpos + 1, 4, hipMemcpyHostToDevice, st));
        hipLaunchKernelGGL(k_embed, dim3(1), dim3(256), 0, st, m.embd, d.dprev, d.x, C);
        hipLaunchKernelGGL(k_rms_n, dim3(1, 1), dim3(256), 0, st, Norms{{d.x}, {d.tok_norm}, {d.x}}, C, d.eps);
        hipLaunchKernelGGL(k_dmask, dim3(ew(C), B - 1), dim3(256), 0, st, d.mask, d.x, C, B);
        for (auto & l : d.L) {
            hipLaunchKernelGGL(k_rms_n, dim3(1, B), dim3(256), 0, st, Norms{{d.x}, {l.attn_norm}, {d.h}}, C, d.eps);
            mmv(B, {MV{&l.wq, d.h, d.q, 0}, MV{&l.wk, d.h, d.k, 0}, MV{&l.wv, d.h, d.v, 0}});
            hipLaunchKernelGGL(k_dqkv, dim3(m.H + m.Hkv, B), dim3(128), 0, st, d.q, d.k, d.v, l.qn, l.kn, m.H, m.Hkv,
                               d.dn0, d.theta, d.eps, l.Kc, l.Vc, l.cap);
            const int span = l.win ? l.win + B : n0 + B;
            const int ns = std::min(64, std::max(1, span / 256));
            hipLaunchKernelGGL(k_dattn, dim3(m.H, B, ns), dim3(128), 0, st, d.q, l.Kc, l.Vc, l.cap, d.dn0, B, l.win,
                               m.Hkv, d.part, ns);
            hipLaunchKernelGGL(k_dcomb, dim3(m.H, B), dim3(m.D), 0, st, d.part, ns, d.att, m.D);
            mmv(B, {MV{&l.wo, d.att, d.t0, 0}});
            hipLaunchKernelGGL(k_add, dim3(ew(B * C)), dim3(256), 0, st, d.x, d.t0, B * C);
            hipLaunchKernelGGL(k_rms_n, dim3(1, B), dim3(256), 0, st, Norms{{d.x}, {l.ffn_norm}, {d.h}}, C, d.eps);
            mmv(B, {MV{&l.gate, d.h, d.gg, 0}, MV{&l.up, d.h, d.uu, 0}});
            hipLaunchKernelGGL(k_silu_mul, dim3(ew(B * d.F)), dim3(256), 0, st, d.gg, d.uu, d.gg, B * d.F);
            mmv(B, {MV{&l.down, d.gg, d.t0, 0}});
            hipLaunchKernelGGL(k_add, dim3(ew(B * C)), dim3(256), 0, st, d.x, d.t0, B * C);
        }
        (void) Ck;
        hipLaunchKernelGGL(k_rms_n, dim3(1, B), dim3(256), 0, st, Norms{{d.x}, {d.out_norm}, {d.hn}}, C, d.eps);
        // the target's head on rows 1..nd (no softcap: the Markov bias comes first)
        mmv(nd, {MV{&m.head, d.hn + C, d.dlog, 0}});   // rows of vocab_pad (the head's N)
        CK(hipMemsetAsync(d.state, 0, d.R * 4, st));
        for (int i = 1; i <= nd; ++i) {
            hipLaunchKernelGGL(k_mk_joint, dim3(1), dim3(256), 0, st, d.state, d.w1, d.dprev, d.hn + (size_t) i * C,
                               d.frnn, d.joint, d.R, C);
            mmv(1, {MV{&d.jp, d.joint, d.jo, 0}});
            hipLaunchKernelGGL(k_mk_upd, dim3(1), dim3(d.R), 0, st, d.jo, d.jb, d.state, d.mk, d.R);
            mmv(1, {MV{&d.w2, d.mk, d.bias, 0}});
            hipLaunchKernelGGL(k_mk_pick, dim3(1), dim3(1024), 0, st, d.dlog + (size_t) (i - 1) * m.vocab_pad, d.bias,
                               d.b2, m.vocab, d.dtoks + i - 1, d.dprev);
        }
        CK(hipMemcpyAsync(hpos, d.dtoks, (size_t) nd * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        std::memcpy(out, hpos, (size_t) nd * 4);
        return true;
    }
    // feed up to TM tokens at once (a prompt chunk); all_logits: every token's logits (else only the last one's, in
    // logits[0]); out gets them when given
    // rows: per token, its picture row (C floats, host) or null for text
    void chunk(const int * toks, int T, bool all_logits, std::vector<float> * out, const float * const * rows = nullptr) {
        hpos[0] = pos;
        std::memcpy(hpos + 1, toks, (size_t) T * 4);
        CK(hipMemcpyAsync(dpos, hpos, 4, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(dtok, hpos + 1, (size_t) T * 4, hipMemcpyHostToDevice, st));
        img_on = false;
        if (rows) {
            if (!dimg) { CK(hipMalloc(&dimg, (size_t) TM * m.C * 4)); CK(hipMalloc(&dimgm, (size_t) TM * 4)); }
            std::vector<int> mk((size_t) T, 0);
            for (int t = 0; t < T; ++t)
                if (rows[t]) {
                    mk[(size_t) t] = 1;
                    CK(hipMemcpyAsync(dimg + (size_t) t * m.C, rows[t], (size_t) m.C * 4, hipMemcpyHostToDevice, st));
                    img_on = true;
                }
            CK(hipMemcpyAsync(dimgm, mk.data(), (size_t) T * 4, hipMemcpyHostToDevice, st));
            CK(hipStreamSynchronize(st));   // (mk is on the stack)
        }
        body(T, all_logits);
        img_on = false;
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
        if (img_on) hipLaunchKernelGGL(k_img_put, dim3(T), dim3(256), 0, st, x, dimg, dimgm, C);
        float * cur = x;
        std::vector<float *> blocks;
        int visit = 0;
        auto note = [&](const float * c) {
            for (int k = 0; k < 3; ++k) if (fvisit[k] == visit) feat[k] = c;
            ++visit;
        };
        for (int il = 0; il < m.core0; ++il) { note(cur); layer(T, il, il, cur, blocks, nullptr, pool_i); }
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
                note(cur);
                if (p == 0) { layer(T, slot, ip, cur, blocks, nullptr, pool_i); continue; }
                mix(T, bank, bank_out, m.L[ip].msg_q, 4, msgs);
                layer(T, slot, ip, cur, blocks, msgs, pool_i);
            }
        }
        for (int il = m.core1; il < m.n_phys; ++il) { note(cur); layer(T, il, il, cur, blocks, nullptr, pool_i); }
        mix(T, blocks, cur, m.out_res_q, 1, t1, NormW{{m.out_norm, nullptr, nullptr, nullptr}});
        feat[3] = t1;
        if (T == 1) mv(m.head, t1, logits);
        else if (all_logits && T <= MMV_T) mmv(T, {MV{&m.head, t1, logits, 0}});
        else if (all_logits) {
            GSegs S{};
            S.n = 1; S.tstart[0] = 0; S.tstart[1] = (m.head.N + 63) / 64;
            S.s[0] = Seg{m.head.w, t1, logits, m.head.rb, m.head.N, m.head.K, m.head.type, 0, m.vocab_pad};
            hipLaunchKernelGGL(k_gemm_w, dim3(S.tstart[1], (T + 63) / 64), dim3(256), 0, st, S, T);
        } else mv(m.head, t1 + (size_t) (T - 1) * C, logits);
        const int nl = all_logits ? T : 1;
        hipLaunchKernelGGL(k_softcap, dim3(ew(nl * m.vocab_pad)), dim3(256), 0, st, logits, nl * m.vocab_pad, m.softcap);
    }
};

// ---------------------------------------------------------------- serve (the protocol of src/inkling/inkling_serve.cpp)
struct LineQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool closed = false;
    std::atomic<bool> stop{false};
    void run() {
        char * line = nullptr;
        size_t cap = 0;
        ssize_t n;
        while ((n = getline(&line, &cap, stdin)) >= 0) {
            std::string s(line, (size_t) n);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (s == "STOP") { stop = true; continue; }
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(s);
            cv.notify_one();
        }
        std::free(line);
        std::lock_guard<std::mutex> lk(mu);
        closed = true;
        cv.notify_one();
    }
    bool pop(std::string & s) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return closed || !q.empty(); });
        if (q.empty()) return false;
        s = q.front();
        q.pop_front();
        return true;
    }
};
bool parse_ids(const std::string & s, std::vector<int> & out) {
    out.clear();
    const char * p = s.c_str();
    while (*p) {
        char * e = nullptr;
        const long long v = std::strtoll(p, &e, 10);
        if (e == p) return false;
        out.push_back((int) v);
        p = e;
        if (*p == ',') ++p;
        else if (*p) return false;
    }
    return !out.empty();
}
void say(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void say(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}
struct Sampling {
    // temp 0 (greedy) when a request sends none: the server does not forward temperature=0 (serve/server.py
    // sampling_keys), so the default has to be greedy, as hy4_serve's is
    float temp = 0.0f, top_p = 1.0f, min_p = 0.0f, rep = 1.0f, freq = 0.0f, pres = 0.0f;
    int top_k = 0, last_n = 64;
    uint64_t seed = 0;
};
// the same sampler as almond-inkling's (and hy4_sample)
int sample(const float * logits, int V, const Sampling & s, const std::vector<int> & hist, std::mt19937_64 & rng) {
    std::vector<float> l(logits, logits + V);
    if (s.rep != 1.0f || s.freq != 0.0f || s.pres != 0.0f) {
        std::unordered_map<int, int> cnt;
        const size_t from = s.last_n > 0 && hist.size() > (size_t) s.last_n ? hist.size() - (size_t) s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) if (hist[i] >= 0) ++cnt[hist[i]];
        for (const auto & [t, c] : cnt) {
            if (t < 0 || t >= V) continue;
            float & x = l[(size_t) t];
            if (s.rep != 1.0f) x = x > 0 ? x / s.rep : x * s.rep;
            x -= s.freq * (float) c + s.pres;
        }
    }
    if (s.temp <= 0.0f) return (int) (std::max_element(l.begin(), l.end()) - l.begin());
    if (s.top_k <= 0 && s.top_p >= 1.0f && s.min_p <= 0.0f) {
        // no truncation asked: draw from the whole softmax (two passes, no sort). The sorted path below keeps the best
        // 4,096; this one keeps every token, so it is the exact distribution
        const float mx = *std::max_element(l.begin(), l.end()), it = 1.0f / s.temp;
        double tot = 0;
        for (int i = 0; i < V; ++i) { const float e = expf((l[(size_t) i] - mx) * it); l[(size_t) i] = e; tot += e; }
        std::uniform_real_distribution<double> u(0.0, tot);
        double r = u(rng), acc = 0;
        for (int i = 0; i < V; ++i) { acc += l[(size_t) i]; if (r <= acc) return i; }
        return (int) (std::max_element(logits, logits + V) - logits);
    }
    std::vector<int32_t> idx((size_t) V);
    for (int i = 0; i < V; ++i) idx[(size_t) i] = i;
    const size_t cut = std::min<size_t>(s.top_k > 0 ? (size_t) s.top_k : (size_t) V, 4096);
    std::partial_sort(idx.begin(), idx.begin() + (ptrdiff_t) cut, idx.end(),
                      [&](int32_t a, int32_t b) { return l[(size_t) a] > l[(size_t) b]; });
    const size_t k = cut;
    const float mx = l[(size_t) idx[0]];
    std::vector<double> p(k);
    for (size_t i = 0; i < k; ++i) p[i] = std::exp((double) (l[(size_t) idx[i]] - mx) / s.temp);
    size_t keep = k;
    if (s.min_p > 0.0f)
        for (size_t i = 0; i < k; ++i) if (p[i] < s.min_p * p[0]) { keep = std::max<size_t>(1, i); break; }
    if (s.top_p < 1.0f) {
        double acc = 0, tot = 0;
        for (size_t i = 0; i < keep; ++i) tot += p[i];
        for (size_t i = 0; i < keep; ++i) {
            acc += p[i];
            if (acc >= s.top_p * tot) { keep = i + 1; break; }
        }
    }
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng), acc = 0;
    for (size_t i = 0; i < keep; ++i) {
        acc += p[i];
        if (r <= acc) return idx[i];
    }
    return idx[0];
}
// A GENI request's pictures: the embeddings file holds one record per picture in prompt order (int32 'SVE1', n,
// nx, ny, n_embd, then n x n_embd floats: almond-vision's format). The chat template writes a picture as
// <|content_image|><|unused_200054|><|end_message|>; the server repeats the placeholder once per image token, and
// picture k's rows take the k-th run of it (llama.cpp's mtmd puts the rows right after <|content_image|> the same
// way, with ordinary positions). Each row's cache key is negative, from its picture's key (the server's img_keys=,
// or a hash of its rows) and its index, so two pictures are never taken for each other.
constexpr int kImageTok = 200054;
bool image_rows(const std::string & file, const std::vector<std::string> & ikeys, int C, const std::vector<int> & ids,
                std::vector<int> & keys, std::vector<float> & buf, std::vector<const float *> & rows, std::string & why) {
    FILE * fp = std::fopen(file.c_str(), "rb");
    if (!fp) { why = "cannot open the embeddings file " + file; return false; }
    struct Rec { int32_t n; size_t row0; uint64_t h; };
    std::vector<Rec> recs;
    buf.clear();
    for (;;) {
        int32_t hd[5];
        if (std::fread(hd, 4, 5, fp) != 5) break;
        if (std::memcmp(hd, "SVE1", 4) != 0 || hd[4] != C || hd[1] <= 0) {
            std::fclose(fp);
            why = "the embeddings file is not an almond-vision record set for this model (n_embd " + std::to_string(hd[4]) + ")";
            return false;
        }
        const size_t row0 = buf.size() / (size_t) C;
        buf.resize(buf.size() + (size_t) hd[1] * C);
        if (std::fread(buf.data() + row0 * C, 4, (size_t) hd[1] * C, fp) != (size_t) hd[1] * C) {
            std::fclose(fp);
            why = "the embeddings file is cut short";
            return false;
        }
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the picture's key, or its rows
        if (recs.size() < ikeys.size()) for (char ch : ikeys[recs.size()]) h = (h ^ (uint8_t) ch) * 1099511628211ull;
        else {
            const uint8_t * b = (const uint8_t *) (buf.data() + row0 * C);
            for (size_t i = 0; i < (size_t) hd[1] * C * 4; i += 61) h = (h ^ b[i]) * 1099511628211ull;
        }
        recs.push_back({hd[1], row0, h});
    }
    std::fclose(fp);
    rows.assign(ids.size(), nullptr);
    size_t k = 0;
    for (size_t i = 0; i < ids.size();) {
        if (ids[i] != kImageTok) { ++i; continue; }
        size_t j = i;
        while (j < ids.size() && ids[j] == kImageTok) ++j;
        if (k >= recs.size() || (size_t) recs[k].n != j - i) {
            why = "the prompt's picture " + std::to_string(k + 1) + " has " + std::to_string(j - i) +
                  " image tokens, the embeddings file " + (k < recs.size() ? std::to_string(recs[k].n) : "none");
            return false;
        }
        for (int32_t t = 0; t < recs[k].n; ++t) {
            rows[i + t] = buf.data() + (recs[k].row0 + (size_t) t) * C;
            const uint64_t hk = recs[k].h ^ ((uint64_t) (t + 1) * 0x9E3779B97F4A7C15ull);
            keys[i + t] = -1 - (int) (hk & 0x3FFFFFFF);
        }
        ++k;
        i = j;
    }
    if (k != recs.size()) {
        why = "the embeddings file has " + std::to_string(recs.size()) + " pictures, the prompt " + std::to_string(k);
        return false;
    }
    return true;
}
int serve(Model & m, Runner & run, const std::vector<int> & eos, const Sampling & defaults, int n_draft, float draft_tmax) {
    std::vector<int> hist;                 // the tokens in the cache
    std::deque<Runner::Ckpt> ckpts;        // prompt-end checkpoints, the newest last
    constexpr size_t kMaxCkpt = 4;
    auto clear = [&] { run.clear(); hist.clear(); ckpts.clear(); };
    auto save_ckpt = [&] {
        if (!ckpts.empty() && ckpts.back().n == (int) hist.size()) return;
        Runner::Ckpt c{(int) hist.size(), {}};
        run.save_state(c.data);
        while (ckpts.size() >= kMaxCkpt) ckpts.pop_front();
        ckpts.push_back(std::move(c));
    };
    auto restore_ckpt = [&](size_t limit) -> size_t {
        while (!ckpts.empty() && (size_t) ckpts.back().n > limit) ckpts.pop_back();
        if (ckpts.empty() || ckpts.back().n == 0) return 0;
        run.load_state(ckpts.back().data, ckpts.back().n);
        hist.resize((size_t) ckpts.back().n);
        return hist.size();
    };
    std::vector<float> img_buf;            // the current request's picture rows
    LineQueue lq;
    std::thread rd([&] { lq.run(); });
    rd.detach();
    std::mt19937_64 rng(std::random_device{}());
    const long long n_ctx = run.ctx;
    say("INFO engine=%s model=inkling context=%lld mtp=0 native=1 kv=%s\n", ALMOND_VERSION, n_ctx,
        run.kvt == 1 ? "q8_0" : run.kvt == 2 ? "f16" : "f32");
    say("READY %lld stop\n", n_ctx);
    std::string line;
    const float * lg = nullptr;
    while (lq.pop(line)) {
        if (line.empty()) continue;
        if (line == "QUIT") break;
        if (line == "CONVCLEAR") { clear(); say("CLEARED 0 0\n"); continue; }
        // VLEND / VRECLAIM (between requests, as the GLM engine): the on-demand image encoder is about to start on
        // this card - the last expert tensors' memory is released for it ("VLENT <bytes>") - and has ended
        // ("VRECLAIMED", or an ERR while its memory is not back yet: the server asks again)
        if (line == "VLEND") {
            const char * e = std::getenv("ALMOND_VISION_LEND_MB");
            if (!e) e = std::getenv("ALMOND_GLM_VISION_LEND_MB");
            const size_t want = (size_t) (e ? std::atof(e) : 3072.0) * 1048576 + (256u << 20);
            if (g_lend.empty()) { say("ERR nothing to lend (the experts are not in lendable memory)\n"); continue; }
            say("VLENT %zu\n", lend_out(want));
            continue;
        }
        if (line == "VRECLAIM") {
            if (lend_back()) say("VRECLAIMED\n");
            else say("ERR the lent memory is not free yet\n");
            continue;
        }
        if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
            const bool save = line[0] == 'S';
            const std::string file = line.substr(save ? 5 : 8);
            const auto t0 = std::chrono::steady_clock::now();
            size_t bytes = 0;
            bool ok;
            if (save) ok = run.save_file(file, hist, bytes);
            else {
                clear();
                std::vector<int> t;
                ok = run.load_file(file, t, bytes);
                if (ok) hist = t; else clear();
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ok) say("%s %zu %zu %.1f\n", save ? "SAVED" : "RESTORED", hist.size(), bytes, ms);
            else say("SERR %s 0 the cache file could not be %s\n", save ? "write" : "read", save ? "written" : "read");
            continue;
        }
        if (lend_owed() && !lend_back()) { say("ERR the experts' memory lent to the image encoder is not back yet\n"); continue; }
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (geni) line = "GEN " + line.substr(5);   // GENI <max_new> [keys] [img_keys=..] <embeddings file> <ids>
        if (line.rfind("GEN ", 0) != 0) { say("ERR unknown command\n"); continue; }
        std::vector<std::string> f;
        for (size_t i = 4; i < line.size();) {
            size_t j = line.find(' ', i);
            if (j == std::string::npos) j = line.size();
            if (j > i) f.push_back(line.substr(i, j - i));
            i = j + 1;
        }
        std::vector<int> ids;
        if (f.size() < (geni ? 3u : 2u) || !parse_ids(f.back(), ids)) { say("ERR bad GEN line\n"); continue; }
        std::string efile;
        std::vector<std::string> ikeys;
        if (geni) {
            efile = f[f.size() - 2];
            f.erase(f.end() - 2);
            for (const std::string & x : f)
                if (x.rfind("img_keys=", 0) == 0)
                    for (size_t p = 9; p <= x.size();) {
                        size_t q = x.find(',', p);
                        if (q == std::string::npos) q = x.size();
                        ikeys.push_back(x.substr(p, q - p));
                        p = q + 1;
                    }
        }
        const int max_new = std::max(1, std::atoi(f[0].c_str()));
        Sampling s = defaults;
        for (size_t i = 1; i + 1 < f.size(); ++i) {
            const size_t eq = f[i].find('=');
            if (eq == std::string::npos) continue;
            const std::string k = f[i].substr(0, eq);
            const double x = std::atof(f[i].c_str() + eq + 1);
            if (k == "temperature") s.temp = (float) x;
            else if (k == "top_p") s.top_p = (float) x;
            else if (k == "top_k") s.top_k = (int) x;
            else if (k == "min_p") s.min_p = (float) x;
            else if (k == "penalty_repeat") s.rep = (float) x;
            else if (k == "penalty_freq") s.freq = (float) x;
            else if (k == "penalty_present") s.pres = (float) x;
            else if (k == "penalty_last_n") s.last_n = (int) x;
            else if (k == "seed") s.seed = (uint64_t) x;
        }
        if (s.seed) rng.seed(s.seed);
        bool bad = false;
        for (int t : ids) if (t < 0 || t >= m.vocab) bad = true;
        if (bad) { say("ERR a token id is outside the vocabulary\n"); continue; }
        // keys: what the cache matches a prompt on (the token, or for a picture's row a negative key of the picture
        // and the row); rows: per token its picture row
        std::vector<int> keys(ids);
        std::vector<const float *> rows;
        if (geni) {
            std::string why;
            if (!image_rows(efile, ikeys, m.C, ids, keys, img_buf, rows, why)) { say("ERR %s\n", why.c_str()); continue; }
        }
        if ((long long) ids.size() + 1 > n_ctx) {
            say("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), n_ctx);
            continue;
        }
        lq.stop = false;
        // the cache is kept only when the prompt extends it (the convolution state cannot be cut back)
        size_t keep = 0;
        if (!hist.empty() && hist.size() < keys.size() && std::equal(hist.begin(), hist.end(), keys.begin())) keep = hist.size();
        else {
            size_t same = 0;
            while (same < hist.size() && same < keys.size() && hist[same] == keys[same]) ++same;
            keep = restore_ckpt(std::min(same, ids.size() - 1));
            if (keep == 0) clear();
            else say("INFO checkpoint restored at %zu (shared prefix %zu)\n", keep, same);
        }
        if (keep > 0) say("RESUME %zu\n", keep);
        const auto t0 = std::chrono::steady_clock::now();
        bool cancelled = false;
        for (size_t at = keep; at < ids.size();) {
            const size_t n = std::min((size_t) run.TM, ids.size() - at);
            const bool pic = !rows.empty() && std::any_of(rows.begin() + (ptrdiff_t) at, rows.begin() + (ptrdiff_t) (at + n),
                                                          [](const float * r) { return r != nullptr; });
            if (n == 1 && !pic) run.step(ids[at], nullptr);
            else run.chunk(ids.data() + at, (int) n, false, nullptr, pic ? rows.data() + at : nullptr);
            run.dspark_inject((int) n, (int) at);
            if (at + n == ids.size()) lg = run.fetch_logits();
            hist.insert(hist.end(), keys.begin() + (ptrdiff_t) at, keys.begin() + (ptrdiff_t) (at + n));
            at += n;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            say("PP %zu %zu %.0f %.1f\n", at, ids.size(), ms, (double) (at - keep) / std::max(ms, 1e-3) * 1000.0);
            if (lq.stop && at < ids.size()) { cancelled = true; break; }
        }
        const double pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (cancelled || lq.stop) {
            clear();
            say("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu\n", ids.size(), pp_ms, keep);
            continue;
        }
        save_ckpt();   // the prompt's end: where the next prompt of this conversation will part from the cache
        const auto t1 = std::chrono::steady_clock::now();
        int produced = 0;
        std::string finish = "length";
        std::vector<int> h(hist);
        auto emit = [&](int y) {
            say("T %d\n", y);
            ++produced;
            h.push_back(y);
            if (std::find(eos.begin(), eos.end(), y) != eos.end()) { finish = "stop"; return false; }
            if (produced >= max_new) return false;
            if (lq.stop) { finish = "cancel"; return false; }
            return true;
        };
        int y = sample(lg, m.vocab, s, h, rng);
        bool more = emit(y);
        long long drafted = 0, accepted = 0;
        // DSpark: greedy drafts after y, checked position by position with the request's own sampler; the batch
        // keeps the draft up to the first token the target did not sample itself (the same distribution as without).
        // A verification of n + 1 tokens costs about 1.8 single steps here (each token reads its own experts), so
        // while the accepted share (a moving average) is under 25% the next 32 tokens go without drafting
        double share = 0.5;
        int plain_left = 0;
        while (more) {
            if ((long long) hist.size() + 1 > n_ctx) break;
            // at temperature 1.0 too few drafts hold to pay for the verification (docs/inkling/ENGINE.md, step 6)
            if (run.dr && n_draft > 0 && plain_left == 0 && s.temp <= draft_tmax) {
                const int base = (int) hist.size();
                const int nd = (int) std::min<long long>({(long long) n_draft, n_ctx - base - 2, (long long) (max_new - produced - 1)});
                int dt[SPEC_T];
                if (nd > 0 && run.dspark_draft(y, base, nd, dt)) {
                    int b[SPEC_T];
                    b[0] = y;
                    for (int i = 0; i < nd; ++i) b[i + 1] = dt[i];
                    const float * L = run.verify(b, nd + 1);
                    int kept = 0;
                    for (int i = 0; i <= nd; ++i) {
                        hist.push_back(b[i]);
                        kept = i + 1;
                        y = sample(L + (size_t) i * m.vocab, m.vocab, s, h, rng);
                        more = emit(y);
                        if (!more || i == nd || y != dt[i]) break;
                    }
                    drafted += nd;
                    accepted += kept - 1;
                    run.rollback(nd + 1, kept);
                    run.dspark_inject(kept, base);
                    share = 0.75 * share + 0.25 * (double) (kept - 1) / nd;
                    if (share < 0.25) { plain_left = 32; share = 0.5; }
                    continue;
                }
            }
            if (plain_left > 0) --plain_left;
            run.step(y, nullptr);
            run.dspark_inject(1, (int) hist.size());
            lg = run.fetch_logits();
            hist.push_back(y);
            y = sample(lg, m.vocab, s, h, rng);
            more = emit(y);
        }
        if (drafted > 0) say("INFO dspark drafted=%lld accepted=%lld\n", drafted, accepted);
        const double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        say("DONE %d %zu %.1f %.1f %s 0 0 %zu\n", produced, ids.size(), pp_ms, gen_ms, finish.c_str(), keep);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string path, score, dump;
    int gpu = 0, ctx = 4096, n_dump = 0, n_bench = 0, chunk = 1;
    std::string dspark;   // --dspark DRAFT.gguf (opt-in): Unsloth's DSpark drafter
    int n_draft = 4;      // --dspark-n N: tokens drafted a step (at most the drafter's block - 1)
    float draft_tmax = 0.8f;   // --dspark-max-temp T: draft only for requests at temperature <= T
    bool spec_test = false;   // --spec-test: the text through verify(4) / rollback(2), checks the speculative path
    bool serve_mode = false;   // --serve: the engine's line protocol on stdin / stdout
    std::vector<int> eos;
    Sampling defaults;
    int kv_gpu = -1;   // --kv-gpu N: the full attention layers' K/V and attention on card N (long contexts)
    int kvt = 0;   // --kv-q8 / --kv-f16: the KV cache in q8_0 / F16 (default F32)
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
        else if (a == "--kv-q8") kvt = 1;
        else if (a == "--spec-test") spec_test = true;
        else if (a == "--serve") serve_mode = true;
        else if (a == "--inkling") {}
        else if (a == "--vision") {}   // (the server says images may come: GENI is always read)
        else if (a == "--vram-reserve-mib") (void) next();   // (the server's, for engines that size pools: none here)
        else if (a == "--ubatch") chunk = std::atoi(next());
        else if (a == "--eos") eos.push_back(std::atoi(next()));
        else if (a == "--temp") defaults.temp = (float) std::atof(next());
        else if (a == "--top-p") defaults.top_p = (float) std::atof(next());
        else if (a == "--dspark") dspark = next();
        else if (a == "--dspark-n") n_draft = std::atoi(next());
        else if (a == "--dspark-max-temp") draft_tmax = (float) std::atof(next());
        else if (a == "--kv-f16") kvt = 2;
        else if (a == "--kv-gpu") kv_gpu = std::atoi(next());
        else { std::fprintf(stderr, "almond-inkling-native: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (path.empty() || (score.empty() && !serve_mode)) {
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
    if (serve_mode) {
        Runner run(m, ctx, kvt, chunk > 1 ? std::min(chunk, 1024) : 1024, false, kv_gpu);
        size_t fr = 0, tot = 0;
        CK(hipMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "almond-inkling-native: context %d, KV %s, chunks of %d, %.2f GiB of VRAM free\n", ctx,
                     kvt == 1 ? "q8_0" : kvt == 2 ? "f16" : "f32", run.TM, fr / 1073741824.0);
        if (kv_gpu >= 0) {
            CK(hipSetDevice(kv_gpu)); CK(hipMemGetInfo(&fr, &tot)); CK(hipSetDevice(gpu));
            std::fprintf(stderr, "almond-inkling-native: full attention K/V on card %d, %.2f GiB free there\n", kv_gpu,
                         fr / 1073741824.0);
        }
        if (!dspark.empty()) {
            run.dspark_load(dspark);
            n_draft = std::max(1, std::min(n_draft, run.dr->B - 1));
            CK(hipMemGetInfo(&fr, &tot));
            std::fprintf(stderr, "almond-inkling-native: DSpark drafter %s, %d tokens a step, %.2f GiB of VRAM free\n",
                         dspark.c_str(), n_draft, fr / 1073741824.0);
        }
        return serve(m, run, eos, defaults, dspark.empty() ? 0 : n_draft, draft_tmax);
    }
    std::vector<int> ids;
    { std::ifstream f(score); int v; while (f >> v) ids.push_back(v); }
    if ((int) ids.size() > ctx) ids.resize(ctx);
    if (n_pf > 0) {
        Runner pr(m, ctx, kvt, std::max(2, std::min(chunk, 1024)), false, kv_gpu);
        std::vector<int> t;
        while ((int) t.size() < n_pf) t.insert(t.end(), ids.begin(), ids.end());
        t.resize((size_t) n_pf);
        const int c = std::max(2, std::min(chunk, 1024));
        pr.chunk(t.data(), c, false, nullptr);   // warm-up chunk (pointer sets, kernels)
        pr.pos = 0;
        for (auto & sl : pr.slots) CK(hipMemset(sl.hist, 0, (size_t) 3 * (2 * m.Hkv * m.D + 2 * m.C) * 4));
        const auto tp = std::chrono::steady_clock::now();
        for (int i = 0; i < n_pf; i += c) pr.chunk(t.data() + i, std::min(c, n_pf - i), false, nullptr);
        const double ps = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
        std::printf("BENCH prefill %d tokens in chunks of %d: %.1f tok/s (%.2f s)\n", n_pf, c, n_pf / ps, ps);
        return 0;
    }
    Runner run(m, ctx, kvt, std::max(2, std::min(chunk, 1024)), true, kv_gpu);
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
    chunk = std::max(1, std::min(chunk, 1024));
    if (spec_test) {
        run.enable_spec();
        for (size_t i = 0; i + 1 < ids.size();) {
            const int T = (int) std::min<size_t>(4, ids.size() - 1 - i), keep = std::min(T, 2);
            const float * l = run.verify(ids.data() + i, T);
            for (int t = 0; t < keep; ++t) score_one(i + t, l + (size_t) t * m.vocab);
            run.rollback(T, keep);
            i += (size_t) keep;
        }
        chunk = 1 << 30;   // skip the loop below
    }
    for (size_t i = 0; chunk < (1 << 30) && i + 1 < ids.size();) {
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

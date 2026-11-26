// bench_hy4_copy - the host->card paths the expert staging uses, measured on their own.
//
// Why: the decode round moves the routed experts of the RAM-resident layers from host RAM into a card on every step,
// and Avalon's fork reports that its staged host copies only got fast once they were taken off SDMA and the legacy
// stream (docs/ENGINES_WATCH.md).  This measures what this machine gives, per device and per size, without loading a
// model and without touching the running engine: it allocates its own buffers.
//
//   bench_hy4_copy [device] [MiB] [mapped]     # card 1 sits on a PCIe x2 uplink; "mapped" adds the kernel arms
//
// What it says (2026-11-22, 12 reps, HIP events, the same pinned host buffer for the two memcpy arms):
//
//   card 0   memcpy on the legacy stream   28.0 - 28.7 GB/s   memcpy on a stream of its own   the same
//   card 1   memcpy on the legacy stream    3.3 GB/s          memcpy on a stream of its own    3.3 GB/s
//
// which is the link's roof (PCIe 4.0 x16 and x2), and the same the plain hipMemcpy test in docs/hy4/SPEED_WORK.md
// found: a copy that has to cross the link cannot be made faster by how it is copied or on which stream it runs.
//
// The kernel arms - a copy kernel reading host memory the card can address, which is Avalon's shape - are here to
// record that they do not work on this machine.  They launch, they time beautifully (12.9 GB/s on the x2 card,
// 107 GB/s on the x16 one, which no PCIe link of this generation carries), and the bytes they deliver are wrong:
// each arm fills its own source with its own tag and reads it back, and every kernel arm fails that check.  With the
// alias from hipHostGetDevicePointer the reads go somewhere else; passing the host pointer instead takes the process
// down with a GPU page fault (exit 139).  Off unless "mapped" is given, and the check is the point of the program:
// a copy that moves nothing is the fastest copy there is.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TRY(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); return 1; } } while (0)

// What the box has left, so this bench can refuse instead of being the last gram.  Learned on 2026-11-22 at 16:24:
// with the engines already at 121 of 122 GB, the pinned buffers of a 128 MiB run were enough for almond-engine to
// invoke the OOM killer, and the kernel killed qwen-host (anon-rss 57.4 GB).  Pinned pages are not reclaimable.
static long long mem_available_kib() {
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char key[64]; long long v = 0, out = -1;
    while (std::fscanf(f, "%63s %lld kB", key, &v) == 2) if (!std::strcmp(key, "MemAvailable:")) { out = v; break; }
    std::fclose(f);
    return out;
}

// A 16-byte-per-lane copy: what limits this is the traffic, not the arithmetic.
static __global__ void copy16(uchar4 * __restrict__ dst, const uchar4 * __restrict__ src, size_t n16) {
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        dst[i] = src[i];
    }
}

int main(int argc, char ** argv) {
    const int dev      = argc > 1 ? std::atoi(argv[1]) : 1;
    const size_t mib   = argc > 2 ? (size_t) std::atoi(argv[2]) : 128;
    const bool  mapped = argc > 3;
    const size_t bytes = mib << 20;
    const int reps = 12;

    // h_plain, and for the kernel arms h_reg + h_mapped + four windows of h_big: seven times the size, all pinned.
    const size_t pinned = bytes * (mapped ? 7 : 1);
    const long long avail = mem_available_kib();
    if (avail >= 0 && avail * 1024LL < (long long) pinned + (8LL << 30)) {
        std::fprintf(stderr, "bench_hy4_copy: MemAvailable %.1f GiB, il bench ne vuole %.2f GiB di pinned piu 8 GiB di "
                             "margine.  Non parto: le pagine pinned non sono recuperabili e con gli engine su "
                             "l'OOM killer colpisce un altro processo.\n",
                     avail / 1048576.0, pinned / 1073741824.0);
        return 3;
    }

    TRY(hipSetDevice(dev));
    hipDeviceProp_t p{};
    TRY(hipGetDeviceProperties(&p, dev));
    std::printf("# card %d (%s), %zu MiB, %d CU, memory %d MHz @ %d bit\n",
                dev, p.name, mib, p.multiProcessorCount, p.memoryClockRate, p.memoryBusWidth);

    std::vector<uint8_t> pattern(bytes);
    for (size_t i = 0; i < bytes; ++i) pattern[i] = (uint8_t) (i * 31u + 7u);

    uint8_t * d = nullptr;
    TRY(hipMalloc(&d, bytes));

    uint8_t * h_plain = nullptr, * h_reg = nullptr, * h_reg_dev = nullptr, * h_mapped = nullptr, * h_mapped_dev = nullptr;
    uint8_t * h_big = nullptr, * h_big_dev = nullptr;
    TRY(hipHostMalloc(&h_plain, bytes, hipHostMallocDefault));                     // pinned, not device-visible
    if (mapped) {
        // The real thing: host RAM, page-locked and registered so the card's own kernels can read it over the link.
        // (hipHostMalloc with the Mapped flag is a trap on ROCm: it answers from device memory, so the copy below
        // would measure a card-internal copy, not the link - the fourth arm shows it.)
        if (posix_memalign((void **) &h_reg, 4096, bytes) != 0) { std::fprintf(stderr, "posix_memalign\n"); return 1; }
        std::memset(h_reg, 0, bytes);
        TRY(hipHostRegister(h_reg, bytes, hipHostRegisterDefault));
        TRY(hipHostGetDevicePointer((void **) &h_reg_dev, h_reg, 0));
        TRY(hipHostMalloc(&h_mapped, bytes, hipHostMallocMapped));
        TRY(hipHostGetDevicePointer((void **) &h_mapped_dev, h_mapped, 0));
        // four windows of the same size but different content, so a timed sweep cannot be served by the 48 MB of L2
        // (each window is re-read only after 192 MB of other traffic) and cannot pass a check on a stale copy.
        if (posix_memalign((void **) &h_big, 4096, 4 * bytes) != 0) { std::fprintf(stderr, "posix_memalign big\n"); return 1; }
        for (int w = 0; w < 4; ++w)
            for (size_t i = 0; i < bytes; ++i)
                h_big[(size_t) w * bytes + i] = (uint8_t) ((i * 31u + 7u) ^ (unsigned) (w * 0x5au));
        TRY(hipHostRegister(h_big, 4 * bytes, hipHostRegisterDefault));
        TRY(hipHostGetDevicePointer((void **) &h_big_dev, h_big, 0));
    }
    std::memcpy(h_plain, pattern.data(), bytes);
    if (mapped) std::memcpy(h_reg, pattern.data(), bytes);
    if (mapped) std::memcpy(h_mapped, pattern.data(), bytes);

    hipStream_t st = nullptr;
    TRY(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));

    struct Arm { const char * name; int kind; hipStream_t stream; };   // 0 memcpy, 1 kernel on registered RAM, 2 kernel on hipHostMalloc, 3 kernel, fresh windows out of DRAM
    const Arm arms[] = {
        { "memcpy/legacy",     0, nullptr },
        { "memcpy/stream",     0, st },
        { "kernel/registered", 1, st },
        { "kernel/hostalloc",  2, st },
        { "kernel/fresh-512MB", 3, st },
    };

    hipEvent_t e0, e1;
    TRY(hipEventCreate(&e0)); TRY(hipEventCreate(&e1));
    // Each arm writes its own content into the source it reads from, and checks that content back out of the device
    // afterwards. Without the per-arm tag the check proves nothing: every arm would compare against what the previous
    // arm already put in the destination - which is how a kernel copy that moves nothing looked correct once.
    auto fill = [](uint8_t * p, size_t n, unsigned tag) {
        for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) ((i * 31u + 7u) ^ tag);
    };
    for (const Arm & a : arms) {
        if (a.kind && !mapped) continue;
        const unsigned tag = 0x5au * (a.kind + 1u);
        uint8_t * src_host = a.kind == 1 ? h_reg : a.kind == 2 ? h_mapped : a.kind == 3 ? h_big : h_plain;
        const size_t src_n = a.kind == 3 ? 4 * bytes : bytes;
        fill(src_host, src_n, tag);
        TRY(hipStreamSynchronize(a.stream));
        auto launch = [&]() -> hipError_t {
            const int block = 256, grid = 32 * p.multiProcessorCount;
            if (a.kind == 1) {
                copy16<<<grid, block, 0, a.stream>>>((uchar4 *) d, (const uchar4 *) h_reg_dev, bytes / 16);
                return hipGetLastError();
            }
            if (a.kind == 2) {
                copy16<<<grid, block, 0, a.stream>>>((uchar4 *) d, (const uchar4 *) h_mapped_dev, bytes / 16);
                return hipGetLastError();
            }
            if (a.kind == 3) {
                copy16<<<grid, block, 0, a.stream>>>((uchar4 *) d, (const uchar4 *) h_big_dev, bytes / 16);
                return hipGetLastError();
            }
            return hipMemcpyAsync(d, h_plain, bytes, hipMemcpyHostToDevice, a.stream);
        };
        for (int i = 0; i < 2; ++i) { hipError_t e = launch(); if (e != hipSuccess) {
            std::printf("  %-14s lancia male: %s\n", a.name, hipGetErrorString(e)); break; } }
        TRY(hipStreamSynchronize(a.stream));
        TRY(hipEventRecord(e0, a.stream));
        for (int i = 0; i < reps; ++i) {                                  // kind 3 reads a fresh window every rep
            if (a.kind == 3) {
                const size_t off = (size_t) (i % 4) * bytes;
                const int block = 256, grid = 32 * p.multiProcessorCount;
                copy16<<<grid, block, 0, a.stream>>>((uchar4 *) d, (const uchar4 *) h_big_dev + off / 16, bytes / 16);
            } else if (launch() != hipSuccess) break;
        }
        TRY(hipEventRecord(e1, a.stream));
        TRY(hipEventSynchronize(e1));
        float ms = 0;
        TRY(hipEventElapsedTime(&ms, e0, e1));
        const double s = ms / 1e3 / reps;
        const uint8_t * want = a.kind == 3 ? h_big + (size_t) ((reps - 1) % 4) * bytes : src_host;
        std::vector<uint8_t> chk(64);
        bool ok = true;
        for (size_t at : { (size_t) 0, bytes / 2, bytes - 64 }) {
            TRY(hipMemcpy(chk.data(), d + at, 64, hipMemcpyDeviceToHost));
            if (std::memcmp(chk.data(), want + at, 64) != 0) ok = false;
        }
        std::printf("  %-14s %7.2f ms  %6.1f GB/s%s\n", a.name, s * 1e3, bytes / s / 1e9, ok ? "" : "   (il contenuto non torna)");
    }

    TRY(hipFree(d)); TRY(hipHostFree(h_plain));
    if (mapped) { TRY(hipHostUnregister(h_reg)); std::free(h_reg); TRY(hipHostFree(h_mapped));
                  TRY(hipHostUnregister(h_big)); std::free(h_big); }
    TRY(hipStreamDestroy(st));
    return 0;
}

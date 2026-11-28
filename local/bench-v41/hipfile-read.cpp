// hipfile-read FILE MODE(hipfile|posix) N GAP_MS [DEV]: N reads of one 10 MiB blob at random 4 KiB-aligned offsets of FILE
// into VRAM, one every GAP_MS.  hipfile: hipFileRead straight into a registered device buffer; posix: O_DIRECT pread
// into pinned host memory, then hipMemcpy to the device (the engine's file tier).  Prints the driver's status (whether
// local NVMe P2P is supported or the library falls back to its compatibility path) and mean / median / p90 / max ms.
// Build: hipcc -O2 -o hipfile-read hipfile-read.cpp -lhipfile
#include <hip/hip_runtime.h>
#include <hipfile.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: hipfile-read FILE hipfile|posix N GAP_MS [DEV]\n"); return 2; }
    const char *path = argv[1]; bool hf = !strcmp(argv[2], "hipfile"); int n = atoi(argv[3]); int gap = atoi(argv[4]);
    hipSetDevice(argc > 5 ? atoi(argv[5]) : 0);
    const size_t B = 10u << 20;
    int fd = open(path, O_RDONLY | O_DIRECT); if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    void *dbuf; if (hipMalloc(&dbuf, B) != hipSuccess) { fprintf(stderr, "hipMalloc failed\n"); return 1; }
    hipFileHandle_t fh = nullptr; void *hbuf = nullptr;
    if (hf) {
        hipFileError_t e = hipFileDriverOpen();
        if (e.err != hipFileSuccess) { fprintf(stderr, "hipFileDriverOpen: %s (hip %d)\n", hipFileGetOpErrorString(e.err), e.hip_drv_err); return 1; }
        hipFileDriverProps_t p{}; hipFileDriverGetProperties(&p);
        printf("hipFile driver %u.%u, status 0x%x (NVMe %d, NVMeP2P %d), control 0x%x (compat allowed %d), features 0x%x\n",
               p.nvfs.major_version, p.nvfs.minor_version, p.nvfs.driver_status_flags,
               !!(p.nvfs.driver_status_flags & (1u << hipFileNVMeSupported)), !!(p.nvfs.driver_status_flags & (1u << hipFileNVMeP2PSupported)),
               p.nvfs.driver_control_flags, !!(p.nvfs.driver_control_flags & (1u << hipFileAllowCompatMode)), p.feature_flags);
        hipFileDescr_t d{}; d.type = hipFileHandleTypeOpaqueFD; d.handle.fd = fd;
        e = hipFileHandleRegister(&fh, &d);
        if (e.err != hipFileSuccess) { fprintf(stderr, "hipFileHandleRegister: %s\n", hipFileGetOpErrorString(e.err)); return 1; }
        e = hipFileBufRegister(dbuf, B, 0);
        if (e.err != hipFileSuccess) { fprintf(stderr, "hipFileBufRegister: %s\n", hipFileGetOpErrorString(e.err)); return 1; }
    } else if (hipHostMalloc(&hbuf, B) != hipSuccess) { fprintf(stderr, "hipHostMalloc failed\n"); return 1; }
    std::mt19937_64 rng(42); std::vector<double> t;
    for (int i = 0; i < n; i++) {
        off_t off = (off_t)(rng() % ((st.st_size - B) / 4096)) * 4096;
        double a = now_ms(); ssize_t r;
        if (hf) r = hipFileRead(fh, dbuf, B, off, 0);
        else { r = pread(fd, hbuf, B, off); if (r == (ssize_t)B) hipMemcpy(dbuf, hbuf, B, hipMemcpyHostToDevice); }
        double ms = now_ms() - a;
        if (r != (ssize_t)B) { fprintf(stderr, "read %d returned %zd (errno %d)\n", i, r, errno); return 1; }
        t.push_back(ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(gap));
    }
    std::vector<double> s = t; std::sort(s.begin(), s.end()); double m = 0; for (double x : t) m += x;
    printf("%s: %d reads of 10 MiB, gap %d ms: mean %.2f, median %.2f, p90 %.2f, max %.2f ms\n", argv[2], n, gap, m / n,
           s[n / 2], s[n * 9 / 10], s[n - 1]);
    if (hf) { hipFileBufDeregister(dbuf); hipFileHandleDeregister(fh); hipFileDriverClose(); }
    return 0;
}

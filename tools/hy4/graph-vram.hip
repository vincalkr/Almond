// tools/hy4/graph-vram.hip - what the HIP runtime takes on a card for a captured graph, and what it does when the
// card has no room left.
//
// Why: hy4 dies inside libamdhip64 at a `movntdq %xmm0,(%rdi)` with rdi = 0 - the runtime copying into a buffer it
// does not have - in the moment right after ggml prints "CUDA graph warmup complete", which is where ggml starts a
// capture and instantiates the graph (docs/hy4/GGUF_NOTES.md, "The engine dies and the cards are loaded again"). The
// engine's own accounting leaves 0.45 GiB free on card 0 at 256K. This asks the same questions with no model and no
// ggml: how much room a capture and an instantiation take, whether recapturing gives the room back, how much a graph
// kept alive costs, and whether a runtime allocation that cannot be answered comes back as an error or as a signal.
//
// build:  hipcc -O2 -o /tmp/graph-vram tools/hy4/graph-vram.hip
// run:    /tmp/graph-vram [device] [fill_MiB] [graphs] [nodes] [recaptures]
//           e.g. /tmp/graph-vram 0 0 8 4096 20            a card left as it is
//                /tmp/graph-vram 0 23552 8 4096 20         card 0 filled to ~0.45 GiB free, as hy4 leaves it
//                /tmp/graph-vram 0 24300 8 4096 20         filled past the point the engine has
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void probe(float * p, int v) { if (p) p[0] = (float) v; }

static bool ok = true;

static void ck(const char * what, hipError_t e) {
    if (e == hipSuccess) return;
    ok = false;
    std::printf("  !! %s: %s\n", what, hipGetErrorString(e));
}

static double free_mib(int dev) {
    size_t fr = 0, tot = 0;
    if (hipMemGetInfo(&fr, &tot) != hipSuccess) return -1;
    return (double) fr / 1048576.0;
}

// a graph of `nodes` kernel launches, built the way ggml builds one: capture a stream
static hipGraph_t capture(int nodes, float * buf, hipStream_t stream) {
    hipGraph_t g = nullptr;
    ck("hipStreamBeginCapture", hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
    for (int i = 0; i < nodes; ++i) probe<<<64, 64, 0, stream>>>(buf, i);
    ck("hipStreamEndCapture", hipStreamEndCapture(stream, &g));
    return g;
}

int main(int argc, char ** argv) {
    const int dev       = argc > 1 ? std::atoi(argv[1]) : 0;
    const long fill_mib = argc > 2 ? std::atol(argv[2]) : 0;
    const int n_graphs  = argc > 3 ? std::atoi(argv[3]) : 8;
    const int n_nodes   = argc > 4 ? std::atoi(argv[4]) : 4096;
    const int recaps    = argc > 5 ? std::atoi(argv[5]) : 20;

    ck("hipInit", hipInit(0));
    ck("hipSetDevice", hipSetDevice(dev));
    double f0 = free_mib(dev), tot = 0;
    { size_t fr = 0, t = 0; hipMemGetInfo(&fr, &t); tot = t / 1048576.0; }
    std::printf("device %d: %.0f MiB free of %.0f\n", dev, f0, tot);

    float * buf = nullptr;
    ck("hipMalloc(buf)", hipMalloc(&buf, 4));

    // fill the card to what the engine leaves it, with plain allocations ggml's pool would hold
    std::vector<void *> fill;
    if (fill_mib > 0) {
        long got = 0;
        while (got < fill_mib * 1024 * 1024) {
            void * p = nullptr;
            if (hipMalloc(&p, 256ull << 20) != hipSuccess) break;
            fill.push_back(p);
            got += 256ull << 20;
        }
        std::printf("filled %.0f MiB: %.0f MiB free\n", fill.size() * 256.0, free_mib(dev));
    }

    hipStream_t stream = nullptr;
    ck("hipStreamCreateWithFlags", hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // 1. one graph, captured and instantiated `recaps` times: what a recapture costs, and what it gives back
    std::printf("\n1) one graph of %d nodes, captured and instantiated %d times over\n", n_nodes, recaps);
    {
        double before = free_mib(dev);
        hipGraph_t g = capture(n_nodes, buf, stream);
        hipGraphExec_t ex = nullptr;
        ck("hipGraphInstantiate", hipGraphInstantiate(&ex, g, nullptr, nullptr, 0));
        std::printf("   captured + instantiated: %.0f -> %.0f MiB free (%+.0f MiB)\n", before, free_mib(dev),
                    free_mib(dev) - before);
        for (int i = 0; i < recaps; ++i) {
            const double b = free_mib(dev);
            if (ex) ck("hipGraphExecDestroy", hipGraphExecDestroy(ex));
            ex = nullptr;
            if (g) ck("hipGraphDestroy", hipGraphDestroy(g));
            g = nullptr;
            g = capture(n_nodes, buf, stream);
            ck("hipGraphInstantiate", hipGraphInstantiate(&ex, g, nullptr, nullptr, 0));
            ck("hipGraphLaunch", hipGraphLaunch(ex, stream));
            std::printf("   recapture %2d: %.0f -> %.0f MiB free (%+.0f)\n", i + 1, b, free_mib(dev), free_mib(dev) - b);
        }
        if (ex) ck("hipGraphExecDestroy", hipGraphExecDestroy(ex));
        if (g) ck("hipGraphDestroy", hipGraphDestroy(g));
        ck("hipDeviceSynchronize", hipDeviceSynchronize());
        std::printf("   all released: %.0f MiB free\n", free_mib(dev));
    }

    // 2. the same, but updating the executable instead of re-instantiating it (ggml's cudaGraphExecUpdate path)
    std::printf("\n2) one graph of %d nodes, %d captures with hipGraphExecUpdate\n", n_nodes, recaps);
    {
        hipGraph_t g = capture(n_nodes, buf, stream);
        hipGraphExec_t ex = nullptr;
        ck("hipGraphInstantiate", hipGraphInstantiate(&ex, g, nullptr, nullptr, 0));
        const double before = free_mib(dev);
        for (int i = 0; i < recaps; ++i) {
            const double b = free_mib(dev);
            if (g) ck("hipGraphDestroy", hipGraphDestroy(g));
            g = capture(n_nodes, buf, stream);
            hipGraphNode_t err_node = nullptr;
            hipGraphExecUpdateResult res = hipGraphExecUpdateSuccess;
            hipError_t e = hipGraphExecUpdate(ex, g, &err_node, &res);
            if (e != hipSuccess || res != hipGraphExecUpdateSuccess)
                std::printf("   update: %s / %d\n", hipGetErrorString(e), (int) res);
            else ck("hipGraphLaunch", hipGraphLaunch(ex, stream));
            std::printf("   update %2d: %.0f -> %.0f MiB free (%+.0f)\n", i + 1, b, free_mib(dev), free_mib(dev) - b);
        }
        std::printf("   graph + exec kept: %.0f -> %.0f MiB free\n", before, free_mib(dev));
        if (g) ck("hipGraphDestroy", hipGraphDestroy(g));
        if (ex) ck("hipGraphExecDestroy", hipGraphExecDestroy(ex));
    }

    // 3. graphs kept alive at the same time (ggml keeps one per graph key, evicting only after 10 s idle)
    std::printf("\n3) %d graphs of %d nodes kept at the same time\n", n_graphs, n_nodes);
    {
        std::vector<hipGraph_t> gs;
        std::vector<hipGraphExec_t> xs;
        const double before = free_mib(dev);
        for (int i = 0; i < n_graphs; ++i) {
            hipGraph_t g = capture(n_nodes, buf, stream);
            hipGraphExec_t ex = nullptr;
            hipError_t e = hipGraphInstantiate(&ex, g, nullptr, nullptr, 0);
            std::printf("   graph %2d: %.0f MiB free%s\n", i + 1, free_mib(dev), e != hipSuccess ? " (!)" : "");
            if (e != hipSuccess) { hipGraphDestroy(g); break; }
            gs.push_back(g);
            xs.push_back(ex);
        }
        std::printf("   kept: %.0f -> %.0f MiB free\n", before, free_mib(dev));
        for (hipGraphExec_t x : xs) hipGraphExecDestroy(x);
        for (hipGraph_t g : gs) hipGraphDestroy(g);
        ck("hipDeviceSynchronize", hipDeviceSynchronize());
        std::printf("   all released: %.0f MiB free\n", free_mib(dev));
    }

    std::printf("\n%s\n", ok ? "every call answered, no signal" : "some call failed (printed above)");
    for (void * p : fill) hipFree(p);
    hipFree(buf);
    hipStreamDestroy(stream);
    return ok ? 0 : 1;
}

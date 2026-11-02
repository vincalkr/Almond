// include/almond/qwen/kernels/window_limits.hpp - how many rows one verify / batch window can hold.
//
// Until 0.2.7 a window held at most 8 rows.  The CUDA and HIP engines now hold up to 16 (four pipelined groups of 16
// slots = 64 slots on a four-card split: docs/BATCHING.md), so every compile-time maximum that follows the window
// (the verifier's per-row arrays, the expert pool's per-token tables, the resident plan's one-thread-per-entry block)
// is derived from ONE number.  It is a macro because the SYCL and Metal builds compile the same shared headers and
// must not change: they get the default, 8.  The CUDA / HIP CMake sets ALMOND_VERIFY_MAX_T from the cache variable
// ALMOND_WINDOW_MAX_T (8 or 16; -DALMOND_WINDOW_MAX_T=8 builds an engine whose windows are the 0.2.7 ones).
//
// The rule every user of the number follows: a window of up to kVerifyNarrowT rows runs EXACTLY the code it ran at 8
// (the same kernel instantiations, the same buffer sizes); the wider code is reached only by a window of 9..16 rows.
#pragma once

#ifndef ALMOND_VERIFY_MAX_T
#define ALMOND_VERIFY_MAX_T 8
#endif

namespace almond::qwen::kernels {

/// The window every 0.2.7 kernel was built for: windows of up to this many rows keep their instantiations.
inline constexpr int kVerifyNarrowT = 8;
/// The most rows one window can hold in this build (8 or 16).
inline constexpr int kWindowMaxT = ALMOND_VERIFY_MAX_T;

static_assert(kWindowMaxT == 8 || kWindowMaxT == 16, "ALMOND_VERIFY_MAX_T is 8 or 16 (the wide kernels are built for 16)");

}  // namespace almond::qwen::kernels

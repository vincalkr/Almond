# Hy4-flash: speed work in progress (handover, 2026-11-20)

Where the Hy4 engine (`src/hy4`, branch `hy4-flash`) stands and what is being tried, so the work can go on in
another session or with another model. Measured numbers and the reasoning behind each step are in
`docs/hy4/GGUF_NOTES.md`; this file is the short state.

## The configuration (`local/hy4-local.json`, `-vision`)

`--hy4 ... --gpus 0,1 -c 262144 --engram-place auto --mtp --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj
--fused-indexer --gather-attn --kv-q8 --ubatch 1024 --prefetch-experts`, env `ALMOND_MMVF_ROWS=1`. Last commits:
0145d9a (reserve fallback, 2,048-token steps), a8acd66 (`HY4_COPY_DEBUG`), cb2ead2 (`--prefetch-experts`),
e6ffc8d (`hy4-local-fast.json`), 3f985c0 (`GRAPH_NODES`), d9c82b1 / 42d1b2b (RDNA3 MMA diagnosis).

Speed now (256K, two RX 7900 XTX, card 1 on PCIe x2, 9900X with DDR5 at ~65 GB/s, greedy, MTP):
short prompt ~24 tok/s; 16K prompt read at ~480 tok/s, then ~21 tok/s; the 66K numbers below are from before
`--prefetch-experts` (it turns itself off there) and were not re-measured.
29 of 42 layers keep their routed experts in VRAM at 256K; 13 run on the CPU.

## Where the time goes

- Decode, one MTP round (~73 ms, 1.76 tokens): GPU kernels ~36 ms, CPU experts of the 13 RAM layers ~21 ms
  (~80 MB a layer for a 2-token verify, at the RAM's bandwidth; more threads and 2 MB pages do not help, and
  `--host-moe` only hides the shared expert of the same layer, ~2,8% of the round - see the section at the end).
- Prompt, 16K tokens in 1,024-token steps, before `--prefetch-experts` (~41 s, `HY4_STEP_STATS=1`): graph compute
  38 s, of which ~25 s are kernels on the two cards and ~10 s are copies: ggml's op offload copies the whole
  routed-expert tensors of the 13 RAM layers (~17 GB) to card 0 at every step, at full bus speed (16 steps x 17 GB in
  ~10 s, ~27 GB/s; a plain hipMemcpy test gives 28.4 GB/s to card 0 and 3.3 GB/s to card 1 on its x2 link, pinned or
  not). Waiting for the Engram rows read from the SSD: 2.4 s. With the Engram table in RAM the wait is gone and
  `--prefetch-experts` hides part of the copies: the prompt is ~34 s (480 tok/s) and compute ~38 s.

## Done in this round (2026-11-20, in the working tree or committed with this file)

- `hc_mix`: the iHC mixes (`hc_*_fn`, F32 [12288, 8]) in prompt steps as streams x fn, transposed (they ran as F32
  GEMMs with 128x128 tiles, 1.2 ms each, ~85 a step): 16K prompt 383 -> 389 tok/s, KL 0.087 (noise level).
- The host builds the causal / SWA masks and the indexer's `mask_base` only when the graph uses them: no measurable
  change (392 tok/s).
- `--pin-experts` (opt-in): the CPU's mapped weights registered as pinned memory: the offload copies become fewer
  and longer but take the same 10 s (395.5 vs 395.4 tok/s).
- `HY4_STEP_STATS=1`: prints the Engram read / wait and graph compute times at exit.
- The prompt-step reserve fallback was broken: a failed `ggml_backend_sched_reserve` left its VRAM buffers allocated
  (`ggml_gallocr_reserve_n_impl` frees a buffer and then fails its realloc, and the other backends keep the failed,
  larger graph's sizes), so every smaller retry had less VRAM and failed too. At 256K `--ubatch 2048` cascaded
  2048 -> 1024 -> 512 -> 256 -> 128 and ended at 128-token steps (1,114 MiB buffers). Each attempt now gets a fresh
  scheduler, freed on failure; 2048 falls back to 1024 in one step (953 / 867 MiB buffers, 1.08 / 0.98 GiB free).
- 2,048-token steps measured: they need 1,775 / 1,733 MiB of compute buffers on the two cards, which fit at 256K only
  with **26 routed-expert layers in VRAM instead of 29** (27 still falls back). Same 16K prompt, 400 generated tokens,
  greedy, one run each: prompt 426.8 -> 486.8 tok/s (+14%), decode 20.96 -> 20.03 tok/s (-4.4%), `HY4_STEP_STATS`
  compute 56.42 -> 52.53 s. Quality: a 4K PPL check of 26 against 29 expert layers at the same 1,024 steps (4,094
  scored rows) gives KL 0.00000 and 100% top-1 agreement; the greedy ids of the random 256K prompt are not reproducible
  even in two runs of the unchanged baseline, so they say nothing. Use `--ubatch 2048 --expert-layers-gpu 26` to try
  it. Left off.
- `--prefetch-experts` (on in the configurations): overlap the op-offload copies with the graph's compute. `HY4_COPY_DEBUG=1`
  showed ggml copies 13 layers x gate/up 297 + down 648 MiB into two reused device regions just before each split. A
  660 MiB staging buffer on a private stream plus the scheduler's copy callback stages the next layer while this one
  computes. The key is that the mmap'd expert pages must be `hipHostRegister`ed (the flag does it): a micro-benchmark
  gives `hipMemcpyAsync` returning in 45 ms from plain mmap against 17 us when registered. 16K prompt, 256K, two
  alternate pairs: 431.3 / 431.3 -> 479.3 / 480.7 tok/s (+11%), `HY4_STEP_STATS` compute 42.0-42.3 -> 38.1-38.2 s;
  4K PPL KL 0.00000 / 100% top-1. Needs ~1 GiB free on card 0 after the compute buffers: at 64K with 31 experts only
  0.38 GiB is free, and the flag turns itself off rather than OOM. See GGUF_NOTES.
- `--ubatch 2048 --expert-layers-gpu 26 --prefetch-experts` (`local/hy4-local-fast.json`, router id `hy4-flash-fast`):
  the two prompt gains add up. Same 16K prompt, 100 generated tokens: prompt 480.3 -> 528.9 tok/s, decode 21.67 ->
  20.62; `HY4_STEP_STATS` compute 35.29 s, all 336 copies staged. Against the configuration before this round (426.8
  tok/s) it is +24% prompt for ~ -2% decode. It moves 3 expert layers to RAM, so it is an alternative, not the default.
- `--ubatch 4096` does not fit: the 4,096-step masks need 3,480 MiB on card 1 (against 1,733 at 2,048) and the reserve
  halves it back to 2,048. Building the 4,096 graph also exhausted the old 24K-node `GRAPH_NODES` context (a ggml
  assert), now 49K so the larger `--ubatch` falls back cleanly instead of crashing. Not faster.

## Next for the prompt

1. Overlap the offload copies with compute: done with `--prefetch-experts` above (+11% prompt, no quality cost). What
   is left is the ~4 s of copies still exposed (a 660 MiB staging holds one layer; two layers would need ~1.3 GiB more
   VRAM), and the 64K case where the flag is off because card 0 is short of memory. The staging is now charged to card
   0's placement budget (`place()`) instead of being allocated after the compute buffers: re-measure the 256K pair of
   numbers and the 64K case with it (docs/hy4/GGUF_NOTES.md, "The engine dies and the cards are loaded again"). The
   256K pair since: 16K prompt 434.3 tok/s and decode 24.54 with `--reserve-mib 2400` and the graph ceilings
   (27 of 42 layers' experts in VRAM, 1.84/2.04 GiB free at load); the 64K case is still not re-measured.
2. Longer prompt steps: the fallback bug is fixed and 2,048 steps are measured above (+14% prompt for -4% decode and
   3 fewer expert layers in VRAM, PPL checked). The remaining reason it is not the default is that tradeoff, not the
   reserve. A way to fit the 1.7 GiB card-1 buffer without giving up expert layers would keep both gains.
3. The Engram table in RAM: <= 6% of the prompt (16K prompt: 2.4 s of 41 s waiting for the rows read from the
   SSD). `--engram-place auto` keeps it on the SSD at 256K because it also counts the file pages of the weights
   copied to VRAM (~70 GiB). Tried on 2026-11-20 with `--engram-place ram`: the machine paged (~2,800-3,300 major
   faults a second, kswapd busy, no progress in 10 minutes), as on 2026-11-18. What was found:
   - the engine's RSS was 108 GB: the file pages of the weights already uploaded to VRAM stayed mapped, although
     `MADV_PAGEOUT` was called on them after the upload;
   - `--pin-experts` does not lock anything on ROCm (`hipHostRegister` leaves `VmLck` / `VmPin` at 0) and `mlock`
     is limited to 8 MB here (`ulimit -l`), so the experts the CPU reads cannot be protected that way.
   The cause: the weights were copied to VRAM straight from the mapping, and ROCm locks the source pages of such a
   copy and keeps them locked, so neither `MADV_PAGEOUT` nor `MADV_DONTNEED` could drop them (43.5 GiB "released",
   the RSS unchanged, 62 GiB of shard 1 mapped). Committed (b55d55e): `set_staged()` copies a weight through
   a 64 MB heap buffer, and `release_pages()` drops its pages from the mapping (`MADV_DONTNEED`, whole pages) and
   the page cache (`posix_fadvise`). RSS after loading (table on the SSD): 65.9 -> 25.2 GB, about the experts the
   CPU reads; the upload got faster too (28.5 -> 15.5 s). Then at 256K, 16K prompt: the table in RAM (read in 4.8 s)
   pages no more than on the SSD (17,637 / 17,886 major faults, the loading's), prompt 393.6 -> 417.8 tok/s (the 2.47
   s of waiting for rows gone), decode 20.72 / 19.49 tok/s (noise). `--engram-place auto` no longer counts the VRAM
   weights' pages: RAM when the table and the CPU's experts fit with 12 GiB to spare (committed).

## The graph churn (2026-11-21): measured, and capped

The 256K run printed 3,728 `CUDA graph warmup reset` against 3,779 `warmup complete` in 148 s: ggml found node
properties that moved on nearly every compute, so the graph was captured again instead of launched. Measured with
`GGML_CUDA_GRAPH_STATS=1` / `GGML_CUDA_GRAPH_DIFF=4`: 8,192 computes, 1,624 captures, 16 keys, and **1,548 of the
1,874 trunk nodes change their `data` pointer between two computes** (615 of 1,604 in the draft; 36/95 and 57/126 in
the small split graphs, `src1-data`). Each ROCm capture/instantiation keeps ~2 MiB on the card (measured with
`tools/hy4/graph-vram.hip`), so the cards ran out of room and the engine segfaulted (docs/hy4/GGUF_NOTES.md, "The
seven deaths of 2026-11-21").

`GGML_CUDA_GRAPH_RECAPTURE_MAX=200` stops capturing the six churning keys. 16K prompt at 256K, greedy, alternate
pairs: prompt 445.3 / 469.3 -> 469.4 / 469.2 tok/s, decode 23.28 / 22.77 -> 23.37 / 23.75 (at 400 tokens, 22.00 /
22.85 -> 23.03). So the prompt is unchanged and the decode +2-4%: the capture was worth less than the compute it
skipped, and 4,000-token runs now finish. On in the three hy4 configurations.

The way to get the rest back looked like making the graph stable - bucket `n_pool` / `n_top` / `n_sel` the way `n_kv` is
bucketed to 256, so the nodes keep still and one capture is reused. Tried the same evening (`local/hy4-graph-stability-
check.sh`; the change itself, `HY4_POOL_PAD`, is parked in `local/hy4-pool-pad-wip.patch` and not in the tree): it
changes **nothing**. The graph counters are the same to the unit (`12,288 computes: 480 launched, 80 captured, 11,728
direct, 16 keys`, the same 30 graphs switched off by the ceilings) and the speed is the same - prompt 467.0 against
477.3 / 477.0 tok/s, decode 25.99 against 24.98 / 25.64, the differences the usual VM noise. Switching the graphs off
altogether (`GGML_CUDA_DISABLE_GRAPHS=1`, so all 12,288 computes run node by node) gives prompt 477.6 and decode
25.20: the same speed again, and on the prompt the graph path costs ~2% rather than paying for itself.

Why: the diff said it already - what moves between two computes is the nodes' `data` **pointers** (1,548 of 1,874), not
their shapes, and a key switched off by the ceiling stays switched off. Bucketing the shapes cannot stop pointer churn.
So the graph is not where the decode time is, and there is nothing to win making it stable; the time is the experts of
the 13 RAM layers reading ~1.0 GB a round at the RAM's bandwidth, and the host work around them.

## Next candidates (estimates, not measured)

- Prompt: the sliding layers' attention (`flash_attn_tile<576,512>`, ~5.6 s of a 16K prompt) was tried and reverted.
  On RDNA3 `ggml_cuda_get_best_fattn_kernel` only takes the MMA F16 path for `Q->ne[0] <= 256` (MFMA is CDNA-only);
  the 576/512 MMA instances and RDNA config cases are compiled, but widening the gate behind an env var crashed
  reproducibly at the first graph compute (ROCm `pthread_create() failed`). Needs a proper kernel fix. The IQ2_XXS
  expert GEMMs (`mul_mat_q<16>`) are next in size.
- Decode: a fresh rocprofv3 trace of ~19 tokens at 256K shows 31,436 kernel launches and only 417 ms of kernel time
  in a 1,162 ms window (64% gaps), but the cost is not the small launches: a fused `rms_norm + weight` op (new ggml op
  `GGML_OP_HY4_RMS_NORM_MUL`, one block per row) cut the graph from 4,961 to 4,704 nodes and changed nothing
  (476.9 -> 479.7 tok/s, KL 0.035), so it was reverted. The kernel time is `mul_mat_vec_q` Q8_0 114 ms,
  `flash_attn_tile<576,512>` 56 ms, `mul_mat_vec_q_moe` IQ2_XXS 69 ms, `mul_mat_vec_bf16_rows` 20 ms. Fewer kernels is
  the lever, but the norm fusion shows the small ops are not what the round costs.
- Decode: the length of the MTP chain is a knob now, `--mtp-adaptive <pct>`: the draft model's own top-1 probability
  for the guess it just made is the estimate of that guess being accepted, so the chain stops where the running
  estimate drops under the bar, and every fourth round drafts anyway so a quiet stretch cannot lock the gate shut.
  `HY4_MTP_STATS=1` prints the histogram of the chains offered and of where they die, which is how the bar gets set.
  Built 2026-11-22 and **not measured**: neither the speed, nor the KL 0.00000 that greedy ought to show against a
  run without the gate.  Off by default (`mtp_adaptive_pct = 0`) until it is.
- With a quality cost (measured, off): `--ram-requant q3_k` +5-7% at KL 0.012.
- Not worth it (measured): hot-expert cache in the ~2 GB left (top 60 experts of a RAM layer cover 14.7% of its
  picks); a smaller `--reserve-mib` (an expert layer is ~1.3 GB); DFly2 (`--dfly`, ~2 tokens a round, never faster
  than MTP); `--down-requant`; wave64; one card; a fused `rms_norm + weight` kernel (nodes 4,961 -> 4,704, no speed
  change, KL 0.035: the small launches are not the decode cost); a **stable graph** (tried the same evening, worth
  nothing: the graph churn is pointers, and the graphs switched off cost nothing); the **dense Q8_0 mat-vec at verify
  widths** - `build-hip-fused/bench_hy4_q8` (`local/hy4-q8-bench.sh`) puts both kernels on the shapes the engine really
  multiplies, six weights each so the 48 MB of L2 cannot flatter the repeat, and at 2-4 tokens a step both reach
  ~700 GB/s: at the memory roof. Those 6-7 ms a round of Q8_0 are bytes, not kernels (docs/hy4/GGUF_NOTES.md, "The
  same for Q8_0 rows").

### What Strata's Hy4 engine does, read 2026-11-21 (MIT source, not measured here)

Strata 0.2.18 (2026-11-21) runs Hy4-Flash natively with its own engine, `strata-hy4`, and the source is public:
`engine/src/hy4flash/run/` is a hand-written HIP forward of the 42 layers (`hy4_graph.hip.h`, 155 kB) with its own
kernels, indexer, Engram reader and speculator. It is a different pack (SQ-64/96/128, its own `S_K` expert codes, hot
experts on the card) and its quoted speeds are for RTX PRO 6000 / 5090 / RX 9070 XT / Strix Halo / M5 Max - no RX 7900
XTX number, so not comparable with our 23.4-24.5 tok/s. Three mechanisms in it are where we have a wall:

- **The KV grows instead of being reserved.** `Runner`: "starts at `kv_alloc_pos` positions (16,384 by default on
  discrete cards) in VRAM, grows in 16K increments in VRAM up to `kv_dev_limit` (or max_ctx), and spills cleanly to
  pinned host memory if VRAM runs out". We reserve the whole 3.59 GiB of `--kv-q8` for 262,144 rows at load, every
  session, whatever the session reads: that room is 2,7 GiB of experts we never get back at 16K (`place()` charges
  it before choosing, and the room says so). A KV that starts at 16K rows, grows by 16K and spills the coldest rows
  to pinned host is the same shape of win as one more expert layer, ~+2% decode, and it removes the "no room at long
  context" class for good.
- **CPU experts as jobs, with the file reads hidden behind the compute of earlier waves** (`hy4_cpu_experts.h`): a
  pinned-thread pool, one arena for the resident expert bodies (huge-page advised, mlock), activations quantized once
  to `Q8_1_G64`, and for the cold experts a wave schedule - "wave 0 = resident experts + cold cache hits, then the
  missed cold experts, read by reader threads with large preads (O_DIRECT by default) ... issued when the job started
  and overlap the earlier waves' compute" - with a 256 MB LRU slot cache. Their stats carry exactly the number we
  would need: `phase_s[5]` (plan / quantize z / gate-up / down / combine) and `cold_wait_s`, "seconds the pool waited
  for reads (**not hidden** behind the compute of earlier waves)". Our 13 RAM layers are the same problem with no
  overlap at all: 21 ms of a 73 ms round sits in front of the GPU.
- **Speculation is a tree with the attention and the indexer told about it.** `TreeSpec`: "up to 8 nodes rooted at
  anchor node 0, topological order `parent[i] < i`" with an ancestor table, handed to the MLA (`to_mla_pos()`) and to
  the indexer (`to_idx_tree()`); `Tree-8 CUDA Graph capture and replay across all 38 recomputed rows, bitwise-exact
  with ungraphed execution` (release notes). Alongside the MTP draft there is a **suffix drafter**
  (`spec/suffix_drafter.cpp`, ~120 lines: a hash of the last three tokens -> the WAYS most recent positions, extend
  the longest match, propose what followed) and a policy (`spec/draft_policy.cpp`) that *measures* the round time per
  window size and context bucket and drafts only where acceptance against that measured cost pays, with priors by
  match length (0.75 / 0.88 / 0.93 / 0.96) and forced probes so a cheap window is never priced out forever. We have
  neither the tree nor the free drafter, and our window size is a static flag.

What does **not** transfer: their `S_K`/AVX-512 VNNI expert kernels (our CPU experts already run AVX-512 and read at
48,4 GB/s against the ~65 GB/s the RAM gives, so the bus is the limit, not the kernel); their rank tables (the
`.strata` file stores each layer's 512 experts hot-to-cold and `make_mask_rank_range` puts "the coldest 400 of every
layer in RAM, the hottest 112 on the GPU" - with LatentMoE's routing nearly uniform over 512 experts, our measured
top-60-of-a-layer covering 14.7% of the picks says per-expert ranks buy little where whole layers are bought instead);
and their Tree-8 at 38 rows, which costs ~2,4x our 2-row check on this box (the row cost table above).

## The decode pipeline: where CPU expert time can and cannot hide (2026-11-21)

The idea was "run the CPU experts beside the GPU". The dependency graph says what that can mean. One decode round at
256K is 73 ms: GPU kernels ~36, the experts of the 13 RAM layers ~21, host/sync ~16, and it yields 1.76 tokens
(24.1 tok/s). Inside a layer the chain is strict - attention, router, routed experts (CPU for a RAM layer), then
`rup` - so the expert time of layer L is on the critical path of the whole round: no stream, event or second card can
move it, because layer L+1's attention needs its answer.

What **can** run beside it, in the same layer, with no dependency at all: the **shared expert**. The layer's answer is
`out = moe + shexp` (`hy4_moe.cpp`: `sgate/sup/sdown` with `shexp`, added at the end), and `shexp` does not read
anything the routed experts produce. The mechanism for using the host for the routed part already exists: `--host-moe`
(`ggml-cuda/host-moe.cu`) - the GPU stream publishes `x/ids/wts` into mapped pinned memory and then runs
`host_moe_wait_copy`, a kernel that spins on the flag and copies the answer back, with no scheduler split and no
copies. It measured as not helping, and the reason is in its position: the publish+wait node is built *before* the
shared expert (`hy4_engine.cpp`: `ggml_host_moe` then `ffn_routed_up`, then `ffn_gate_shexp/up_shexp/down_shexp`), so
the stream stops there and the independent branch never starts beside the host. Splitting it into publish-early /
wait-late is ~40 lines of ggml and would give the overlap.

But the ceiling of that overlap is small, and the arithmetic is worth writing down before the lines are written: what
the host can hide behind is only the GPU work *of the same layer*, because the next layer's attention needs this
layer's answer. A RAM layer's own GPU work is its shared expert - 3 x 3072 x 13824 = 127 M weights, 135 MB in Q8_0,
~0,27 ms a layer at the card's bandwidth - while its host expert work is 31 experts x 2,54 MB = 79 MB at 48,4 GB/s =
1,63 ms. So the window hides ~0,27 of 1,63 ms a layer: 13 layers x 0,27 = **3,5 ms of the 73, +4,8% decode**, not the
20 first hoped. `rdown`/`rup` add ~0,02 ms a layer to the window. Worth having after the byte wins, not before them.

The bytes are the wall: a round reads 31 experts x 2,54 MB x 13 layers = **1,02 GB of expert weights from RAM**, which
at 48,4 GB/s is exactly the 21 ms measured, while the same 31 picks x 29 layers on the cards cost 3,6 ms of kernel
time (`mul_mat_vec_q_moe` IQ2_XXS, 69 ms per 19 tokens). What lowers the bytes, in the order of what is measured:
`--ram-requant q3_k` on `ffn_down_exps` of the RAM layers (+5-7%, KL 0.012, already in the
engine); one expert layer more in VRAM (1,3 GB, ~+2%, and the room is in the KV: Strata's growable/spilling KV, our
parked `--ipool-f16`, a shorter `-c` profile); and only then the publish/wait split above.

Why the other kind of overlap - more rows through the trunk, so the CPU work hides behind the GPU work of another row
- does not pay on this box, measured: the cost of a check with k rows (64K, mean of 12) is 55.0 / 68.6 / 90.8 / 100.4 /
119.0 / 132.0 / 152.4 / 167.1 ms for k = 1..8, so a k-row round has to reach 1.76 x cost(k)/68.6 tokens to break even:
2.33 at 3 rows, 2.58 at 4, 4.29 at 8. The reason the curve is steep is the same wall: 16 picks a token over 512
experts of ~2.5 MB, so the *union* of experts a round must read is 31 per layer at 2 rows (79 MB a layer, 1.00 GiB
over the 13 RAM layers), 60 at 4 rows (1.94 GiB) and 113 at 8 rows (3.65 GiB) - 3.6x the CPU bytes for 4x the rows.
Wide speculation only pays if those bytes hide: the same two tracks first, the tree after.

Plan, in the order the arithmetic gives: (1) the byte wins - `--ram-requant q3_k` measured, and the room for one or
two expert layers more (a KV that grows and spills, `--ipool-f16`, a shorter `-c` where the session does not need
256K); (2) the publish-early / wait-late split of `--host-moe`, ~40 lines of ggml, +4,8% by this arithmetic and
+2,8% of the round measured, below; (3) only then a wider verify, because with the CPU bytes still at 1,00 GiB a round
its break-even is 2,33 tokens a round at 3 rows and 4,29 at 8, which no drafter reaches here (DFly2 with 5 guesses got
2,31).

## The publish / wait split of `--host-moe`, built and measured (2026-11-22)

Built as planned: `GGML_OP_HOST_MOE_PUBLISH` (ggml.h, ggml.c - the op, its names, `GGML_OP_COUNT` 104 -> 105),
`ggml_host_moe_publish(c, x, ids, w, bell, tag)` publishing and returning a one-element I32 **ticket** that carries the
bell pointer and the layer tag in its `op_params`, and `ggml_host_moe(c, ticket, x, bell)` waiting on that ticket - so
the wait is ordered after whatever is built between the two, and `x` rides along only for the shape. In ggml-cuda
`ggml_cuda_op_host_moe_publish` publishes (`x->ne[1]` rows) and `ggml_cuda_op_host_moe` waits and copies
(`dst->src[1]->ne[1]`); the kernels are the ones already there. `hy4_engine.cpp ffn()` takes the new order for a host
layer only: publish, then the shared expert (`ffn_gate_shexp / up_shexp / down_shexp`) inside the window, then the
wait; the graph of a run **without** `--host-moe` is node for node what it was. Kept as
`tools/v41/ggml/host-moe-publish.patch` (on top of `hy4-long-context.patch`, which carries the op itself).

Measured with `local/hy4-host-moe-check.sh pair`: the configuration of record above, 16,384 real-text ids (the repo's
own docs and sources, `local/hy4-text-ids.py`) at 256K, greedy, 1,500 tokens, two alternate pairs.

| arm | prompt tok/s | decode tok/s | rounds | tokens a round | ms a round |
|---|---|---|---|---|---|
| as shipped | 464.3 / 447.4 | 20.79 / 21.59 | 795 / 795 | 1.89 / 1.89 | 90.7 / 87.4 |
| `--host-moe`, split | 471.5 / 471.1 | 21.76 / 20.99 | 789 / 834 | 1.90 / 1.80 | 87.4 / 85.7 |

- The prompt: 455.8 -> 471.3 tok/s, **+3.4%**, both pairs the same way.
- The decode **round**: 89.0 -> 86.5 ms, **-2.8%** - but the two arms overlap (as shipped 90.7 / 87.4, split
  87.4 / 85.7), so that is the whole signal, not more.
- The decode **token**: a wash (20.98 against 20.98 tok/s). Greedy with MTP forks between runs of the *same* config -
  two baseline runs already differ from their 11th token (`... 12 2935 77 2411` against `... 12 77505 63 13525`) - so
  the acceptance (1.89 / 1.89 against 1.90 / 1.80) moves the token count as much as the speed does. The round is the
  unit that means something here; `tok/s` on 1,500 tokens is not a two-sample statistic.
- What the host reports is the prediction: 1,579 / 1,501 us of expert compute per call, 13 host layers = ~20 ms of the
  ~88 ms round, against the 21 ms estimated above.

So the window is real and smaller than its ceiling: ~2,5 ms of the round hidden where the arithmetic said at most
3,5. It stays opt-in and out of the configuration of record, for two reasons: 2,8% of a round is next to the spread of
a single pair, and `--host-moe` is not quality-neutral in the way the plan assumed - the host computes the products
with the CPU MoE kernels, not the GPU ones, and "the same text" (GGUF_NOTES, *Where the MoE runs*) cannot be checked
at this length because the runs fork by themselves. If it ever goes into the configuration it needs the KL
(`--ppl-save` / `--ppl-ref`), which has not been run for it. The byte wins stay first: `--ram-requant q3_k` is measured
at +5-7% and still is not in the configuration.

**What has NOT been tested - resume here (2026-11-22).**

- **Quality: nothing.** No `--ppl-save` / `--ppl-ref` run, no KL, no perplexity for the host expert path. The
  "the same text" sentence in GGUF_NOTES (*Where the MoE runs*) was never actually checked, and the host computes the
  products with the CPU MoE kernels, not the GPU ones. **Done the same day** - see "`--host-moe` re-checked against a
  reference that repeats" below: the path is bit-identical, KL 0.00000.

### Run, and the answer is: the path does not repeat (2026-11-22)

`local/hy4-host-moe-quality-check.sh` (four arms, 8,190 scored rows each, `-c 8192` with
`--expert-layers-gpu 29` so the same 13 layers sit on the CPU as the 256K configuration puts them) and
`local/hy4-determinism-check.sh` (five arms, each pair saving and reading its own reference):

| arm | KL against the reference | top-1 | ppl |
|---|---:|---:|---:|
| reference, save (chunk 1 / 2) | - | - | 35.7327 (31.8600 / 35.7327) |
| **the same flags again** (control) | **0.14530** | 88.17% | 35.5282 |
| `--host-moe`, graphs as usual | 0.13567 | 88.18% | 36.0442 |
| `--host-moe`, `GGML_CUDA_DISABLE_GRAPHS=1` | 0.13454 | 88.32% | 35.8203 |

| same flags again, in other settings | KL | top-1 | ppl |
|---|---:|---:|---:|
| graphs off, prefetch on | 0.12347 | 88.42% | 36.0012 against 35.4946 |
| graphs on, no prefetch | 0.13881 | 88.46% | 35.9419 against 36.4138 |
| graphs off, no prefetch (against the graphs-on run) | 0.13234 | 88.57% | 35.6168 |

**Two runs of one configuration differ by KL ~0.12 - 0.14 and perplexity +-1.3%.** So:

- `--host-moe` changes nothing measurable - 0.1357 and 0.1345 are at or under the control's own 0.1453 - and the same
  for graphs on against off. But that is a **floor, not a proof**: a 0.01-0.05 effect, including a ring slot that
  desynchronizes under replay, sits under it and this test cannot see it.
- The drift is inside one forward pass, not in the windowing: chunk 1 alone is 31.8600 against 31.2560.

### Found it: `--fused-indexer` (2026-11-22, `local/hy4-drift-hunt.sh`)

The hunt takes one ingredient out at a time and runs the rest twice; a pair that prints KL 0.00000 names the ingredient:

| pair, same flags twice | KL | top-1 | ppl of the two runs |
|---|---:|---:|---|
| `--engram-place ram` | 0.14632 | 88.50% | 35.6307 / 35.8373 |
| `--engram-place ssd` | 0.12915 | 88.89% | 35.3498 / 35.3803 |
| **without `--fused-indexer`** | **0.00000** | **100.00%** | 35.7561 / 35.7561 |
| without `--fused-hc` | 0.12958 | 88.50% | 36.1732 / 36.0996 |

**The fused lightning indexer is what does not repeat.** With `--fused-indexer` out, two runs are the same to the last
digit of perplexity; everything else tried is excluded - the Engram's placement (RAM or SSD, so the page cache and the
63.4 GB table are not it), `--fused-hc` (the one ggml op here with grid-wide atomics - not it), the graphs and the
expert prefetch. What the two paths are: `hy4_engine.cpp:871-878` scores the pool with `ggml_lightning_indexer` (one
kernel, no `[n_pool, nt, NH]` tensor - that tensor is 5 GB of compute buffer at 256K, which is why the flag is in the
configuration) against `hy4_engine.cpp:880-886` building `relu(q . k)` per head with `mul_mat` and adding
`in.pool_mask` in F32. Two differences to look at, in order: the fused path casts the mask to **F16**
(`ggml_cast(c, in.pool_mask, GGML_TYPE_F16)`) while the unfused one adds it as **F32**; and under HIP the op takes the
plain-float kernel (`lightning-indexer.cu:618-640` keeps the WMMA instance for NVIDIA only) which tiles `n_kv` in
blocks of 32 and reduces across 64 heads in groups. The kernel's own accumulators are initialized
(`score_k[K_VECS_PER_WARP] = { 0.0f }` at :313, out-of-range lanes zeroed at :293/308), so the next step is not a read
of the kernel: it is to dump the scores and the selected pool indices for two runs and see whether the *scores* move
or the *picks* do - about ten lines in `hy4_engine.cpp` beside `route_stats` (which already dumps the expert picks,
`HY4_ROUTE_STATS`).

- **What this costs the documentation:** less than it looked at the time. The correction, same day: the KL tables of
  `docs/hy4/GGUF_NOTES.md` (lines 455-470 and 612-624) were taken at a **2,048 to 4,096 token window**, and their own
  "the same configuration again" row reads **0.000 / 100%** - with a pool that short the pick does not land on a pile
  of ties, so those numbers stand. The drift is a **long-pool** effect: at 8,192 rows, and at the 256K configuration,
  the same run does not repeat. Everything measured there has to be taken again - which from today it can be, because
  the pick is deterministic by default. For the one number that was still open, `--ram-requant q3_k` against the same
  run without it:

| the requant against a reference that repeats | KL | top-1 | ppl |
|---|---:|---:|---:|
| 4,096-token window (2026-11-20, `docs/hy4/GGUF_NOTES.md`) | 0.012 | 95.8% | 12.70 -> 12.79 |
| 8,190 rows of `local/hy4-text-ids.py` text, pick deterministic | **0.02874** | 92.81% | 35.5641 -> **35.4973** |

  Same direction: a little less agreement at the long window, perplexity slightly *better*. It is still not shipped -
  the user's call, 2026-11-22 - but it is a verified number now, not one under a floor.

### The cause is the pick, not the score kernel: `ggml_top_k` (same day, `local/hy4-indexer-sort-check.sh`)

`--fused-indexer` builds its picks with `ggml_top_k`, the unfused path with `ggml_argsort_top_k` (`hy4_engine.cpp:888-891`).
The CUDA/HIP `ggml_top_k` is a radix select: the keys that tie with the k-th are placed by
`atomicAdd(&state->equal_count, 1)` (`ggml-cuda/top-k.cu:201,204`), so **which** of the tied keys wins is up to the
scheduler. Ties are not an edge case here: every dead pool slot carries the same mask value, so the boundary of the
top-k sits on a pile of equal scores. `--indexer-argsort` (new flag) keeps the fused score kernel and pays a real sort
for the picks instead:

| two runs, same flags | KL | top-1 | ppl |
|---|---:|---:|---:|
| `--fused-indexer`, picks through `ggml_top_k` | 0.12667 | 88.51% | 35.4566 / 35.6838 |
| `--fused-indexer --indexer-argsort` | **0.00000** | **100.00%** | 35.5641 / 35.5641 |

What it costs at the configuration of record (16K real text, 256K, greedy, 750 tokens): prompt **471.0 -> 470.3 tok/s**
(the sort is free there; the prompt is the tight measurement, 0.15%), decode **83.6 -> 84.3 ms a round, +0.8%** - which
is under the ~2% the clock itself jitters along an identical path (see "How to measure"), so call it "not measurable, at
worst about a per cent" rather than a measured 0.8%. The per-token figure (-2.1%, 21.22 -> 20.78 tok/s)
is not a cost: it is the drift itself, seen from the other side - with `ggml_top_k` the two runs pick differently, so
they draft and accept differently (422 rounds at 1.77 tokens against 428 at 1.75).

So the shipped configuration can be made reproducible for under 1% of a round. **Flipped to the default the same day,
at the user's go-ahead**: the pick is sorted unless `--indexer-topk` buys the old behaviour back, and `--indexer-argsort`
is still accepted as a no-op for old command lines. Verified with the shipped flags and no indexer flag at all, twice:
**ppl 35.5641 both runs, KL 0.00000, top-1 100.00%** (`bash local/hy4-indexer-sort-check.sh default`). The published
record numbers (prompt 471-477 tok/s, decode 21.0-21.6) were taken with `ggml_top_k`; with the sorted pick the same
configuration gives 470.3 prompt and 84.3 against 83.6 ms a round.

The same hazard is live in two other places, neither measured: `src/ds41/ds41_dense.cpp:741,744` selects the routed
experts of the dense DeepSeek-V4.1 path with `ggml_top_k` over the router scores, and `src/qwen35/qwen35.cpp:1057`
takes its sampling candidates with it. Ties are rarer on raw logits and on sigmoid+bias router scores than on a pool
mask, but the shape of the failure is the same - the pick, and so the answer, is not the pick of the next run.

### The ties were still taken in thread order: `top_k_radix_ties` (2026-11-25)

The index ramp did not make every run repeat: 8K of the 16K document (`--ppl`, `-c 8192`, two cards, reserve 1700,
Engram table in RAM) gave ppl 41.1452, 41.6646, 41.6577 or 40.9848 from one run to the next, with or without
`--prefetch-experts`. Dropping one option at a time, two runs each: only without `--fused-indexer` (41.1164 four
times out of four) did the runs agree. The ramp, `idx x 1e-15`, is below a float's resolution for any score that is
not 0, so two pools with the same non-zero score still tie, and the radix top-k kernel placed the keys equal to the
k-th with `atomicAdd`: which of them made the cut followed the order the threads ran in.

The vendored kernel (`top-k.cu`, `tools/v41/ggml/topk-ties.patch`) now places those keys in a separate pass, one block
a row, lowest column first; the keys above the k-th still go in with `atomicAdd`, in any order, which is the same set
every run (the indexer sorts its picks afterwards). Measured with the same 8K:

| Run | ppl, 4-6 runs |
|---|---|
| before, `--fused-indexer` | 41.1452 / 41.6646 / 41.6577 / 40.9848 |
| after, without `--prefetch-experts` | 41.1452 six times out of six |
| after, with `--prefetch-experts` | 41.1452 three times, 41.1477 once |

With `--prefetch-experts` a routed expert is computed from its staged copy on card 0 or on the CPU depending on
when the copy lands, and the two do not round the same way, so that run-to-run difference stays; compare quality
without it. Speed at 256K (16K prompt, 500 tokens, reserve 1700, Engram in RAM): prompt 457.5 tok/s against 458.0,
85-87 ms a round of generation against 86 ms before (the runs took another token path, 1.73 tokens a round, so the
tok/s do not compare directly).

### `--host-moe` re-checked against a reference that repeats (2026-11-22, `local/hy4-host-moe-quality-check.sh` again)

With the pool pick sorted (the default now), the same four arms print:

| arm | KL against the reference | top-1 | ppl |
|---|---:|---:|---:|
| reference, save | - | - | 35.5641 |
| **the same flags again** (control) | **0.00000** | **100.00%** | 35.5641 |
| `--host-moe`, graphs as usual | **0.00000** | **100.00%** | 35.5641 |
| `--host-moe`, `GGML_CUDA_DISABLE_GRAPHS=1` | **0.00000** | **100.00%** | 35.5641 |

So the host expert path is **bit-identical** to the scheduler's GPU -> CPU -> GPU split over 8,190 rows and 13 CPU
layers, and the ring slot a captured graph bakes in did not desynchronize from the host thread over those rows - the
risk the section above could not see. The quality objection to `--host-moe` is gone: there is nothing to distinguish.

What is left is speed, and the speed figures in this file were taken under the drift: the publish/wait split was
-2.8% a round at the configuration of record and +2.3% at `hy4-local-fast`, with the prompt +3.4%. Those want taking
again before the flag is decided either way: `bash local/hy4-host-moe-check.sh pair` (~18 min), and the plain
`--host-moe` against the shipped path the same harness measures with its other mode.

**Taken again the same day, and the difference is gone.** Three runs at the configuration of record (`pair`'s `BASE`,
16K real text, 256K, greedy, 1000 tokens, one arm thrown away to warm the Engram):

| run | arm | rounds | tokens a round | decode a round |
|---|---|---:|---:|---:|
| 1 | `--host-moe` | 557 | 1.80 | 89.8 ms |
| 2 | as shipped | 560 | 1.79 | 88.9 ms |
| 3 | `--host-moe` | 557 | 1.80 | **87.7 ms** |

The two `--host-moe` runs followed **exactly** the same path - 557 rounds, 443 drafts accepted, the same text - and
still differ by 2.3% in wall clock. So the machine's own jitter is ~2% and the flag's effect is inside it: neither the
-2.8% a round nor the +3.4% prompt of yesterday's pair, which was the pick's drift, can be told apart from nothing.
`--host-moe` therefore stays off on the speed question, with a much better reason than it had yesterday: it is
bit-identical and it is not measurably faster. (The prompt of these runs, 310-359 tok/s, is not comparable with the
471 of the rest of this file: `pair`'s `BASE` uses `--engram-place auto` and the Engram pages come from the SSD; keep
`ram` for prompt numbers.)

### The other configuration, measured: the split loses there

`HM_UBATCH=2048 HM_EXTRA="--expert-layers-gpu 26" bash local/hy4-host-moe-check.sh pair` - `local/hy4-local-fast.json`
at 16K real text, 256K, greedy, 1,500 tokens. This pair is clean: the MTP trajectory is the same to the token in both
arms (826 rounds, 673 of 826 accepted, 1.81 tokens a round either way).

| arm | prompt tok/s | decode tok/s | ms a round |
|---|---:|---:|---:|
| as shipped | 518.7 | 20.13 | 90.1 |
| `--host-moe`, split | 526.4 | 19.69 | **92.2** |

**+2.3% per round**, the opposite sign of the configuration of record. There the split put 13 layers on the host, here
16 (`host-moe: 13,216 calls, 1550.6 us` each = 24.8 ms of the round): the host work grows with the layers while the
window - the same layer's shared expert, 0.27 ms each - does not, and the wait lands in front of `ffn_routed_up` on
three more layers. The prompt still gains (+1.5%). Net: `--host-moe` is not a feature to turn on, in either
configuration; it is a mechanism kept because it is the only way to put host work beside card work at all.
- **The graph path: run, and it cannot be read at this resolution** - see the tables below. The delicate part remains
  real: the publish node inside a captured graph runs on the host at capture time only, and the ring slot the wait
  reads is **baked into the graph** (`ggml-cuda/host-moe.cu`, and `st.graph_slot` advances per replay to match) - a
  desync between the host's `seq` and the replay's slot gives a stale answer with no crash and no error, only wrong
  tokens. What would still show it: a check that does not go through the model's numerics, e.g. the engine writing the
  `seq` it published and the `flag` the wait saw per layer and comparing them (cheap, in `hy4_engine.cpp`), or the same
  comparison against a deterministic baseline once the run-to-run drift below is understood.
- **Other configurations**: `hy4-local-fast.json` measured, and the split loses there (below). Still untested: the 64K
  case where `--prefetch-experts` switches itself off, and DFly2 with `--host-moe`.
- **No unit test covers it**: `src/hy4/test_hy4_moe.cpp` and the other `test_hy4_*` do not know `--host-moe`, which
  needs mapped pinned memory and only exists in the engine path.
- **The `--serve` path** (the router, session save/restore, `local/hy4-crash-check.sh`'s fini deaths) with hy4 +
  `--host-moe`: never run.

### The host → card copy path, measured on its own (2026-11-22, `src/hy4/bench_hy4_copy.cu`)

The last item of the queue was Avalon's claim that its staged host copies only got fast once they were taken off
SDMA and the legacy stream (`docs/ENGINES_WATCH.md`).  Our own note on the same subject was two numbers from a plain
`hipMemcpy` test: 28.4 GB/s to card 0, 3.3 GB/s to card 1, its x2 uplink.  `bench_hy4_copy` settles it without
loading a model - it allocates its own buffers, so the running engine is left alone:

```
$ ./build-hip-fused/bench_hy4_copy 1 64 mapped      # card 1, 64 MiB, 12 reps, HIP events
  memcpy/legacy    20.29 ms     3.3 GB/s
  memcpy/stream    20.29 ms     3.3 GB/s
  kernel/registered     5.22 ms    12.9 GB/s   (il contenuto non torna)
  kernel/hostalloc      5.21 ms    12.9 GB/s   (il contenuto non torna)
  kernel/fresh-512MB    5.22 ms    12.9 GB/s   (il contenuto non torna)
```

| | card 0 (x16) | card 1 (x2) |
|---|---|---|
| `hipMemcpyAsync`, legacy stream | 28.0 - 28.7 GB/s | 3.3 GB/s |
| the same on a non-blocking stream of its own | 28.2 - 28.7 GB/s | 3.3 GB/s |
| a copy kernel reading host memory the card addresses | 102 - 111 GB/s, **wrong bytes** | 12.5 - 12.9 GB/s, **wrong bytes** |

The two real paths are at the link's roof and the stream they run on changes nothing, at 4, 16, 64, 96 and 128 MiB
alike: the expert staging of `--prefetch-experts`, measured in the engine at ~27 GB/s, is already there.  So Avalon's
first half - get it off the legacy stream - buys nothing here, and its second half cannot be tried honestly, because
the kernel that reads registered host memory does not work on this machine.  It launches without an error, it times
at rates no PCIe generation of this box carries, and it is wrong: each arm fills the buffer it reads from with its
own tag and reads the destination back, and every kernel arm fails that check.  Read the host pointer directly
instead of the alias from `hipHostGetDevicePointer` and the process dies of a GPU page fault (exit 139).

That check is the reason the program exists.  The first version compared every arm's destination against the same
pattern, so the bytes the previous arm had already left there made the next arm look correct: a kernel copy that
moves nothing was measured at 107 GB/s and "passed".  The lesson is the general one - a copy that moves nothing is
the fastest copy there is, and a speed number without a content check is not a measurement.

**Closed**: the copies are not where the decode round's time can be won.  What is left in that direction is moving
fewer bytes (the expert placement of `docs/hy4/VRAM_SIZING.md`) or overlapping them, not copying them differently.

What the bench costs to run, since that cost fell on someone else: with `mapped` it pins seven times the size it
copies - one buffer per arm plus four windows - and pinned pages are not reclaimable.  At 16:24 on 2026-11-22, with
the engines already at 121 of 122 GB, such a run was enough for `almond-engine` to invoke the OOM killer, and the
kernel killed `qwen-host` (anon-rss 57.4 GB).  It now reads `MemAvailable` before allocating and refuses with exit 3
unless the box has room for its own pinned bytes plus 8 GiB.  The kernel arms also take the process down inside the
runtime when they are given a host pointer instead of the alias (`general protection fault ... in
libamdhip64.so.7.14.60850`, four times at 15:57): they stay in the file as the trap, not as a path.

## How to measure

The knobs that repeat on this machine, and the ones that turned out to be jitter: the engine flags are in `almond.sh`'s
`HY4_ENGINE_ARGS`, and the environment in `almond.sh.env`.

- Build: `./build.sh` (a new `.cu` in `ggml-cuda` needs `cmake third_party/llama.cpp/build-hip` first).
- Speed: `build-hip-fused/almond-engine <config args> --temp 0 -n 400 --prompt-ids IDS` (prints prompt and
  generated tok/s and the MTP rounds). A VM on this machine adds noise: compare alternate runs.
- **Wall clock still jitters ~2% even along an identical path** (2026-11-22: two `--host-moe` runs with the same 557
  rounds, the same 443 accepted drafts and the same text took 50.03 and 48.87 s for 1,000 tokens, 2.3% apart). The pick
  made the *path* repeat; it did not make the clock repeat. Anything under ~2-3% per round needs repeats or a different
  metric - the prompt is the tight one (471.0 against 470.3, 0.15%), and a kernel's own cost is better read from
  `rocprofv3` than from a rate.
- An environment-knob A/B (does it change the numbers, what is it worth): `bash local/hy4-knob-ab.sh NAME=value -n 500`
  - a perplexity pass for the KL against the shipped configuration, then the speed pair at the configuration of record.
- Quality: `--ppl FILE -c 4096` with `--ppl-save` once and `--ppl-ref` after (KL and top-1 against it); a change of
  kernels alone moves KL by ~0.1 on this 2-bit model. `HY4_PPL_BINS=N` prints perplexity per N positions.
- Long contexts: `--prompt-ids @FILE` reads the ids from a file (a 66K prompt does not fit in argv).
- A real-text prompt: `python3 local/hy4-text-ids.py 16384 /tmp/hy4-p16k-text.ids` - the repo's own docs and sources
  through `tools/almond_tokenizer.py`, byte for byte reproducible. It needs `.venv/bin/python` (the system python has
  no `regex`).
- Harnesses in `local/`: `hy4-host-moe-check.sh [pair]` (as shipped against `--host-moe`, the configuration of record,
  one command per arm), `hy4-host-moe-quality-check.sh`, `hy4-determinism-check.sh`, `hy4-drift-hunt.sh`,
  `hy4-indexer-sort-check.sh [default]`, `hy4-graph-stability-check.sh`, `hy4-prefetch-check.sh`, `hy4-crash-check.sh`.
  All of them want the whole machine and refuse to run while an `almond-engine --serve` holds the cards: unload the
  model through the router (`/free`, or the CLI's `free_gpu`) rather than stopping the serve stack, which takes the
  user's session with it. **Since 2026-11-22 the user asked that the engine not be taken at all unless they say so** -
  ask first before a run that would interrupt them.
- For a decode A/B, the unit is the **round** (decode seconds / MTP rounds), not tok/s: greedy forks between runs of
  the same config, so the acceptance and with it the token count moves by more than a few percent of speed. Two
  samples per arm show nothing below ~5%.
- A **pseudo-random** id prompt is a bad decode harness: on `/tmp/hy4-p16k.ids` (16,384 random ids, regenerated
  2026-11-21) the model settles on repeating one token and the MTP draft hits 749 of 751 - 2.00 tokens a round, decode
  25.99 tok/s where real text gives 23.4-24.5. Use it for the prompt and for memory/room measurements, never for
  acceptance or decode.
- Do not kill the engine with `pkill -f` patterns (they matched the agent's own shell); stop background loops
  with their task, not only the engine, or the loop starts the next run.

## The decode regression of b07acb2, and its fix (2026-11-23)

b07acb2 made the indexer's pick a full sort (`ggml_argsort_top_k` over every pool, n_past / 4 of them, for every
token of 12 layers) to make the engine repeat itself. It costs nothing at 2K and grows with the context: real-text
prompt, 300 greedy tokens, the configuration's flags at 256K, ms per MTP round at 32K (rounds vary with acceptance, so
round time is the comparison): bedf452 76.4 / 77.5, the branch's head 81.2 / 81.2 (+5%), `--indexer-topk` 77.7 /
77.2. In long sessions this is where 22-24 tok/s became ~19.

Fix: the ties are broken by the pool's index (score - idx x 1e-15, below a float's resolution for any score that is
not exactly 0, so it decides only among equal ones - the ReLU's zeros above all), `ggml_top_k` picks the set, and only
those n_top picks are sorted by index (a 511-wide argsort), so the set and the attention's summation order are the
same on every run. Measured: 78.2 / 78.0 ms a round at 32K; two runs over the 32K text in 4,096-token windows give the
same perplexity (42.5465) and KL 0.00000 / top-1 100.00% over 16,376 rows; against `--indexer-topk` KL 0.12563,
top-1 88.40% (the distance between two runs of that unrepeatable path).

## CPU + PCIe "miss-sharing" in decode, measured (2026-11-23): not worth it as built

Idea (Strata 0.2.20's "async PCIe miss-sharing"): a RAM layer's experts split between the CPU and card 0, which gets
its share over PCIe x16 (28 GB/s) and computes it while the CPU does the rest, so the two bandwidths add up.
`--pcie-share N` (with `--host-moe`, off): the host thread hands up to N of a step's unique experts to a worker that
copies their gate/up/down slices from the mapped file to staging tensors on card 0 (a ggml backend of its own) and
runs them there; the CPU's slots for those experts are given weight 0 and another of the token's ids.

2K real-text prompt, 300 greedy tokens, the configuration at 256K, one run each (`HY4_HM_STATS=1`):

| N | CPU per call | card 0 per call | tok/s | ms a round |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,584 us | - | 23.77 | 78.6 |
| 4 | 1,537 us | 1,202 us | 23.45 | 77.5 |
| 8 | 1,681 us | 1,650 us | 21.31 | 79.1 |
| 12 | 2,078 us | 2,026 us | 22.30 | 83.1 |

Why: card 0's share is slow (4 experts in 1.2 ms against ~0.5 estimated: 12 synchronous copies from the pageable
mapping, ~100 us each of launch and staging), and the CPU barely saves anything (-3% for 4 of ~31 experts instead of
-13%: the swapped slots still compute, and from 8 experts on the copies compete with the CPU for the RAM's bandwidth).
Doing it well would need batched asynchronous copies and a CPU product that skips the slots; the ceiling was +5-8%,
and the bandwidth contention seen here eats into it. Left in the tree, off.

## A crash with the Engram table in RAM: --reserve-mib 1400 -> 1700 (2026-11-25)

With `--engram-place ram` at 256K (16K real-text prompt, `--temp 0 -n 500`), every run ended in a segmentation fault in
the first generation steps, the old binary as the new one. Under gdb: `hipGraphExecUpdate` inside
`ggml_cuda_graph_evaluate_and_capture`, from `Hy4Engine::Impl::step`. The start log had warned: 275 MiB left on the
cards with the compute buffers in (card 0 0.26 GiB, card 1 0.76 GiB); the runtime's graph update finds no room there.
The engine runs on ROCm 7.14 here.

| `--reserve-mib` | expert layers in VRAM | free after buffers (card 0 / 1) | prompt | generation |
|---|---|---|---|---|
| 1400 | 29 of 42 | 0.26 / 0.76 GiB | crash | crash |
| 1400, `--min-free-mib 600` | 29 | not reachable at any prompt step | crash | crash |
| 1700 | 28 | 1.66 / 0.58 GiB | 458.0 tok/s | 20.76 tok/s (1.79 tokens a round) |
| 2000 | 26 | 1.66 / 3.00 GiB | 435.0 tok/s | 19.94 tok/s (1.82) |

The configurations take 1700: the smallest of these that ran, one expert layer fewer on the cards than 1400.

### Warp sums through DPP: no gain, not kept (2026-11-25)

`warp_reduce_sum` / `warp_reduce_max` in the vendored ggml with DPP row moves and `v_permlanex16` in place of
`ds_bpermute` on RDNA3 (ENGINES_WATCH, sweep of 2026-11-24). With the repeatable top-k above, without
`--prefetch-experts`: KL from the build without it 0.00000, top-1 100.00% over 8190 rows (8K of the 16K document).
Speed at 256K (16K prompt, 500 tokens, reserve 1700, Engram in RAM, both on the same token path, 1.81 tokens a round),
two runs each:

| Build | Prompt tok/s | Generation tok/s |
|---|---|---|
| ds_bpermute | 426.3 / 424.1 | 21.13 / 21.16 |
| DPP | 424.9 / 427.4 | 21.07 / 21.15 |

The same within the noise, so the change was not kept.

### `hy4-local-fast.json` removed (2026-11-25)

To keep at most two configurations per model, and since a chat waits on generation more than on the prompt: the fast
configuration's `--ubatch 2048 --expert-layers-gpu 26` gave the prompt +10% (480.3 -> 528.9 tok/s, 16K) for -5% of
generation (21.67 -> 20.62), moving three expert layers to RAM. `local/hy4-local.json` (and its `-vision` twin) stay.


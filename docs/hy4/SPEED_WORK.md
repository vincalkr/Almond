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
  (~80 MB a layer for a 2-token verify, at the RAM's bandwidth; more threads, 2 MB pages and `--host-moe` do not
  help), syncs / launch gaps / host work ~16 ms.
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
- With a quality cost (measured, off): `--ram-requant q3_k` +5-7% at KL 0.012.
- Not worth it (measured): hot-expert cache in the ~2 GB left (top 60 experts of a RAM layer cover 14.7% of its
  picks); a smaller `--reserve-mib` (an expert layer is ~1.3 GB); DFly2 (`--dfly`, ~2 tokens a round, never faster
  than MTP); `--down-requant`; wave64; one card; a fused `rms_norm + weight` kernel (nodes 4,961 -> 4,704, no speed
  change, KL 0.035: the small launches are not the decode cost); a **stable graph** (tried the same evening, worth
  nothing: the graph churn is pointers, and the graphs switched off cost nothing).

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

## How to measure

- Build: `./build.sh` (a new `.cu` in `ggml-cuda` needs `cmake third_party/llama.cpp/build-hip` first).
- Speed: `build-hip-fused/almond-engine <config args> --temp 0 -n 400 --prompt-ids IDS` (prints prompt and
  generated tok/s and the MTP rounds). A VM on this machine adds noise: compare alternate runs.
- Quality: `--ppl FILE -c 4096` with `--ppl-save` once and `--ppl-ref` after (KL and top-1 against it); a change of
  kernels alone moves KL by ~0.1 on this 2-bit model. `HY4_PPL_BINS=N` prints perplexity per N positions.
- Long contexts: `--prompt-ids @FILE` reads the ids from a file (a 66K prompt does not fit in argv).
- A **pseudo-random** id prompt is a bad decode harness: on `/tmp/hy4-p16k.ids` (16,384 random ids, regenerated
  2026-11-21) the model settles on repeating one token and the MTP draft hits 749 of 751 - 2.00 tokens a round, decode
  25.99 tok/s where real text gives 23.4-24.5. Use it for the prompt and for memory/room measurements, never for
  acceptance or decode.
- Do not kill the engine with `pkill -f` patterns (they matched the agent's own shell); stop background loops
  with their task, not only the engine, or the loop starts the next run.

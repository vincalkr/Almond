# Hy4-flash: speed work in progress (handover, 2026-11-20)

Where the Hy4 engine (`src/hy4`, branch `hy4-flash`) stands and what is being tried, so the work can go on in
another session or with another model. Measured numbers and the reasoning behind each step are in
`docs/hy4/GGUF_NOTES.md`; this file is the short state.

## The configuration (`local/hy4-local.json`, `-vision`)

`--hy4 ... --gpus 0,1 -c 262144 --engram-place auto --mtp --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj
--fused-indexer --gather-attn --kv-q8 --ubatch 1024 --prefetch-experts`, env `ALMOND_MMVF_ROWS=1`. Last commits:
7e5817a (indexer fix, gather attention, 256K), 261e5fa (`--kv-q8`), 5a34724 (`--ram-requant`, off).

Speed now (256K, two RX 7900 XTX, card 1 on PCIe x2, 9900X with DDR5 at ~65 GB/s, greedy, MTP):
short prompt ~24 tok/s; 16K prompt read at ~480 tok/s, then ~19-22 tok/s; 66K prompt ~377 tok/s, then ~16-19.
29 of 42 layers keep their routed experts in VRAM at 256K; 13 run on the CPU.

## Where the time goes

- Decode, one MTP round (~73 ms, 1.76 tokens): GPU kernels ~36 ms, CPU experts of the 13 RAM layers ~21 ms
  (~80 MB a layer for a 2-token verify, at the RAM's bandwidth; more threads, 2 MB pages and `--host-moe` do not
  help), syncs / launch gaps / host work ~16 ms.
- Prompt, 16K tokens in 1,024-token steps (~41 s, `HY4_STEP_STATS=1`): graph compute 38 s, of which ~25 s are kernels
  on the two cards and ~10 s are copies: ggml's op offload copies the whole routed-expert tensors of the 13 RAM layers
  (~17 GB) to card 0 at every step, at full bus speed (16 steps x 17 GB in ~10 s, ~27 GB/s; a plain hipMemcpy test
  gives 28.4 GB/s to card 0 and 3.3 GB/s to card 1 on its x2 link, pinned or not). Waiting for the Engram rows
  read from the SSD: 2.4 s.

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
   VRAM), and the 64K case where the flag is off because card 0 is short of memory.
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
   the RSS unchanged, 62 GiB of shard 1 mapped). Fixed in the working tree: `set_staged()` copies a weight through
   a 64 MB heap buffer, and `release_pages()` drops its pages from the mapping (`MADV_DONTNEED`, whole pages) and
   the page cache (`posix_fadvise`). RSS after loading (table on the SSD): 65.9 -> 25.2 GB, about the experts the
   CPU reads; the upload got faster too (28.5 -> 15.5 s). Then at 256K, 16K prompt: the table in RAM (read in 4.8 s)
   pages no more than on the SSD (17,637 / 17,886 major faults, the loading's), prompt 393.6 -> 417.8 tok/s (the 2.47
   s of waiting for rows gone), decode 20.72 / 19.49 tok/s (noise). `--engram-place auto` no longer counts the VRAM
   weights' pages: RAM when the table and the CPU's experts fit with 12 GiB to spare (committed).

## Next candidates (estimates, not measured)

- Prompt: the sliding layers' attention (`flash_attn_tile<576,512>`, ~5.6 s of a 16K prompt): a WMMA kernel for
  576/512 heads, maybe -50% of it, +10-15% prompt. The IQ2_XXS expert GEMMs (`mul_mat_q<16>`) are next in size.
- Decode: fewer kernels per layer (fusing norms, iHC, routing), maybe -5..-8 ms a round, +12-20%.
- With a quality cost (measured, off): `--ram-requant q3_k` +5-7% at KL 0.012.
- Not worth it (measured): hot-expert cache in the ~2 GB left (top 60 experts of a RAM layer cover 14.7% of its
  picks); a smaller `--reserve-mib` (an expert layer is ~1.3 GB); DFly2 (`--dfly`, ~2 tokens a round, never faster
  than MTP); `--down-requant`; wave64; one card.

## How to measure

- Build: `./build.sh` (a new `.cu` in `ggml-cuda` needs `cmake third_party/llama.cpp/build-hip` first).
- Speed: `build-hip-fused/almond-engine <config args> --temp 0 -n 400 --prompt-ids IDS` (prints prompt and
  generated tok/s and the MTP rounds). A VM on this machine adds noise: compare alternate runs.
- Quality: `--ppl FILE -c 4096` with `--ppl-save` once and `--ppl-ref` after (KL and top-1 against it); a change of
  kernels alone moves KL by ~0.1 on this 2-bit model. `HY4_PPL_BINS=N` prints perplexity per N positions.
- Long contexts: `--prompt-ids @FILE` reads the ids from a file (a 66K prompt does not fit in argv).
- Do not kill the engine with `pkill -f` patterns (they matched the agent's own shell); stop background loops
  with their task, not only the engine, or the loop starts the next run.

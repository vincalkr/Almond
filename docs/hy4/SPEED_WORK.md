# Hy4-flash: speed work in progress (handover, 2026-11-20)

Where the Hy4 engine (`src/hy4`, branch `hy4-flash`) stands and what is being tried, so the work can go on in
another session or with another model. Measured numbers and the reasoning behind each step are in
`docs/hy4/GGUF_NOTES.md`; this file is the short state.

## The configuration (`local/hy4-local.json`, `-vision`)

`--hy4 ... --gpus 0,1 -c 262144 --engram-place auto --mtp --fused-hc --dense-q8 --reserve-mib 1400 --fused-proj
--fused-indexer --gather-attn --kv-q8 --ubatch 1024`, env `ALMOND_MMVF_ROWS=1`. Last commits: 7e5817a (indexer fix,
gather attention, 256K), 261e5fa (`--kv-q8`), 5a34724 (`--ram-requant`, off).

Speed now (256K, two RX 7900 XTX, card 1 on PCIe x2, 9900X with DDR5 at ~65 GB/s, greedy, MTP):
short prompt ~24 tok/s; 16K prompt read at ~390 tok/s, then ~18-19 tok/s; 66K prompt ~377 tok/s, then ~16-19.
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

## Next for the prompt

1. Overlap the offload copies with compute (copy layer L+1's RAM experts while layer L computes, as Basalt does):
   up to ~10 s of 41 s at 16K, ~+30%. Needs a prefetch inside or around ggml's scheduler.
2. Longer prompt steps (2,048 tokens): half the copies, ~+12%; the compute buffers do not fit at 256K today, and the
   halving retry after a failed reserve never recovers (even 64 tokens fail afterwards: likely a buffer left
   allocated by the failed attempt).
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

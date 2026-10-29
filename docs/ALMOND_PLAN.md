# Almond: the plan

Almond is ds41e (../ds41e) made multi-model. Today it runs what ds41e runs, renamed (`strata` -> `almond` in the code,
`STRATA_*` -> `ALMOND_*` in the environment, `include/almond/`); it builds with `./build.sh` and the server tests pass.
`models/` is a symlink to `../ds41e/models`. Nothing below is built yet; nothing below has a measured number yet.

## Rules that stay

- The default path computes every expert the router picks (no route bias, no dropped experts, no lossy skip). A lossy
  option may exist, opt-in, with its KL against the full model written down.
- A new feature is opt-in; a default changes only where it measured better, with the numbers.
- The box: two RX 7900 XTX (ROCm/HIP, gfx1100), Ryzen 9 9900X, 122 GiB RAM, NVMe. NVIDIA-only or Metal-only code is a
  reference to port, not something to link.

## Models, in order

1. **DeepSeek-V4.1-Flash-1017** (what ds41e runs, GSQ-RCO IQ2_XXS pack): unchanged, the reference.
2. **GLM-5.3-Flash** (321 B, 18 B active): first new model.
3. **Qwen3.8-Flash-Next** (125 B MoE): second new model. Smaller than the two above: on 48 GB of VRAM plus RAM most of
   it is in VRAM, so it should be far faster.

## Where each piece comes from

All five sources are MIT. Their code is read and ported by hand; nothing downloaded is run.

| source | what it has | what Almond takes |
| --- | --- | --- |
| ds41e (ours) | DeepSeek-V4.1 on HIP: VRAM / peer card / RAM arena / NVMe tiers, exact routing, Engram, vision, server | the base: tiers, kernels, server |
| Strata (../Strata) | DeepSeek-V4.1 engine (`src/dsv41`), Qwen 3.5/3.6 (`src/qwen35`), many kernels, CUDA/HIP/SYCL/Metal | kernels and pack tools where they beat ours; Qwen's layer code as a reference |
| project-maya (../project-maya) | GLM-5.3 on HIP (`src/core/glm_*`: fast path, prefill, DFlash, MTP, CPU lane), MiniMax M3.1, Qwen-3.8-Flash-Next (GGUF, text) | GLM's model and attention code (it already runs on this box, Maya-M / Maya-S24); its `TRUNK_KEEP` / `MISS_SKIP` only as opt-in lossy options |
| Xolotl (github.com/Indras-Mirror/Xolotl, a Strata fork) | DeepSeek-V4.1, GLM-5.3, Qwen3.8-Flash-Next, Qwen3.8-27B; NVIDIA only | Qwen3.8-Flash-Next's model code and its numbers as a target; not its GLM defaults (`--route-bias 0.25` + soft prune: lossy) |
| ds4 (github.com/antirez/ds4) | DeepSeek V4 Flash/PRO, GLM 5.2/5.3, Qwen3.8 Flash Next; Metal and CUDA, ROCm on Strix Halo; exact routing | a small, readable reference for each architecture (attention, router, MTP) to check ours against |
| llama.cpp (`third_party/llama.cpp/ggml`) | ggml and its HIP kernels (MMQ, MMVQ, flash attention) | stays the kernel base; GGUF readers for the new architectures |

Which piece is "the best" for a model is decided by a measurement on this box, not by the source: each candidate is
built behind a flag and compared (tok/s, prompt tok/s, KL / top-1 against the reference) before it becomes a default.

## Phases

1. **Rename** (done): Strata -> Almond, builds, server tests.
2. **A model interface in the engine.** Split `tools/v41/engine` into what every MoE model shares (the expert tiers:
   VRAM cache, peer card, RAM arena, NVMe reads; the pool; the sampler; the serve loop) and what is DeepSeek's (MLA /
   compressed attention, HC mixes, Engram, router). A `Model` with `load`, `prefill`, `decode_step`, `route(layer)`,
   `expert_layout`; DeepSeek becomes the first implementation. Gate: the same tok/s as ds41e (3 alternated runs,
   Promessi Sposi and 17K) and the same forced-token argmax.
   **Done (2026-10-29):** `tools/v41/engine/moe_trunk.hpp` (`MoeTrunk`, `TrunkConfig`, `TrunkInfo`, `make_trunk`);
   `Ds41Dense` implements it; the driver and the conversation stores use only the interface.  Measured with
   `local/bench-v41/ab-ds41e.sh` (ab-hip-spin.sh's chat flags, power-tune on, forced on the reference streams):

   | arm | 17K, 128 tokens (tok/s) | Promessi Sposi, 300 tokens (tok/s) | argmax diffs (17K / Sposi) |
   | --- | --- | --- | --- |
   | ds41e | 19.18 / 19.15 / 19.09 | 23.44 / 23.52 / 23.50 | 1 / 2 |
   | Almond | 19.02 / 19.00 / 19.12 | 23.14 / 23.03 / 23.49 | 1 / 0-2 |

   Within the run-to-run noise (Almond -0.5% / -1.1% on the means, its third round equal); the same diffs.
3. **GLM-5.3-Flash.** Port project-maya's GLM layers onto the interface, experts through Almond's tiers. Read GGUF
   (GSQ-RCO 3.0bit, 117.5 GB; Maya-M for comparison). Gate: KL / top-1 against project-maya on the same pack, then speed
   against project-maya on this box.
   **First version done (2026-10-29), exact, not yet fast:** `tools/v41/engine/glm_trunk.cpp` (llama.cpp's glm5-next
   graph as a MoeTrunk), the experts through Ds41MoeTier (`Ds41FileBlobs` reads llama.cpp's stacked expert tensors;
   Q4_K down added to the grouped GPU kernel). Checked on GSQ-RCO 3.0bit (`pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF`,
   117.5 GB, `models/glm53-gsq-rco`) against llama.cpp (c3df4b7, HIP, `llama-perplexity -c 512 --chunks 1`) on the
   first 512 tokens of `docs/ds4/V41_EXACT_FAST.md`, the second half scored as llama-perplexity scores it:

   | engine | PPL | top-1 |
   | --- | ---: | ---: |
   | llama.cpp | 7.1997 (+/- 1.13) | |
   | Almond | 7.2401 | 0.574 |

   Per-layer sums (`ALMOND_GLM_DEBUG=1` against `llama-eval-callback`, 20 tokens) agree within ~1% through layer 7.
   Prompt 12.1 tok/s (chunked, --prefill-chunk 64, arena 20 GiB, graphs rebuilt per call, no VRAM cache during the
   prompt).
   **Cached decode graphs (2026-10-29):** each layer's graphs are built once and kept (the MLA layers per 256 cells
   of context), allocated once in their own buffers. Exact: the same PPL to four decimals (10.5520 over all 511
   tokens, top-1 0.546). Decode, 128-token prompt, 64 tokens, arena 60 GiB, peer card 22 GiB: 2.91 -> 3.58 tok/s;
   the prompt 2.1 -> 15.0 tok/s. Still slow because only 8.7% of the experts hit VRAM (no GLM rank file) and the peer
   card computes no rows.
   **The routed sum on the card (2026-10-29):** `GlmTrunk::routed_dev()` gives the tier a device tensor, so decode
   takes the tier's `run_to_dev` path, the one where the peer card computes its experts (it computed none before).
   PPL the same (10.5520). Decode, same command: 3.58 -> 6.04 tok/s (peer 4964 rows over 64 tokens; file reads
   5973 -> 1009).
   **Fused HC ops (2026-10-29):** the hyper-connection mixes, Sinkhorn and post through ggml's DeepSeek-V4 ops
   (`ggml_dsv4_hc_comb/pre/post`, `ggml_dsv41_hc_mixes`; llama.cpp's glm5-next has the same fused path), instead of
   ~300 stock-op kernels a layer. `ALMOND_GLM_HC_COMPOSITE=1` = the stock ops. PPL 10.5520 -> 10.5312 (top-1 0.546 ->
   0.548: the fused kernels round differently). `ALMOND_GLM_PROF=1`, 130 tokens: a KDA layer's graph 1.81 -> 1.26 ms,
   an MLA layer's 2.21 -> 1.32 ms, finish 0.16 -> 0.06 ms; decode 6.45 -> 7.77 tok/s (both with the profiler on).
   An expert rank from the 17K prompt's routes (`tools/v41/engine/glm_rank.py`, `local/glm-rank.bin`: 30 GiB cover
   87% of the selections) measured no faster (5.62 against 6.04/5.70 tok/s without), so it is not used: GLM's routing
   follows the text. The rank's first 35 GiB cover 28.5% of the doc prompt's routes (a uniform pick: ~35%), and the
   per-expert counts of the two texts (a C++ review, 17K tokens; this markdown, 192) correlate at -0.05 to -0.22 by layer.
   Where a token goes now (130 tokens, `ALMOND_DEC_PROF=1`): attn_router 41-53 ms, the expert tier 63-78 ms (CPU
   experts ~31 ms: 210 of the 336 picks a token; 11% VRAM hits), finish 2.6 ms. The trunk's GPU work is ~0.55 ms a
   layer (rocprofv3) against ~1.0-1.3 ms of wall time: the rest is the dispatch gap of ~80 kernels a layer; ggml's
   HIP graphs change nothing (`GGML_CUDA_DISABLE_GRAPHS=1`: 1.03 vs 1.05 ms a KDA layer). Missing: the layer-major batch, the next-layer prediction, cached graphs, conversation reuse, the serve
   path's chat template, project-maya's packs (`glm5next`: their GGUF names one key prefix differently, read already).
   **project-maya's engine, vendored (2026-10-29): the GLM path that runs.** The trunk above has a floor: ~45 ms a
   token of attention and router alone, where project-maya's whole token on this box is ~28 ms (Maya-M, 82% VRAM
   hits, `NOTES-AMD-2x7900.md`). Its engine reads our GGUF as it is, through its pack (`local/glm-pack.sh`, 1 GB, a few
   seconds), so it is vendored instead of ported piece by piece: `engines/maya` (project-maya v1.1.37, 7ee8f27, MIT,
   with its pinned ggml), built by `./build.sh` into `build-maya-hip/maya`, served by Almond's server through
   `tools/glm/serve/serve_glm.py` (GLM's `<arg_key>` tool calls translated into the server's form, as serve_ds41.py
   does for DSML) and `./run.sh --model glm` (`local/glm-local.json`, 128K context). It splits the layers over the two
   cards (0-22 / 23-44), each card's experts in its own LRU pool (~4300 slots), a pinned RAM tier and a CPU lane.
   Exact routing (no `TRUNK_KEEP`). Same GGUF, the 512-token doc, 510 positions scored:

   | engine | PPL | top-1 | second half |
   | --- | ---: | ---: | ---: |
   | llama.cpp | | | 7.1997 |
   | Almond trunk (above) | 10.3334 | 0.547 | 7.2401 |
   | project-maya's engine | 10.3788 | 0.545 | 7.0003 |

   (KL from the Almond trunk 0.063 mean, 0.035 median; argmax agree 88.4%.) Decode, 128-token prompt, 130 tokens,
   greedy, a cold start: 7.77 -> 20.71 tok/s (VRAM hits 11% -> 66%, CPU experts 210 -> 97 a token). Through the
   server (`./run.sh --model glm`, power-tune on, thinking off, 700 tokens of code three times): 21.1, 25.1, 25.4
   tok/s; a short Italian answer 24.7 then 28.8 (85% hits). Tool calls (OpenAI, streamed, the tool's result back),
   Anthropic thinking: checked by hand. The Almond trunk (`--trunk glm` in ds41_generate) stays as the reference.
   **MTP drafts (2026-10-29, on by default):** the GSQ-RCO GGUF has no draft block (nextn 0); project-maya's Maya-M
   has one (blk.45, NextN), and the engine takes a block from another quant of the same model (`STRATA_GLM_MTP_GGUF`:
   it reads the trunk's hidden state and the embedding). `local/glm-mtp-extract.py` copies blk.45 out of Maya-M's third
   shard into `models/glm53-gsq-rco/GLM-5.3-Flash-Maya-M-NextN.gguf` (3.6 GB, 29 tensors), and `local/glm-local.json`
   names it. The second card drafts, the first verifies. Server, power-tune on, temperature 0, thinking off, 600
   tokens each, a fresh start each arm (three times the same code prompt, an Italian history prompt, an English
   explanation):

   | arm | code 1 | code 2 | code 3 | Italian | English | drafts accepted |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | no drafts | 23.6 | 26.3 | 26.3 | 27.3 | 25.6 | |
   | Maya-M's shard | 29.8 | 36.1 | 36.2 | 33.5 | 28.5 | 82.1% |
   | the extracted block | 24.7 | 35.7 | 35.8 | 32.3 | 27.8 | 81.3% |
   | the same, default config | 29.9 | 36.4 | 35.5 | 33.4 | 29.8 | 82.6% |

   (tok/s.) Greedy output is not token for token the one without drafts: on the 128-token doc the two part at token
   12, "64" against "128" (a near tie: the verify window's batched kernels round differently from the one-token
   path, as the prompt path does). Tool calls checked again with drafts on (3 of 3, the integer argument typed).
4. **Qwen3.8-Flash-Next**, the same way (Xolotl, Strata and ds4 as references).
5. **The server stays in Python** (`serve/`, ~9,900 lines: OpenAI / Anthropic API, streaming, chat templates, tool
   calls, web app, MCP, the engine's process). A token's time is all in the engine; the server only moves text and
   JSON, so rewriting it would change no tok/s. Each new model adds its chat template and tool-call format here. If one
   day Almond is to ship as a single binary with no `.venv`, a Go server can replace it endpoint by endpoint over the
   same stdin/stdout protocol (Go over C++: far less code for HTTP, SSE, JSON and templates), with the Python tests
   turned into Go tests on the same fixtures.

## Open questions

- Engine binary: one `almond` with `--model deepseek41|glm53|qwen38`, or one binary per model sharing libraries.
- GLM pack: GSQ-RCO 3.0bit GGUF directly, or project-maya's pack format.

# Inkling 2 Mini in Almond: plan

Status 2026-11-23, later: phases 1 and 3 done on llama.cpp's own graph (`almond-inkling`), with Unsloth's UD-Q4_K_XL
chosen by the user instead of IQ4_NL (half the KL at the same size, measured). The speed estimate below did not hold:
30 tok/s measured against 120 estimated, because the decode is bound by small kernels, not by bytes. What was built
and measured is in `NOTES.md`. The rest of this file is the plan as first written. Every number below
carries its source; numbers marked "estimate" are arithmetic from those sources and have to be replaced by
measurements.

## 1. The model

Sources: model card https://huggingface.co/thinkingmachines/Inkling-2-Mini, its `config.json`, the announcement
https://thinkingmachines.ai/news/inkling-2-mini/, the Transformers page
https://huggingface.co/docs/transformers/en/model_doc/inkling2_loop.

- Released 2026-11-20 by Thinking Machines Lab, Apache 2.0 with an acceptable use policy. Text and image in, text out.
- Looped MoE: 24 stored layers, hidden 2304. Layers 6-17 run 3 times (`loop_config`: core 6..18, 3 passes, gate
  rank 64), so a token goes through 48 layer visits. Passes 2 and 3 read the previous pass through a gate.
- 26.4B parameters in total, 3.4B active, 5.4B "effective" active once the loop is counted (announcement).
- MoE: 128 routed experts, top 8, no shared expert, expert FFN 1152, sigmoid routing with gate bias, route scale 8,
  soft caps on gate (4) and up (25) (config).
- Attention: 24 query heads, 8 KV heads, head dim 128. 20 of the 24 layers use a 1024-token window; layers 5, 11, 17
  and 23 are full attention (derived from `local_layer_ids`). Full-attention layers add a relative position bias and
  log scaling; every layer has an attention sink and a short convolution (kernel 4). Each pass keeps its own KV and
  convolution cache (48 cache layers, Transformers page).
- Vocabulary 201,024 (200,058 used). Context 1,048,576 tokens.
- Draft head: `mtp.safetensors` (636 MB), a 3-layer block-drafter (`block_size` 8, Markov head) that the GGUF and
  Strata pages call DSpark. It drafts 8 tokens per call.
- Vision: 23-layer ViT (hidden 1024, patch 20), 318.3M parameters (Strata page).
- Native weights: NVFP4 experts in layers 0-19, MXFP8 in 20-23; one 19.5 GB safetensors file (model card).
- Chat format: `<|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>...
  <|end_message|><|content_model_end_sampling|>`, `reasoning_effort` none..max or 0.0-0.99 (Transformers page,
  ggml-org card). Default effort is "high" per ggml-org and "xhigh (0.9)" per AtomicChat: open. The tool-call format
  is not written on any page read; it is in `chat_template.jinja` (6.87 kB) and llama.cpp PR #32833 (inkling parser).

KV cache per token (derived, not stated): 8 full-attention visits (layers 5 and 23 once, 11 and 17 three times) x 8
KV heads x 128 x 2 (K,V) = 16,384 values = 32 KiB at BF16, about 17 KiB at q8_0. The 40 window visits are capped at
1024 tokens: about 160 MiB per sequence at BF16. 256K context at q8_0: about 4.3 GiB.

### Packs that exist

| Pack | Size | KL vs BF16 | Source |
|---|---|---|---|
| ggml-org Q8_0 | 27.8 GB | 0.0030 | https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF |
| ggml-org IQ4_NL (4-bit experts 0-19, rest Q8_0) | 17.6 GB | 0.0095 | same |
| AtomicChat AD-Q6_K | 23.79 GB | 0.0037 | https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF |
| AtomicChat AD-Q5_K | 19.07 GB | 0.0052 | same |
| AtomicChat AD-Q4_K | 16.53 GB | 0.0080 | same |
| AtomicChat AD-IQ4_XS | 15.00 GB | 0.0101 | same |
| AtomicChat AD-IQ3_S | 13.34 GB | 0.0205 | same |
| AtomicChat DSpark drafter Q8_0 | 481 MiB | - | same |
| Strata SQ-15 / SQ-20 | 14.35 / 19.35 GB | top-1 96.0% / 100% | https://huggingface.co/strata-engine/Inkling-2-Mini-Strata |
| mmproj Q8_0 / BF16 | 0.35 / 0.64 GB | - | ggml-org, AtomicChat |

The whole model fits in VRAM. This is the main difference from every other Almond MoE: no RAM or NVMe tier is needed.

## 2. What the listed projects offer

- llama.cpp (https://github.com/ggml-org/llama.cpp): `inkling2_loop` merged (#32704, 2026-11-21), chat parser
  (#32833), merges fix (#32799). Open: DSpark fixes (#32773), looped layers first in device memory (#32777), MoE gather
  at 28% of prefill GPU time (#32846), garbled output with IQ4_NL + cpu-moe (#32830), CUDA assert past ~79K (#32804).
  Our vendored copy (`third_party/llama.cpp`, 9128b28) has no Inkling code yet.
- Strata (https://github.com/Niko1221/Strata): 0.2.20 runs Inkling with a 4-tier planner and the DSpark drafter;
  DSpark fix 6.5 -> 61.9 tok/s on 12 GB cards; 0.2.19: 250.3 tok/s RTX 5090, 99.0-106.5 Radeon AI PRO R9700. No 7900
  XTX figure. RDNA3 WMMA prefill and DP4A decode kernels.
- Xolotl (https://github.com/Indras-Mirror/Xolotl): NVIDIA-only Strata fork. Inkling on one RTX 4090: 127 tok/s
  decode, 136 with DSpark, 3,500-4,190 prefill, 512K context on one card, k8v4 KV. The nearest reference for one 24 GB
  card.
- project-maya (https://github.com/mw00/project-maya): Inkling as an experimental preview (UD-Q4_K_XL), no speed
  figures; useful 2x 7900 XTX numbers for other models.
- antirez/ds4 (https://github.com/antirez/ds4): Metal/CUDA engine, no Inkling, no gfx1100. MTP/DSpark as opt-in is the
  same policy as ours.
- elpis-on-7900xtx (https://github.com/thefahien/elpis-on-7900xtx): Qwen3.8-27B EXL3 + DFlash2 on one 7900 XTX, code
  178 tok/s vs 85 llama.cpp Vulkan. Shows block drafters pay most on code.
- exllamav3-rocm (https://github.com/phoenixhaxor/exllamav3-rocm): EXL3 and WMMA attention on gfx1100; Qwen3.8-27B 3.5
  bpw DFlash2 157 tok/s at 2K, 107 at 99K. No MoE, no Inkling.
- exllamav3-rocm-moe (https://github.com/phoenixhaxor/exllamav3-rocm-moe): hot experts in VRAM, AVX-512 CPU experts.
  Not needed when everything fits.
- Avalon (https://github.com/freriadoc/Avalon): llama.cpp fork, MTP with adaptive draft length, q8_0/q4_0 KV, QSA
  sparse attention. Adaptive draft length applies to DSpark here.
- Speedy (https://github.com/DevDock-AI/Speedy): NVIDIA Strata fork, two-card layer split; AMD untested. No Inkling.
- llama.cpp-rdna3-int4 (https://github.com/SILVERTHRONE32/llama.cpp-rdna3-int4): INT4 WMMA GEMMs on gfx1100,
  Qwen3.5-4B prefill 11,012 tok/s on a 7900 XT. Candidate for Inkling prefill later; costs accuracy on its fast preset.
- Strata-EXL3-fork (https://github.com/nasone32/Strata-EXL3-fork): two 7900 XTX with peer-GPU expert offload. Not
  needed here.
- kapitsa (https://github.com/nace-ai/kapitsa): Apple only; an NVFP4 GGUF of Inkling (19.1 GB, KL 0.004). That pack
  keeps the native 4-bit grid: worth checking whether it loads in ggml on HIP.

## 3. Approach

**Engine path: a ggml-graph engine in `src/inkling`, shaped like `src/hy4`**, not a hand-written HIP engine like
`src/qwen` or `src/glm`. Reasons:

- Upstream llama.cpp already has the graph (`inkling2_loop`), the chat parser and GGUF packs with KL numbers. Porting
  that graph and checking against it is far less work than kernels from scratch, and `src/hy4` already does the same
  (graph built in `hy4_graph.cpp`, HIP-specific pieces alongside, PR comparison in `tools/hy4/hy4_pr_compare.py`).
- The model is small (3.4B active): what matters is launch overhead and the 48 small layer visits, which ggml with
  HIP graphs handles; a hand-written engine is the later step only where a profile shows a graph op as the cost.
- First step: vendor the needed upstream commits (#32704, #32799, #32807, #32833) into `third_party/llama.cpp`, with
  patches kept in `tools/v41/ggml/` as today. This touches the shared ggml build: coordinate with the hy4 work.

**Quantization and placement: one card.** IQ4_NL (17.6 GB) or AD-Q5_K (19.07 GB) on card 0 with q8_0 KV:
17.6 + 4.3 (256K KV) + 0.16 (window cache) + 0.5 (drafter) = about 22.6 GB. The experts are native NVFP4, so a 4-bit
repack loses little (KL 0.0095). Card 0 only because card 1 sits on PCIe x2: no weights cross it on the hot path.

- Card 1 options, measured later, not default: (a) Q8_0 (27.8 GB) layer split across both cards (only a 2304-wide
  activation crosses per stage, ~4.6 KB at BF16, but decode runs the stages one after the other, so it is slower
  than one card); (b) a second independent server/slot on card 1 for throughput; (c) the vision encoder.
- Keep the looped core (layers 6-17) and the drafter on the faster/higher-precision side if anything has to move;
  the core is read 3 times per token.

**Speculative decoding:** DSpark from `mtp.safetensors` (or AtomicChat's 481 MiB GGUF), 8-token blocks, verified in
one batched pass of the main model. Verify cost: experts touched grow with batch, so the verify of 8 tokens reads
more expert bytes than one token; measure accepted tokens per verify on code and prose, and add an adaptive stop
(as `--mtp-adaptive` in hy4 and Avalon). Opt-in, as every drafter here.

**Attention:** only 4 of 24 layers are global, 8 visits per token. At long context these 8 visits dominate; a gather
or sparse pass like hy4's indexer is not needed at first. Window layers use a ring buffer of 1024 tokens.

**Almond entries:** `local/inkling-local.json` (+ `-vision`), `local/fetch.sh inkling` item (pack + drafter +
mmproj), router entry in `local/router.json`, `tools/inkling/serve/serve_inkling.py`, `./almond serve --model
inkling` and CLI alias, chat template and parser in `serve/` (thinking channels, reasoning effort, tool calls) with
unit tests, `--inkling` flag in the engine dispatch, docs in `docs/MODELS.md`.

## 4. Speed estimate (estimate, not measured)

Bytes per token at decode, IQ4_NL pack:

- Routed experts: 8 experts x 3 matrices x 2304 x 1152 = 63.7M weights per visit x 48 visits = 3.06B weights read per
  token (config). At ~4.5 bits (IQ4_NL) for visits in layers 0-19 and 8.5 bits for 20-23: about 1.8 GB.
- Non-expert weights (attention, router, conv, loop gates) per visit, plus the 201K x 2304 output head (463M weights,
  0.49 GB at Q8_0): roughly 0.4-0.6 GB more. The head is the single biggest dense read.
- Total about 2.3-2.5 GB per token. At 800-900 GB/s (card 0, measured in this repo for other models) the ceiling is
  320-390 tok/s; at the 50-60% of peak this repo's other engines reach, about 160-230 tok/s at short context.
- Cross-check: Xolotl reports 127 tok/s on an RTX 4090 (1,008 GB/s) and Strata 99-106 on an R9700 (640 GB/s).
  Scaled by bandwidth, those point to 100-140 tok/s on a 7900 XTX for code like theirs. Plan target: 120 tok/s
  without a drafter at 4K context, above 150 with DSpark on code.
- At 256K context the 8 global visits read 256K x 17 KiB = 4.3 GB per token at q8_0, more than the weights: decode
  drops to about half. This is where a sparse/gather pass on the global layers would pay.

Prefill: 2 x 5.4B = 10.8 GFLOP per token. At 40-60 TFLOPS sustained on gfx1100 WMMA that is 3,700-5,500 tok/s before
attention and MoE gather; Xolotl's 3,500-4,190 on a 4090 and llama.cpp #32846 (gather 28% of prefill) suggest
2,000-3,500 tok/s here at first.

Assumptions: one sequence, card 0 alone, no RAM or PCIe traffic on the hot path, kernels at the same fraction of peak as
our hy4/qwen paths, numbers from the cited pages taken as given.

## 5. Phases

1. **Upstream reference.** Vendor the Inkling commits into ggml/llama.cpp, build, run `llama-cli` and
   `llama-perplexity` on the IQ4_NL pack on card 0 (when the cards are free). Done: same KL vs Q8_0 as ggml-org's
   0.0095 within noise, decode and prefill tok/s at 4K/32K/128K written in `docs/inkling/`.
   Needs downloads: IQ4_NL 17.6 GB, Q8_0 27.8 GB (reference for KL), drafter 0.5 GB: ask before.
2. **Engine skeleton `src/inkling`.** Graph from upstream, loop with per-pass caches, q8_0 KV, window ring buffer,
   `--inkling` dispatch. Done: greedy output equal to llama.cpp for 256 tokens; PPL on fixed documents matches.
3. **Server, router, CLI, fetch, configs.** Template, parser (thinking, tool calls), unit tests passing with
   `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .`. Done: `./almond -m inkling` chats and calls
   a tool.
4. **Speed on one card.** Profile bytes per token and time per op; HIP graph capture of the whole decode round; fused
   router + top-k + expert gather; head in Q8_0 vs Q6_K measured. Done: measured tok/s vs the 120 target, table in
   `docs/inkling/`.
5. **DSpark.** Drafter load, batched verify, adaptive stop. Done: accepted tokens per verify and tok/s on code and
   prose, opt-in flag, default changed only if better on both.
6. **Long context.** 256K and 1M configs, sparse/gather pass on the 4 global layers if decode at 256K is under half of
   short. Done: tok/s at 32K/128K/256K, KL vs dense attention.
7. **Card 1 and vision.** Measure Q8_0 layer split vs one card, second slot on card 1, mmproj. Done: numbers, and
   a config only for what measured better.

## 6. Open questions

- Tool-call format and default reasoning effort: read `chat_template.jinja` and PR #32833.
- Whether the NVFP4 GGUF (kapitsa, 19.1 GB) runs in ggml on HIP and how fast compared with IQ4_NL; gfx1100 has no FP4
  hardware, so it would be a dequant kernel.
- The loop gate and attention residual reads: cost per visit not known until profiled.
- DSpark acceptance on this model: no page gives it; Xolotl's 127 -> 136 tok/s suggests a small gain at their setting.
- Upstream open bugs (#32804 long-context assert, #32830 garbled IQ4_NL) may touch our path.
- Merging upstream ggml changes into the vendored tree has to be timed with the hy4 branch, which builds from it.

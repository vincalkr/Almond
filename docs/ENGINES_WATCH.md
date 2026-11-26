# Engines and forks worth watching (swept 2026-11-22, 2026-11-24)

The trees that work on the models Almond runs, on cards like ours, and what each of them carried in the last weeks.
**Every number here is theirs, quoted as they print it** - none was measured on this machine, and the models/quantizations
differ, so they are a map, not a scoreboard. For our own numbers on the same models: `docs/MODELS.md` (Qwen3.8-Flash-Next
mean 87.4 / 88.4 tok/s through the server, 49-78 tok/s in agent sessions at 8-63K; Qwen3.8-27B 73.3 tok/s at 192K) and
`docs/hy4/SPEED_WORK.md`.

## ggml-org/llama.cpp - the tree our ggml comes from

`third_party/llama.cpp/VENDORED.md`: ggml 0.26.0 at `3cf03257` (2026-09-20), copied without history, and only two upstream
fixes ported by hand (`tools/v41/ggml/UPSTREAM_FIXES.md`: #30103 the flash-attn tile race, #29953 the MMQ out-of-bounds,
applied 2026-10-14). Two months and ~500 commits behind, and the delta is where the free risk is. The ones that touch our
path (HIP/RDNA3 and MoE), from `ggml/src/ggml-cuda` since the base:

| PR | date | what | ours |
|---|---|---|---|
| #30021 / #30022 | 11-13 / 11-17 | tune the GCN MMQ configs, use stream-k on AMD GCN | not ported - our prompt path is MMQ-heavy |
| #32344 | 11-16 | fix the MoE regression on HIP introduced by #29948 | not ported - check whether #29948 is in our tree at all |
| #31188 | 10-28 | fuse MoE (MUL_MAT_ID) gate/up and GLU in MMQ | not ported; our hy4 fuses them in the model graph (`--fused-proj`) |
| #31288 / #29181 | 10-28 / 10-17 | top-k MoE fusion for 384 experts; grouped experts top-k fusion | we have `tools/v41/ggml/topk-moe-384.patch` in the same file - compare before porting |
| #31047 | 11-05 | linearize the MoE data layout for MMQ/MMF | not ported |
| #32067 | 11-12 | skip empty MoE tiles in the MMQ stream-k fixup | not ported |
| #30885 | 10-23 | **fix the bf16 WMMA operand vector type for gfx11/gfx12** | correctness, RDNA3: check `mmf`/bf16 paths |
| #31005 | 11-05 | sanitize post-bias NaNs in MoE selection | correctness |
| #30804 | 10-23 | chunk graphs to amortize the graph launch | skip: measured here that graph launches are worth ~nothing (`SPEED_WORK`, *The graph churn*) |
| #31617 | 11-15 | refactor the `_sync` warp primitives | the same direction as the DPP work below, upstream-wide |

`ggml/src/ggml-cpu` in the same window is all ARM/SVE/SpacemiT - nothing for x86-avx512, so our CPU experts (`48,4 GB/s`
of the ~65 the RAM gives) get nothing from upstream.

## freriadoc/Avalon - read this one first

481 stars, pushed 2026-11-22, "fast fork of llama.cpp designed for Qwen3.8-Flash-Next", tuned on RDNA3 with ROCm 7.2,
gfx1100. Their headline: **98 tok/s on an RX 7900 XT (PCIe 3.0, Ryzen 7 3800XT, 64 GB DDR4)**; on an **RX 7900 XTX 24 GB,
Ryzen 9 5900X, PCIe 4.0, 64 GB DDR4-3200**: 131k context, `-ub 19072`, `GGML_SCHED_PREFETCH_EXPERTS=6` -> prefill
**2,024 - 2,043 tok/s**, generation **112.9 - 122.8 tok/s** (median 116.5); a later version with `--moe-cache-cpu-overlap 2`
-> prefill **2,113 - 2,130**. Against Strata with both engines' own drafters: 44% faster on a short prompt, 30% at 250k
filled. Our own mean on the same model is 87.4 - 88.4 (server, three prompts, thinking low).

What they have that we do not:

- **Hosted MoE, split inside the block**: "the GPU keeps each MoE block, computes the cached experts while the CPU
  computes the missing ones", then "generation, up / gate and down of the routed experts in one launch, their chunks'
  entries in the plan's table, the shared expert"; "hosted MoE plan's chunk list made by the host, **entry kernel timeline
  stamps**"; "the hosted MoE entry kernel waits for the plan with a **short pause**" (not a spin). So they measure how
  much of the CPU time actually hides - the number we could not see in our publish/wait split.
- **The expert tiers**: experts loaded with O_DIRECT when they fit, else an **arena of registered slots** sized from the
  RAM free at start (`GGML_CUDA_MOE_ARENA_MARGIN_MB`, default 4096), the GPU cache filled from the arena only, and
  "during a prompt, part of the arena is lent to the prefill's reads and restored before" - plus a `prefill ring` that
  is used only for the prompt blocks the hosted path leaves, for packed layers only.
- **MTP length chosen token by token** (up to 4): "a draft token is kept while its chance of acceptance pays for its
  verification, the chance from the draft's confidence and the request's acceptance so far, the **cost from the step's
  clock and the experts it runs on the CPU**". Against 2 fixed drafts: -0.2% on replays of ten texts, +4-6% on the
  server's short answers (`LLAMA_MTP_ADAPT=0` off). This is exactly the break-even we computed for hy4 (2,33 tokens a
  round at 3 rows, 4,29 at 8) turned into a policy.
- **HIP copy plumbing**: "route all staged host copies and one-shot ops **off SDMA and legacy streams** on HIP", "the 2D
  host copy kernel reads the source with its pitch; kernel copies of a pinned source **only while a hosted block is
  queued**", "fix 2-GPU multi-card MoE execution and host copy deadlocks on RDNA2". Our `--prefetch-experts` measures
  27 GB/s of hipMemcpy to card 0 (28,4 with a plain test): a copy kernel on a compute stream is a direct A/B.
- **RDNA3 prompt work**: "a prompt's products in **FP16 on RDNA3**, the placement plan at the graph's live maximum"; a
  **chunked prompt GDN kernel, RDNA3-only** (GDN is Qwen3.8's linear attention); "the QSA attention of a long prompt per
  query, in int8 on the K/V rows as stored"; "a prompt's QSA layer, Q's norm / rope / rotation in one pass"; "the QSA
  decode step in one op, its graph fixed below the indexer's context, the small graphs apart"; "five ports from Basalt
  and Strata for the MTP step, on by default"; "a prompt's hyper-connection gate and mix in one kernel" (that one NVIDIA
  FP16 tensor cores, opt-in).
- Not ours to take: their GGUF layout and the cut LM head (65k frequent tokens for the draft head), and everything in
  `llama/` / `common/` - we vendor `ggml/` only, so all of this transfers as ideas or as ggml patches, not as a merge.

## mw00/project-maya - v1.1.73, pushed 2026-11-22

Runs GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash, Hy4-Flash (220B) on NVIDIA and AMD from 8 GB of VRAM. Three
things in this release are ours in shape:

- **A calibrated, per-token split between copying an expert in and computing it on the CPU**: "for each token, the GPU
  copies the missing experts in or the CPU computes them beside it, **split by rates measured once on the PC and saved**
  (`~/.maya/ik2-offload-calib.txt`). An expert gives the same result wherever it runs, so the answers do not depend on
  the split." We choose the split per *layer* once at load (29 of 42 in VRAM at 256K) and `--host-moe` is all-or-nothing.
- **Hy4-Flash on two GPUs with the IOMMU translating (#1100)**: the layer split's direct GPU-to-GPU copy faults there,
  so direct copies run "only where the driver offers them, the IOMMU allows them and **a test copy arrives intact**;
  otherwise the hand-offs go through pinned RAM" (`HY4_HOST_STAGE=1` forces, `HY4_PEER=1` tries anyway). Our card 1 is on
  PCIe x2 - the same shape of problem. Also #1084: a draft step whose numbers were all invalid read outside the
  vocabulary; it now ends the draft.
- **Two AMD GPUs and the expert cache (#1086, #1027)**: updating the expert tables between windows waited for the whole
  stream of the second GPU and on two AMD cards that wait never returned - now it waits "only for the last reader of the
  table input it overwrites".
- Drafting chosen by a one-time measurement of no-drafting / prompt-lookup / drafter, kept per PC.

## Indras-Mirror/Xolotl - not the finetuning framework, pushed 2026-11-22

"Run DeepSeek-V4.1-Flash, all 552B, as a local coding agent on one gaming PC (RTX 4090 + 110 GB RAM)". The commits of
today are decode-speed work on a model they call Inkling-2-Mini:

- "long-context decode **30 -> 55 tok/s at 477K** (lane-per-cell attention, parallel split merge, positional masks)";
  "decode attention at any context - one chunk plan, **splits capped at 512**".
- "prep kernel writes q8_0/q4_0 rows + **in-kernel Hadamard**; rotate only the Q4 legs"; "INT8 K/V Hadamard rotation on
  by default (`STRATA_KV_ROT=0` turns it off)"; 512K on one 4090 with **k8v4**.
- For us: Hadamard-rotated int8 (and k8v4) KV is the cheap cousin of our `--kv-q8` - it buys quality back at the same
  bytes, which is what would let a longer context fit in the same VRAM; the split-KV cap at 512 is a hint for our
  gather-attn at 256K.

## Niko1221/Strata - v0.2.18 (2026-11-21 17:09) and the commits after it

Read in full last session (`SPEED_WORK`, *What Strata's Hy4 engine does*): growable KV that spills to pinned host
memory, CPU expert jobs with the SSD reads hidden behind earlier waves, Tree-8 speculation with a suffix drafter and a
cost learner, hot-to-cold rank tables. New since that read, 2026-11-21 22:40: **"hy4flash: Tree-8 CUDA graphs, k2_vnni
1.5x speedup, 8-thread hybrid prefill & full vcache"** and **"hy4: add AVX-VNNI + S8_K fast CPU kernels, cgroup limits,
and Docker shape-engine support"**. The S8_K part is the same wall we keep hitting: their CPU experts run on **8-bit
weights**, i.e. the byte win - which for us is `--ram-requant q3_k`, measured +5-7% at KL 0.012 and still not in the
configuration. A faster CPU kernel does not move our 48,4 GB/s of RAM. The 8-thread hybrid prefill and the "full vcache"
are prompt-side and worth a look when the prefill is the complaint.

## nasone32/Strata-EXL3-fork - `amd-exl3-community`, 2026-11-21

"Experimental AMD Linux EXL3 fork of Strata: optimized peer prefill/decode, portable launchers, **RDNA3 measurements**",
gfx1100, and candidate settings for **two 24 GiB RX 7900 XTX**: `--expert-cache 165 --peer-slots 10200 --prefill 8192
--max-context 65536`. The release of 2026-11-21 is "**Qwen EXL3 peer residency and automatic cache sizing**": the peer
card owns its experts and they are *not* duplicated in pinned RAM (`--resident peer`), `--peer-slots` sized from the VRAM
actually free (with a retry at a smaller subset when the allocation does not fit), `--reserve 768 --peer-reserve 2048`,
"registered expert arena measured **40.10 GiB instead of 56.72 GiB** (4.05 bpw)", "optimized native serial prefill kernel
defaults on gfx1100". Our qwen serve carries `--expert-cache auto --vram-reserve-mib 400 --pcie-frac 0.00`: automatic
peer sizing and a per-card reserve are directly comparable, and their numbers say the pinned-RAM copy of the peer's
experts is bytes we pay twice.

## thefahien/elpis-on-7900xtx - 2026-11-02

Recipe and benchmarks for **Qwen3.8-27B EXL3 4.00bpw + DFlash2 on an RX 7900 XTX**: **71.6 tok/s decode after a
51,491-token prompt** (68.3 whole-request), prefill **1,100 tok/s** on that prompt, "the 7900 XTX reaches 53-71% of the
3090's speed", and a llama.cpp-Vulkan table beside it. Our dense qwen27 path: 73.3 tok/s at 192K (with MTP 3) - so at
parity with their EXL3+DFlash2 on the same card, while DFlash2 on our tree measured 26.6 against MTP's 34.5
(`docs/MODELS.md`). Their `docs/BENCHMARKS.md` is a clean definitions page (tok/s = completion tokens / whole-request
wall time) - worth stealing for our own tables.

## phoenixhaxor/exllamav3-rocm and -rocm-moe - last pushed 2026-09-26 / 2026-10-23

RDNA3 port of exllamav3; the `-moe` branch does **Qwen3.8-Flash-Next with CPU/RAM/SSD offload on one 7900 XTX + 60 GB**
(7950X3D, ROCm 7.2.4): **83 tok/s code, 45-50 prose with MTP-2 and learned placement**, prefill **~1,120 / 1,010 tok/s**
on fresh 27.8K / 62.7K prompts, no drafting 53 / 46, and llama.cpp on the same GPU IQ4_XS `--n-cpu-moe 36` at 17.5 as a
reference. "RDNA3 kernels for the MoE decode path, the gated-residual hyper-connections and the router" and *learned
placement* of experts (their profile): our equivalent is `--expert-profile` + `--expert-cache auto`, and our profile is
learned too (`--expert-profile-save`). Nothing newer than our last read.

## SILVERTHRONE32/llama.cpp-rdna3-int4 - branch `rdna3-int4`, 2026-11-05

"Rotated GPTQ INT4 inference for Qwen on AMD RDNA3: 4B, 9B and 27B kernels, benchmarks and profiling" - on a 7900 XT:
**2.27x prefill / 1.53x decode over Q4_K_M at 27B** (2,013 / 50.1 tok/s), no speculative decoding. The new quantization
is theirs, but the kernel work is on our tree's files and arch-independent of the quant:

- `ggml-cuda (HIP/RDNA3)`: **4-wave INT4 GEMM launch for large weights** (`M*K >= 30M`, `GGML_INT4_WN4_MIN` overrides);
- decode: **fold `flash_attn_combine_results` into the tile kernel** (`GGML_FA_FOLD=0` disables);
- decode: **DPP reductions** in the GDN decode kernel and in the matvec epilogue (no shared memory / no shuffle round
  trip); RMS norm + rotation **keeping the normalized values in registers**; interleaved quantisation and hoisted
  epilogue loads in the rotation matvec; the 8-column gate matvec as **two launches of 4** because the 8-column kernel
  spilled registers; q/k/v projections **in one launch** (k, v to scratch); a `mm-skinny-rdna3.cu`; input scheduling in
  `ggml-backend.cpp`;
- `docs/rdna3/PROFILING.md` ranks decode/prefill costs and documents a **launch-gap investigation** - the same thing our
  11,728 direct computes vs 480 graph launches in hy4 point at.

For our dense Qwen3.8-27B path these are the cheapest A/Bs in the whole sweep: they are HIP patches to `fattn-tile.cuh`,
the matvec epilogues and the GDN decode kernel, each behind an env switch.

## antirez/ds4 - last pushed 2026-11-08

The reference engine our `tools/v41` work is written against. The last two weeks are Qwen and Apple: half-aligned Q2
weight rows on Metal, resident Qwen arithmetic when experts stream from SSD, prefill speed kept with speculative
decoding, Qwen v2 checkpoints. Nothing for ROCm/discrete AMD since our read.

---

## Sweep of 2026-11-24 (commits since 11-22; their numbers, not measured here)

- **mw00/project-maya v1.1.77-78.** Three same-bits HIP changes for Hy4 on RX 7900 XTX, by @Antero128:
  #1164 replaces warp sums' `__shfl_xor` (`ds_bpermute_b32` through LDS on HIP) with DPP / `v_permlanex16` register
  moves, same xor order and bits (`ds_bpermute` in hy4's code objects 166,904 -> 32,536). #1151 packs Hy4's short
  attention-head rows (`k_b` Q8_0 k=192, `v_b` Q6_K k=512) 2-4 to a warp: kernel +34%, end to end +0.5%. #1150 turns on
  weight read-ahead on gfx1100: +1.7%. #1168: under `--tp` with the CPU lane, a token window's sums on one host thread
  (helper threads were spinning on the lane's cores): 24.8 -> 39.4 tok/s on Hy4 Maya-M. Maya-S (all experts in VRAM,
  `--tp 2` on two 7900 XTX) reports 108 tok/s at 64K. #1154 Inkling DSpark: on an RTX 3090 at 262K, FP8 KV, new code 124
  -> 200 tok/s, prose 123 -> 134, same tokens; drafting up to 1M with the experts that do not fit in RAM.
  **For us:** our vendored ggml's `warp_reduce_sum` still uses `__shfl_xor_sync` on HIP (no DPP): #1164's change applies
  to it and to `src/inkling/native`, bit for bit.
- **Indras-Mirror/Xolotl (Inkling, RTX 4090).** Decode 113-127 tok/s, 136-140 with DSpark; prefill 3,500 at 8K and
  4,190 at 32K served. 512K on one 24 GB card with `--kv-type k8v4` (q8_0 keys, q4_0 values with a randomised Hadamard
  rotation of the 4-bit leg): KL from f16 3.6e-4 (q8_0) and 8e-4 (k8v4), PPL unchanged; decode 127 / 118 / 100 / 56
  at 7K / 30K / 117K / 477K. Long context 30 -> 55 tok/s at 477K from one lane per cell in the decode attention, a
  parallel split merge and positional masks. Fused loop controllers (BF16, 3 kernels each) 119 -> 127; per-expert
  scales folded into the MoE finish and a BF16 router 127 -> 130. Also an Inkling LoRA trainer.
- **freriadoc/Avalon (RX 7900 XT).** ROCm 10.2.0 (TheRock) against 7.2.4: prefill 3,225 / 3,330 against 2,888 /
  3,112 tok/s at 32K / 131K, same decode. Experts' host memory in 4 KiB pages, not transparent huge pages: compaction
  moved registered pages and KFD stalled the GPU (4 runs in 6 hung with 2 MiB). The MTP draft's late experts loaded
  with O_DIRECT into anonymous memory (page-cache pages were dropped and refaulted, run hung). Captured graphs kept by
  count, not dropped after 10 s (our vendored ggml predates that sweep, #21611, so not our problem).
- **ggml-org/llama.cpp.** Inkling: #32855 mix() weighted sum, #32940 clip with `ggml_l2_norm`; HIP MMQ configs for
  VEGA20 / CDNA1 (not ours); #32500 fused RMS_NORM + MUL + ROPE (CUDA); #32859 faster BPE tokenization.
- **Niko1221/Strata v0.2.21.** Inkling CUDA: prefill MoE 64 tokens of an expert per tile (same bits, sm_80-89), L2
  prefetch of FP4 experts on sm_120, auto prefill chunk by free memory, K/V growth into a host tail. Nothing for HIP.
- **nace-ai/kapitsa 0.2.40.** Inkling on Metal: 1M context with KV growth, Q8_0 packing of BF16 dense projections at
  load (fast profile), grouped MoE prefill with FlashAttention. Nothing for HIP.
- **DevDock-AI/Speedy 0.5.5.** A coding-tuned MTP drafter for Qwen3.8-Flash-Next (+10% decode on Signal AP-Q4_K_XL,
  +4% on UD-Q4_K_XL), on Hugging Face `DevDock-AI/Qwen3.8-Flash-Next-Speedy-MTP`.
- No commits since 11-22: antirez/ds4, elpis-on-7900xtx, exllamav3-rocm(-moe), SILVERTHRONE32, Strata-EXL3-fork.

## What this sweep would change here, in the order it costs least

1. ~~**`--ram-requant q3_k` into the configuration** (hy4)~~ - **decided against, 2026-11-22**: the quality cost is not
   worth +5-7%. Note also that its KL 0.012 sits under the run-to-run floor now measured (KL 0.12-0.14 for identical
   flags, `local/hy4-determinism-check.sh`), so it could not have been confirmed anyway.
2. **An adaptive MTP length from the measured step cost** (hy4 and qwen `--spec`): Avalon's policy, and our own cost
   curve (55.0 - 167.1 ms for k = 1..8 rows) is already in `SPEED_WORK`. Days, no new mechanism.
3. **Copy kernel vs SDMA for the expert staging on HIP** (hy4 `--prefetch-experts`, qwen): Avalon routes staged host
   copies off SDMA and legacy streams on HIP; our 27 GB/s hipMemcpy number is the baseline to beat.
4. **The SILVERTHRONE32 kernel list on the dense qwen27 path**: folded attention combine, DPP reductions, fused
   qkv launch, skinny matmul - each one a small patch with its own env switch, on files we already vendor.
5. **Automatic sizing of the peer expert pool and the VRAM reserve** (qwen, two cards with card 1 on x2): the shape
   nasone32's release and maya's `--vram-reserve-mib` fix both landed on.
6. **A per-token, measured copy-vs-CPU split per expert** (hy4's 13 RAM layers): what maya calibrates once per PC and
   Avalon does inside the MoE block with timeline stamps. This is the real answer to the publish/wait result (+2,8% of
   a round): the split has to be by expert and by measured rate, not by layer and by flag. Most work, most room.

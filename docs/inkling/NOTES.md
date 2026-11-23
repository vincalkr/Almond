# Inkling 2 Mini: what runs and what was measured (2026-11-23)

The plan is in `PLAN.md`. This file holds what was built and measured. Machine: RX 7900 XTX card 0 (gfx1100, ROCm),
Ryzen 9 9900X, 122 GiB DDR5. Everything ran on card 0 alone unless a line says otherwise.

## What runs

- Pack: Unsloth's `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.63 GB, https://huggingface.co/unsloth/Inkling-2-Mini-GGUF),
  chosen by the user. Its routed experts are stored as MXFP4 (ggml type 40 in the kernel trace below), the rest Q8_0.
  `local/fetch.sh inkling` (aria2c, 4 connections) downloads it and extracts the tokenizer.
- Engine: `build-inkling/almond-inkling` (`src/inkling/inkling_serve.cpp`). It is llama.cpp's own `inkling2_loop`
  graph (master `30ec35ae`, pinned in `build.sh`) behind the engine's serve protocol (READY / GEN / PP / T / DONE /
  SAVE / RESTORE). It is a separate executable: libllama brings its own ggml, which cannot link next to the vendored
  ggml in `almond-engine`. Writing our own graph (`PLAN.md`, phase 2) was not done.
- Server: `tools/inkling/serve/serve_inkling.py` maps the model's message tokens to the server's `<think>` /
  `<tool_call>` text, and its effort levels none .. max to the template's 0.0 .. 0.99. The template's
  `{% generation %}` tags are removed before Jinja reads it. Tests: `tools/inkling/serve/test_serve_inkling.py`.
- Configuration `local/inkling-local.json`: 131,072 context, q8_0 KV, ubatch 128, `swa_full` off. Router entry
  `inkling`, `./almond -m inkling`.

## Quality (measured first)

Reference: ggml-org's Q8_0 (27.8 GB; ggml-org gives KL 0.0030 against BF16 for it). It does not fit one card, and
llama.cpp cannot split this model's layers over two cards yet: its looped layers share weights, and
ggml-org/llama.cpp#32751 is still open. The reference therefore ran with its weights in RAM (`-ngl 0`), with card 0
computing the batches. Text: this repository's English docs (`local/inkling-measure.sh`: MODELS.md, V41_EXACT_FAST.md,
AMD_HIP_NOTES.md, HANDOVER.md; 43,709 tokens), 2048-token chunks, `llama-perplexity --kl-divergence`.

| Pack | Size | PPL | KL vs Q8_0 mean / 99% | Same top token |
|---|---|---|---|---|
| Q8_0 (reference) | 27.8 GB | 6.1598 | - | - |
| Unsloth UD-Q4_K_XL | 17.63 GB | 6.1675 (+0.13%) | 0.00728 / 0.0624 | 95.90% |
| ggml-org IQ4_NL | 17.63 GB | 6.1722 (+0.20%) | 0.01416 / 0.1170 | 94.31% |

At the same size, UD-Q4_K_XL is half the KL of IQ4_NL. Unsloth's card gives KLD 0.0107 and top-1 98.0% against their
own UD-Q8_K_XL. That is a different reference and a different text, so the two do not compare directly.

Tokenizer: the server's tokenizer (`tools/almond_tokenizer.py`, pre-split `inkling2` = llama.cpp's GPT-4o split with
ignore_merges) gives the same ids as `llama-tokenize` on all 43,709 tokens of that text.

Sampling: with the card's defaults (temperature 1.0, top_p 1.0; top_k was 50 then), a tool reply over
`{"high_c":17,"low_c":8}` wrote "1.7°C / 0.8°C" in 1 of 3 samples. At temperature 0 it was right ("17°C / 8°C"),
and the other two samples were right too. This is sampling, not the engine. The configuration keeps the card's
values; a lower temperature or a min_p is a quality choice for the user.

Recommended values: the original repo's `generation_config.json` gives only temperature 1.0; Unsloth's guide
(https://unsloth.ai/docs/models/inkling-2-mini) gives temperature 1.0, top_p 1.0, top_k 0 (off), min_p 0.0; ggml-org's
card gives reasoning_effort high as the default. No source gives a penalty or a max-token value. The configuration
follows these: it leaves top_k out (the server's config check takes only 1..64), which almond-inkling takes as off
(it then samples over the 4,096 most likely tokens).

## Speed

`llama-bench`, UD-Q4_K_XL, card 0, q8_0 KV, flash attention (`local/inkling-measure.sh speed`, 2 runs each):

| Depth | Prefill 512 | Prefill 4096 | Decode (128 tokens) |
|---|---|---|---|
| 0 | 1,396 tok/s | 1,065 tok/s | 30.6 tok/s |
| 8,192 | 776 | 731 | 30.2 |
| 32,768 | 395 | 388 | 29.0 |

Through the server (ubatch 128):
- A 40-token prompt decoded 594 tokens at 30.6 tok/s.
- A 21,725-token prompt (docs/MODELS.md plus an English question) prefilled at 444.7 tok/s (48.8 s) and decoded at
  29.3 tok/s.

This is a quarter of the plan's estimate (120 tok/s). Bytes are not the limit. A kernel trace (rocprofv3, 33 decode
steps, HIP graphs off, which costs 10%: 27.9 against 31.1 tok/s) gives:
- about 24 ms of GPU time per token, in about 7,400 kernels per token;
- the quantized matvecs (Q8_0 dense 23.7%, MXFP4 experts 10.9%) take about a third;
- the rest is thousands of 1.5-3.7 µs element-wise kernels: mul, add, scale, rms_norm, concat, tanh, soft_max,
  copies, the short convolutions, and the depth mixing (`mix`) and loop controllers (`control`) of every layer visit.

With HIP graphs on, launches are not the cost: each small kernel takes its time on the GPU. The way up is fusing
those per-visit chains (the mixes, the controllers, the clip / norm / convolution of Q, K and V) into a few kernels.
That is phase 4 of the plan, in our own engine.

## Memory on one card

- KV at q8_0 (8 full-attention visits): 1,088 MiB at 64K, 2,176 at 128K, 4,352 at 256K (17 KB a token). This matches
  the plan's estimate.
- With llama.cpp's default `swa_full` the 40 window visits also keep the whole context: 21.7 GB at 256K, which does
  not fit. `almond-inkling` turns it off.
- The compute buffer grows with context x ubatch. The relative-position bias is gathered as n_kv x n_tokens x 24
  heads. Sizes: 6.5 GB at 64K/512, 4.8 GB at 96K/256, 3.2 GB at 128K/128, 4.0 GB at 160K/128.
- 256K does not fit even at ubatch 128. 160K/128 loads. The configuration keeps 128K/128 for margin. The cost is
  prefill: 444.7 tok/s at 21.7K through the server against 731 for 4096-token batches at 8K depth in llama-bench.

## DSpark in almond-inkling (opt-in)

`local/inkling-dspark-local.json` (`./almond -m inkling-dspark`, router entry beside `inkling`) runs
`build-inkling/almond-inkling-dspark`: `src/inkling/inkling_serve.cpp` built with `-DALMOND_DSPARK` on llama.cpp PR
#32218 (head `e4b1cb5`, `build-inkling/llama.cpp-dspark`, built by `build.sh` once the drafter is downloaded). The
default `almond-inkling` and its configuration are unchanged.

- Drafts: `--dspark DRAFT.gguf --dspark-n 3`, llama.cpp's `common_speculative` with greedy drafts (the drafter's
  argmax; probabilistic drafts lost the whole gain at temperature 1.0, above).
- Check: the target decodes [last token, drafts] in one batch, samples every position with the request's own sampler
  (the same `sample()` as without the drafter, penalties included) and keeps the drafts up to the first one it did not
  sample. Each kept token is a token the target sampled itself, so the output follows the target's distribution.
- Cut back: the context is made with `n_rs_seq = 3`, llama.cpp's per-step snapshots of the recurrent (convolution)
  state for inkling2 (`llm_arch_supports_rs_rollback`), so the rejected tail is removed with `llama_memory_seq_rm` -
  no checkpoint copy and no re-decode (the same idea as Strata's snapshot + records). The drafter's cache is cut with
  it. The prefix reuse between turns works as before.

Through the server (`serve_inkling.py`, card 0, 128K context, q8_0 KV, the two prompts above, 600 tokens, reasoning
off):

| | temp 0 | temp 1.0 |
|---|---|---|
| `inkling` (no drafter) | 29.2 / 29.3 | 29.4 / 29.4 |
| `inkling-dspark` | 41.7 / 44.2 | 46.3 / 41.4 |

A three-turn chat at temperature 1.0 (a name and an age to remember, then counting to 30) answered right on every
turn at 41-57 tok/s. Not measured yet: a long prompt with DSpark, tool calls with DSpark, and PPL / KL with it on (the
argument for the same distribution is the construction above, plus the near-tie check at temperature 0).

## Prompt checkpoints (2026-11-23)

The cache used to be kept only when a prompt extended it exactly. An agent's next prompt re-renders the model's last
turn (its thinking and tool call as the template writes them), which is not the tokens the model produced, so every
file it read restarted the prefill from 0 (seen in `local/router.log`: 17,037 tokens reused on one turn, the next
20,037-token prompt read from 0).

`almond-inkling` now copies the state that cannot be cut back (llama.cpp's `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`: the
convolution state and the window layers' cache) to RAM at the end of every prompt, keeping the last 4, as
llama-server's context checkpoints do. A prompt that parts from the cache goes back to the newest checkpoint inside
the shared prefix and reads only the rest. Three agent-like turns (a 4K-token file, then a 5K-token file, then a
question), card 0, temperature 1.0:

| | turn 2 (9,596 tokens) | turn 3 (9,839) |
|---|---|---|
| before | all read again | all read again |
| `inkling` now | 4,157 from the cache | about 240 read |
| `inkling-dspark` now | 4,157 from the cache | about 240 read |

All answers were right. The checkpoints' size in RAM was not measured (a window of 1,024+ tokens over 40 window
visits at q8_0 plus the convolution rows: an estimate of under 100 MB each).

## Not done

- **DSpark drafter:** wired since 2026-11-23 as an opt-in (next section). Measured first
  with llama.cpp PR #32218 (`pwilkin/inkling2-dspark`, head `e4b1cb5`, built apart in `build-inkling/llama.cpp-dspark`)
  and its own `llama-server`. Card 0, UD-Q4_K_XL, q8_0 KV, 16K context, two prompts (Python code with tests; a long
  explanation), 600 tokens each, reasoning off, `--spec-draft-sampling probabilistic`:

  | | temp 0 | temp 1.0 |
  |---|---|---|
  | no drafter | 30.5 tok/s | 29.5 |
  | DSpark, draft 3 | 50.3 / 47.3 (accepted 362/709, 345/762) | 30.3 / 30.0 (341/770, 340/776) |
  | DSpark, draft 7 | 48.3 / 43.9 | 27.9 / 27.2 |

  So +55-65% greedy, nothing at the card's temperature 1.0 with probabilistic drafts.

  The cause is the draft sampling, not the rollback. Same setup, draft 3, `--spec-draft-sampling greedy` (the draft
  takes its argmax; the target still samples at the request's temperature and checks the draft against it):

  | | temp 0 | temp 1.0 |
  |---|---|---|
  | DSpark, draft 3, greedy drafts | 49.8 / 46.8 | **42.4 / 39.8** (accepted 332/799, 312/855) |

  +35-44% at temperature 1.0. The log shows no speculative checkpoint restore in that run (only the prompt cache's
  context checkpoints); mean accepted length 2.3-2.5 of 3, acceptance by position 0.64-0.68 / 0.39-0.50 / 0.26-0.35.

  Same tokens? Temperature 0, three prompts, 400 tokens each, token ids compared (`/completion`, reasoning none). Two
  runs without the drafter gave identical tokens. With DSpark the text first differs at token 36, 25 and 77. At each of
  those points the run without the drafter had its top two tokens within 0.06-0.11 in log-probability (about 1.06-1.12x
  apart). The target checks the drafts in batches of 4 instead of 1, and those kernels round differently, which flips
  near-ties. That is the batch numerics already noted for the other engines in AGENTS.md, not a different
  distribution. Speculation is exact in distribution by construction; a stronger check (PPL or KL with the drafter on)
  was not run.

  How Strata (Niko1221/Strata, `engine/src/inkling2`, its own HIP engine, read 2026-11-23) does it: the drafter reads
  4 target features (visits 5, 23, 41, 48), drafts up to 7 with the length backed off by an EMA of acceptance, verifies
  a beam tree of up to 12 nodes when greedy and a chain otherwise (rejection sampling at temp > 0). Rollback is cheap:
  before verify it snapshots the 3-step convolution history of every visit and records each step's inputs; a rejection
  rebuilds snapshot + accepted steps in 2 kernels, with no re-decode. No speed or acceptance numbers for Inkling are
  published there.
- **Images:** the mmproj is downloaded (`inkling-vision`), but `GENI` answers an error.
- **Relative-position bias:** Strata computes it over a fixed 1,024-position window (`rel_extent`) inside its fused
  QRKV kernel, so its scratch does not grow with context. llama.cpp gathers it over the whole n_kv, which is what limits
  us to 128K (above). Not checked here whether the window is exact for this model.
- **Own engine (`src/inkling` graph on the vendored ggml) and fused kernels:** not started. They are what the speed
  table above needs.

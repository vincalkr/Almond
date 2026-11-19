# Hy4-flash GGUF, the IQ2-AH build

What the two GGUF shards actually contain, measured on 2026-11-18. Source: `AngelSlim/Hy4-flash-GGUF`, build
`Hy4-flash-IQ2_AH`. The metadata and tensor directories were read from the first 8 MB of each shard over HTTP Range
with `tools/hy4/hy4_inspect.py`; no model data was read. File sizes are the repository's.

| shard | size | tensors |
| --- | ---: | ---: |
| `Hy4-flash-IQ2_AH-00001-of-00002.gguf` | 71.6 GB | 1,415 |
| `Hy4-flash-IQ2_AH-00002-of-00002.gguf` | 63.4 GB | 1 |
| total | 135 GB (125.73 GiB) | 1,416 |

## Quantization

The build name says "IQ2-AH" but no tensor has a type called IQ2_AH. Shard 1's types are **IQ2_XXS** on the 82
`ffn_gate_exps` / `ffn_up_exps` tensors, **Q4_K** on the 41 `ffn_down_exps`, **Q8_0** on `attn_gate` and
`attn_output`, **BF16** on the MLA projections, the shared experts and the latent up/down, and **F32** on the small
norms, gates and scales. All of those are already in the vendored ggml (`ggml.h:392-429`) and have HIP dequant and
MMQ instances (`CMakeLists.txt:132`).

Shard 2 is one tensor, `per_layer_token_embd.weight`, shape `[96, 660,006,416]` = 63,360,615,936 elements, **type
43 = F8_E4M3** (FP8, 1 byte per element). Upstream llama.cpp added this type after the vendored ggml's
`GGML_TYPE_COUNT = 43`; the engine can read the table with its own byte-to-float conversion instead of patching ggml
for it, as `src/ds41/ds41_engram.cpp` already does for the pack's iq4_nl rows.

So the 2-bit applies to the routed experts only. The Engram table is 8-bit and is half the download: 63 GB of the
125.73 GiB. A smaller model would have to quantize that table, which is a separate change with a quality cost to
measure.

## Geometry from the metadata

`general.architecture` is `hy_v4_flash`; `block_count` 43 — 42 trunk layers and one MTP draft block — with
`leading_dense_block_count` 1. Hidden 3072, dense FFN 13824, vocab 120,832, context 1,048,576. MLA: q_lora 1024,
kv_lora 512, key/value length 576/512 (latent 256/256), rope dim 64, rope bases 1e7 and 1e4 for the sliding layers.

Attention is hybrid: `attention.head_count` lists 32 (sparse) and 64 (sliding) heads per layer,
`sliding_window_pattern` marks the sliding layers and `sliding_window` is 512. The sparse layers carry the indexer:
16 heads x 128 dims, top 2048, kpool 4, keeping the current group. iHC has 4 streams (eps 1e-6, magnitude 2).

MoE: 512 experts, 16 routed per token, 1 shared, expert FFN 1536, latent width 1536, gating function 2, weights
scale 3.998, norm on. Engram: layers 1, 11, 21, 31 (1-based), trigrams, 8 heads per order, 96 dims per head,
64 head buckets, 12 multipliers, pad id 120002, and a 120,832-entry token map.

Sampling in the file: temperature 0.9, top_p 1.0, top_k -1. Tokenizer `gpt2` with `hy_v4` pre-split, BOS 120000,
EOS 120025; a 17 KB chat template is in the metadata.

## Tensor names

Every trunk block has `attn_q_a/q_b`, `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`, `attn_gate`, `attn_output`,
`attn_norm`, `attn_sinks`, `post_attention_norm`, `post_ffw_norm` and the six iHC tensors (`hc_attn_*`, `hc_ffn_*`).
MoE layers add `ffn_gate_inp`, `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_routed_down/up`, the three
`ffn_*_shexp` and `exp_probs_b.bias`; sparse layers add `indexer.*` and `indexer_compressor_*`. The four Engram
layers add `engram_gate`, `engram_hc_scale`, `engram_kv`, `engram_kv_b`, `engram_lookup_norm`, `engram_pre_gamma`.
Block 42 carries `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm` and `nextn.shared_head_norm`.

These names differ from DeepSeek's (`blk.<L>.engram_embd.weight` with type iq4_nl, plus the `DS41_META` overlay
arrays), so `src/hy4/hy4_geometry.hpp` reads the arrays straight from this file's header: no overlay is needed.

## What is implemented (2026-11-18)

- `almond-engine --hy4 -m <shard> --info` opens both shards, reads the geometry and checks the tensor map against
  the file; it prints the numbers above. `--list-tensors` prints the tensor directory with numeric type ids. Checked
  on the downloaded 71.6 + 63.4 GB files: geometry and tensor map pass.
- `src/hy4/hy4_engram.hpp` has the n-gram hash from llama.cpp PR #32517 and an FP8 E4M3 byte-to-float conversion.
  `build-hip-fused/test_hy4_engram <shard>` checks the layout against the metadata (four modules at layers 1/11/21/31,
  16 columns each, bucket ranges tiling all 660,006,416 rows), checks that hashes stay inside their buckets and are
  deterministic, checks the FP8 decoding at zero / subnormal / one / NaN, and reads two real rows plus a hashed row
  from shard 2. All checks pass; the values are finite and rows differ.
- `src/hy4/hy4_attn.cpp` runs the single-key gated MLA attention of layer 0: Q/K/V projections, q/k latent norms,
  nope absorption into the KV latent space, value absorption through `wv_b`, the Q8_0 output gate and projection.
  RoPE is not applied yet: with one key the softmax is 1 and the rotation cancels in q·k, so this verifies the weight
  layout and the projections. `test_hy4_attn` + `tools/hy4/hy4_attn_ref.py` compare against numpy with Q8_0
  dequantization: mean absolute difference 0.0042, max 0.034 over 3,072 outputs on 2026-11-18. The remaining gap is
  the accumulation order of ggml's Q8_0 dot products.
- `src/hy4/hy4_attn.cpp` also contains a multi-token version with NORM RoPE, a causal mask, sinks and ggml's flash
  attention. `test_hy4_attn_multi` runs on three tokens; `tools/hy4/hy4_attn_multi_ref.py` recomputes it with numpy.
  Measured on 2026-11-18: max absolute difference 2.80, mean 0.36 over 9,216 outputs. A staged dump showed each graph
  stage is internally consistent (dumped intermediates reproduce the final output to 5e-7); the residual difference
  comes from the mixed-precision accumulation of BF16/F16/Q8_0 products, which numpy does not reproduce bit for bit.
  The structure (RoPE NORM, causal mask, sinks in the softmax denominator, per-head `wv_b`) is therefore confirmed,
  but not to the 1e-6 tolerance of the earlier blocks. A HIP build of PR #32517 was made outside the repository (its
  libraries compile) but no runnable executable was linked, so there is no second engine as an oracle yet.
- `src/hy4/hy4_forward.cpp` runs the whole trunk in one pass: token embedding repeated over the four iHC streams,
  42 layers (layer 0 dense SwiGLU, layers 1-41 LatentMoE, Engram on 1/11/21/31), then the iHC head, `output_norm`
  and the 120,832-way output projection. `test_hy4_forward` runs two tokens end to end in about 31 s on the CPU;
  the logits are finite (range -12.87..13.38) and greedy token ids are printed by the test. This is an integration
  milestone, not a validated model: there is no KV cache, no k-pool indexer/sparse selection (full causal attention
  over the batch), no 512-token SWA window, no chat template, sampling or server, and no end-to-end numerical
  comparison against PR #32517. The rope base follows `sliding_window_pattern` (1e7 sparse, 1e4 SWA) and the draft
  block 42 is excluded from the trunk.
- `almond-engine --hy4 -m <gguf> --prompt-ids 10,20 -n 2 --temp 0` runs a generation loop over the single-pass
  forward and samples with temperature/top-p (defaults from the GGUF: 0.9 / 1.0). Measured on 2026-11-18: step 0
  took 33.3 s for a 2-token context, step 1 109.5 s for 3 tokens (a cold page cache and a per-step weight copy;
  the KV cache and the in-place weights below replaced both). The ids are the interface; tokenization and the chat
  template belong in a Python shim.
- `src/hy4/hy4_layer.cpp` chains the verified pieces into one full trunk layer in the PR's order: Engram (when the
  layer owns one), iHC pre, `attn_norm`, gated MLA attention, iHC post, iHC pre, `ffn_norm`, LatentMoE,
  `post_ffw_norm`, iHC post. `test_hy4_layer` runs layer 1 (the first MoE layer, with Engram) on two tokens with real
  weights; the output is finite and deterministic across runs. This is an integration check, not an independent
  numerical comparison of the composed layer: each stage was verified separately (see above) and the dense FFN of
  layer 0 is still unimplemented.
- `src/hy4/hy4_moe.cpp` runs a one-token LatentMoE with layer 1's real weights: router to 512 experts, sigmoid
  gating, top-16 selection, weight normalization and scale 3.998, `ffn_routed_down` into the 1536-wide latent space,
  per-expert SwiGLU through `ggml_mul_mat_id` on the IQ2_XXS/Q4_K stacked tensors, `ffn_routed_up`, and the BF16
  shared expert. `test_hy4_moe` + `tools/hy4/hy4_moe_ref.py` compare against numpy from a staged dump: routing ids and
  weights are exact, SwiGLU differs by 6e-8, and the final output by 9.6e-5 mean / 4.2e-4 max absolute on 2026-11-18. The expert weights stay in ggml; the reference checks the routing and combination, not their
  dequantization.
- `src/hy4/hy4_hc.cpp` runs the iHC pre/post of PR #32517 as ggml graphs, using layer 0's real `hc_attn_*` weights.
  `test_hy4_hc` dumps the inputs and outputs; `tools/hy4/hy4_hc_ref.py` recomputes them with numpy. Measured on
  2026-11-18: max absolute differences 1.2e-6 (mixes), 1.8e-7 (pre gates), 2.4e-7 (reduced streams), 1.5e-8 (post
  gates) and 1.2e-7 (post output).
- `src/hy4/hy4_graph.cpp` runs PR #32517's `build_engram` as a ggml graph on the CPU, loading the six F32/BF16 weight
  tensors from the GGUF and gathering the FP8 rows. `test_hy4_graph <shard> out.bin` dumps four fixed tokens' inputs
  and output; `tools/hy4/hy4_engram_ref.py <shard1> <shard2> out.bin` recomputes the same projection with numpy from
  the GGUF weights. Measured on 2026-11-18: max absolute difference 2.1e-6, mean 1.5e-7 over all 49,152 output
  elements, zero elements over 1e-4. The numpy reader uses `mmap` and reads only the 64 hashed FP8 rows, so it never
  loads the 63 GB table. Two layout traps were found the hard way and are handled in both readers: GGUF's `[ne0, ne1]`
  shapes need transposing for numpy's C order, and ggml tensors have `ne0` fastest, so the dumped flat arrays must be
  reshaped as `[nt, ne1, ne0]` and transposed back.

## Checked against PR #32517 (2026-11-18)

The PR (commit `0a054d0b0`) was built with HIP for gfx1100 in a separate worktree outside this repository. With the
experts in RAM (`-ot exps=CPU`) and the rest on cards 0 and 1 it generated coherent text from this GGUF at 10.4 tok/s
(47 tokens, prompt "The capital of France is", temperature 0, one run).

`tools/hy4/hy4_pr_dump.cpp`, linked against the PR's `libllama`, saved the activations named by its graph callbacks (`hc_init`,
`engram_out-L`, `attn_out-L`, `hc_attn_out-L`, `ffn_out-L`, `l_out-L`, `result_output`) for the 4-token prompt
"The capital of France" (ids 802, 8778, 299, 12749; CPU, one batch). `HY4_DUMP_DIR=<dir>` makes `almond-engine --hy4`
save the same tensors under the same names; `tools/hy4/hy4_pr_compare.py` compares them. The comparison found three bugs, now fixed:

- `hy4_forward.cpp` copied the int64 token ids into the I32 embedding index with `memcpy`, so the model read
  `[802, 0, 8778, 0]`.
- `hy4_layer.cpp` gave every token the Engram history of the first one, and the hash used raw previous ids where the
  PR maps them through `engram.token_map` first. `prev_ids` is now `[nt * (n_gram-1)]` raw ids, newest first.
- `hy4_engram_forward_cpu` returns `x + delta`, and `hy4_layer.cpp` added that to `x` again.

After the fixes, relative L2 difference of the streams against the PR: 2.9e-3 after layer 0, 1.2e-2 after layer 11,
2.0e-2 after layer 21, 1.1e-1 after layer 41; logits 9.2e-2 (cosine 0.9958). The greedy token is the same at all four
positions and the top-10 lists share 10, 8, 10 and 10 ids. The difference grows slowly with depth and has no jump at
any layer, which points at accumulation order and activation quantization, not at a structural error; that is not
proven. The earlier 2.80 max difference of `test_hy4_attn_multi` against numpy was the numpy side: the same
attention matches the PR at layer 0. This covers 4 tokens only, so the 512-token window and the k-pool indexer,
which do not act at that length, are not checked.

## KV cache (2026-11-18)

`Hy4KvCache` (`hy4_forward.hpp`) keeps, per trunk layer, each position's key as F32: the normed 512-wide latent and
the 64 roped dims (576 floats, 2.3 KB per layer per token, 97 KB per token over 42 layers). The value is the latent
part of the same row. `almond-engine --hy4` now runs the prompt once and then one token per step; `--no-kv-cache`
reruns the whole prefix as before. The attention mask now applies the 512-token window on the sliding layers
(`sliding_window_pattern`), which the earlier path lacked. The sparse layers still attend to every key: the PR's
indexer selects every pool while the context is at most `top_k - kpool * kpool_select_tail` tokens, so the result
is the same up to that length; past it the engine warns and the attention is not the model's.

Checked on 2026-11-18, CPU, prompt ids 802,8778,299,12749, greedy, 8 tokens: the cached and uncached runs produced
the same 8 ids (341 15641 13 12749 341 259 6252 297). Both took about 12 s per step, because every layer copied its
weights out of the mapped GGUF at every step (the stacked experts alone are about 1.3 GB per layer); see the next
section.

The PR's own greedy run of the same prompt gives 341 15641 13 3477. At that step the PR in a single 7-token batch
ranks 12749 first (15.20 against 14.62 for 3477), as this engine does (15.92 against 13.44); only the PR's
token-by-token decode picks 3477. Over those 7 tokens the stream difference against the PR stays at most 0.8% at
position 0 through all 42 layers, and reaches 8-31% at the later positions by layer 41. The increases appear at the
MoE output of one token at a time (for example token 5 at layer 2: 1.0% to 8.9%), which is what a different expert
in a near-tied top-16 looks like; the expert ids were not compared, so that is not proven.

## Weights read in place (2026-11-18)

`hy4_weight` (`src/hy4/hy4_weights.hpp`) makes each weight once as a ggml tensor whose data points into the mapped
GGUF, in a no_alloc context that lives as long as the process; the attention, the dense FFN, the LatentMoE and the
output head use it instead of copying. The Engram weights (BF16 converted to F32) are still copied at every step.
Measured on 2026-11-18, CPU only (12 threads, `HIP_VISIBLE_DEVICES=` empty), prompt ids 802,8778,299,12749, greedy,
with the file already in the page cache from earlier runs: 0.6 s for the 4-token prompt, 0.405 s per token mean over
63 decode steps (2.5 tok/s), against 12 s per token before. The 8 first ids are the same as before the change. The
64 tokens read "The capital of France is Paris. France is a country in Europe. Paris is the capital of France. What
is the capital of France? ..." and then repeat the question, close to the PR's own greedy text. The PR on the same
machine with the experts in RAM and the rest on two cards runs at 10.4 tok/s.

## The engine on the cards (2026-11-18)

`src/hy4/hy4_engine.cpp` (`Hy4Engine`) is the path `almond-engine --hy4` runs; the CPU blocks above stay as the
reference it was checked against. One ggml graph per step over the HIP backends and the CPU (`ggml_backend_sched`,
with op offload: a prompt batch runs the experts that live in RAM on the card).

- Placement: the trunk is cut into one layer range per card; each card holds its layers' attention, shared expert,
  norms, iHC and Engram weights and their caches, and its first layers keep their routed experts in VRAM while they
  fit. The other layers' experts are read in place from the mapped GGUF and run on the CPU; those pages are touched
  once at load (16.5 GiB in 2 s), and the file pages of the weights copied to VRAM are given back (`MADV_PAGEOUT`).
  With 4K of context on two 24 GB cards: experts of 29 of 42 layers in VRAM; with 64K: 26.
- KV: the 30 sliding-window layers keep a ring of 1,024 rows (a ubatch of 512 never overwrites a row its own tokens
  still see); the 12 sparse layers keep every position, plus the indexer's caches: value|gate of the last tokens (a
  ring) and the pooled key of each group of 4.
- The k-pool indexer is PR #32517's `build_kpool_select`: pooled keys (softmax over the group with the learned
  positional term, RMS norm, RoPE on the last 64 dims at the group's first position), a score per visible pool (ReLU
  of q.k per head, weighted by a projection of the input), the best 511 pools plus the token's own group, and the mask
  of those cells for the layer's flash attention. It acts only past 2,048 tokens: before that every pool is selected.
- Engram (`--engram-place auto|ram|ssd`): the 59 GiB FP8 table is read per row with O_DIRECT from the SSD (no page
  cache), or whole into RAM (5.6 s) when it fits next to the experts read from RAM and the file pages of the weights
  in VRAM. On this machine (122 GiB) `auto` picks the SSD: with the table in RAM the experts were paged from disk at
  every token and a 128-token run did not finish in 15 minutes. The row reads run while the graph is built.
- `--dense-q8` (opt-in) stores the 2-D BF16 matrices as Q8_0: one more expert layer in VRAM, the same speed (15.6
  tok/s), perplexity 27.80 against 27.93; the BF16 matrices are small and their cost is kernel launches, not bytes.

Measured on 2026-11-18, cards 0 and 1 (RX 7900 XTX), 12 CPU threads, Engram from the SSD:

| | Almond | PR #32517 (`-ngl 99 -ot exps=CPU`) |
| --- | ---: | ---: |
| generation, 4-token prompt, 128 tokens (4K context) | 14.9 tok/s | 10.4 tok/s |
| generation after a 15,000-token prompt (64K context) | 11.8 tok/s | |
| prompt, 15,000 tokens | 237 tok/s | |
| perplexity, `local/glm-ref/doc.txt`, 512-token chunks | 27.93 | 28.46 |
| perplexity, 60 KB of `docs/ds4/V41_EXACT_FAST.md`, 512 / 1,024 / 2,048 | 26.48 / 32.98 / 39.73 | 26.25 / 32.59 / 39.82 |
| the same text, 4,096-token chunks (indexer active) | 36.42 | 34.75 (`-ub 256`: 37.40, `-fa off`: 35.77) |

The greedy tokens of the 4-token prompt are the CPU path's (341 15641 13 12749 341 259 6252 297). At 4,096 the PR
itself moves between 34.75 and 37.40 with the ubatch size or flash attention, and Almond's 36.42 is inside that;
with the indexer off (`HY4_FULL_ATTENTION=1`, a diagnostic) Almond gives 35.91.

Generation spends about 37 ms per token on the cards (some 4,100 kernels, many small) and about 2.4 ms more for each
layer whose experts are on the CPU (10 layers in VRAM: 9.4 tok/s, 20: 12.2, 29: 15.5).

## The server

`./almond serve --model hy4` (`local/hy4-local.json`; `local/hy4-local-vision.json` when the image encoder and
`build-vision-hy4` are there, as for the other models; `tools/hy4/serve/serve_hy4.py`, router entry `hy4`, chat
`./almond -m hy4`). The tokenizer directory is extracted from the GGUF by `tools/almond_tokenizer.py` (pre-split
`hy_v4`, the same regexes as `joyai-llm`) together with the model's chat template. The shim maps the reasoning levels
(none / low / high, the default / max), ends a turn on `<｜hy_end｜>` (120025) and rewrites Hy4's tags
(`<think:6124c78e>`, `<tool_call:6124c78e>NAME<arg_key:…>K</arg_key:…><arg_value:…>V</arg_value:…>`) to the server's
forms; `tools/hy4/serve/test_serve_hy4.py` checks the rewrite. The engine speaks the serve protocol (GEN / RESUME / PP /
T / DONE, STOP, CONVCLEAR) and reuses the cached prefix of a prompt. Checked on 2026-11-18 through
`/v1/chat/completions`: a question answered with its reasoning apart at 14.9 tok/s, and a tool call returned as
`get_weather({"city": "Rome"})` with `finish_reason` `tool_calls`.

## MTP (`--mtp`, 2026-11-18)

The draft block (layer 42) is PR #32517's graph_mtp: `eh_proj` over the RMS-normed embedding of the next token and the
trunk's final state (iHC head + output_norm) before it, then an ordinary sparse layer (gated MLA with its own KV and
k-pool indexer, LatentMoE) with a plain residual, `shared_head_norm` and the shared output head. Its dense weights and
caches sit on the last card, its experts (6.75 GB) in RAM. Every trunk step also runs the block's attention input
over the step's (state, token) pairs, but only as far as its cache writes; a draft is a one-token graph of the block
alone. Generation runs [token, draft] as one trunk step and keeps the draft when the token sampled after the first
one is the draft (exact for any sampling: the kept tokens are the ones sampling would give); otherwise the cache is
cut back by one. With greedy sampling the tokens are those of the same placement without the block: loading it moves
one expert layer to the CPU (28 in VRAM instead of 29 at 4K), and that alone changes the greedy continuation of the
4-token test prompt from its second token (both give 341 5586 23990 15641 341 269 8778 7370).

Measured on 2026-11-18 through the serve protocol, 64K context, three chat prompts rendered with the model's template
(reasoning low: a B-tree explanation, a Python parser with tests, a short story in Italian), 400 tokens each:

| | no MTP | `--mtp` | drafts kept |
| --- | ---: | ---: | ---: |
| greedy | 14.30 / 14.42 / 14.25 tok/s | 18.68 / 18.23 / 18.34 tok/s | 75% / 73% / 71% |
| temperature 0.9, seed 7 | 14.18 / 14.15 / 14.23 tok/s | 16.60 / 16.43 / 17.56 tok/s | 58% / 58% / 65% |

So `local/hy4-local.json` runs with `--mtp`.

## Conversations on disk and chained drafts (2026-11-18)

`SAVE <file>` / `RESTORE <file>` (the server's `conversation_dir`) write and read the conversation in the cache: its
tokens, the MTP states, each full layer's KV rows up to the current position, the sliding layers' and the indexer's
rings whole, and the pooled keys of the completed groups (a temporary file, then a rename). Checked on 2026-11-18 with
a 39-token chat prompt and 8 greedy tokens: SAVE 43.7 MB in 21 ms, CONVCLEAR, RESTORE in 14 ms, then the prompt plus
those tokens resumed 46 cached tokens and continued with the same 8 tokens as without the round trip
(4527 4548 297 5681 11 430 7231 303).

`--mtp-n N` chains N guesses (each block step fed the previous one's state and guess) and keeps the longest prefix
the trunk agrees with. Measured on the three chat prompts above (400 tokens, 64K context):

| | N = 1 | N = 2 | N = 3 |
| --- | ---: | ---: | ---: |
| greedy, tok/s | 18.22 / 18.37 / 18.35 | 17.11 / 17.12 / 16.21 | 14.15 / 16.55 / 14.59 |
| temperature 0.9, tok/s | 16.62 / 16.54 / 17.48 | 13.64 / 14.32 / 15.25 | 13.19 / 14.05 / 14.93 |

The chained guesses are kept far less often (N = 2 greedy: 213 of 372 against 172 of 228), so N = 1 stays the
default. With N = 3 the greedy text drifts from N = 1's after a while: a four-token verify batch takes other GPU
kernels than a two-token one, and their rounding differs (the engine is not bit-reproducible, AGENTS.md).

## Images (2026-11-18)

The image encoder is AngelSlim's `mmproj-Hy4-flash-bf16.gguf` (1.04 GB, `local/fetch.sh hy4-vision`), read by the same
`almond-vision` as GLM's and Qwen's (`tools/glm/vision/vision.cpp`) built against PR #32517's `libmtmd`, which has
the `hy_v4_flash` projector (ViT, RMS-norm adapter, 2x2 conv merge, a newline row per line of patches):
`./build.sh` fetches the PR at commit 0a054d0b0 into `build-vision-hy4/llama.cpp` and builds
`build-vision-hy4/bin/almond-vision` (checked from a clean clone on 2026-11-18); it runs on the CPU (the cards hold the
model).
A picture is `<｜hy_image▁start｜>` + one row per encoder token + `<｜hy_image▁end｜>`, as mtmd frames it; the server
expands the template's single pad to one per row and sends GENI. The rows' positions are Tencent's 2-D ones (see "Picture positions"
below), where the PR gives them consecutive positions. In the engine an image position
takes its row instead of the token embedding, gets no Engram delta (the PR masks those positions), counts as no token
in its neighbours' n-grams, and carries a negative key from the picture's key and the row index, so the cache never
takes one picture for another of the same size.

Checked on 2026-11-18 through `/v1/chat/completions`: a 448x336 test picture (a red square, a blue circle, the text
"HELLO 42") became 165 rows in 278 ms on the CPU; the answer named the red square on the left, the blue circle on the
right and read "HELLO 42", at 18.3 tok/s with MTP.

## Where generation time goes, and what changed it (2026-11-19)

A kernel and HIP API trace (rocprofv3) of greedy generation at 64K context, no MTP, about 70 ms a token: some 36 ms of
kernels on the cards, about 21 ms of gaps of a few microseconds between ~4,000 kernels a token, and about 24 ms of the
cards waiting on the routed experts in RAM (a median gap of 1.14 ms a layer, of which the CPU's MoE is 0.57 ms: a
micro-benchmark of one layer's 16 experts, 12 threads; 24 threads take 5.4 ms). The HIP graphs are reused (4,042 of
4,096 split computes launched from a captured graph); OpenMP spin-waiting and `HIP_FORCE_DEV_KERNARG` change nothing
measurable; the Engram rows from the SSD cost about 5% against the table in the page cache.

- `--fused-hc`: iHC through the vendored ggml's fused DeepSeek-V4 ops (`ggml_dsv41_hc_mixes`, `ggml_dsv4_hc_pre`,
  `ggml_dsv4_hc_post` with no comb), Hy4's eps inside the gates' sigmoid folded into the base at load: 13.84 -> 15.35
  tok/s; perplexity 27.51 (512-token chunks, against 27.93) and 39.70 (60 KB, 2,048-token chunks, against 39.79).
- The placement now tries every cut between the cards and keeps a layer with its experts in RAM off every card but
  the last: with one to three of them on card 0, 13.6-14.7 tok/s; with all on card 1, 16.0-17.1 (two runs each). Card
  1 sits on a PCIe x2 link (its port reports x16, the card's own switch), so this is latency, not bandwidth.
- `--dense-q8` and `--reserve-mib 1400` (the compute buffers take 1.33 GiB a card at 64K): experts of 29 layers in
  VRAM instead of 26, 15.5 -> 16.8-17.1 tok/s; perplexity 28.82 / 40.74. That +2.4-4.8% is inside the noise of this
  2-bit model: an exact conversion of the same BF16 matrices to F16 moved the 2,048-chunk perplexity to 41.18 and was
  not faster (15.4-15.9 tok/s); that option was removed on 2026-11-19.
- The SwiGLUs are written as `ggml_swiglu_split` (the same values; no measurable change), and a step that completes no
  indexer group pools the last one again, so the graph keeps one shape (no change either: the graphs were reused).

Through the serve protocol, the three chat prompts of the MTP section, 400 tokens, `local/hy4-local.json` as it is
now (`--mtp --fused-hc --dense-q8 --reserve-mib 1400`, 64K):

| | 2026-11-18 (`--mtp`) | `--mtp --fused-hc` | + `--dense-q8 --reserve-mib 1400` |
| --- | ---: | ---: | ---: |
| greedy, tok/s | 18.22 / 18.37 / 18.35 | 19.01 / 19.19 / 18.89 | 21.34 / 21.42 / 20.94 |
| temperature 0.9, tok/s | 16.62 / 16.54 / 17.48 | 17.20 / 17.74 / 18.37 | 18.49 / 19.86 / 20.46 |

A 15,000-token prompt with that configuration: read at 262 tok/s, then 64 tokens at 14.8 tok/s.

## Sampling and template, checked against Tencent's files (2026-11-19)

- Sampling: the model card (tencent/Hy4-flash) recommends temperature 0.9 and top_p 1.0, and `generation_config.json`
  has top_k -1 (off); `local/hy4-local.json` sends the same. The engine's serve path took a GEN line without
  `temperature=` as the GGUF's 0.9, while serve/server.py leaves the key out for a greedy request (temperature 0), as the
  other engines read it: such requests were sampled at 0.9. A missing key is now its neutral value (greedy, top_p 1).
- Template: the GGUF's `tokenizer.chat_template` is Tencent's `chat_template.jinja` with the special tokens' suffix
  `:6124c78e` instead of `:opensource`; the ids are the same (120000 `hy_start`, 120025 `hy_end`, 120029/120030 the
  think tags, 120035 tool calls, 120118-120120 the image tokens, in both tokenizers), so the model reads the same
  prompt. The GGUF's template also accepts `reasoning_effort` "low", which Tencent's refuses (the card names high, the
  default, and no_think; the template max as well): the server maps low, minimal and medium to high.
- Architecture: Tencent's config.json and card agree with what the engine runs (12 sparse / 30 SWA layers, 32 / 64
  heads, q_lora 1,024, kv_lora 512, QK 192 + 64, V 256, RoPE bases 1e7 / 1e4, a 512-token window of 511 preceding +
  current, 2,048 sparse tokens by groups, 512 experts with 16 routed and 1 shared, Engram at layers 1, 11, 21, 31). One
  difference to PR #32517: the original rotates with interleaved M-RoPE (mrope_section [12, 10, 10]); the PR uses
  NORM rope for this architecture, which is the same for text (t = h = w) but gives picture rows consecutive positions.
  The engine follows the original (next section).

## Picture positions: Tencent's interleaved M-RoPE (2026-11-19)

From transformers PR #50903 (`modeling_hy_v4_flash.py`, the model card's reference): the rotation pairs consecutive
dims (rotate_half over 0::2 / 1::2, as NORM), and pair i takes the height plane when i % 3 == 1, the width plane when
i % 3 == 2, below pair 30, and the time plane otherwise. A picture of H lines of W patches plus a newline token per
line, starting at position p0, puts row (y, x) at t = p0, h = p0 + y, w = p0 + x; the text after it starts at
p0 + max(H, W + 1), so a token's rotation position is no longer its index in the cache.

- `almond-vision --hy4-rows` (or `ALMOND_VISION_HY4_ROWS=1`, set in `local/hy4-local-vision.json`): mtmd gives this
  projector no grid, so the record's grid is taken from n = H * (W + 1) and the picture's aspect ratio (the test
  picture, 448x336: 165 rows -> 15 x 11). Without the flag nothing changes (GLM's and Qwen's encoders).
- The engine keeps (t, h, w) for every cached position (also in the session files, now version 2). A step whose
  rotations all have t = h = w (every step of text) runs the plain rotation as before; a step with picture rows, or
  pooling an indexer group whose first member is one, runs `ggml_rope_multi` in IMROPE mode with t, h, w and t again
  (ggml's IMROPE gives the pairs past the sections the fourth plane), around a move of the pairs to ggml's (i, i + 32)
  order and back. This covers the attention, the indexer's queries and pooled keys, and the MTP block.

Checked on 2026-11-19: the 60 KB perplexity text at 2,048-token chunks gives 41.72 against 41.62 before (text
positions unchanged; the difference is this engine's run-to-run noise); through the server the test picture is
described as before, with the shapes' places more precise ("in the upper half", "HELLO 42" in the lower left), at
21.9 tok/s.

## The cards' clocks during generation (2026-11-19)

A kernel trace of greedy generation (`--fused-hc --dense-q8`, 64K) counts about 3,100 kernels a token (from ~4,000):
the fused top-k MoE routing runs (`topk_moe_cuda`, the routing written in build_moe_ffn's order), the iHC fused ops
too. The largest item is now `mul_mat_vec_q` at about 17 ms a token: Q8_0 3,072-row products at ~35 us (~140 GB/s)
and the IQ2_XXS experts at ~80 us for 16 experts (~60 GB/s). Sampling the clocks while generating: card 0 (PCIe x16)
runs at about 1,300 MHz and 60-80 W, card 1 at 2,200-2,300 MHz and ~105 W, both at the driver's `auto` level. Card 0
holds the first 15-16 layers. `local/gpu-perf.sh compute` (the COMPUTE power profile, minimum active clock 1 GHz)
needs root; measured on 2026-11-19 with both cards set to it: greedy without MTP 17.43 / 17.31 tok/s (against 17.72
at `auto`), through the server with MTP 20.85 / 22.77 / 21.78 greedy and 19.94 / 20.93 / 21.11 at temperature 0.9
(against 20.94-21.42 and 18.49-20.46): inside the run-to-run spread, so the clock is not what holds card 0 back and
`auto` stays.

## Wave64 kernels (`GGML_HIP_WAVE64`, 2026-11-19)

llama.cpp issue #20934 measured ggml's HIP kernels compiled with `-mwavefrontsize64` on gfx1100 (the files that need
wave32 kept so: WMMA / mma.cuh users, softmax, the FA vec instances): +0-16% generation on ten models. The vendored
ggml-hip has it as an opt-in CMake option (off by default; AMD does not support the flag on RDNA), built here into
`third_party/llama.cpp/build-hip-w64` (71 files kept at wave32) and the engine into `build-hip-w64`. Measured on
2026-11-19 (`--fused-hc --dense-q8 --reserve-mib 1400`, 64K, greedy, 400 tokens): 17.62 / 17.79 tok/s against 17.41 /
17.27 with the normal build (+2%); the perplexity run (2,048-token prompt batches) had not finished after 7 minutes,
where the normal build takes 48 s, so some prompt-batch kernel is far slower or stuck in wave64. Not used.

## An expert cache, estimated before writing it (2026-11-19)

`HY4_ROUTE_STATS=<file>` makes the engine count each layer's routed experts and write them when it ends. Over the
three chat prompts (about 1,300 tokens, 20,656 picks a layer), the most used 64 / 128 / 256 / 384 of a layer's 512
experts take 46% / 67% / 88% / 97% of its picks (layer means; in-sample, so optimistic). The routing is flat: half of a
layer's experts for 90% of its picks.

A cache of the hottest experts across all 41 MoE layers in today's VRAM (about 360 a layer) would leave every layer
with some misses, and the graph, built before the routing is known, would need its GPU -> CPU -> GPU step in all 41
layers: about 25 ms a token at the ~0.6 ms each such step costs on ROCm here, against about 15 ms for today's 13 layers
in RAM. Keeping the 29 whole layers and caching the 13 RAM layers' hottest experts in the 1-2 GB left would cut the
CPU's share (about 40 experts a layer, ~35% of the picks), not the steps: about 2 ms a token. Not written; tandem.cpp's
1.9-3.3x are measured against llama.cpp with every expert on the CPU, not against a VRAM already full of whole layers.

## The matrix-vector kernels (2026-11-19)

A micro-benchmark of Hy4's shapes on card 0 (ggml's `mul_mat` / `mul_mat_id`, one token, a different weight tensor at
each repetition so the 96 MB Infinity Cache does not hold them): Q8_0 1,536 -> 3,072 in 23-29 us, 3,072 -> 1,536 in
29-32 us (157-171 GB/s), 1,024 -> 16,384 in 53 us, 8,192 -> 3,072 in 44 us (600 GB/s), IQ2_XXS 16 of 512 experts in
32 us (303 GB/s); a 256 -> 64 product, the fixed cost of the quantize + product pair, takes 11 us. So a third of a
small product is its launch. Letting RDNA3 take ggml's `small_k` rows-per-block rule changed nothing (reverted). The
fused gate/up/SwiGLU products do run fused (`mul_mat_vec_q<..., true>` for the Q8_0 shared experts and the IQ2_XXS
routed ones). What is left at about 3,100 kernels a token is their count; fewer of them means writing fused kernels
of our own (Tencent cites "megakernels" for its 150 tok/s), not tuning ggml's.

## One product for the attention's input (`--fused-proj`, 2026-11-19)

With `--dense-q8` the four matrices that read a layer's attention input (`attn_q_a`, `attn_kv_a_mqa`, `attn_gate`, and
`indexer.attn_k` on the sparse layers) are all Q8_0 with 3,072 columns; `--fused-proj` stacks them at load into one
matrix (`blk.L.attn_in_cat`: the same Q8_0 rows) and the graph takes one product and views of its rows, one input
quantization and one launch instead of four of each. Measured on 2026-11-19, 64K: greedy without MTP 18.18 / 18.33
tok/s against 17.39 / 17.23 (the same first tokens); perplexity on the 60 KB text at 2,048-token chunks 41.7157
against 41.7167; through the serve protocol with MTP, the three chat prompts: 21.40 / 23.00 / 22.31 tok/s greedy
(against 20.94-21.42) and 21.26 / 21.65 / 21.59 at temperature 0.9 (against 18.49-20.46). On in both
configurations.

One card or two: with `--gpus 0` alone (the x16 card, the rest in RAM) the routed experts of 10 layers fit in VRAM
against 29 on both cards, and greedy generation is 14.20 tok/s against 17.51 (200 tokens, 64K, no MTP): a layer whose
experts are in RAM costs about 0.7 ms when the rest of the step is on the same card, but 32 of them outweigh card 1's
x2 link. Both cards stay.

## Fewer copies, and the MoE's input stacked too (2026-11-19)

A kernel trace with `--fused-proj` counted about 2,490 kernels a token (from ~4,000 at the start of these notes):
quantize + Q8_0 product ~384 each, RMS norms ~366, device memcpys ~270, F32 / BF16 products ~217. The memcpys were
`ggml_cont` of tensors already laid out contiguously (in a one-token step most views are); the graph now copies only
where the layout needs it. `--fused-proj` also stacks `ffn_routed_down`, `ffn_gate_shexp` and `ffn_up_shexp` (one
product for the MoE's input, the shared expert's SwiGLU on views of it). Measured on 2026-11-19, 64K, greedy, no MTP:
18.60 / 18.53 tok/s with the copies skipped, 19.12 / 18.88 with the MoE stacking as well (from 18.18 / 18.33);
perplexity on the 60 KB text at 2,048-token chunks 41.85, then 41.63 (from 41.72: noise). Through the server with
MTP, the three chat prompts: 21.48 / 22.88 / 21.82 greedy and 20.74 / 21.42 / 21.75 at temperature 0.9, the same as
before within the spread: the verify step runs two tokens, where the views need their copies back.

## `ffn_down_exps` requantized for the cards (`--down-requant`, 2026-11-19)

The routed experts' `down` is Q4_K in this GGUF (648 MiB a layer, more than half of a layer's 1.21 GiB of experts;
`gate` and `up` are IQ2_XXS). `--down-requant q3_k|iq3_s|iq3_xxs|q2_k` dequantizes it at load and quantizes it again
for the layers whose experts go to the cards (those in RAM keep the file's Q4_K): more layers fit in VRAM. Measured on
2026-11-19 with `--fused-hc --dense-q8 --reserve-mib 1400 --fused-proj`, 64K:

| | experts in VRAM | load | greedy, no MTP | server + MTP, greedy | server + MTP, temperature 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K (the file) | 29 of 42 layers | ~45 s | 19.12 / 18.88 | 21.48 / 22.88 / 21.82 | 20.74 / 21.42 / 21.75 |
| `q3_k` | 33 | 45 s | 20.48 | 24.64 / 24.74 / 24.81 | 23.14 / 23.26 / 23.12 |
| `q2_k` | 36 | 110 s | 20.54 | 24.97 / 27.26 / 27.33 | 24.63 / 25.10 / 26.47 |

Quality, with `--ppl-save` / `--ppl-ref` (the scored rows' top-64 log-probs of a reference run, then the KL over those
64 and how often both rank the same token first; the 60 KB text at 2,048-token chunks, 7,161 rows; the reference is
the configuration above with the file's Q4_K):

| run against the reference | KL | top-1 agreement | perplexity |
| --- | ---: | ---: | ---: |
| the same configuration again | 0.000 | 100% | 41.72 |
| without `--fused-proj` (the same Q8_0 rows) | 0.003 | 98.99% | 41.74 |
| without `--fused-hc` (iHC's eps in another place) | 0.184 | 85.95% | 39.79 |
| without `--dense-q8` (BF16 matrices, fewer expert layers in VRAM) | 0.202 | 85.04% | 40.23 |
| `--down-requant q3_k` | 0.303 | 81.37% | 37.58 |
| `--down-requant q2_k` | 0.484 | 75.74% | 37.57 |

This 2-bit model moves about 0.2 in KL from changes that compute the same thing in another order, which is also why
its perplexity on 7,161 tokens moves by several percent; `q3_k` is a little past that, `q2_k` about two and a half
times. Off by default: faster, measurably further from the file's weights.

## What is not here

Video: Almond's server takes no video input for any model, so the template's video markers are not reached. The CPU path's numbers above are from before the
engine on the cards existed.

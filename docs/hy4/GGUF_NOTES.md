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
kernels than a two-token one, and their rounding differs - which is not the same thing as a run failing to repeat: with
the pool pick sorted (the default since 2026-11-22) one configuration repeats at KL 0.00000 over 8,190 rows, AGENTS.md.

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

## DFly2, Tencent's block drafter (`--dfly`, 2026-11-19)

Hy4-flash ships a drafter of its own, DFly2 (AngelSpec, `model.dfly_layers.*` in `model-00148-of-00148.safetensors` of
tencent/Hy4-flash, 1.69 GB BF16, in `models/hy4-dfly2`). `tools/hy4/dfly_convert.py` writes it as a GGUF (90 tensors,
`kv_b_proj` split into per-head key and value halves like the trunk's `attn_k_b` / `attn_v_b`; the confidence head is
left out, as vLLM leaves it out). `--dfly FILE` (with `--mtp`, which it turns on; `--mtp-n` is then the block's guesses,
5 by default) loads it on the last card, Q8_0 under `--dense-q8`; the placement still kept 28 expert layers in VRAM.

One drafter step is the block [the next token, then 120009 (mask) for each guess] at n_past: slot 0 gets the MTP-style
fusion of its embedding with the trunk's output-normed state before it, through the learned gate; five layers of
pre-norm gated MLA (sigmoid output gate, learnable sink, post-projection norm) and a SwiGLU FFN. Layer i reads trunk
layer 25, 29, 33, 37, 41's cache: the context rows the trunk's indexer picked for the last token (the attention mask
rows of the trunk's last 8 positions are kept per layer for this; after RESTORE every row is used) and the block's own
rows, written at n_past.. (the check that follows writes them again); the block attends both ways. Final norm, then per
slot the hidden correction with the previous token's embedding and the trunk's head, greedy, each guess feeding the
next slot. The guesses are checked like the MTP chain (exact for any sampling).

What it costs to check k tokens in one step (64K context, the same configuration, 1024 tokens of context, mean of 12):

| tokens in the step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 55.0 | 68.6 | 90.8 | 100.4 | 119.0 | 132.0 | 152.4 | 167.1 |

Greedy, 400 tokens, the three chat prompts above (`--prompt-ids`, tokens per round / tok/s):

| | prompt 1 | prompt 2 | prompt 3 |
| --- | ---: | ---: | ---: |
| MTP (one guess) | 1.64 / 21.61 | 1.71 / 22.54 | 1.66 / 22.48 |
| DFly2, 1 guess | 1.58 / 21.32 | 1.67 / 22.69 | 1.45 / 19.64 |
| DFly2, 2 guesses | 1.89 / 21.12 | 2.08 / 23.29 | 1.63 / 18.24 |
| DFly2, 3 guesses | 1.93 / 19.76 | 2.22 / 22.65 | 1.57 / 16.35 |
| DFly2, 5 guesses | 1.98 / 14.19 | 2.31 / 16.44 | 1.56 / 11.45 |

The guesses read well (the first rounds of prompt 1 kept 5 of 5; misses are mostly other words for the same thing,
"guardian" for "keeper"), and the greedy text was the MTP run's. But against this 2-bit trunk a block keeps about 2
tokens a round, where a 6-token check costs 2.4 times a 2-token one (the 14 layers whose experts are in RAM read more
experts per extra token); the break-even with MTP for 5 guesses is about 3.3 tokens a round. So `--dfly` stays off;
MTP with one guess remains the default.

## A matrix-vector kernel for short BF16 rows on RDNA3 (`ALMOND_MMVF_ROWS=1`, 2026-11-19)

A kernel trace (rocprofv3, 64 greedy tokens, 64K, no MTP) put the per-head BF16 products of every layer's attention,
`attn_k_b` (rows of 192 values) and `attn_v_b` (rows of 512), at 26-60 us each, 200-350 GB/s, about 3.9 ms a token:
ggml's `mul_mat_vec_f` runs one block of 96 or 256 threads per row, most of them idle or in the block's reduction.
`mul_mat_vec_bf16_rows` (vendored ggml, kept as `tools/v41/ggml/mmvf-rows.patch`, opt-in with
`ALMOND_MMVF_ROWS=1`) gives a row to one RDNA3 wave of 32 lanes and 8 rows to a block; each lane reads 8 BF16 (16
bytes) at a time and the wave sums with shuffles; up to 8 columns of y, no ids, no fusion. Measured:

- those products: 18.7 us on average (5,292 calls), about 1.6 ms a token;
- greedy with MTP, the three chat prompts, 400 tokens: 21.65 / 22.63 / 22.25 -> 23.24 / 24.36 / 23.15 tok/s;
  without MTP, 64 tokens: 18.71 -> 18.78 and 19.18 -> 19.89;
- the greedy text: the same 32 first ids on two of the three prompts, another on the third (the sums run in
  another order);
- one token per step (`-c 512 --ubatch 1`, doc.ids): perplexity 28.04 -> 28.60, KL 0.107 and top-1 agreement
  87.5% against the run without it (the same run repeated: KL 0.00000), within the 0.18-0.20 / ~85% that equivalent
  variants of this 2-bit model differ by (see `--down-requant`). In 8-token steps it is not used.

`local/hy4-local.json` and `local/hy4-local-vision.json` set it in their `env`; the other models do not.

## The same for Q8_0 rows (`ALMOND_MMVQ_ROWS=1`, measured, left off, 2026-11-19)

RDNA3's mmvq table gives a one-token Q8_0 row 8 waves (64 blocks a pass), so rows of 1,024-3,072 values leave a
quarter to a half of the threads idle. `mul_mat_vec_q8_0_rows` (vendored ggml, `tools/v41/ggml/mmvq-rows.patch`,
opt-in) takes a row with one wave (4 lanes a block, dp4a against the Q8_1 input), 2 or 4 waves for rows past 2,048 /
4,096 values, up to 8 columns, no ids, no fusion. Per call (rocprofv3, 64K): `attn_q_b` 1,024 -> 16,384 47 -> 22 us,
the stacked attention input 83 -> 67 us, the output head 496 -> 431 us; about 2.5 ms of kernels a token less.

- Without MTP (prompt 2, 200 tokens): 19.75 / 20.17 -> 20.86 / 21.24 tok/s (+5.5%).
- With MTP the trunk's steps have 2 tokens, where ggml's table already gives a row one wave: the two kernels take the
  same time (the stacked input at ~850 GB/s), and the three prompts, run alternately twice with a VM busy on the CPU,
  gave 22.57 / 24.00 / 23.33 / 23.20 / 24.09 / 23.50 without it against 22.16 / 22.83 / 23.06 / 22.82 / 22.62 / 24.20
  with it. The greedy text was the same on all three.

The configurations use MTP, so it stays off there.

**Re-measured 2026-11-22, now that a run repeats** (`bash local/hy4-knob-ab.sh ALMOND_MMVQ_ROWS=1`, 16K real text,
256K, 500 tokens, the shipped MTP configuration unless said otherwise). The kernel is **bit-identical** to ggml's at
one token a step - `--ppl` over 8,190 rows gives KL 0.00000, top-1 100.00%, perplexity 35.5641 both - and with
speculation switched off (`HY4_NO_SPEC=1`) it is worth **+10.1%**: decode 16.34 -> **17.99 tok/s**.

At the trunk widths of a verify batch it does not win, at any setting of the waves-per-row split
(`ALMOND_MMVQ_WPR`, a probe added to `tools/v41/ggml/mmvq-rows.patch` for this, 0 = the heuristic, 1/2/4 forced):

| one MTP round, 500 tokens | rounds | tokens a round | ms a round |
| --- | ---: | ---: | ---: |
| ggml's `mul_mat_vec_q` (the knob off, the shipping case) | 289 | 1.73 | **84.7** |
| `mul_mat_vec_q8_0_rows`, heuristic split | 278 | 1.79 | 85.5 |
| same, 1 wave a row | 289 | 1.73 | 85.5 |
| same, 2 waves a row | 273 | 1.83 | 88.9 |
| same, 4 waves a row | 288 | 1.73 | 92.4 |

So the 2026-11-19 reading was right: the kernel wins where a step carries one token and loses where it carries the two
to four of a verify batch, whatever the split - the heuristic's 4 waves for long rows is the worst of them (92.4).

**Then it was measured where it can be measured cheaply**: `build-hip-fused/bench_hy4_q8` (`src/hy4/bench_hy4_q8.cpp`,
driven by `local/hy4-q8-bench.sh`) runs those products on their own - the dense shapes the engine really multiplies,
**six different weights per shape** so the 48 MB of L2 cannot flatter the repeat, one ggml graph per measurement, a
variant read in seconds instead of a model load. Microseconds a product (reps 6, 8 iterations, the free card; the same
numbers came out of three separate processes):

| the weight (K × N) | 1 token a passo, ggml | 1 token a passo, nostro (1 onda) | 3 tokens, ggml | 3 tokens, nostro |
| --- | ---: | ---: | ---: | ---: |
| `attn_q_b` 1024 × 8192 | 25.0 | **14.9** | 16.8 | 16.8 |
| `attn_output` 8192 × 3072 | 62.8 | **48.2** | 49.6 | 48.2 |
| `ffn_down` 13824 × 3072 | 80.1 | **65.6** | 66.6 | 66.0 |
| `attn_gate` 3072 × 8192 | 47.4 | 42.5 | 43.7 | 43.7 |
| `attn_q_a` / `kv_a` 3072 × 1024 | 13.1 | 12.9 | 13.7 | 13.6 |
| `ffn_up` / `ffn_gate` 3072 × 13824 | 70.7 | 83.5 | 63.8 | 64.5 |

Two things follow, and they close this line of work.

- **At two to four tokens a step both kernels are at the roof, not at a disadvantage**: 45,1 MB of weights in ~64 us is
  ~700 GB/s (with the L2 helping; the HBM peak is 504 GB/s). The bytes a step moves are the bytes of the weights, and
  they do not depend on how many tokens the step carries - which is why a deeper verify batch costs almost nothing and
  why no kernel can take this time back. Fewer bytes can: what `--dense-q8` puts on the cards is ~85 MB a layer (the
  attention projections and the shared expert - **only layer 0 has a dense FFN**, the other 42 are MoE with a shared
  expert), so ~3,7 GB a step, 6-7 ms of an ~87 ms round, which is the 114 ms of `mul_mat_vec_q` in 19 tokens that the
  rocprofv3 trace of 2026-11-21 counted.
- **At one token a step ours is 18-40% faster on four of the six shapes and worse on the wide one** (`ffn_up` 70.7 ->
  83.5 at one wave, 63.4 at four). **One wave a row is now what the code chooses** (`mmvq-rows.patch`: it beat 2 and 4
  waves at every shape but that one, and the engine's own pair agreed - 85.5 ms a round at 1 wave against 92.4 at 4);
  `ALMOND_MMVQ_WPR` still forces 2 or 4 for the next attempt. The flag stays off in the configurations, whose steps are
  2-4 tokens wide - there the two kernels perform the same.

## The sparse layers' indexer, and 256K (2026-11-20)

**A bug from e4d4d56 to here.** `--fused-proj` stacks `indexer.attn_k` into `blk.L.attn_in_cat`, and the engine
decided whether a layer has an indexer by looking that tensor up: with `--fused-proj` (on in the configurations) the
12 sparse layers ran without their indexer, full attention over every position instead of the 2,048 the indexer
picks, and their indexer caches were not even made. Below 2,048 positions it is the same; past them the model ran in
a way it was not trained for, and every token read the whole cache. `has_indexer()` now looks up
`indexer.attn_q_b`, which is never stacked. (Same 4,096-token window, 8-token steps: perplexity of positions
2,048-4,095 12.62 with the bug, 12.87 with the indexer.)

**Placement.** It charged every layer a full layer's KV rows; the 30 sliding layers keep a ring and only the
indexer layers have indexer caches. Counted per layer: 256K, 20 -> 28 expert layers in VRAM (8.7 GB had been left
free on card 1); 64K, 28 -> 30.

With the indexer on, the mask path got slower (the sparse mask rebuilt over every cache row each step, the indexer
scores as an [n_pool, tokens, 16 heads] tensor: 5 GB of compute buffer at 256K, 512-token steps). New, opt-in:

- `--fused-indexer`: the scores through the vendored ggml's `ggml_lightning_indexer` (sum over heads of
  w * relu(q . k) plus the mask in one kernel; extended to 16 heads, the tile and vector kernels, F32/F16/BF16 keys)
  and the picks with `ggml_top_k` instead of a full argsort. Against the unfused indexer, 4,096 tokens: KL 0.082,
  top-1 91.4%, perplexity 13.39 / 13.31 (the kernel takes q and k as F16).
- `--gather-attn`: a sparse layer attends over the picked rows only: they are gathered from the cache
  (`ggml_get_rows_type`, F16 straight out) and each token is its own batch of 2,048 rows, in groups of 32 tokens in a
  prompt step; the cost no longer grows with the context. Against the reference: KL 0.100, top-1 90.7%.
- The prompt step halves itself when its compute buffers do not fit.

`--ubatch 1024` (configurations): ggml copies the weights of the experts kept in RAM to a card for every prompt step
of 32 tokens or more (`GGML_OP_OFFLOAD_MIN_BATCH`), ~17 GB a step here, so longer steps pay it less often; 2,048 does
not fit at 256K.

Measured on 2026-11-20 (two cards, `-c 262144`, the configuration's other flags, greedy with MTP, one run each; a VM
was using the CPU):

| | 16K prompt, tok/s | then decode, tok/s | 66K prompt, tok/s | then decode, tok/s |
| --- | ---: | ---: | ---: | ---: |
| before (indexer off by the bug, 512-token steps) | 178.7 | 14.49 | 204.1 | 15.43 |
| indexer on, mask path (256-token steps, the 512 do not fit) | | | 141.1 | 11.24 |
| + `--fused-indexer --gather-attn` (512-token steps) | 286.2 | 19.35 | 265.3 | 18.67 |
| + `--ubatch 1024` | 383.6 | 18.51 | 376.5 | 16.01 |

A short prompt at 256K: 23.8 tok/s (28 expert layers in VRAM), at 64K 24.2 (30).
The 66K decode at 1,024-token steps was noise: 16K prompt, two alternate runs, 512 / 1,024-token steps: prompt
286.9 / 384.8 and 285.9 / 381.8 tok/s, decode 19.24 / 18.29 and 18.43 / 18.42 tok/s.

`--kv-q8` (configurations): the full layers' KV rows as Q8_0 (the sliding rings stay F16); the gather and, below
top-k, a get_rows of the rows in use give the attention its F16. At 256K one more expert layer in VRAM (28 -> 29),
23.55 / 23.94 -> 24.21 / 24.44 tok/s (short prompt, alternate runs); against the reference KL 0.107, top-1 90.4%.
Re-measured 2026-11-21 with the graph ceilings on and `--reserve-mib 2400`, 16K prompt and 1,500 greedy tokens,
alternate configuration otherwise identical: with it 27 of 42 layers' experts in VRAM, 434.3 prompt / 24.54 decode
tok/s; without it (F16) 25 of 42, 434.0 / 22.88. So the Q8 buys two expert layers and +7.3% decode, and costs the
KL above. The room it saves is fixed at load, not per session: 12 full layers x (576+512) rows x 262,144 = 6.38 GiB
in F16, 3.59 GiB in Q8_0 - 2.79 GiB, about two expert layers.
A smaller `--reserve-mib` does not help: 1,100 and 900 leave ~1 GB a card free and the same 29 layers (an expert
layer is ~1.3 GB), 700 puts 31 in VRAM and the compute buffers no longer fit.

`--ipool-f16` (the indexer's pooled keys as F16 instead of F32) was tried the same afternoon and **left out**: what
it frees is the pool itself and nothing more - the cards gained 80 MiB (card 0) and 120 MiB (card 1) at 256K and no
expert layer moved into VRAM (27 of 42 with and without) - and an F16 pool drops the fused indexer: the keys are cast
to F32 inside the graph at every compute. Same 16K prompt at 256K: 434.3 -> **46.6 tok/s** prompt, 24.54 -> 23.15
decode. And there was nothing to win in quality either: against the F32 pool at 4K over 8,188 scored rows, KL
0.00000 and top-1 agreement 100.00%. To do it properly the fused `ggml_lightning_indexer` would have to take F16
pool keys itself.

`--ram-requant TYPE` (opt-in, off): `ffn_down_exps` of the layers whose experts stay in RAM (Q4_K in the file)
requantized at load, the cards' layers untouched. 2026-11-20, the configuration's flags at 256K: quality on a
4,096-token window against the same run without it (which repeats exactly: KL 0.00000), speed on a short prompt with
MTP, 400 tokens, two alternate runs:

| down experts in RAM | KL | top-1 | perplexity | tok/s |
| --- | ---: | ---: | ---: | ---: |
| Q4_K (the file) | 0 | 100% | 12.70 | 22.99 / 22.44 |
| Q3_K | 0.012 | 95.8% | 12.79 | 24.11 / 24.06 |
| IQ3_XXS | 0.021 | 94.7% | 12.84 | 23.47 / 23.12 |
| Q2_K | 0.035 | 92.7% | 13.00 | 23.66 / 23.72 |

Q3_K is the best of them: +5-7%, at a cost below the KL ~0.1 that a change of kernels gives.

**Not shipped (2026-11-22, the user's call): the quality cost is not worth +5-7%.** Two caveats found while checking
whether to ship it, and both resolved the same day. (1) The KLs in the table above were measured against a reference
taken as repeating, and the engine was found not to repeat at long context (**KL 0.12 - 0.14, perplexity +-1.3%** over
8,190 rows; `docs/hy4/SPEED_WORK.md`, *Run, and the answer is: the path does not repeat*). The cause turned out to be
the indexer's pool pick going through `ggml_top_k`, whose radix-select kernel places the keys tied with the k-th by
`atomicAdd` and the long pool is all such ties; **it is fixed and is now the default** (sorted pick, `--indexer-topk`
for the old way), and the configuration repeats at **KL 0.00000 / top-1 100.00%**. The table above was taken at a
4,096-token window, where the pick never lands on ties and the run does repeat - so its numbers stand, and re-measured
at the long window (8,190 rows of `local/hy4-text-ids.py` text, the configuration's flags, 13 layers' experts in RAM)
`--ram-requant q3_k` gives the same direction: **KL 0.02874, top-1 agreement 92.81%, perplexity 35.5641 -> 35.4973**.
(2) The speed gain was measured on a short prompt (400 tokens, `-c 256K`); the 16K real-text configuration runs
~21 tok/s, not ~23.

Tried and left out: gathering the sliding layers' window rows in short steps (22.2 / 22.3 tok/s with, 23.0 / 22.5
without); the per-head `attn_k_b` / `attn_v_b` as F16 for the prompt's GEMMs (267.6 -> 267.0 tok/s).
`--host-moe` (opt-in): the experts in RAM computed by a host thread the graph waits on through mapped memory
(`ggml_host_moe`, the GLM engine's doorbell), in place of the scheduler's split: the same text, +1% with MTP (the
CPU's expert products take 0.93 ms a layer, 40.7 MB of experts at ~44 GB/s against a measured 65 GB/s peak; more
threads do not help, 2 MB pages for the experts did not either). Split into publish-early / wait-late on 2026-11-22
(`GGML_OP_HOST_MOE_PUBLISH` + a ticket tensor, so the layer's shared expert runs inside the window; the graph without
`--host-moe` is unchanged): +3.4% on the prompt and -2.8% on the decode round, ~2,5 ms of the ~88 ms it costs, against
the 3,5 ms the shared expert could hide at most - so still opt-in, and it has never had the KL run against the
reference. `docs/hy4/SPEED_WORK.md`, *The publish / wait split*. The ggml part is
`tools/v41/ggml/host-moe-publish.patch`, on top of `hy4-long-context.patch`.

The ggml changes are kept as `tools/v41/ggml/hy4-long-context.patch`.

## Where a long prompt's time goes (2026-11-20)

16K tokens at 256K in 1,024-token steps, ~41 s (`HY4_STEP_STATS=1`, rocprofv3 kernel and copy traces): ~25 s of
kernels on the two cards; ~10 s of copies: ggml's op offload (`GGML_OP_OFFLOAD_MIN_BATCH`, 32) copies the routed
experts of the 13 layers kept in RAM (~17 GB) to card 0 at every step, at full bus speed (a hipMemcpy test gives
28.4 GB/s to card 0, 3.3 GB/s to card 1 on its x2 link, pinned or pageable alike); 2.4 s waiting for the Engram rows
read from the SSD.

- The iHC mixes (`hc_*_fn`, F32 [12288, 8]) ran in prompt steps as F32 GEMMs with 128x128 tiles (1.2 ms a call, ~85
  a step); taken as streams x fn and transposed they go to the matrix-vector kernel: 383 -> 389 tok/s, KL 0.087.
- Building the step's masks on the host only when the graph uses them, and `--pin-experts` (the CPU's mapped weights
  registered with `hipHostRegister`): no measurable change (392-395 tok/s).
- The Engram table in RAM (`--engram-place ram`) at 256K first paged the machine as on 2026-11-18. The cause: the
  weights were copied to VRAM straight from the mapping, ROCm locks the source pages of such copies and keeps them,
  and the process held 62 GiB of shard 1 (RSS 108 GB with the table) that neither `MADV_PAGEOUT` nor
  `MADV_DONTNEED` could drop. Now a weight goes through a 64 MB heap buffer and its pages are dropped from the
  mapping and the page cache: RSS after loading 65.9 -> 25.2 GB (the CPU's experts), upload 28.5 -> 15.5 s; the
  table in RAM then pages no more than on the SSD, prompt 393.6 -> 417.8 tok/s, decode within noise.
  `--engram-place auto` chooses RAM again when the table and the CPU's experts fit with 12 GiB to spare.

## The prompt-step reserve, and 2,048-token steps (2026-11-20)

`--ubatch` sizes the compute buffers for the largest graph, a step at the end of the 262K context: `Impl::init` builds
that graph and calls `ggml_backend_sched_reserve`, halving the step when it does not fit. A failed reserve left its
buffers allocated: `ggml_gallocr_reserve_n_impl` frees a buffer and then fails its realloc, and the other backends keep
the sizes of the failed, larger graph. The smaller retry then had less VRAM than the first and failed too. Measured at
256K with `--ubatch 2048` (the configuration's other flags), the attempts asked for 1,732 MiB on card 1, 2,576 MiB on
card 0, **6,782 MiB** on card 0 (the leftover), 2,080 MiB, and ended at 128-token steps with 1,114 / 8 MiB buffers.
`Impl::init` now makes a fresh scheduler for each attempt and frees it on failure; 2048 falls back to 1024 in one step,
with 953 / 867 MiB buffers and 1.08 / 0.98 GiB free on the two cards.

A 2,048-token step needs 1,775 MiB on card 0 and 1,733 MiB on card 1 (against 953 / 867 at 1,024). With 29 routed-expert
layers in VRAM card 1 has only ~1.85 GiB free before the buffers, and the reserve fails; at 26 layers it has ~2.7 GiB
and 2,048 steps run the whole prompt (27 layers still fails). The cost is the three expert layers moved to RAM, plus a
layer cut that puts more of the trunk on the x2 card (16/27 -> 14/29 layers per card). Measured on 2026-11-20 with the
configuration's other flags, a 16,384-token prompt of fixed pseudo-random ids, 400 generated tokens, greedy, one run
each:

| VRAM experts / step | prompt (16K) | decode (400) | `HY4_STEP_STATS` compute |
| --- | ---: | ---: | ---: |
| 29 / 1,024 | 426.8 tok/s (38.38 s) | 20.96 tok/s | 56.42 s |
| 26 / 2,048 | 486.8 tok/s (33.66 s) | 20.03 tok/s | 52.53 s |

So +14% on the prompt for -4% in decode. Quality: a 4K PPL check of the same ids, 26 against 29 expert layers at the
same 1,024 steps (4,094 scored rows), gives KL 0.00000 and 100% top-1 agreement, so moving three expert layers to RAM
does not change the output distribution here. The greedy ids of the random 16K prompt at 256K are not reproducible
even across two runs of the unchanged baseline (first id 59718 against 81), so they are not a quality signal. Left off:
the configuration keeps 1,024 steps and 29 expert layers; `--ubatch 2048 --expert-layers-gpu 26` runs the faster one.
Note that this flag pair OOMs on card 1 at 4K (the placement puts 31 layers there, only 0.35 GiB free after the 2,048
buffers); it is for the 256K configuration it was measured on.

## Overlapping the offload copies (`--prefetch-experts`, 2026-11-20)

`HY4_COPY_DEBUG=1` prints the op-offload copies of the experts in RAM that ggml makes at every prompt step: per layer,
`ffn_gate_exps` and `ffn_up_exps` (297 MiB each) and `ffn_down_exps` (648 MiB), 13 layers. Gate and up are copied into
the SAME device region, down into another: those ~945 MiB of destinations are reused by all 13 layers, one split at a
time, just before that split's compute.

`--prefetch-experts` stages one copy at a time in a 660 MiB device buffer on a private HIP stream; the scheduler's copy
callback (`ggml_backend_sched_set_copy_callback`) turns each host->device copy into a short device->device copy from
staging, and issues the next copy on the stream while the current split computes. The first attempt gained nothing
because `hipMemcpyAsync` from the pageable mmap'd weights blocks: a micro-benchmark of a 1.2 GiB copy gave the call
returning in 45 ms from plain mmap against 17 us after `hipHostRegister` on the same mapping (copy itself 44 ms on card
0, 380 ms on card 1's x2 link). The flag now registers the mapped expert pages (`hipHostRegister`, as `--pin-experts`
does) and only then prefetches. A single staging buffer suffices because gate, up and down are copied in sequence;
two buffers (to stage two layers) would need ~1.3 GiB more VRAM and do not fit at 256K.

Measured on 2026-11-20, 256K, the configuration's flags, greedy, a 16,384-token prompt of fixed pseudo-random ids and
100 generated tokens, alternating runs:

| | prompt 16K | `HY4_STEP_STATS` compute |
| --- | ---: | ---: |
| baseline (`--pin-experts`) | 431.3 / 431.3 tok/s (37.99 s) | 42.00 / 42.33 s |
| `--prefetch-experts` | 479.3 / 480.7 tok/s (34.08-34.18 s) | 38.12 / 38.20 s |

So +11% on the prompt; decode is inside the noise (22.27 / 20.64 against 21.90 / 21.82 tok/s). A 4K PPL check of
the same ids against the same config gives KL 0.00000 and 100% top-1 over 4,094 rows: the staging delivers exactly the
same weights. The flag is on in both configurations. It needs about 1 GiB free on card 0 after the compute buffers:
at 256K with 29 VRAM expert layers there is ~1.1 GiB and all 585 copies are staged (0 fallbacks); at 64K with 31 layers
there is only 0.38 GiB, and `prefetch_init` turns the prefetch off instead of letting the graph compute run out of memory.

The gain adds to the 2,048-token steps of the previous section: `local/hy4-local-fast.json` (router id `hy4-flash-fast`)
runs `--ubatch 2048 --expert-layers-gpu 26 --prefetch-experts`. Same 16K prompt and 100 generated tokens: prompt
480.3 -> 528.9 tok/s, decode 21.67 -> 20.62, all 336 copies staged, `HY4_STEP_STATS` compute 35.29 s. Against the
configuration before these two changes (426.8 tok/s prompt) that is +24% prompt for about -2% decode; the three fewer
expert layers in VRAM make it an alternative rather than the default. (2026-11-21: `place()` charges the staging to
card 0's budget instead of the staging eating what `--reserve-mib` leaves, so card 0 should end with its ~1.1 GiB free
*plus* the staging rather than 0.45 GiB between the staging and cuBLAS - expected, not measured again yet. The 64K
case, where the flag turned itself off, has to be measured too.)

A 4,096-token step does not fit: its compute buffers need 3,480 MiB on card 1 (1,733 MiB at 2,048), so the reserve
halves the step to 2,048. Building the larger graph also ran out of the 24K-node graph context (`ggml_new_object`
assert); `GRAPH_NODES` is now 49K, so `--ubatch 4096` falls back cleanly rather than aborting. No speed gain.

## The wide-head flash attention on RDNA3: tried, crashes (2026-11-20)

The sliding layers' attention runs `flash_attn_tile<576,512>` (~5.6 s of a 16K prompt). ggml also has an MMA F16 flash
attention with compiled 576/512 instances and RDNA config cases (`fattn-mma-f16.cuh`), but on RDNA3
`ggml_cuda_get_best_fattn_kernel` takes that path only for `Q->ne[0] <= 256`: MFMA is CDNA-only and the AMD WMMA branch
is capped at 256. An opt-in env var was added to widen the gate to 576/512 and the kernel was rebuilt. An `AMD_LOG_LEVEL=3` trace of a
1,024-token prompt shows the launch of `flash_attn_ext_f16<576, 512, 2, 16, false, true, false>` with grid 1,024,
block 128 and 37,376 bytes of dynamic shared memory after a successful `hipFuncSetAttribute`, then rocclr creating a
~12 MB hostcall buffer and aborting on `pthread_create() failed` inside `libamdhip64`. Disabling HIP graphs
(`GGML_CUDA_DISABLE_GRAPHS`) or reducing the CPU threads does not help, and a rocgdb backtrace shows only the runtime's
internal abort frames. The crash is in ROCm 7.14's runtime, not in a ggml-side check (the kernel has no device-side
assert or printf). Forcing the small ncols=8 instance (9 KiB of shared memory instead of 37 KiB) did not help either, so it is not the
LDS limit. Reverted; the attention stays on the tile kernel. A fix needs a ROCm-level investigation.

## Decode: where the launches are, and what did not help (2026-11-20)

A rocprofv3 kernel trace of ~19 generated tokens at 256K, `local/hy4-local.json` as it is now, shows 31,436 kernel
dispatches and 417 ms of kernel time in a 1,162 ms window — 64% is gaps between kernels. By count the leaders are
`quantize_q8_1` (3,862), `copyBufferRectAligned` (3,467), `mul_mat_vec_q` Q8_0 (3,246), `rms_norm_f32` (3,342) and
`k_bin_bcast` (1,628, the learned norm weight); by time the matvecs and attention: `mul_mat_vec_q` 114 ms,
`flash_attn_tile<576,512>` 56 ms, `mul_mat_vec_q_moe` IQ2_XXS 69 ms, `mul_mat_vec_bf16_rows` 20 ms.

Tried a fused `rms_norm + weight` op (`GGML_OP_HY4_RMS_NORM_MUL`: one block per row, warp-shuffle sum of squares,
`__fmul_rn` twice so the two products round as the stock composite does; CPU fallback in ops.cpp). It is active
(the 1,024-token graph goes 4,961 -> 4,704 nodes) but the decode does not move: 16K prompt 476.9 -> 479.7 tok/s, both
inside the noise, and the quality moves (4K KL 0.035 against the composite, top-1 84.17% — the sum-of-squares order).
So the small launches are not what a decode round costs, and the op was reverted. The remaining decode levers are the
matvec and flash-attention kernels themselves (the WMMA path is blocked, see above), not the norm/quantize launches.

## The engine dies and the cards are loaded again (2026-11-20)

"What looks like the engine loading card 0's VRAM again" is the process dying and the server starting it again at the
next request: every start uploads `41.3 GiB of weights in 27 s`, reads the 59.0 GiB Engram table into RAM (4.9 s) and
registers 23.2 GiB of CPU weights - about 50 s before the first token. `local/router.log` names hy4's log in five
`the engine stopped unexpectedly (exit code -11)`: four on 2026-11-20, and one around 2026-11-17, long before
`--prefetch-experts` existed. The kernel log (`journalctl -k`) has the four of the 20th, 12:28:50, 20:43:58, 21:09:40
and 21:16:47, all at one instruction of one library:

    almond-engine[...]: segfault at 0 ip 000078d8238b9f78 sp ... error 6 in libamdhip64.so.7.14.60850-0000000[...]

That address is file offset 0x6b8f78 (the r-x segment maps 0x1ae000 -> 0x1af000) and it is a non-temporal copy loop
inside the HIP runtime; the faulting instruction is `movntdq %xmm0,(%rdi)` with `rdi = 0`, so the runtime is copying
into a buffer it does not have. The engine's log says nothing at that moment - the last lines of all three evening
runs are `ggml_backend_cuda_graph_compute: CUDA graph warmup complete`, the router was mid-generation
(`[high] thinking: ... 17.8 tok/s`) - and `ulimit -c` is 0, so there is no core to read.

`--prefetch-experts` is not the cause: the 12:28 crash is from a run without the flag and without registered pages,
same instruction. It does take room that nothing accounted for, though: the 660 MiB staging was allocated after the
compute buffers, and at a byte-for-byte identical layout (`ROCm0 953 MiB ROCm1 865 MiB CPU 304 MiB`) card 0 went from
`1.10 GiB free` to `0.45 GiB free`; one of those runs prints `ROCm pool[0]: alloc of 44.89 MiB failed, flushing
107.58 MiB of cached buffers and retrying`. What the four crashes have in common is host memory rather than VRAM:
59.0 GiB of Engram table in RAM, 41.3 GiB uploaded, 23.2 GiB registered and so unreclaimable, on 122 GiB with no swap
(at 23:50:48 the kernel OOM-killed an almond-engine with `anon-rss:86120808kB`). A runtime allocation failing inside
its own copy path is the plausible way to reach that null pointer. Not confirmed: it needs a run with
`--engram-place ssd`, and one where the machine is not holding two engines at once.

Fixed in the prefetch while here (not re-measured yet):

- `place()` charges the staging to card 0's budget, so `--reserve-mib` is no longer eroded by it; the 64K case, which
  turned the prefetch off for lack of room, may now fit instead.
- a copy larger than the staging is left to ggml: the largest expert copy is `ffn_down_exps` at 648 MiB against a
  660 MiB staging, and `hipMemcpyAsync` does not check - it would have written past the end of the buffer.
- the recorded copy order is kept per step length, and a step that does not match it goes to ggml whole and is
  recorded again for the next step. Before, the record was cleared in the middle of a step, and since `prefetch_first`
  skips steps under 32 tokens without re-arming, every decode step appended its 39 expert copies to that record
  without limit and the prefetch stayed disarmed until the next prompt step.
- a HIP failure on the private stream now says which call (`prefetch off: ...`), and `--verbose` prints
  `prefetch: 660 MiB staged on card 0, X GiB free with it` and, beside the pinned weights,
  `X GiB of RAM available after it`. The `prefetch: N copies staged, M fallbacks` summary printed only in `~Impl`,
  which a signal skips: none of the crashed runs shows it.
- `serve/server.py`'s death note no longer blames RAM for every death: on a signal it names it (SIGSEGV, SIGABRT,
  ...) and says where the faulting instruction is (`journalctl -k --since '-2 hours' | grep -i segfault`); the RAM
  hint stays for SIGKILL. That is why four segfaults were read as "running out of RAM" for a day.

## The seven deaths of 2026-11-21, and the graphs that never settle

This boot's kernel log (`journalctl -k`) has seven segfaults, 10:13:05, 10:16:27, 10:24:57, 10:28:28, 10:32:25,
10:37:05 and 10:39:38, all `almond-engine[...]: segfault at 0 ... error 6 in libamdhip64.so.7.14.60850-0000000` at
the same address in the library. The kernel prints the faulting file offset in the `[...]`: `0x6b8f78`. Checked
against the library: its r-x segment is file offset `0x1aea70` at vaddr `0x1afa70` (so file = vaddr - 0x1000) and the
instruction there is `movntdq %xmm0,(%rdi)` with `%rdi = 0` - one loop of a 16-bytes-at-a-time non-temporal copy that
ends with `sfence` and a read of the last byte. The runtime is writing into a buffer it does not have.

All seven are runs of the 256K configuration: `local/serve-hy4-engine.log` has nine `engine started` lines between
09:59 and 10:37, and pairing every segfault with the start before it gives lives of 2.5 to 6.5 minutes. The two runs
in `local/serve-hy4-fast-engine.log` (10:04:00, 10:04:50) are not among them.

What each of them was doing, from its own log: the last line is `ggml_backend_cuda_graph_compute: CUDA graph warmup
complete`, and that is exactly where ggml starts a capture (`cudaStreamBeginCapture`, then `cudaGraphInstantiate` or
`cudaGraphExecUpdate`). And the graphs never settle: the run started 10:37:10 and dead at 10:39:38 (148 s) printed
**3,728 `CUDA graph warmup reset` and 3,779 `CUDA graph warmup complete`**. Counting decode rounds only (one MTP
round ~73 ms, the prompt not counted), those 148 s hold about 2,000 rounds and so roughly 8,000 graph computes over
the two cards (draft and trunk, two backends each) - so `ggml_cuda_graph_update_required` finds different node
properties on nearly every one, and the ~4,700-node graph is captured over and over. `shape()` is why some of that is
real: `n_pool`, `n_top` and `n_sel` grow with the position (`n_kv` is rounded to 256, the others are not).

The room the runtime is asked for is small and the cards have almost none: the 256K configuration prints
`prefetch: 660 MiB staged on card 0, 0.45 GiB free with it` and `compute buffers: ROCm0 953 MiB ROCm1 865 MiB CPU 304
MiB; card 0 0.45 GiB free; card 1 1.01 GiB free`; the `-fast` configuration, which is not among the deaths, leaves
`card 0 4.75 GiB free; card 1 0.71 GiB free`. So: a copy loop in a runtime allocation that could not be answered,
inside a capture loop that runs thousands of times, on a card with 460 MiB left. `local/serve-hy4-engine.log` already
shows the pool being squeezed: `ROCm pool[0]: alloc of 70.88 MiB failed, flushing 106.27 MiB of cached buffers and
retrying`. Host RAM is not the story for these seven: the engine prints `46.0 GiB of RAM available after it` at the
pin, and the kernel log has no OOM entry for them.

Reproduced and measured the same day (below, "What the measurement said"): the cards do run out of room, the capture
loop is what eats it, and a ceiling on the captures stops both the death and the creep.

Added to measure it, all off by default:

- `GGML_CUDA_GRAPH_DIFF=N` (vendored ggml, `tools/v41/ggml/graph-recapture-guard.patch`): prints the first N nodes
  whose properties changed and what moved - `data`, `ne`, `nb`, `srcK-data/ne/nb`, or the node count. That names the
  tensor that forces the recapture.
- `GGML_CUDA_GRAPH_RECAPTURE_MAX=N`: after N property changes, when they are more than a quarter of that graph's
  computes, the graph is not captured any more: its captured graph and its executable are destroyed and the graph runs
  node by node. A capture of 4,700 nodes costs more than the launch it saves, and it is what asks the runtime for
  room. `GGML_CUDA_GRAPH_STATS=1` (already there) counts launched / captured / direct.
- `HY4_VRAM_STATS=N` (`src/hy4/hy4_engine.cpp`): free VRAM per card every N steps, so a session's creep is visible
  instead of inferred.
- `--min-free-mib N` (hy4): the prompt step halves - the same fallback a compute buffer that does not fit uses - until
  every card would have N MiB left with that step's compute buffers and, on card 0, the prefetch staging in. It counts
  what the reserve plans rather than what `hipMemGetInfo` says, because ggml's pool caches what a failed attempt
  frees. With `--verbose` the engine now warns under 512 MiB:
  `hy4: warning: N MiB left on the cards with the compute buffers in: the HIP runtime may find no room there` - for
  the 256K configuration N is the order of 460, the `0.45 GiB` the load line prints once the 953 MiB step buffers and
  the 660 MiB staging are counted. Without the flag the placement is exactly as it was.
- `tools/hy4/graph-vram.hip` (`hipcc -O2 -o /tmp/graph-vram tools/hy4/graph-vram.hip`): the same questions with no
  model and no ggml - what a capture + `hipGraphInstantiate` takes on a card, what recapturing gives back, what K
  graphs kept alive at once cost, whether `hipGraphExecUpdate` leaks, and what the runtime does when the card has no
  room: an error code, or a signal like the engine's. It takes a `fill_MiB` argument to put the card where hy4 leaves
  it (`/tmp/graph-vram 0 23552 8 4096 20`).
- `local/hy4-crash-check.sh`: 16,384-token prompt and 3,000 greedy tokens, four runs - as it is, with
  `GGML_CUDA_GRAPH_RECAPTURE_MAX=200`, with `--min-free-mib 1024`, and as it is again - with the cards' used VRAM
  sampled every 5 s.

### What the measurement said (2026-11-21, this machine)

`-c 262144`, the 16,384-id prompt of `local/hy4-prefetch-check.sh`, greedy, `--ubatch 1024 --prefetch-experts`,
power-tune on, the cards to themselves.

- **It reproduces.** Unchanged, the engine dies of SIGSEGV (exit 139) between 1,536 and 1,920 generated tokens -
  17.9K to 18.3K tokens into the session - with graph warmup lines as its last words, as the server's runs have.
- **The room really goes.** `HY4_VRAM_STATS=64`: card 0 free MiB at steps 64, 128, 192, 256, 320, 384, 448 =
  100, 66, 32, 76, 44, 10, 0; card 1 at steps 64..768 = 704, 640, 562, 498, 436, 376, 300, 238, 174, 102, 38, 0.
  After both read 0 the death follows within a minute.
- **Why ggml recaptures.** `GGML_CUDA_GRAPH_STATS=1` over 8,192 computes: 4,719 launched, **1,624 captured**, 1,849
  direct, 16 keys. `GGML_CUDA_GRAPH_DIFF=4` says what moves: **1,548 of the 1,874 trunk nodes** (615 of 1,604 in the
  draft graph) change their `data` pointer between two computes, and the two small split graphs 36/95 and 57/126
  through `src1-data`. Almost nothing of the graph is stable, so it is captured again and again - ~1,600 captures a
  session, ~50 a second while the prompt runs.
- **What the runtime keeps.** `tools/hy4/graph-vram.hip` with 4,096-node graphs, the card deliberately left with
  968 MiB free (hy4 leaves it 465): every capture + `hipGraphInstantiate` cycle keeps about **2 MiB on the card**, and
  `hipGraphDestroy` / `hipGraphExecDestroy` do not give it back (20 re-captures: -10 MiB; 20 `hipGraphExecUpdate`:
  -18 MiB; 8 graphs kept at once: 2 MiB each). Down to that 968 MiB it answers with error codes, never a signal.
  1,624 captures x ~2 MiB is ~3.2 GiB - which is what the cards had.
- **The ceiling fixes it.** `GGML_CUDA_GRAPH_RECAPTURE_MAX=200`: 6 of the 16 keys print "201 recaptures in 389
  computes: not capturing this graph any more"; as ggml destroys those graphs the room comes back (card 0 204 MiB and
  card 1 452-678 MiB free at 17.4-19.9K tokens, against 0 and 0) and the run finishes 4,000 tokens, where without it
  the engine dies at ~1,500-1,900.
- **And costs nothing.** Same prompt, alternate pairs, 1,000 tokens: baseline prompt 445.3 / 469.3 tok/s, decode
  23.28 / 22.77; capped prompt 469.4 / 469.2, decode 23.37 / 23.75. At 400 tokens: baseline 464.6 / 22.00 and
  469.5 / 22.85, capped 469.4 / 23.03. So the prompt unchanged and the decode +2-4%: re-capturing 1,874 nodes every
  other compute was worth less than running those nodes.

Now in the configurations: `GGML_CUDA_GRAPH_RECAPTURE_MAX=200` in the `env` of `local/hy4-local.json`,
`local/hy4-local-vision.json` and `local/hy4-local-fast.json`. `--min-free-mib 1024` cannot be met at 256K (card 0
has 465 MiB with the 1,024-token step and the staging, and halving the step never reaches 1 GiB), so the flag now
takes the largest step that fits and says so instead of failing the load. Whether a larger `--reserve-mib` (one
expert layer less in VRAM) buys anything above the ceiling: not measured.

### The 140K death of the same afternoon: the ceiling was per graph

The session that resumed the 124,817-token conversation (`local/serve-hy4-engine.log`, run started 12:19:12,
`GGML_CUDA_GRAPH_RECAPTURE_MAX=200` in) got to ~140K tokens and died. Its log says why the ceiling was not enough:
**17 different graphs reached their 200 recaptures** ("201 recaptures in 475 computes: not capturing this graph any
more") - the ceiling is per graph *key*, and a new key comes with every 256-token bucket of `n_kv`, so what those 17
graphs hold is the order of 17 x 200 x 2 MiB ~ 6.8 GiB. Then ggml's pool starts thrashing
(`ROCm pool[0]: alloc of 44.89 MiB failed, flushing 105.87 MiB of cached buffers and retrying`, then
`64.58 MiB failed, flushing 72.21 MiB`), and the kernel, on card 0 (PCI 03:00.0):

```
nov 21 12:30:39 amdgpu 0000:03:00.0: [drm] *ERROR* Not enough memory for command submission!
```

After that the engine stopped making progress but stayed alive, holding its VRAM, until it was killed:

```
nov 21 12:35:12 amdgpu 0000:16:00.0: VM memory stats for proc almond-engine(1298498) ... is non-zero when fini
```

That is the 2.99 GiB the cards kept until `./stop.sh`: nothing is leaked by the driver, the room belonged to a
process that had stopped working but had not exited. A dead engine takes its cards with it only when it exits.

Two changes from that, both for the room, not for speed:

- `GGML_CUDA_GRAPH_CAPTURE_MAX=N` (vendored ggml): a ceiling for the **whole process** next to the per-graph one -
  after N graph rebuilds (captures or updates) of any kind, nothing is captured any more. In the hy4 configurations:
  128, with `GGML_CUDA_GRAPH_RECAPTURE_MAX` down from 200 to 16. Bounded room either way: ~128 x 2 MiB rather than
  17 x 200 x 2 MiB.
- `--reserve-mib 2400` was tried in `local/hy4-local.json` and `local/hy4-local-vision.json` (was 1400) and **put
  back to 1400**: once the captures are bounded the room stops being the problem. Both measured at 256K with a 16K
  prompt: 2400 leaves the cards 1.84 / 2.04 GiB free with 27 of 42 layers' experts in VRAM, 434.3 prompt and 24.54
  decode tok/s; 1400 leaves 0.45 / 1.01 GiB, 29 of 42 layers, 469 prompt and 23.4-23.75 decode. So 2400 buys about
  1 GiB of slack on each card for two expert layers and 7% of the prompt. The session it was meant for - resumed at
  17:43 with the ceilings on and 1400 - read prompts up to 171,646 tokens (past the 140K where it had died) and its
  log holds **no** `ROCm pool[0] ... failed, flushing` line at all, where the 140K death had two within a minute of
  dying.

Measured after both, the same 16,384-token prompt at 256K, greedy, 1,499 tokens: the captures stop at **82** (12,288
computes: 432 launched, 82 captured, 11,774 direct; 18 keys, 44 graphs switched off by the two ceilings) and the
room **stops moving**: `card 0 1628/24560 MiB free card 1 1834/24560` from step 512 to step 1408, where before the
ceiling the same trace fell 100 -> 0 MiB in 448 steps. Speed: prompt 434.3 tok/s against the 469 of `--reserve-mib
1400`, decode 24.54 against 23.37-23.75 - the prompt is the price of the reserve, not of the ceilings (the ceilings
alone were free: measured +2-4% decode above). So the order to try if a session is not enough: ceilings first (free),
reserve second (a few percent) - and the configurations ship 1400. The flat trace above is the 2400 run's; at 1400 the
same question was answered by the real session instead (171,646 tokens read, no pool flush line), not by
`HY4_VRAM_STATS`.

## What is not here

Video: Almond's server takes no video input for any model, so the template's video markers are not reached. The CPU path's numbers above are from before the
engine on the cards existed.

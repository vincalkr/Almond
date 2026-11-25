# Inkling 2 Mini: our own engine (design, 2026-11-24)

`almond-inkling` runs llama.cpp's `inkling2_loop` graph: about 7,400 kernels a token, 30 tok/s on card 0
(`NOTES.md`). This file is the plan for Almond's own engine for the model, written for the RX 7900 XTX (gfx1100), with
no ggml and no other engine's files. The techniques come from reading project-maya's `src/ik2` and Strata's
`engine/src/inkling2` (both MIT); the code is written here.

What it is measured against: project-maya's README reports 109.1 tok/s decode and 2,267 tok/s prefill for UD-Q4_K_XL
at 128K on an RX 7900 XTX (a user's report, not measured here).

## The model, as llama.cpp computes it (the reference)

From `build-inkling/llama.cpp/src/models/inkling2.cpp` and `inkling2-loop.cpp` (master `30ec35ae`), with the GGUF's
values for UD-Q4_K_XL:

- 24 stored layers, hidden 2304, 24 query heads and 8 KV heads of 128, no RoPE. Layers 5, 11, 17, 23 are full
  attention, the rest a 1,024-token window. Loop: layers 6-17 run 3 times (48 visits); each pass of the loop starts
  again from the prelude's output and blocks.
- Embedding Q8_0 (201,024 padded rows; the first 200,058 are the vocabulary), then RMS norm (`token_embd_norm`).
- Block residuals: every 6 layers (`ip % 6 == 0`, `ip` the stored index) the layer's input is pushed to `blocks` and
  that layer does not add its attention output to the stream. `mix(blocks, cur, query)`: RMS-normalise (no weight,
  eps 1e-6) every block and cur, scores = query . state / sqrt(2304), softmax over the states, sum of p_i * normalised
  state_i. SWA layers mix one query, full layers three (one each for Q, K, V); layer 0 has none.
- Loop messages (passes 1 and 2): `mix(bank, bank_out, msg_query)` with 4 queries (q, k, v, ffn), where `bank` is
  the previous pass's blocks after the first core block and `bank_out` its output. `control(x, m)`: g = down .
  [rms(x); rms(m)] (rank 64), x - tanh(forget . g) * x + tanh(gain . g) * m. Weights F32.
- Attention inputs: SWA: norm (`attn_norm`) once (three times when the loop's messages make Q, K, V differ); full:
  `attn_{q,k,v}_inp_norm`. Q = wq x0, K = conv(wk x1), V = conv(wv x2), R = attn_rel x0 (16 per head).
- Short convolution, 4 taps, per channel: out = x + (w0 h-3 + w1 h-2 + w2 h-1 + w3 x) [+ dynamic], times
  1/sqrt(n*4+1) (n = 2 with the dynamic taps of the full layers: per KV head, `attn_{k,v}_conv_proj` . x1 gives 4
  taps). The state is the last 3 inputs per channel: K, V, attention output, FFN output.
- clip(x, s, th): x*s, then x * clamp(th/(||x||+1e-6), 0, 1). Q with `attn_q_clip`, K with none (th 22.627), R with
  `attn_rel_clip` (th 8). V: RMS norm without weight (eps 1e-6). Full layers: Q and R times
  tau = 1 + 0.1 * log(max(1, (pos+1)/8192)).
- Sinks: sink = `attn_sink_proj` . x0 per head (BF16). Bias of a key at distance d = pos_q - pos_k:
  R . `attn_rel_bias`[d] - sink when d < extent (1,024 on every layer), else -sink. Attention: softmax over the keys
  of (q.k/sqrt(128) + bias) with one extra zero logit (the sink), times V. Output times sigmoid(`attn_gate` . x0),
  then conv(wo . out) with `attn_post_conv`.
- MoE: mix with `ffn_res_query` (plus the ffn message on loop passes), norm, router F32: sigmoid(logits); pick the
  top 8 of sigmoid + `exp_probs_b`; weights = the unbiased sigmoids normalised to sum 1, times 8. Experts: gate_up
  (2304 -> 2x1152) times its per-expert scale, SITU = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25), down times its scale.
  Layers 0-19: NVFP4 experts; 20-23: Q8_0. Then conv with `ffn_post_conv` and the residual.
- Output: mix with `output_res_query`, norm, Q8_0 head, softcap 30, drop the padding.

## Engine shape

- `src/inkling/native/`: a GGUF reader (mmap, the tensors uploaded as they are: Q8_0 and NVFP4 blocks), HIP kernels
  and a driver that speaks the engine's serve protocol (the same lines as `almond-inkling`). Built by `build.sh` into
  `build-inkling/almond-inkling-native`; an opt-in configuration until it measures better than the llama.cpp path.
- Step 1, correctness: plain kernels, one op at a time, F32 activations, a token at a time. Checked against
  llama.cpp on the same file: PPL on `local/inkling-measure/docs.txt` (llama.cpp: 6.1675) and greedy text.
- Step 2, decode speed: fuse per visit (mix + norm; controller; projections with their convolutions; clip / norm;
  attention with the bias inside the kernel and split-K decoding; router + top-8; experts gate_up + SITU; down +
  combine + conv + residual), the whole token as one HIP graph.
- Step 3, prompts: batched kernels (WMMA for the experts and the attention), the bias still inside the kernel, so the
  buffer does not grow with the context (the 128K limit today).
- Quality first: every step keeps the PPL and KL against the Q8_0 reference (`NOTES.md`); anything that changes the
  output measurably stays off by default.

## Step 1 measured (2026-11-24)

`build-inkling/almond-inkling-native` (`src/inkling/native/ink_native.hip`, `--score IDS.txt`), plain kernels, F32
activations, the weights as the GGUF stores them. `almond-inkling --score` runs llama.cpp on the same ids with the
same rule (PPL of the second half's next tokens; `--cpu-weights` keeps the Q8_0 in RAM). Text: the first 1,024
tokens of `local/inkling-measure/docs.txt`, card 0:

| | PPL | same top token as Q8_0 | KL from Q8_0 (128 positions, full logits) |
|---|---|---|---|
| llama.cpp, Q8_0 (reference) | 4.1994 | - | - |
| llama.cpp, UD-Q4_K_XL | 4.1911 | 996 / 1,023 | 0.00641 |
| ours, UD-Q4_K_XL | 4.2009 | 988 / 1,023 | **0.00520** |

KL between the two engines on the same file: 0.00530 (max 0.044). Ours is closer to the reference than llama.cpp's
run of the same file: llama.cpp quantizes the activations to 8 bits for its quantized matmuls, ours keeps them in F32.
Speed of these plain kernels, one token at a time: 26.3 tok/s (llama.cpp: 30.6).

## Step 2, decode (2026-11-24)

Same text and check as step 1; `--bench 300` times the decode alone (300 tokens fed after the 1,024, no logits read
back), card 0. Every row keeps the step-1 output: logits within 2.5e-5, KL from Q8_0 0.00520, PPL 4.2009.

| Change | Decode tok/s at 1K | Kernels a token |
|---|---|---|
| step 1 (plain) | 26.3 | 2,789 |
| the token as one HIP graph, the position on the device | 29.2 | 2,789 |
| vector loads in the row dots, a parallel router | 45.9 (at 78 tokens) | 2,789 |
| several matvecs a launch; norms in the mixes; one kernel for clip/norm/cache write; convolution with the residual; controllers in groups; no copies | 36.1 (65.5 at 78) | 944 |
| flash-decoding attention (3 query heads a KV head, splits, relative bias once a visit) | 64.3 | 1,040 |
| NVFP4 values from an LDS table, more attention splits on the window | 75.6 | 1,048 |
| split-K for the controllers' down projections and the router; 1,024-thread mixes | 82.6 | 1,040 |
| NVFP4 scales from a table in global memory | **84.2** | 1,040 |

llama.cpp on the same card and file: 30.6 tok/s. Tried and dropped (slower): the E2M1 values computed instead of
looked up (the experts went from 51 to 54 us), the expert inputs copied to LDS (36 KB a block cut the occupancy),
the scale table built in LDS by every block. The Q8_0 rows repacked at load (`T_Q8P`, aligned 8-byte loads) did not
change the speed and are kept (simpler loads).

Where the time goes now (rocprofv3, 10.2 ms of GPU time a token at 1K): experts gate_up 2.1 ms (540 GB/s of
24 MB) and down 1.2 ms, the projections 2.1 ms (about 575 GB/s), the head 0.53 ms (930 GB/s, at the bandwidth),
mixes 0.9 ms, attention 0.8 ms; about 1.9 ms more between kernels. Not done yet: KV cache in F16/q8 (F32 today: 128K
does not fit), the serve protocol, batched prompts.

## Step 3, prompts (2026-11-24)

`--bench-prefill N --chunk C` times a fresh N-token prompt (the text repeated) fed in chunks of C, only the last
logits read; card 0, UD-Q4_K_XL. Correctness: the 1,024-token text in chunks against token by token, logits of the
511 scored positions (`--dump-logits`): every row below keeps PPL 4.2009, the same top token at all 511 positions, KL
from the token-by-token run at most 2.4e-10, and KL from Q8_0 0.00520 (128 positions). The first chunked version
(FP32 GEMMs) differed from position 908 on (KL max 1.6e-3, a near-tie in the expert pick); the matrix-core version
below no longer does.

| Change | 4K tok/s | 32K tok/s |
|---|---|---|
| chunks of 128, FP32 tiled GEMM, experts grouped by expert | 331.6 | - |
| projections, experts and head on the matrix cores (WMMA 16x16x16 F16: the weights as their exact quantized integers, the activations as two F16 parts, each 16-wide K step times its block scale in F32) | 708.6 | - |
| chunks of 512 | 1,028.5 | 513.2 |
| prompt attention: 16 tokens x 3 query heads a block share each key and value tile in LDS (F32, same online softmax) | 1,131.7 | 847.5 |
| each wave reuses its decoded weights for 32 entries (experts) / 32 tokens (projections) | 1,289.2 | - |
| chunks of 1,024 | 1,456.8 | 1,051.4 |
| a compact list of expert tiles (no empty blocks), the next K step's weights and activations loaded while this one computes | 1,523.0 | - |
| attention: float4 reads, the values read once for the wave's 6 rows | **1,618.4** | **1,205.2** |

llama.cpp through the server, same card and file: 444.7 tok/s at 4K (ubatch 128). Decode is unchanged (84.1 tok/s
at 1K depth). Tried without a gain: tiles of 64 entries for the experts, the half-waves sharing the weight decode
(kept: no slower, half the decode work).

Where the time goes at 4K in chunks of 1,024 (rocprofv3): the experts 43%, the attention 20%, the projections 18%.
The experts run at about 50 TFLOPS counting both F16 parts; the two parts are what keeps the activations at F32
precision, and are kept. At 32K the attention grows; next steps there: the scores and P . V on the matrix cores with
the same split, and query tiles of 32.

## Step 4, behind the server (2026-11-24)

`almond-inkling-native --serve` speaks almond-inkling's protocol (GEN / PP / T / DONE, RESUME, STOP, CONVCLEAR,
SAVE / RESTORE) with the same sampler; the chat template, tool calls and effort levels stay in
`tools/inkling/serve/serve_inkling.py`, unchanged. A prompt that extends the cache keeps it; otherwise the newest of 4
prompt-end checkpoints inside the shared prefix is restored (the convolution histories and the window layers' rings,
about 360 MB in RAM each at 128K with chunks of 1,024: a full attention layer's rows below the checkpoint are never
written again). SAVE writes that state plus the full layers' rows below the cache's end. Checked: after
SAVE / CONVCLEAR / RESTORE a greedy continuation is the same as the whole prompt read from scratch; an agent's turns
reuse the previous prompt (115 -> 2,499 -> 4,883 tokens, the earlier ones cached).

KV cache. The model is sensitive to it: on the 1,024-token text, KL from the same engine with F32 KV is 0.00481 with
q8_0 KV (`--kv-q8`) and 0.00191 with F16 (`--kv-f16`); llama.cpp's own q8_0 KV against its F16 KV: 0.00464, the same
loss. At 128K, F32 does not fit one card; F16 does (0.91 GiB free with chunks of 1,024), so the configuration uses F16.

Against Q8_0 (128 positions of the text, full logits), as each path is served:

| | KV | KL from Q8_0 | same top token |
|---|---|---|---|
| llama.cpp (configuration removed) | q8_0 | 0.00573 | 122 / 128 |
| ours (`local/inkling-local.json`) | F16 | **0.00557** | **125 / 128** |
| ours, F32 KV (not servable at 128K) | F32 | 0.00520 | 123 / 128 |

Speed through the server (`/v1/chat/completions`, streamed, temperature 1.0, top_p 1.0, card 0, 128K configuration,
256 tokens out; prefill = prompt tokens / time to the first token, decode = the rest):

| prompt | llama.cpp prefill | ours prefill | llama.cpp decode | ours decode |
|---|---|---|---|---|
| 772 tokens | 461 tok/s | 1,320 tok/s | 29.2 tok/s | 64.3 tok/s |
| 5,658 | 553 | 1,502 | 29.1 | 55.2 |
| 47,659 | 337 | 1,047 | 27.2 | 54.2 |

The decode through the server is below `--bench`'s 84 tok/s: every token copies the 200K logits back and sorts them on
the CPU (the same sampler as llama.cpp's path), and the decode at depth reads more keys. Chat, a tool call and a
four-turn agent run answered correctly. Quality is the same or better and every speed is higher, so this is the
default; the llama.cpp configuration (`inkling-llama`) was removed on 2026-11-24 as slower.

## Step 5, profile and fixes (2026-11-24)

Card 0, the served configuration (128K, F16 KV, chunks of 1,024). Engine numbers from `--serve` (GEN, DONE's timings,
no HTTP).

Decode, rocprofv3 at 1K depth: 11.1 ms of GPU time a token, about 1,040 kernels. Projections (`k_mmv`) 2.64 ms,
expert gate_up 1.97 and down 1.12, attention 1.10, mixes 0.87, the head 0.53, the rest small kernels (router,
controllers, norms, relative bias, split merge) at 4-7 us each. The weights read a token are about 3.1 GB (experts
1.7 GB, projections 1.3 GB, head 0.2 GB): at 960 GB/s that bounds the decode near 9 ms, about 110 tok/s, so the
matvecs at 485-580 GB/s and the small kernels are where the rest goes.

| Change | served decode, 1K / 6K, temperature 1 | 
|---|---|
| before | 65.4 / 57.5 tok/s |
| attention splits from the keys present (read on the device, the graph adapts) instead of the context size: at 128K a 1K-deep decode read all its keys in one split per head; splits of at least 32 keys | 77.6 / 76.4 |
| logits to pinned memory; with no truncation asked (top_k 0, top_p 1, min_p 0) the token is drawn from the whole softmax in two passes, no sort (the sorted path kept the best 4,096: this one is the exact distribution) | **78.7 / 77.4** (temperature 0: 83.2) |

Logits unchanged (KL 7e-12 from the token-by-token path at 64K context). Tried and dropped, slower or no change: 2 or
4 rows a wave in the matvecs (x read once for them: 79.4 and 70.9 against 83.2), unrolled matvec loops (80.4), the
router as one wave with no block barriers (same speed).

Prefill:
- Matrix-core GEMMs: the activations' low F16 part is kept times 2^12 (exact) and summed in its own accumulator, so
  values below about 0.12 no longer lose it to F16 subnormals. KL from the token-by-token path (F32 KV, 511
  positions) went from max 2.3e-10 to 6.5e-11.
- Prompt attention on the matrix cores for the F16 cache (`k_attn_pf_w`): keys and values are exact in F16; Q and
  the probabilities as two parts (the low one scaled the same way); one wave per 16 (token, head) rows. Its output
  is within 1e-5 (values up to 5) of the FP32 kernel on the same inputs, the size of FP32 summation-order noise.
  32K: 1,205 -> 1,286 tok/s; 4K: 1,613.
- One F16 part for the activations (no low part) was 15% faster and cost KL 0.0018 from the token-by-token path:
  dropped.

The F16 cache makes a whole-text diff unusable as a check: K and V come out of the prompt GEMM and the decode matvec
within 1e-7 of each other, a few values round to the other F16 neighbour, and the model carries that on (prompt vs
token by token with F16 KV: KL 0.0005-0.0015, already with the FP32 attention kernel; two chunk sizes with the same
kernel: 0.0003). The check is KL from Q8_0: 0.00566 (124/128 same top token) here, 0.00557-0.00575 for every F16
variant, against 0.00573 for llama.cpp's served q8_0 path.

Where it stands against project-maya's reported 109 tok/s decode / 2,267 tok/s prefill (not measured here): 83 / 1,613
at 4K on the engine. Decode is bound by the matvecs' bandwidth and ~1,000 launches; prefill by the experts' GEMM at
about 50 TFLOPS with the two activation parts, and at 32K by the attention's softmax work per 16 keys. Not done yet:
DSpark in this engine, fewer launches per decode token, larger key blocks in the prompt attention.

## Step 6, DSpark in this engine (2026-11-24)

`--dspark FILE [--dspark-n N] [--dspark-max-temp T]` (opt-in, `local/inkling-dspark-local.json`,
`./almond -m inkling-dspark`): Unsloth's DSpark drafter for this model, written here from llama.cpp PR #32218's
dflash / dspark graphs (no llama.cpp code linked):
- Context: after every target pass the drafter reads the target's residual stream at the inputs of visits 5, 23 and
  41 and the final norm's output (the GGUF's `dflash.target_layers` 5, 23, 41, 48), mixes them per layer
  (`ctx_mix`, RMS norm), and keeps K (normed, NeoX rope, base 1e6) and V of every committed token in its own F16 cache
  (layers 0 and 2: a 1,025-token window ring; layer 1: the whole context). The same mix with `markov_ctx_*` is the
  Markov head's context state of the last committed token.
- Draft: the block [last token, 7 mask rows] through the 3 dense layers (Q/K norms, rope, attention with no causal
  mask over the context and the block, SiLU FFN), the target's head on rows 1..n, then the recurrent Markov head
  (`joint_proj`: state, `markov_w1[prev]`, the row's hidden, the context state -> gate / candidate / output;
  `markov_w2` adds its bias; the argmax is the draft token and the next `prev`). Greedy drafts.
- Check: the target feeds [y, drafts] as one batch (a captured HIP graph per batch size), samples every position with
  the request's own sampler and keeps the drafts up to the first token it did not sample itself; the output follows
  the same distribution as without the drafter. The convolution states of the rejected tail are rebuilt from the
  state before the batch and the batch's inputs, which the verification records (`k_spec_roll`); the KV rows past
  the kept tokens are rewritten later. `--spec-test` runs a text as verify(4) / keep 2: KL from the token-by-token path
  max 2e-11 over 511 positions. At temperature 0 with F32 KV the 300 generated tokens are the same, token for token,
  with and without the drafter. With the served F16 KV they can part at a close pick (here at token 129, top two 0.15
  apart): the verification batch computes K / V with another summation order (about 1e-7), a few values round to the
  other F16 neighbour, and the model carries that on, as between prompt chunks and the decode (step 3). At that token
  the token-by-token path picks what the drafted run picked.
- Small batches (the verification, the drafter's block) go through a matvec that reads each row once for up to 16
  tokens (`k_mmvT`), and the experts through the decode kernels per entry; the tile GEMMs wasted most of a 64-token
  tile there (a 4-token verification went from 58.7 to about 20 ms).

Engine (`--serve`, card 0, 128K, F16 KV, chunks of 512, 600-token prompt of the measurement text, 400 tokens out):

| temperature | without | `--dspark-n 2` | 3 | 4 |
|---|---|---|---|---|
| 0 | 82.6 tok/s | 109.5 (206/258 accepted) | 115.3 (213/270) | **117.8** (227/303) |
| 1.0 (drafting forced on) | 78.7 | 75.8 (41/140) | 75.2 (57/210) | 73.3 (53/240) |

A verification of 4 tokens costs about 1.8 single steps (each token reads its own 8 experts; grouping the batch's
experts was slower: few are shared), so drafting pays above roughly 40% accepted. At temperature 1.0 about 20% of
the greedy drafts hold, so `--dspark-max-temp` (0.8) leaves those requests undrafted, and within a request a moving
average of the accepted share under 25% turns drafting off for 32 tokens.

Through the server (`/v1/chat/completions`, streamed, a 786 / 5,672-token prompt asking for code, 512 tokens out):

| temperature | without | DSpark |
|---|---|---|
| 0 | 78.5 / 77.0 tok/s | **123.0 / 144.3** |
| 0.6 | 78.5 / 77.0 | 97.5 / 134.9 |
| 1.0 | 78.6 / 77.0 | 78.1 / 76.5 (not drafting) |

Prefill with the drafter: 1,152-1,366 tok/s (chunks of 512, the drafter's 0.5 GB and its cache leave 0.61 GiB free)
against 1,287-1,520.

Found on the way: the server does not forward `temperature=0` (serve/server.py `sampling_keys`: an absent
temperature is the engine's greedy default), but both Inkling engines defaulted to 1.0, so a request at temperature
0 was sampled at 1.0. Both now default to greedy, as hy4_serve does; requests without a temperature still get the
configuration's 1.0 from the server.

## Step 7, fewer launches (2026-11-24)

Fused, with each kernel's arithmetic kept (logits unchanged: KL 3.5e-13 from the token-by-token path, the same as
before): the one-token K / V convolution and the relative bias (every query head, once R is clipped) into `k_qkvr`
(96 launches a token fewer), the controllers' output and the RMS norm after them into `k_ctrl_apply_n` (48 fewer).
Decode at 1K depth, 128K F16 configuration: 83.2 -> 84.1 tok/s (+1%). Tried and dropped: merging the attention splits
in the last split's block (an atomic counter, so no `k_attn_combine` launch): 72.5 tok/s, the merge of 64 splits by
one block per KV head is slower than its own launch. What is left of the small kernels is the mixes (120 a token,
about 7 us each, dependent on the layer before) and the router / controller matvecs.

## Step 8, decode experts and the prompt's expert GEMM (2026-11-24)

Profile of a decode token at 1K depth (128K, F16 KV, rocprofv3): 10.4 ms of kernels, of which the attention matvecs
(`k_mmv`, 3 a visit) 2.7 ms at about 510 GB/s, the experts' gate/up 1.6 ms at 560 GB/s, their down 1.1 ms at 510 GB/s,
the output head 0.53 ms at 930 GB/s; the rest is small kernels. A plain read of 400 MB runs at 956 GB/s on this card.

- Kept: the decode experts' gate/up (NVFP4) in `k_exp_gu_nv<4>`: four rows a wave share each x load, and the E2M1
  values are decoded with two byte permutes instead of 16 table reads (the same values, products in F32).
  Logits with F32 KV: KL 4.4e-13 from the previous build. Gate/up 42.8 -> 35.3 us a launch.
- Kept: the prompt's expert GEMM (NVFP4) folds each block's E4M3 scale into the weights before the matrix cores (a
  doubled E2M1 value times an E4M3 scale has at most 6 significant bits, so it is exact in F16) and sums 64 values
  on the matrix cores before adding them in F32. Carrying the matrix-core accumulator over the whole row was faster
  (1,663 tok/s) but not exact: KL 1.4e-5 from the token-by-token path (max 1.6e-3), so it is not used. Kept version:
  KL 2.3e-12 (the same as before).
- Tried and dropped: a down kernel walking the 8 entries' blocks as one list (80.1 against 82.6 tok/s); prefetching
  the next layer's attention weights into the Infinity Cache, from a second stream in the captured graph (70 tok/s:
  the branches did not overlap) or from extra blocks of the experts' or the attention's launch (the matvecs went from
  18.5 to 10.8 us, the launch carrying the prefetch got slower by as much: no gain); expert tiles of 64 entries
  (1,082 tok/s); DSpark drafts of 5 tokens (89.3 against 96.7 tok/s with 4).

Engine, 128K, F16 KV, 1K-token prompt (`--bench 300` after `--score`, and `--serve` with 256 tokens out):

| | before | after |
|---|---|---|
| decode, `--bench` | 83.2 tok/s | 85.3 |
| `--serve`, temperature 0, 1K / 3K prompt | 83.0 / 82.3 | 85.0 / 84.4 |
| `--serve`, temperature 1 | 78.9 | 80.8 |
| prefill 4K, chunks of 1,024 | 1,582 tok/s | 1,623 |

With DSpark the greedy text of this prompt changes (the F16 KV rounds a close pick the other way), so the accepted
share changes with it (127/251 against 104/216) and the two runs do not compare token for token: 96.7 / 102.1 before,
94.3 / 98.8 after.

## Step 9, images (2026-11-24)

The llama.cpp configuration was the only one that could have taken images and it never did; this engine now reads
them. The image encoder stays llama.cpp's mtmd (its `inkling2v` projector: a 23-layer ViT with a CLS token, 2-D RoPE,
2 x 2 patch merge and a GELU projector into the model's 2304 channels), built as `build-vision-inkling/bin/almond-vision`
against `build-inkling/llama.cpp`, with Unsloth's mmproj in F32 (the highest precision published: F32 / F16 / BF16;
`local/fetch.sh inkling-vision`), on the CPU, so card 0 keeps its 1.7 GiB for the 128K cache.

- Prompt: the chat template writes a picture as `<|content_image|><|unused_200054|><|end_message|>`; the server shim
  (`tools/inkling/serve/serve_inkling.py`) repeats the placeholder once per image token, only right after
  `<|content_image|>`. llama.cpp's mtmd places the rows right after `<|content_image|>` the same way, with
  ordinary positions (no 2-D positions in the text model).
- Engine (`GENI <max_new> [keys] [img_keys=..] <file> <ids>`): picture k's rows take the k-th run of placeholders;
  each row replaces its token's embedding after the embedding norm, as given (llama.cpp does not normalise raw
  embeddings). The cache matches a picture's rows on negative keys made from the picture's key and the row index,
  so a different picture never reuses a cached prompt; repetition penalties skip them.
- `local/inkling-local-vision.json` is `inkling-local.json` with the encoder; launch.py picks it when the encoder
  and the mmproj are there.

Checked against llama.cpp's server (`llama-server`, the same GGUF and mmproj, card 1) on three test pictures, both
at temperature 0 with the template's default effort, 400 tokens out: the same prompt length (331 / 221 / 544
tokens); shapes and text ("ALMOND 42"), a bar chart (Thursday, 9) and a drawn house. The first two answers are the
same text throughout, thinking included; for the third the thinking parts after 205 characters on a close wording
and the answer is the same sentence. Text without pictures: logits identical to the previous build (KL 0 over 511
positions), the server tests pass.

`local/inkling-dspark-local-vision.json`: the DSpark configuration with the same encoder; launch.py takes an Inkling
configuration's `-vision` twin when the encoder is there, so the router's `inkling-dspark` entry gets it. The encoder
is on the CPU, so card 0 has what the DSpark configuration had: 0.61 GiB free after the drafter. Through the server
at temperature 0 (the three pictures above and a code question, 400 tokens out at most), the answers with and
without DSpark are the same text, character for character; requests (prompt reading included) 58.0 -> 75.0,
49.6 -> 61.3, 41.2 -> 42.3 and 83.7 -> 109.5 tok/s.

## Step 10, the encoder on the card in lent memory; the prompt GEMM tried again (2026-11-24)

Images now work the way GLM's and DeepSeek's do: the encoder runs on the GPU only while a request's pictures are
encoded, in memory the engine lends it, and the engine then has its model back as if the encoder had not been there.

- Lending (`VLEND` / `VRECLAIM`, GLM's protocol): the expert tensors are placed in reserved virtual address ranges
  with their own physical memory (`hipMemAddressReserve` / `hipMemCreate` / `hipMemMap`). `VLEND` unmaps and
  releases the last ones until the encoder's measured need is free (2.5 GB in 2 ms in the test), `VRECLAIM` maps
  new memory at the same addresses and copies the tensors back from the mapped GGUF (87 ms for 2.5 GB). The
  addresses do not move, so the captured graphs stay valid, and the weights come back byte for byte: generations
  before and after a lend are identical, and a request that arrives while memory is lent takes it back first.
  Text speed unchanged (decode 85.2, prefill 1,623 tok/s; logits KL 0). `INK_NO_LEND=1` uses plain allocations.
- The encoder on the GPU was not right by default: against the CPU its rows ended at cosine 0.895 - 0.983. Dumping
  the graph's tensors (`ALMOND_VISION_DUMP=<dir>` in almond-vision) showed two causes: ggml's hipBLAS path picks F16
  products for this F32 encoder (the patch embedding already 2.8e-3 off), and the flash attention reads Q / K / V
  as F16. With `--f32-matmul` (GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 for the encoder's process only, so the ggml
  engines are not touched) and `--no-flash-attn`: cosine 0.9999998, relative error 1e-5 against the CPU without
  flash attention, and the screenshot in 0.59 s instead of 3.5 s.
- Without flash attention the attention's buffers grow with the square of the image tokens: measured 2.79 / 5.80 /
  8.93 GiB at 1,024 / 2,048 / 4,096 tokens (the model stops at 2,704). The configurations ask for 4,096
  (`max_tokens`), so no picture is shrunk, as on the CPU; the engine has about 13.7 GB of experts to lend.
- Through the server (`local/inkling-local-vision.json`, `"gpu": true, "no_flash_attn": true, "f32_matmul": true,
  "on_demand": true`): card 0 at 23,771 MiB before and 23,786 after the requests; a 1920x1080 screenshot's first
  token after 2.53 s against 4.55 s with the CPU encoder, a 1800x1350 photo 2.71 against 5.57 s; small pictures
  0.5 - 0.7 s slower (the encoder starts per request, which reads its 1.3 GB). The answers are the same as with the
  CPU encoder up to a word (the CPU encoder ran with its F32 flash attention); with DSpark
  (`local/inkling-dspark-local-vision.json`, 0.61 GiB free) the same four answers as without, character for
  character.

The prompt's expert GEMM, tried again before this: kept exact, nothing gained. Removing the low part of the
activations from the matrix cores took the kernel from 3,014 to 1,736 us (it is bound by the matrix cores: the
exact two-part product is 42% of it), removing the weights' decoding to 2,187 us; decoding with byte permutes
(the doubled E2M1 values as F16 have a zero low byte) and the E4M3 scale computed instead of read from a table
changed nothing (1,623 tok/s); two to four row fragments a wave, 1,633 / 1,385 / 1,239 tok/s. (A first version of
the computed scale had the exponent off by 2^7 and gave 1,817 tok/s with NaN logits - a reminder to check the
logits before the speed.)

## Step 11, the full attention K/V on the second card (2026-11-24, unfinished)

`--kv-gpu N` (opt-in; the default path and `local/inkling-*.json` are unchanged). The 8 full-context K/V slots
(layers 5, 11, 17, 23, with 11 and 17 visited 3 times by the loop) cost 8 x 8 heads x 128 x 2 (K, V) x 2 bytes =
32 KiB a token in F16; the 40 window slots keep 2,048 rows each (320 MiB) whatever the context. With `--kv-gpu N`
the full slots live on card N and their attention (`k_attn_split` / `k_attn_pf_w` / `k_attn_pf`, then
`k_attn_combine` with the gate) runs there: q, the gate, the relative bias, the sinks and the position are copied
over, the gated output comes back; card 0's `k_qkvr` writes the new rows straight into card N (peer access).
Decode over a far slot uses 256 splits instead of 64. The decode graph is off with far slots (`INK_FAR_GRAPH=1`
captures across both cards: same logits, 21 tok/s against 72 eager). 256K F16 is 8 GiB on card 1, 512K 16 GiB;
1M needs q8_0 (`--kv-q8`, 16.6 GiB on card 1), since 32 GiB of F16 fits neither card.

Checks, cards 0 and 1 (RX 7900 XTX), 32K context, the repository's own sources and docs as text:
- prefill, 4,096 tokens in chunks of 1,024: logits byte-identical to the one-card path (256 rows dumped);
- decode, 1,024 tokens one at a time: logits byte-identical too (KL 0, same top token at 256 / 256), also with the
  graph captured.

Speed (`--bench-prefill N`, F16 KV, chunks of 1,024; decode `--bench 200` at depth 1,024):

| context | one card | `--kv-gpu 1` |
|---|---|---|
| prefill 32K | 1,293 tok/s | 750 tok/s |
| prefill 128K | 775 tok/s | 538 tok/s |
| prefill 256K | does not fit | 390 tok/s (673 s) |
| decode, 32K context, depth 1K | 86.1 tok/s | 72.3 tok/s |
| decode, 512K F16 / 1M q8_0 context, depth 1K | - | 72.2 / 72.7 tok/s |
| 32K text scored, then decode at depth 32K (one card 128K F16; far 1M q8_0) | PPL 4.2943, 601 tok/s, decode 73.8 | PPL 4.2882, 434 tok/s, decode 63.7 |

The copies cost: every far visit of a 1,024-token chunk moves q, the gate and the output (12 MB each) over PCIe,
and decode pays 16 cross-card waits a token. The 1M q8_0 and 512K F16 runners allocate and run. Not measured yet (the run was stopped at 512K): prefill at 512K and 1M,
decode at depth, KL and needle retrieval at 256K / 1M, DSpark / images / checkpoints with `--kv-gpu`, the server
configuration. So 128K on one card stays the default.

### 256K by default, q8_0 KV (2026-11-24, the user's choice)

The configurations now use q8_0 KV (`--kv-q8`, 136 bytes a row against 256 in F16: 17 KiB a token for the 8 full
slots) and 256K on card 0 alone: `local/inkling-local.json` and `-vision` at 262,144 (0.64 GiB of VRAM left), the
DSpark ones at 196,608 (1.33 GiB left with the drafter; at 256K only 0.02 GiB were left). F16 stays in the engine
(`--kv-f16`) but no configuration names it. `local/inkling-1m-local.json` (router entry, opt-in): 1M, q8_0, the full
slots on card 1 (`--kv-gpu 1`, `"gpu": "0,1"`).

Measured, card 0, the repository's sources as text:
- 256K q8_0, one card: decode at depth 32K 73.96 tok/s (128K F16: 73.79), prefill to 32K 1,172 tok/s (1,293);
  PPL on the 32K text 4.2865 (F16 4.2943).
- 256K, full slots on card 1, a 262,000-token text with a passphrase at token 20,000 and the question at the end:
  the 11 answer tokens are each the top prediction, F16 and q8_0 alike; KL of q8_0 from F16 over 256 positions
  at depth ~131K: 0.0108 (max 1.20, same top token 253 / 256); PPL 1.8365 / 1.8360.
- 256K q8_0 on card 0 alone (`local/inkling-local.json`), the same 262,000-token needle text, chunks of 256: the 11
  answer tokens each the top prediction; PPL 1.8336, 277.5 tok/s for the whole scoring run.
- KL from the Q8_0 weights (llama.cpp, `almond-inkling --cpu-weights`), 8,192 tokens of the same text, 256 positions
  from token 4,096: q8_0 KV 0.00706 (250 / 256 same top token), F16 KV 0.00714 (248 / 256). On this text the q8_0 KV
  costs nothing measurable against the quantization's own loss.
- Through the server (`local/inkling-local-vision.json`, 256K q8_0, 0.64 GiB free; `/v1/chat/completions`, streamed,
  temperature 1.0, 256 tokens out; prefill = prompt tokens / time to the first token):

| prompt | prefill | decode |
|---|---|---|
| 5,867 tokens | 1,474 tok/s | 78.9 tok/s |
| 50,319 | 1,015 | 66.5 |
| 189,227 | 501 | 45.4 |
| a 520-token picture request (encoder on card 0 in lent memory) | 204 | 82.5 |

  The picture was described correctly (the llama.cpp logo); at 50K and 189K the model found the passphrase in the
  prompt and reasoned about whether to repeat it.
- `local/inkling-1m-local.json` (1M q8_0, full slots on card 1) filled to 524,000 tokens (the needle text, chunks of
  256, every position's logits): the 11 answer tokens each the top prediction; 163.3 tok/s for the whole scoring
  run (45 min); decode at depth 524K 28.2 tok/s (`--bench 100`). Not measured: a context filled to 1M (about 3 hours
  at this rate), DSpark and checkpoints at 256K through the server.

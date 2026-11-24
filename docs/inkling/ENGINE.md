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
| llama.cpp (`local/inkling-llama-local.json`) | q8_0 | 0.00573 | 122 / 128 |
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
default; `./almond -m inkling-llama` keeps the llama.cpp path.

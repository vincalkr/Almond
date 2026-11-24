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

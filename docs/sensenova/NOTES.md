# SenseNova-Loop-9B: architecture notes

Read from the checkpoint's `config.json` and `modeling_sensenova_loop.py` (sensenova/SenseNova-Loop-9B on Hugging
Face, the remote code that `transformers` 5.18 loads; github.com/OpenSenseNova/SenseNova-Loop is the inference
package around it) and from the header of the official `sensenova/SenseNova-Loop-9B-GGUF` files.

## Body

A Qwen3.5 text decoder (`qwen3_5_text`), the same layer kinds as Qwen3.8-27B with smaller sizes:

| | SenseNova-Loop-9B | Qwen3.8-27B |
|---|---|---|
| physical layers | 32 (3 Gated DeltaNet + 1 gated full attention, repeated) | 64 |
| hidden / FFN | 4096 / 12288 | 5120 / 17408 |
| attention heads / KV heads / head dim | 16 / 4 / 256, partial rotary 0.25 (64 dims) | 24 / 4 / 256 |
| DeltaNet | 16 key heads, 32 value heads, dim 128, conv 4 | 16 / 48 / 128 |
| vocabulary | 248320, untied head | same |
| MTP | 1 block (`mtp.safetensors`, `blk.56.nextn.*` in the GGUF) | 1 block |

RoPE: interleaved M-RoPE (sections 11/11/10), theta 1e7, and **YaRN** factor 2 over an original 262144
(`max_position_embeddings` 524288). YaRN changes the low frequencies at every position, not only past 262144, and
scales the rotated dims by 0.1 ln 2 + 1 = 1.0693 (both q and k), so it has to be applied from the first token.
ggml's rope does the same with `freq_scale` 0.5, `ext_factor` 1, `beta_fast` 32, `beta_slow` 1, `n_ctx_orig` 262144
(the correction range is computed over the 64 rotated dims in both).

## Loop schedule

`loop_settings` = {start 4, end 28, counts [2], eval_count 2}. The execution plan is

    0 1 2 3 | 4 .. 27 | 4 .. 27 | 28 29 30 31

56 layer executions per token from 32 physical layers. Each execution has its own cache: the KV rows of a full
attention layer and the conv window + recurrent state of a DeltaNet layer are kept per *execution*, not per physical
layer (`_VisitCache` maps execution index -> cache slot; the cache is built with `num_hidden_layers = 56`). Both
passes see the same positions. The loop boundaries fall on whole groups of four, so execution i is a full attention
layer exactly when (i + 1) % 4 == 0, as in the plain model.

In the GGUF, `block_count` is 57 (56 executions + the MTP block) and the tensors are named by the *first* execution
that uses them: `blk.0` .. `blk.27` for physical 0..27, `blk.52` .. `blk.55` for physical 28..31, `blk.56` the nextn
block; executions 28..51 have no tensors of their own. Keys `sensenova_loop.loop.{start,end,count}` and
`sensenova_loop.mhc.*` carry the schedule.

## mHC (two residual streams)

`mhc` = {start 4, end 28, streams 2, iterations 20, fixed_identity false, pre_scale 1.0}. Physical layers 4..27 are
`MHCDecoderLayer`s with two FP32 routers each, `mhc_attn` and `mhc_ffn` (weight [8, 8192], bias [8]; GGUF
`blk.N.hc_attn_fn.{weight,bias}`, `blk.N.hc_ffn_fn.*`, F32). On every visit:

- enter: at physical layer 4 the hidden state [E] is copied into both streams, R = [h, h] ([2, E]); otherwise the
  layer reads the [2E] it was handed.
- router(R): flat = R as [2E]; z = flat * rsqrt(mean(flat^2) + 1e-6) (no weight); logits = W z + b (8 values);
  pre = sigmoid(logits[0:2]); post = 2 sigmoid(logits[2:4]); mix = Sinkhorn(logits[4:8] as 2x2 row-major, 20
  iterations in log space, each a row logsumexp then a column logsumexp, then exp).
- branch = pre0 R0 + pre1 R1, then the usual input norm and attention (or DeltaNet).
- merge: R_i <- sum_j mix[i][j] R_j + post_i * branch.
- the same with `mhc_ffn` around post-attention norm and FFN.
- leave: at physical layer 27 the two streams are averaged back to [E]; otherwise R is passed on.

Because 4 enters and 27 leaves, *each* of the two passes enters and leaves the two-stream form; layers 0-3 and
28-31 are plain Qwen3.5 layers. The routers run in FP32 in the reference (`_keep_in_fp32_modules`), the residual
streams in the model dtype.

The 2x2 Sinkhorn is not converged after 20 iterations: the closed-form limit (x = sigmoid((l00 + l11 - l01 - l10)/2)
on the diagonal) differs from the 20-iteration result by up to 0.025 on random logits around the router's initial
bias, so the engine runs the 20 iterations exactly (one small ggml op, `ggml_loop_mhc`).

## GGUF tensor layout differences from qwen35

- `attn_qkv_gate.weight` [4096, 12288]: the DeltaNet's qkv (8192 rows) and its output gate z (4096 rows) in one
  matrix (qwen35 GGUFs have `attn_qkv` and `attn_gate`).
- `ssm_beta_alpha.weight` [4096, 64]: beta (32 rows) and alpha (32 rows) in one matrix.

Checked against the BF16 safetensors (layer 4, Q8_0 dequantized): `attn_qkv_gate` = [in_proj_qkv; in_proj_z] and
`ssm_beta_alpha` = [in_proj_b; in_proj_a] (max error 0.4% of the largest weight, the Q8_0 rounding), with the value
heads in llama.cpp's qwen35 order (HF head k*2 + j at GGUF j*16 + k, the same reorder in qkv's v part, z, beta,
alpha, dt, A and conv1d) - the convention the qwen35 path already reads. `hc_*_fn` are the HF `mhc_*` weights
exactly (F32, same row order).

## DFlash2 drafter

`sensenova/SenseNova-Loop-9B-DFlash2` reads `target_layer_ids` [3, 27, 51, 54], the outputs of those *executions*
(LightLLM's `LayerHiddenCollector.add(layer_index=i, hidden=<layer i's output>)` over the 56 execution layers), then
`final_postnorm`, the last hidden after the model's output norm. 27 and 51 are the mHC exits, whose output is already
the mean of the two streams, so every feature is 4096 wide (fc: 20480 -> 4096). In the GGUF `dflash.target_layers` is
[4, 28, 52, 55] (llama.cpp's convention: the layer whose input it is); the engine's trunk loop runs over executions, so
these index it directly. The config's `layer_types` mark the fifth layer `full_attention`, but LightLLM's DFlash2 layer
uses `sliding_window` for every layer, as the engine does.


## The prompt read again after an agent loop, and `preserve_thinking` (2026-11-26)

With the agent, a new user message after a run of tool calls read the whole conversation again (13,609 tokens in 8 s
in a session; 28,599 tokens in 20.5 s in a test). The model's template (the Qwen3.5 one) drops the reasoning of every
turn before the last user query, so when the next query arrives the turns of the previous tool loop are rendered
again without it: the prompt changes right after the previous query and everything after it is computed again. The
engine's checkpoints cannot help (a checkpoint in front of the answer to the last query was tried: it held, and saved
267 tokens of 28,599); llama.cpp does the same with these models.

`"preserve_thinking": true` in the configuration (opt-in, off in `local/sensenova-local.json`) renders the earlier
turns with their reasoning, as Qwen3.6's template variable of the same name does: the next prompt extends the last one.
The server adds the switch to a template of this family that has none (a no-op while it is not set). Measured on card
1 with the test above (8 tool steps, ~28.7K tokens, then a new query): 20.5 -> 0.79 s for the new query. Not measured:
the answers' quality with the reasoning kept (the model was trained with it dropped) and how much faster the context
fills.

Quality, measured the same day (`local/sensenova-preserve/preserve_ab.py`, card 1, temperature 0): 40 pairs of
MATH-500 problems, turn 1 problem A (its answer generated once and kept for both arms), turn 2 problem B graded. With
turn 1's reasoning dropped (the template) 38 of 40 right, with it kept 36 of 40; the two that differ (pairs 4 and 18)
both lost with it kept, none gained. Turn 2's prompt 482 -> 1,062 tokens on average, its answer 962 -> 1,015 tokens.
Two pairs are within what 40 can tell apart, but nothing points to a gain: `preserve_thinking` stays off by default.
The run planned 50 pairs; the arm without it stopped at pair 40 when the engine went silent for 300 s mid-answer
(2,263 tokens into a reasoning, the server ended it). The kernel log has one shader error on that card (card 1,
`0000:16:00.0`) at the second the silence began (`amdgpu ... sq_intr: error ... type 2`, 20:54:57, 300 s before the
server's message); it is the only `sq_intr` in this boot and in the five before it. The same request sent again twice
on the same card ran through (3,275 tokens, 74.5 tok/s, both times), so it is not the engine's path for that prompt.
Both cards run LACT's profile with a -200 mV voltage offset (`/etc/lact/config.yaml`; its base entry has -100 mV for
card 0), a likely cause of a lone shader error; not proven by a run at a smaller offset.

# Models: thinking levels and sampling

What each model gets, where it comes from, and what was measured.  Everything below is the server's behaviour with the
shipped configurations (`local/ds41-local*.json`, `local/glm-local.json`, `local/qwen-local.json`); a client sees the levels in `/health`,
`/v1/models` (the router) and `./almond`'s `/models`.  Reviewed 2026-10-29.

## The configurations now (2026-11-25)

One or two per model; the sections below keep the measurements of configurations since removed (the Qwen3.8-Flash
IQ3_S ones, `qwen27-local-256k`, the DSpine ones, `hy4-local-fast`, `inkling-dspark`), named as they were then.

| model | router name | configuration | cards | context |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | `ds41` | `local/ds41-local.json` (`-1gpu` with `--gpu 0`, `-vision` twins) | 2 | 256K |
| GLM-5.3-Flash | `glm` | `local/glm-local.json` (`-vision`) | 2 | 256K |
| Qwen3.8-Flash-Next | `qwen` | `local/qwen-local.json`, UD-Q4_K_XL, images on | 2 | 256K |
| Qwen3.8-27B | `qwen27` | `local/qwen27-local.json` (`-vision`), IQ3_S, MTP 3 | 1 | 192K |
| | `qwen27-q8` | `local/qwen27-q8-local.json`, Q8_0, `--tp` | 2 | 256K |
| SenseNova-Loop-9B | `sensenova-loop-9b` | `local/sensenova-local.json`, Q8_0, MTP 3 + DFlash2 5, `--spec auto` | 1 | 256K |
| Hy4-flash | `hy4` | `local/hy4-local.json` (`-vision`) | 2 | 256K |
| Inkling 2 Mini | `inkling` | `local/inkling-local.json` (`-vision`) | 1 | 256K |
| | `inkling-1m` | `local/inkling-1m-local.json`, DSpark, K/V on card 1 | 2 | 1M |

## DeepSeek-V4.1-Flash-1017 (GSQ-RCO IQ2_XXS, `ds41_generate`)

Source: DeepSeek's cloud API reference (api-docs.deepseek.com, create-chat-completion, `deepseek-flash`) and the
model card (`deepseek-ai/DeepSeek-V4.1-Flash-1017`: temperature 1.0, top_p 1.0).  The GSQ-RCO pack's card gives no
sampling and measured its benchmarks at reasoning effort 50.

| | thinking (default) | thinking off |
| --- | --- | --- |
| levels | low 50, high 75 (default), max 100, or 1-100 | none |
| temperature | 1.0, the request's ignored (the API: "no effect") | the request's, 0-2, default 1.0 |
| top_p | the request's, 0.95-1.0, default 1.0 | 1.0 |
| top_k / min_p | off (the engine's 256 best candidates) | off |
| penalties | none (the API ignores them) | none |
| max_tokens unset | 64K, 128K at max | 8K |

Client words: none / off / minimal = thinking off; low = 50; medium, high, xhigh = 75; max = 100; a number 1-100 as
it is.  An Anthropic `budget_tokens` under 8K asks for high, else max.

Beyond the API (in the configuration, not DeepSeek's):

- No thinking cap, as the API (the configurations' `reasoning_budget_by_effort` {low 1024, high 4096}, never measured,
  went in this review); a request's own `reasoning_budget_tokens` still caps it.
- `reasoning_close_retry`: a reply that ends inside its thinking is continued once with the thinking closed.
- `reasoning_loop_recovery: "recover"` does nothing for DeepSeek (it swaps a Qwen template sentence the prompt does
  not have).

`ALMOND_DS41_FREE_SAMPLING=1` passes a request's sampling through as it is (measurements).

## GLM-5.3-Flash (GSQ-RCO 3.0bit until 2026-11-02, then 3.5bit; the engine)

Source: Z.ai's API reference (docs.z.ai, chat completions, the GLM-5.3-Flash series); the GGUF's
`general.sampling.temp` 1.0 and `general.sampling.top_p` 0.95 say the same.

| | |
| --- | --- |
| levels | low, high, max (default); thinking cannot be turned off ("can only be enabled") |
| temperature | 0-1, default 1.0 |
| top_p | 0.01-1, default 0.95 |
| top_k / min_p | off (the engine's widest, 64 candidates) |
| penalties | none |
| max_tokens | at most 131072; unset = the rest of the context up to that |

Client words: none / off / minimal / `enable_thinking: false` / Anthropic's `disabled` = low (the API's nearest);
medium = high; xhigh = max.  An Anthropic `budget_tokens` under 2K asks for low, under 8K high, else max.

Context: `--max-context 262144` since 2026-11-03 (the GGUF's `context_length` is 1,048,576; it was 131072).  Measured
on both cards: the larger K/V leaves 3,452 expert slots in VRAM instead of 3,628; a 400-token reply at low with a short
prompt runs at 30.4-31.2 tok/s (30.5-30.9 at 131072); a 162,393-token prompt (GLM's own sources) is read and answered
in 289 s, decode 22.6 tok/s at that length, the answer on topic.
`ALMOND_GLM_THINK_OFF=1` brings back an Off level (low with the thinking closed at once: not Z.ai's, not measured);
`ALMOND_GLM_FREE_SAMPLING=1` passes a request's sampling through.

Measured, the server with MTP drafts and the expert profile, a fresh start each arm, six 600-token requests (three
code, one Italian prose, one English explanation, after one asyncio question), thinking low, tok/s:

| sampling | 1 | 2 | 3 | 4 | 5 | 6 | mean | drafts accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| temperature 0 (greedy) | 30.3 | 32.3 | 38.0 | 37.3 | 34.6 | 27.3 | 33.3 | 80.8% |
| Z.ai's 1.0 / 0.95, top_k off (the default) | 32.1 | 32.1 | 36.4 | 36.5 | 32.8 | 26.1 | 32.7 | 81.2% |
| 1.0 / 0.95, top_k 20 (the engine default) | 30.0 | 32.4 | 35.3 | 36.0 | 32.1 | 26.8 | 32.1 | 79.5% |

Sampling at the API's values costs the drafts nothing measurable (one run an arm: the means are within 4%).

## Fixed in this review

- ds41_generate kept a request's repetition / frequency / presence penalty for every request after it (one global
  state): each request now starts from the command line's.  Checked: greedy "apple" x10, then a request with
  repetition_penalty 3.0 (2 words), then the first again - the same 10 words.
- The server sent top_k 0 ("off") to DeepSeek as 64; DeepSeek's engine without one takes its 256 best candidates,
  which is what it gets now.

## Loops in the thinking (2026-10-30)

GLM-5.3-Flash at max, "cosa sono le terzine incatenate ? citando i versi": the 3.0bit quant does not remember Dante's
verses and tries again and again ("nella miseria; e, fuor di mia credenza" - hmm, no), each round a few words
different. Before: 16,000 tokens of thinking, no answer (twice).

- the loop guard (`serve/loop_guard.py`, MIT, unchanged; `"loop_guard": true`, `ALMOND_LOOP_GUARD=0/1`):
  a block repeated back to back gets a short repetition penalty, the third loop in the thinking closes it, a loop that
  comes back in the answer ends it. On this prompt it saw one loop only, at 13,831 tokens (the rounds are not exact).
- `"reasoning_loop_recovery": "close"`: the passage detector (#728, 12-word passages seen three times over the last
  2,000 words) closes the thinking with the budget's wrap-up and the model answers. Offline on the saved 16,000-token
  thinking it fires from ~3,200 tokens (coverage 0.52); live it closed at 2,560 tokens and GLM answered (4,191 tokens,
  finish "stop"). A normal max question (a train timetable, 572 tokens) was untouched.
- `clear_thinking` true by default (Z.ai's model card: "for chat scenarios, explicitly pass clear_thinking=true"):
  earlier turns' reasoning is not put back into the prompt; the reasoning after the last user message (tool calls)
  stays. A client's `chat_template_kwargs.clear_thinking` wins.

Both are on in `local/glm-local*.json`; DeepSeek's configurations keep theirs.

## Images

- DeepSeek: the encoder (0.9 GiB) loads on card 0 for a request with images and is freed after; the VRAM expert cache
  reopens into the room. Measured before the fix below: card 0 held 2172 experts before an image, 1872 / 1855 after
  (15.51 / 15.38 GiB free at the reseed against ~16.1 after a text prompt); text decode 23.2 tok/s before, 21.1, 23.7,
  23.7, 22.1, 23.5 after (one run each). Open: where the missing memory goes. Measured 2026-10-30 with
  ALMOND_VRAM_TRACE=1: a 531-token text prompt (layer-major batch) leaves 17.37 GiB free at the reseed (2104 experts),
  a 452-token prompt with a picture 15.46 (1865): ~1.9 GiB, taken during the image prompt's batch, not by the encoder
  (17.26 GiB free right after it was freed). Freeing the dense half's cached graphs and both allocators after it
  changed nothing (15.45): not kept. Clearing ggml-hip's buffer pool
  after the encoder (tried, 2026-10-30) gave back 112 MiB only and the next decode hit an illegal memory access: removed.
- GLM: the encoder (`almond-vision`, llama.cpp's mtmd with GLM-5.3's tower, built by `./build.sh` for HIP;
  `local/fetch.sh glm-vision` gets the mmproj F16 + vocab, 1.1 GB) starts only when a picture arrives, in 2.78 GB the
  engine lends it from the tail of card 0's expert pool (273 experts moved to colder slots, 27 dropped) and takes back
  after (`local/glm-local-vision.json`, which `./almond serve --model glm` picks when the encoder is there). Text
  decode, one
  run each: 29.2, 32.2 before, 29.8, 33.3 after the first picture, 29.2 after the second (cached: no encoder start).
- Qwen3.8-Flash-Next: the encoder (Qwen's mmproj BF16, GPU, `f32_matmul`, `no_flash_attn`) on card 0 only while a
  request's pictures are encoded (`local/qwen-local-vision.json`, `--vram-elastic`, now built on HIP: the expert cache
  through HIP's virtual memory). `VLEND` shrinks the expert cache by the encoder's need + 256 MB, `VRECLAIM` grows it
  back (measured 2026-11-24 on the engine: 3.8 GB lent in 60 ms, back in 149 ms). Precision against an F32 reference:
  the CPU BF16 encoder used before min cosine 0.563, the GPU in F32 0.995-0.9999. Through the server: the encoder
  needs 1.59 GB at 300 image tokens; three pictures answered right (shapes, a bar chart, a house), a 297-token picture
  prompt first token 2.06 s; card 0 used 25.41 GB at start, 25.48 GB after the pictures.

## Sessions on disk

- DeepSeek: `--conversation-disk local/conv-cache-rco` (as in Almond).
- GLM: `ALMOND_GLM_SLOT_KEEP=1`, `ALMOND_GLM_SLOT_DIR=local/glm-slots`: the conversation in the model is saved when
  the engine ends and taken back after a start. Measured: a 7,163-token prompt read in 13.6 s; after a stop and a
  start the same conversation plus a question was taken back from disk in 0.1 s and read 123 new tokens (2.3 s).
- Qwen: `"conversation_dir": "local/qwen-slots"` in its configuration (`conversation_dir_gib`, default 16): the server
  (`tools/qwen/serve/serve_qwen.py`) has the engine SAVE the conversation after each reply, drops that conversation's
  earlier file, and RESTOREs the longest saved prompt a new one starts with when the engine has not read it since it
  started. Measured on card 0 (an 18,071-token prompt, thinking off): the save is 489 MiB in 160-190 ms per reply;
  after a stop and a start the next turn (18,150 tokens) took 0.8 s with the resume (202 ms of it the RESTORE) against
  10.6 s read again.

## GLM: checkpoints in conversations with pictures (`ALMOND_GLM_CKPT_IMAGES`, 2026-11-03)

GLM kept its conversation checkpoints (the system prompt's end, the last turn's start, every 16K tokens) for text
prompts only: in an agent's conversation with a screenshot, a turn that did not continue the previous prompt exactly
went back to the system prompt's slot and read everything after it again. In a dsh session (GLM, tool screenshots)
every turn took back 9,545 tokens and read 40,935 to 83,568 again. With `ALMOND_GLM_CKPT_IMAGES=1` a prompt with
pictures gets checkpoints too, at the turn markers only, each keeping the hash of the pictures before it and taken
back only by a prompt with the same pictures there. Measured (a 4,563-token system prompt, a picture, a turn edited
after it, 6,113 tokens): 4,904 tokens from a checkpoint and 1,208 read in 5.1 s, where it took back 4,563 and read
the picture again. DeepSeek (the longest common prefix, pictures compared by the hash of their bytes) and Qwen (its
checkpoints keep the pictures' keys) had no such limit. On in `local/glm-local*.json`.

## GLM: long agent sessions (`ALMOND_GLM_CKPT_USER`, `ALMOND_GLM_SLOT_SKIP_TAIL`, 2026-11-05)

In a dsh session at 131K-137K tokens with 9,000-28,000 picture tokens three things still sent turns back to the
system prompt's slot or stalled them:
- the pictures' hash was taken over their embeddings, which the encoder recomputes after a restart: the server now
  sends each picture's file key (`img_keys=` in the GENI request) and the hash is taken over those;
- dsh's template drops the earlier steps' reasoning, so a prompt departs right after the last user turn's start,
  and the four checkpoints (least recently used out) all lay past it: `ALMOND_GLM_CKPT_USER=1` keeps a checkpoint at
  the last user turn and never evicts the longest one of those;
- before reading such a prompt the old conversation was saved to `local/glm-slots` (2.0-2.1 GB): 1.6-8 s at 94K-100K
  tokens, 97-140 s at 133K-138K with the RAM full (112 of 122 GB used, the engine 104 GB), the cards idle meanwhile.
  `ALMOND_GLM_SLOT_SKIP_TAIL=8192` skips that save when the checkpoint is the old conversation but for its last 8,192
  tokens or fewer (the dropped reasoning, which does not come back).
Measured in that session: 135,437 of 137,635 tokens from a checkpoint, 2,182 read in 10.1 s (216 tok/s); then turns
at 176K-181K taking back all but 144-5,216 tokens (3-16 s). The skip, measured in the same kind of sessions since:
48 saves skipped (the tail lost 64-7,597 tokens), 16 still made, 0.2-98 s each (median about 25 s) - each a prompt
departing more than 8,192 tokens before the old conversation's end at the end of an agent's run (113,162 of 128,882
tokens in common, 136,007 of 147,263), or a new conversation (a compaction: 19,310 of 158,740).
`ALMOND_GLM_SLOT_SKIP_SAME=0.85` skips those of the first kind too: a prompt sharing at least 85% of the old
conversation's tokens is the same one going on, and its tail is not kept (built, not measured yet).
`ALMOND_GLM_CKPT_WHY=1` prints, for a prompt that misses the snapshot, why each checkpoint was not taken. All on in
`local/glm-local*.json`.

## GLM-5.3-Flash against llama.cpp (2026-10-30)

Is the engine computing the model right, or are GLM's mistakes in an agent session ours? The same tokens through an
independent implementation: llama.cpp's own (ggml-org master 792fcf4, which has `glm5-next`), the same GSQ-RCO 3.0bit
file, experts in RAM (`--cpu-moe`), `llama-perplexity -c 512 --chunks 1` on `local/glm-ref/doc.txt` (its tokens are
`local/glm-ref/doc512.i32`'s, checked with `llama-tokenize`); Almond's engine `LOGP 257` on those tokens (the 255
llama-perplexity scores, the second half of the chunk), three runs:

| | PPL |
| --- | ---: |
| llama.cpp | 7.1997 |
| Almond | 7.1433, 7.0977, 7.0221 |

The same within the engine's run-to-run spread; at 8K too (llama.cpp 5.9528, Almond 5.9706 / 6.0006 / 6.0123 on the
first 8192 tokens of `local/glm-ref/long-docs.txt`, `local/glm-ref/logp.py`).

**The figures GLM garbled were ours: the tokenizer (2026-10-31).** Copying 40 sizes after ~10K tokens of text
(`local/glm-ref/copytest.py 8163 5 10000`, temperature 1.0, five runs, each prompt read anew): Almond 41 of 200 wrong
with MTP, 21 of 200 without; llama.cpp's server on the same file and template 0 of 200, and its prompt 12,471 tokens
against Almond's 13,153. The server tokenized GLM with the default `qwen35` pre-split, one digit a token, where the
GGUF says `glm4`: digits in runs of up to three (`\p{N}{1,3}`) and whole-piece tokens before the merges
(llama.cpp's `ignore_merges`). With `glm4` (`tools/almond_tokenizer.py`, `GlmTokenizer` in `serve_glm.py`) the ids are
llama.cpp's on 7 files, ~300K tokens (README and docs, Python, C++), the prompt is 12,471 tokens, and the copy is
0 of 200 wrong at ~12K, 0 of 120 short. The conversation caches built with the old ids are read again once.

**Both tokenizers against llama.cpp (2026-10-31).** `llama-tokenize` (ggml-org master 792fcf4, `--no-escape
--no-parse-special`) on each model's GGUF against the server's tokenizer: DeepSeek's (`joyai-llm`) gives the same ids
on 7 texts, 305K tokens (README and docs, Python, C++, and a mixed text: accents, Chinese, Japanese, Korean, emoji,
symbols, tabs and runs of spaces); GLM's (`glm4`) the same on those and on the mixed text.

## GLM-5.3-Flash GSQ-RCO 3.5bit (2026-11-02)

The same author's 3.5-bit build (`pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF`, 137.07 GB against 117.48; the card says
+4.49% PPL over Q8_0 against +9.07%, KL 0.071 against 0.142). `local/glm-pack.sh 3.5` makes its pack
(`models/glm53-gsq-rco/pack-3.5` then). It is GLM's model since 2026-11-02: the 3.0bit GGUF, its pack and kept
slots went to the trash, and `local/glm-pack.sh`, `local/fetch.sh glm`, `./almond serve --model glm` take the 3.5bit
(the expert profile it learned in these runs kept). (On 2026-11-11 the 3.0bit came back as a download option
for a day and was dropped again: GLM stays the 3.5bit build.) On the two 7900 XTX, power-tune on: 35.5 GB of experts on the cards as with 3.0, 84.6 GB
pinned in RAM (76 with 3.0), the rest read from the NVMe.

| | 3.0 bit | 3.5 bit |
| --- | ---: | ---: |
| decode, thinking low, 600 tokens, 3 prompts x 2 (`local/glm-ref/speed.py`) | 31.8 tok/s | 27.5 (25.2-28.9) |
| PPL, 16 windows of 512 of `long-docs.txt`, 4,080 tokens scored, 2 runs (`logp.py ... 512 2 CFG 16`) | 14.02, 14.01 | 13.43, 13.38 |
| 40 sizes copied after ~10K tokens (`copytest.py`) | 0 of 200 | 0 of 120 |

One window of 255 tokens (`doc512`) read 7.38-7.43 against 3.0's 7.02-7.14: one short document does not tell two
builds this close apart. Open: an 8K-token `LOGP` on the 3.5-bit model stopped in "returning the expert pool's tail"
(the engine's 180 s watchdog); a 12K prompt through the server read fine.

## GLM: leaving light misses out of the late layers (`ALMOND_GLM_MISS_SKIP_W`, 2026-11-02)

A routed expert not in VRAM whose share of the route's weights is under W is not computed, on layers from
`ALMOND_GLM_MISS_SKIP_FROM` on (the engine's option, from project-maya; the same idea as exllamav3-rocm-moe's tail skip).
GSQ-RCO 3.5bit, two 7900 XTX, power-tune on, `local/glm-ref/ab-miss-skip.sh`: decode through the server
(`speed.py`, 2 rounds of 3 prompts) and the PPL of 16 windows of 512 tokens of `long-docs.txt` (4,080 tokens; run to
run the exact model read 13.38-13.44):

| | tok/s | PPL |
| --- | ---: | ---: |
| exact | 25.9 | 13.44 |
| W 0.12, layers 23+ | 32.1 (+24%) | 13.48 (+0.3%) |
| W 0.16, layers 23+ | 33.8 (+30%) | 13.51 (+0.5%) |
| W 0.12, every layer | 36.7 (+42%) | 14.11 (+5.0%) |

The late layers take it: +24-30% for a PPL within about the run-to-run spread; every layer costs 5%. With W 0.16 from
layer 23 the 3.5bit model decodes faster than the exact 3.0bit did (31.8 tok/s) at a lower PPL (13.51 against 14.01).

## GLM: DFlash2 drafter against MTP (2026-11-02)

`ALMOND_GLM_DFLASH` (the engine's `--dflash`) with `Anbeeld/GLM-5.3-Flash-DFlash2-GGUF` Q8_0 (1.25 GB; 5 layers reading
trunk layers 6, 15, 25, 34, 43, block 8) in place of the MTP block, GSQ-RCO 3.5bit with the late-layer skip, `speed.py`
2 rounds: MTP 34.5 tok/s (76.4% of drafts accepted), DFlash2 26.6 (78.3%). The drafts are as good; the longer verify
windows cost more than they save with the experts across two cards and RAM. MTP stays; the drafter went to the trash.

## Qwen3.8-Flash-Next against another engine for it (2026-11-06)

An independent llama.cpp fork built for this model alone (its own GGUF, "IQ3_KS hybrid", 94.8 GB in 33 shards plus a
1.5 GB MTP head) against `almond-engine --qwen` on our IQ3_S, both on card 0, the same day, `local/glm-ref/speed.py`
(three prompts, temperature 0, 600 tokens, thinking low), two rounds each. It does not load our GGUF: its MoE path
stops at load on our expert types (IQ2_S / Q2_0).

| | the other engine, its GGUF | Almond, our GGUF |
| --- | ---: | ---: |
| MTP drafts (2 / 4) | 129.5 108.2 112.2 / 105.7 117.3 112.4, mean 114.2 | 99.1 84.1 87.1 / 94.4 88.2 89.7, mean 90.4 |
| no MTP (Almond: `--spec 4` without `--mtp`, which it needs) | mean 63.1 | mean 51.8 |

Drafts accepted are alike (about 350 of 600 tokens on either), so the difference is the time of a step: about 21.7 ms
against 27 with drafts, 15.9 against 19.3 without. A kernel trace of one 400-token answer without drafts (rocprofv3,
which slows both): 838 kernels a token against 2,130, the GPU busy 13.8 ms a token against 17.4. Where the 3.6 ms are:

| ms / token on the GPU | other | Almond |
| --- | ---: | ---: |
| hyper-connections | 1.70 | 2.60 |
| experts' matrix-vector products | 1.85 | 2.24 |
| router, top-k and the wait for the CPU's experts | 3.59 | 3.94 |
| experts copied into the cache while decoding (`fetch_blobs`) | - | 0.54 |

The hyper-connections' gap is the weights, not the kernels: its file holds `hc_*_down`, `hc_*_up` and `hc_*_inject` as
Q8_0, ours as BF16 (as the checkpoint): 0.67 against 1.26 GB a token, read at about 390 and 480 GB/s. Its attention
weights are Q8_0 too. Measured what Q8_0 hyper-connections cost: our GGUF with only `hc_*_down`, `hc_*_up` and
`hc_*_inject` requantized from BF16 to Q8_0 (54.82 -> 54.22 GB), the other fork's `llama-perplexity` on the CPU
(`-ngl 0`; its GPU path refuses our expert types), `local/glm-ref/long-docs.txt`, 16 x 512 tokens, against the
unchanged GGUF's logits:

| | Q8_0 hyper-connections | the unchanged GGUF again |
| --- | ---: | ---: |
| PPL ratio | 1.0014 +/- 0.0077 | 1.0000 |
| mean KL | 0.0704 +/- 0.0018 | 0.0000 |
| 99% KL | 0.552 | 0 |
| same top token | 87.1% | 100% |

A KL of 0.07 is the size of a whole quantization step (IQ4_XS against Q8 on another model, 0.077): the
hyper-connections stay BF16.

Prefill, the same prompts to both (slices of `local/glm-ref/long-docs.txt`, a different start each run, max_tokens 1,
client wall time), each engine fresh, two runs a size (tok/s):

| prompt | other engine, its GGUF | Almond, our GGUF |
| --- | ---: | ---: |
| 11,626 / 11,161 tokens | 1,443 / 1,838 | 1,676 / 1,728 |
| 45,780 / 44,940 tokens | 1,952 / 2,053 | 1,836 / 1,848 |

Fewer promotions during decode (`--adapt-every 8`, or `--adapt-swaps 0`) did not help (with drafts, tok/s, two rounds
alternated): base 87.6 / 89.2, every 8 86.8 / 87.7, no swaps 88.5 / 90.4. The defaults stay.

## Qwen3.8-Flash-Next (GSQ-RCO IQ3_S, 2026-11-02)

The model (`qwen4exp`: 48 layers, Gated DeltaNet with sparse attention every fourth layer, hyper-connections, the PLE
n-gram table, 512 experts with 10 routed) as ISTA-DASLab's GSQ-RCO IQ3_S GGUF (two shards, 83.6 GB), its pack
(`tools/qwen/iq_pack.py`: the 1.5 GB of float weights, the tensor index and the tokenizer) and the MTP block from the
BF16 checkpoint's 31 `mtp.*` tensors (`tools/qwen/mtp_fetch.py`, `mtp_pack.py` with Q2_0 experts, `mtp_rt.py`), on
card 0 (`local/qwen-local.json`). The engine is `src/qwen`, the Qwen path of the engine this model ran on before,
brought into `almond-engine` (`--qwen`) on Almond's ggml.

Thinking and sampling as the model card (Qwen/Qwen3.8-Flash-Next) gives them, set by `tools/qwen/serve/serve_qwen.py`:

| | thinking (default) | thinking off |
| --- | --- | --- |
| levels | xhigh (default), medium, low | off (`enable_thinking` false) |
| temperature / top_p / top_k | 1.0 / 0.95 / 20 | 0.7 / 0.8 / 20 |
| presence penalty | 0 | 1.5 |

Client words: high and max are xhigh; none / off are thinking off. A request's own sampling values win.  Measured
(the same question, two runs a level, 6,000 tokens of room): thinking off 38-51 tokens with no reasoning, low 97-103
characters of reasoning, medium 117-224, xhigh 276-315; low to xhigh answered right.

What changed on the way in, beside the names (`almond::qwen`, `ALMOND_*`):

- ggml 0.26's MMQ takes the tile width (`J_best`) where the older one took the columns to optimize for and chose it:
  `src/qwen/prefill/moe_mmq.cu` chooses it as ggml's own host code does (`mmq_tile_width`).
- The window size (`ALMOND_VERIFY_MAX_T` 16) is a PUBLIC definition of `qwen_core`: `generate.cpp` is compiled at
  the top level, and a directory-wide definition left it with other struct layouts than the libraries (the server
  stopped at load on "the stage's layer range ... wrong").
- `ggml_backend_cuda_context` is compiled with `GGML_HIP_GRAPHS` as ggml-hip is (so are DeepSeek's and GLM's now,
  docs/ENGINE_REVIEW.md), its host shim is weak as GLM's and
  DeepSeek's (ggml-hip's definitions win), and the MMQ context lives as long as the process: ggml-hip's destructor
  segfaulted at exit after the run had printed everything.
- The pack files' header (`# almond native experts v3`, `# almond hadamard v1`) is read whatever the word before
  "native experts" (packs written before say `strata`).

Checked against the previous build of the same engine (its binary, on the same files):

- one-shot, the first 512 tokens of `local/glm-ref/doc.txt` (`local/qwen-ref/oneshot.sh`), greedy, 64 tokens: the
  same 64 ids from both, at 50.8-51.0 tok/s against 48.9;
- the pack and the MTP block written again by the tools in `tools/qwen/`: `dense.bin`, `conversions.json`, the
  tokenizer and every row of `index.txt` / `native_experts.txt` the same bytes (the headers say `almond`); the MTP
  block's `dense.bin`, `dense.txt` and `experts.bin` the same bytes; the regenerated pack gives the same 64 ids.

Through the server, the same configuration with either binary (thinking low, temperature 0, 600 tokens, three prompts:
code, Italian, English; two rounds alternated, a fresh start each; the server's unit tests ran beside the second round), tok/s:

| engine | round 1 | round 2 | mean |
| --- | --- | --- | ---: |
| `almond-engine --qwen` | 100.1 79.1 81.1 | 99.3 81.7 82.8 | 87.4 |
| the previous binary | 98.1 84.1 80.5 | 99.3 84.9 83.6 | 88.4 |

The same within the runs' spread (greedy answers differ run to run, and with them the drafts accepted). The decode
expert cache hit 89-91% of lookups; 8,996 experts (17.06 GiB) in VRAM, 546 MiB free with everything loaded. The
previous engine's own log on this machine, in agent sessions at 8-63K context: 49-78 tok/s.

Images: the encoder (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`) on the CPU through `almond-vision` (12 threads, up to 300
image tokens), the configuration `local/qwen-local-vision.json`; a 357-token request with a logo described correctly
(its text read) in 6.7 s with 203 tokens of answer. A tool call (`read_file`) came back as a call with its arguments.

Not measured yet: perplexity (the native path dumps no prompt logits and the engine has no `LOGP`), long-context
copying, an agent session.

## Qwen3.8-27B (dense, GSQ-RCO IQ3_S, `--qwen35`, 2026-11-07)

The dense 27B (GGUF architecture `qwen35`: 64 layers, three Gated DeltaNet layers to one gated full-attention layer,
a dense SwiGLU FFN of 17408, its nextn block in the same file) runs on its own path, `src/qwen35/qwen35.cpp`: ggml
graphs on the vendored ggml (HIP), `./almond serve --model qwen27` (`local/qwen27-local.json`; the server shim is
Flash-Next's `serve_qwen.py`, the tokenizer is Flash-Next's with the 27B's own chat template in
`models/qwen38-27b/tokenizer`). Model files: `models/qwen38-27b/` (the ISTA-DASLab IQ3_S + MTP GGUF, 11.28 GiB, and
optionally Tencent's DSpine drafter, 1.16 GiB, and the BF16 mmproj, 0.87 GiB). `local/fetch.sh qwen27` (and
`qwen27-q8`, `qwen27-dspine`, `qwen27-vision`) downloads them; the tokenizer folder is exported from the GGUF
(`tools/almond_tokenizer.py`: vocab, merges and token types identical to the folder above) with the official chat
template kept in `tools/qwen/data/qwen27_chat_template.jinja` (unsloth's Q8_0 carries a copy without a `| safe`).

Q8_0 (unsloth's `Qwen3.8-27B-Q8_0.gguf`, 27.04 GiB, the same 866 tensors with its nextn block):
`local/qwen27-q8-local.json`, `./almond -m qwen27-q8` (router name `qwen3.8-27b-q8`), cards 0 and 1 (layers 0-29
on card 0), 262144 context, q8_0 KV, MTP 3, its own conversation folder `local/qwen27-q8-slots`. Measured
2026-11-11 against the IQ3_S, `--bench 512` (the fixed prompt, 512 tokens, greedy):

| | IQ3_S, card 0 | Q8_0, cards 0 and 1 |
| --- | ---: | ---: |
| no drafts | 41.7 tok/s | 23.2 tok/s |
| MTP 3 | 101.9 tok/s | 64.8 tok/s |
| MTP drafts accepted | 379 / 399 | 382 / 396 |
| prompt (512 tokens) | ~1030 tok/s | ~965 tok/s |

The Q8_0 reads 2.4x the bytes per token; it is the choice for the closest answers to the original weights, the
IQ3_S for speed and one card.

Images (`local/qwen27-local-vision.json`, picked by `./almond serve --model qwen27` when the mmproj is there and
`almond-vision` is built): the encoder runs on the CPU (`almond-vision`, 12 threads, at most 300 image tokens), the
engine takes `GENI` with its `SVE1` records: each picture's rows replace its run of `<|image_pad|>` tokens in the
trunk's embeddings, with mtmd's M-RoPE positions (t = p, h = p + y, w = p + x; the text after it at p + max(nx, ny));
each image row is matched on its picture's key (the server's `img_keys=`) and index, so a different picture of the
same size is never taken for a cached one; the MTP block reads the picture's rows too. Checked on card 1 with a 640x420 test picture (a red square, a blue circle, "ALMOND 27B"; 260 image
tokens, a 20x13 grid): both shapes, their colours and places and the text described right, in English and Italian
(5.7 s with the encode, 1.1 s with the picture already encoded); a second turn of that conversation reused 293 of
316 tokens; another picture of the same size (a green triangle, an orange square) reused none and was described
right. Real pictures (greedy, thinking off, 500 tokens): `/usr/share/backgrounds/jdituicha-raccoon1-light.jpg` (the
raccoon, the faint "2", the purple-to-orange gradient: right), llama.cpp's `media/matmul.png` (both products, their
row / column-major layouts and the ne labels: right), a 2939x2499 Android Studio screenshot (the IDE,
`MainActivity.kt` and "BUILD SUCCESSFUL" right, but files it could not read made up: at 300 image tokens, the
configuration's limit as for Flash-Next, its small text is not legible). MTP drafts accepted on those three: 307 /
576, 313 / 558, 329 / 513.

Conversations on disk (`conversation_dir`, `local/qwen27-slots`, serve_qwen.py's `SAVE` / `RESTORE` after each
reply): the file holds the tokens, match keys, M-RoPE positions, each DeltaNet layer's live state, the KV rows of
every attention layer up to the conversation's length, the MTP rows not paired yet and the DFlash2 ring (199 MiB for
1,394 tokens, saved in 105 ms). Checked on card 1: a conversation saved, the server restarted, the next turn resumed
the 1,394 tokens from the file in 77 ms and read only its new ones (1,409 of 1,414 reused, the right answer).

Template and thinking: the GGUF's own chat template (it differs from Flash-Next's only in merging several system /
developer messages); the tokenizer is Flash-Next's (the same 248,320 tokens, compared one by one). The levels are the
model card's, the same as Flash-Next's: off (`enable_thinking` false, an empty thinking block), low and xhigh (a
system instruction each), medium (none), xhigh by default, a client's high or max is xhigh; sampling with thinking
1.0 / 0.95 / top_k 20, without 0.7 / 0.8 / 20 and presence 1.5. Checked through the server (greedy, "a train leaves at
9:40 and arrives at 13:15, how long?"): off 9 tokens, low 149, medium 300, xhigh 67, the same right answer each.

How it keeps state: each DeltaNet layer keeps 8 snapshots of its SSM state (`ggml_gated_delta_net` with K = 8) and a
conv window of 11 rows; a verify window of n tokens of which a are kept leaves the state in snapshot n-a, read by the
next graph through `get_rows` with that index as an input, so a rejected draft costs no copy and the graphs stay the
same shape. One live graph per kind (trunk, MTP catch-up, MTP step, DFlash2 block), each with its own scheduler; a
graph whose shape is unchanged is replayed with new inputs.

Correctness: PPL on the repository's docs (`docs/*.md docs/ds4/*.md`, llama-tokenize, 16 chunks of 2048, the second
half of each scored) 6.1552 with an f16 KV, llama.cpp's `llama-perplexity` on the same text and file 6.1542. Greedy
output with MTP or DFlash2 is the same token for token as without (512 tokens checked).

Speed, card 0, greedy, `--bench 512`, prompts: Python ISO-8601 parser with tests / Gated DeltaNet vs softmax attention
/ a 400-word story in Italian, thinking off. llama.cpp 669204ee2 (ROCm, same GGUF, `llama-server`, f16 KV) for
comparison:

| | code | technical | story | mean (harmonic) |
|---|---|---|---|---|
| no drafts, q8_0 KV | 42.2 | | | |
| MTP 3 (default), q8_0 KV | 96.1 | 73.0 | 57.6 | 73.3 |
| MTP 3, f16 KV | 99.6 | 73.9 | 56.6 | 73.5 |
| DFlash2, 5 drafts | 105.4 | 81.3 | 44.5 | 67.6 |
| DFlash2, 7 drafts | 106.7 | 72.9 | 36.9 | 59.6 |
| llama.cpp, no drafts | 43.7 | | | |
| llama.cpp, MTP 3 | 102.8 | 77.2 | 58.7 | 75.5 |
| llama.cpp, DFlash2 7 | 116.2 | 70.0 | 39.4 | 62.2 |

(z-lab's DFlash2 drafter was removed on 2026-11-11: its configuration, router entry and download; DSpine inside code,
below, took its place.) MTP 3 stays the default: DFlash2 is ahead on code and technical prose and far behind on free prose, where its
8-token verify windows (47.8 ms against 23.8 ms for one token) are mostly rejected. A draft probability floor
(`--spec-min-p`) measured worse at every value tried (0.3 / 0.5 / 0.7 with 3, 4 and 5 drafts): a shorter window is
another graph shape to build. The q8_0 KV (the default) costs nothing measurable against f16 here. Where the time
goes in an MTP round (code prompt): the 4-token verify window 32.7 ms, the MTP catch-up 1.9 ms, two draft steps 1.6 ms
each. The verify window is ggml's MMVQ at 4 columns; changing its RDNA3 table (warps / rows per block), decoding the
IQ3_S / IQ3_XXS / IQ2_S / IQ4_XS blocks once for all columns, or unrolling its K loop did not make it faster (32.3 -
40.7 ms; rocprofv3: the kernels read at ~350 GB/s at 4 columns, ~470 GB/s at one, latency-bound), so ggml is
unchanged.

Conversations: the engine keeps up to four checkpoints of the recurrent state in host RAM (the DeltaNet SSM states
and conv rows and the MTP pending row, ~150 MB each), one in front of each prompt's last `<|im_start|>` and one at
its end; the KV caches, the MTP block's KV and the DFlash2 ring are indexed by position and stay right for any prefix.
A request resumes from the cache when it continues it, else from the longest checkpoint that is a prefix of it.
Without them, every agent turn whose prompt rendered the previous answer again (its thinking dropped) was read from
zero: in an agent session through the CLI, 13,539 and 14,514-token prompts read again in 12.5-13.9 s. With them
(card 1, a 4-turn conversation, thinking low, 400 tokens a turn): turn 0 2,159 tokens read in 2.57 s, turns 1-3
reused 2,559 / 2,985 / 3,411 of 2,585 / 3,011 / 3,437 and read the rest in 0.26-0.30 s.

Long context (`--bench 256` after a prompt of the docs' tokens repeated, q8_0 KV, context 262144):

| cards | prompt | prompt read | decode, MTP 3 |
|---|---|---|---|
| 1 | 4K | 1000 tok/s | 67 tok/s |
| 1 | 64K | 717 tok/s | 66 tok/s |
| 1 | 200K | 444 tok/s | 36.6 tok/s |
| 1 | 258K | out of memory (the MTP window's attention scratch) | |
| 2 | 4K | 967 tok/s | 49 tok/s |
| 2 | 258K | 352 tok/s | 25.7 tok/s |

One card holds the weights (11.28 GiB) and a 256K q8_0 KV with the DeltaNet state (10.20 GiB), but not the compute
buffers of a prompt window that deep, so the default configuration is one card at 196608 (192K) and
`local/qwen27-local-256k.json` splits the layers over two cards for the full 262144 (slower at any length: every
token crosses both cards). Past ~100K the decode slows down: with a q8_0 KV the verify window's attention converts the
cache it reads to f16.

### DSpine drafter inside code, MTP elsewhere (`--spec auto`, 2026-11-11)

Tencent's DSpine (`tencent/Qwen3.8-27B-DSpine-b16`; the GGUF `aj9o9/Qwen3.8-27B-DSpine-b16-GGUF` Q4_K_M, 1.16 GiB,
`local/fetch.sh qwen27-dspine`) is the DFlash frame without the convolutions and the selector: 5 layers reading the
trunk at layers 2, 17, 32, 47, 62, causal inside the block, an 8192-token window (the KV ring is 16384 cells), and
after every layer a code bridge (each row reads a unit code from its hidden state, the first row takes its token's
codebook row, row t gets a gated message from row t-1's code); the candidates are scored by redoing the last bridge
once per predecessor candidate (top 16). The engine reads it through `--dflash` (the GGUF says which drafter it is),
at most 7 drafts (the verify window's 8 DeltaNet snapshots).

Measured first in llama.cpp (a fork with DSpine support, same prompts, greedy, 512 tokens): it gains on code and
loses on prose, more with longer drafts.

| IQ3_S, tok/s | MTP 3 | DSpine 3 | DSpine 7 | DSpine 15 |
| --- | ---: | ---: | ---: | ---: |
| Python function with tests | 90.8 | 88.3 | 106.9 | 117.2 |
| technical explanation | 74.9 | 70.9 | 57.4 | 54.2 |
| short story in Italian | 54.4 | 40.8 | 28.6 | 23.0 |
| bash script, explained | 84.3 | 83.4 | 87.3 | 75.0 |

(Q8_0 on two cards: Python 59.6 with MTP 3 against 97.9 with DSpine 15; the story 36.2 against 19.1.)

So `--spec auto` (with `--mtp 3` and the drafter) follows the generated text: inside a ``` block or a `<tool_call>`
the block drafter drafts, elsewhere MTP; the MTP block is caught up every round either way and the drafter's ring is
filled by every verify window. Inside code, when the drafter's tokens per millisecond fall below 0.9x MTP's (measured
on the text before), MTP drafted until the block ended (`ALMOND_Q35_AUTO_GIVEUP`; off by default since 2026-11-16, see below). Every switch is a line in the engine log
(`qwen35: --spec auto: MTP -> DSpine at token N (a ``` block opened; rates ...)`), and each request's first drafter. An earlier rule that chose by the two
drafters' moving rates with trial rounds (MTP and DFlash2) was slower than MTP alone (62.9 against 66.6 tok/s over the
four prompts) and was dropped.

Engine bench on card 0 (`--bench 512 --prompt-ids` of the rendered prompt, thinking off, decode only):

| IQ3_S, tok/s | MTP 3 | DSpine 7 | `--spec auto` (rounds MTP / DSpine) |
| --- | ---: | ---: | ---: |
| Python function with tests | 79.7 | 109.3 | 104.3 (0 / 83) |
| technical explanation | 63.2, 71.6 | 55.1 | 70.2, 70.5 (183 / 0) |
| short story in Italian | 50.4, 56.5 | 29.4 | 54.9, 54.9 (236 / 0) |
| bash script, explained | 88.6 | 88.9 | 85.8 (92 / 38) |

(two values: two runs; the card's run-to-run spread is ~10%.) Greedy, the server's answers with `--spec auto` were
the same text as MTP's on all four; DSpine alone the same on three (the story parts at a near tie, as MTP and DFlash2
do). Configurations: `local/qwen27-local-dspine.json` (`./almond -m qwen27-dspine`, 131072) and
`local/qwen27-q8-local-dspine.json` (`./almond -m qwen27-q8-dspine`, two cards); the defaults keep MTP.

**Removed on 2026-11-25** (both configurations, the router entries and the download item; the engine keeps `--dflash`
and `--spec auto`). Through the server on card 0, temperature 0, the IQ3_S at 192K with DSpine against MTP 3, two runs
each: a Python module with tests (1500 tokens) 96.1 / 96.6 against 92.6 / 92.8 tok/s (+4%), a 500-word story (894
tokens) 53.3 / 53.4 against 54.5 / 54.5, a 182,638-token prompt read at 477 tok/s by both and answered at 33.0 against
33.7. DSpine fits at 192K, but +4% on code and -2% elsewhere did not earn a profile.

### Every configuration through the server, short and ~123K context (2026-11-15)

Each configuration started with `./almond serve --config local/<name>.json --port 8161` on 127.0.0.1 and was sent,
through `/v1/chat/completions` (streamed, the configuration's sampling: temperature 1.0, top-p 0.95, top-k 20): a
two-sentence answer, a Python function with three asserts (one ``` block, ~135 tokens), a 500-word story (650-800
tokens, thinking off), and a `get_weather` tool call. Every configuration answered all four correctly: the same
right function, a well-formed `get_weather({"city": "Rome"})`, and on `-vision` a right one-sentence description of a
PNG logo (361 prompt tokens, 5.5 s with the CPU encoder). Then one long request: 390,000 characters of this
repository's docs and sources plus "summarize what they say about the 27B" (122,927 prompt tokens, a cold cache,
400 tokens generated, thinking on, temperature 0). tok/s are the server's, from the first streamed token to the last:
they include the HTTP and detokenizer overhead and are under the engine bench's numbers above.

| configuration | cards | story tok/s | code tok/s | 123K prefill tok/s (s) | 123K decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen27-local` (IQ3_S, 192K, MTP 3) | 0 | 56.6 | 89.5 | 585 (210.0) | 37.3 |
| `qwen27-local-vision` | 0 | 55.8 | – | 573 (214.7) | 35.0 |
| `qwen27-local-dspine` (131072, `--spec auto`) | 0 | 54.9 | 79.0 | 585 (210.2) | 37.6 |
| `qwen27-local-256k` (IQ3_S, 262144) | 0,1 | 40.3 | 68.4 | 538 (228.4) | 29.8 |
| `qwen27-q8-local` (Q8_0, 262144, MTP 3) | 0,1 | 36.4 | 57.8 | 555 (221.4) | 28.1 |
| `qwen27-q8-local-dspine` (`--spec auto`) | 0,1 | 35.5 | 57.4 | 554 (221.8) | 25.0 |

One run each (the card's run-to-run spread is ~10%). The code answer is one short block, so DSpine drafts a few
rounds only (24 on the Q8_0) and does not show its gain on longer code (the tables above). At 123K decode falls to
about two thirds of the short-context speed on one card, and the second card's link (PCIe 4.0 x2) costs another ~20%.

**The Q8_0 + DSpine configuration did not start** (`./almond -m qwen27-q8-dspine` in the chat, or the router): the
engine split the layers by the two cards' free memory (layers 0-29 on card 0) before it loaded the drafter, which goes
on the last card (1.16 GiB of weights, its KV ring and graph); card 1 then had no room for the 609 MiB compute buffer
(`cudaMalloc failed: out of memory`, `qwen35: cannot allocate a graph`) and the server stopped. The split now takes
the drafter's file size plus 768 MiB off card 1's free memory when `--dflash` is given on two cards (layers 0-30 on
card 0); without a drafter the split is unchanged. Also, `./almond serve --config local/qwen27-q8-local.json` without
`--model qwen27` started DeepSeek's server on the 27B's configuration (an assertion on the end-of-turn token): the
launcher now picks the server from the engine arguments the configuration names (`--qwen35`, `--qwen`, `--glm-pack`)
when `--model` is not given.

### Decode attention over a long q8_0 KV (`GGML_CUDA_FA_Q8_SPLITK=1`, 2026-11-16)

Where the time goes at long context: `--bench 512` on card 0 (IQ3_S, q8_0 KV, MTP 3, greedy, thinking off) with a
122,824-token prompt (this repository's docs and sources) against a 26-token one: the 4-token verify window
(`ALMOND_Q35_PROF=1`, `trunk4`) takes 68.9 ms against 32.4 ms. rocprofv3's kernel trace of the decode puts the
difference in the 16 full-attention layers: with a q8_0 KV and 4 query tokens (24 query rows per KV head) ggml picks
its tile kernel, which needs f16 K and V, so every call first converts the whole K and V views to f16
(`dequantize_block_q8_0_f16`, ~0.5 ms each) and then runs `flash_attn_tile` (1.26 ms): ~2.3 ms per layer, ~36 ms a
round, where reading the layer's 267 MB of q8_0 KV once takes ~0.3 ms at the card's bandwidth.

`ggml-cuda/fattn-q8-splitk.cu` (opt-in, `GGML_CUDA_FA_Q8_SPLITK=1`; the patch in `tools/v41/ggml/fa-q8-splitk.patch`)
reads the q8_0 bytes directly: one block per KV head and 256 context positions, all the head's query rows (1-4 tokens
x 6 heads) at once, K and V staged in LDS with 16-byte loads, an online softmax over 64-position sub-chunks, and a
second kernel that merges the blocks. It runs for q8_0 K and V, head size 256, at most 24 query rows (so not
DSpine's 8-token windows) and at least 16K context positions; anything else goes to ggml's kernels as before. The
first, simple version was slower than ggml's (1.8 ms a call at 123K); what made it faster, measured in a standalone
microbenchmark at 32K positions: staging K/V in LDS, rotating Q's rows in LDS so the lanes of one position do not hit
the same bank (790 -> 497 us), two positions per lane for the scores and two dimensions per thread for P*V, and no
unrolling of the score loop (200 VGPRs, no spills): 387 us at 32K, 1.40 ms at 123K (191 GB/s; still under the
card's bandwidth, the scores' LDS reads limit it).

| context (tokens) | ggml tok/s | `GGML_CUDA_FA_Q8_SPLITK=1` tok/s |
| --- | ---: | ---: |
| 26 | 59.45 | 59.58 (not used under 16K) |
| 8,847 | 66.65 | 65.57 (when it ran at 8K; now not used) |
| 36,847 | 52.46 | 58.11 (+10.8%) |
| 73,418 | 42.97 | 48.44 (+12.7%) |
| 122,824 | 39.34, 39.32 | 44.60 (+13.4%; `trunk4` 59.6 ms) |

(`--bench 512`, one run each unless two values are given; the 8K-73K prompts are shorter cuts of the same text, with
the question at the end.) Output: two runs without it gave the same 512 tokens; with it the text parts at token 48 at
a near tie ("Dense 27B parameter model" against "Dense 27B model") and stays coherent and on the same content to the
end. Since it measured better from 32K up and the same at short context, `local/qwen27-local.json` and its
`-vision` copy set it in their `env`.

The other configurations, same `--bench 512` (each configuration's own arguments; the 26-token story and a
123,595-token cut of the same text, one run each):

| configuration | story, ggml | story, flag | 123K, ggml | 123K, flag |
| --- | ---: | ---: | ---: | ---: |
| `qwen27-local-256k` (IQ3_S, two cards) | 44.12 | 44.10 | 29.68 | 32.22 (+8.6%) |
| `qwen27-q8-local` (Q8_0, two cards) | 39.66 | 39.64 | 28.96 | 30.41 (+5.0%) |
| `qwen27-local-dspine` (one card, `--spec auto`) | 59.04 | 58.97 | 36.71 | 42.43 (+15.6%) |
| `qwen27-q8-local-dspine` (two cards) | 39.09 | 39.09 | 27.23 | 29.78 (+9.4%) |

Better at 123K in all four and the same at short context, so all the 27B configurations set it now. On two cards the
gain is smaller: the attention is a smaller share of a round there (the Q8_0's 27 GB of weights, the hand-over to
the second card).

### llama.cpp with its tensor split, same prompts (2026-11-16)

llama.cpp 669204ee2 (`llama-server`, ROCm, both cards, the user's settings: `-c 131072 -b 2048 -ub 512 -fa on -ctk
q8_0 -ctv q8_0 --spec-type draft-mtp`, the same GGUFs), greedy, 512 tokens, `ignore_eos`, the same story and the same
123,595-token text, tok/s from the server's `timings`:

| llama.cpp | story tok/s | 123K tok/s (prefill tok/s) | MTP drafts accepted at 123K |
| --- | ---: | ---: | ---: |
| Q8_0, `-sm layer`, MTP 2 | 37.85 | stalled at 88K of the prefill (card 1 "100%" at 39 W, memory clock idle, 17 min), killed | - |
| Q8_0, `-sm tensor`, MTP 2 | 51.16 | 16.54 (525.7) | 0 / 1019 |
| Q8_0, `-sm tensor`, MTP 3 | 50.40 | 48.01 (526.8) | 350 / 480 |
| IQ3_S, `-sm tensor`, MTP 2 | 63.40 | 19.33 (523.7) | 0 / 1019 |

Against Almond on the same two cards (the tables above, with `GGML_CUDA_FA_Q8_SPLITK=1`): the Q8_0 39.6 short and
30.4 at 123K, the IQ3_S 44.1 and 32.2. llama.cpp's tensor split (every matrix and the attention heads halved over
the two cards, the cards working at once, a reduction per layer over the x2 link) is +29% on the Q8_0's short story
and +58% at 123K with MTP 3, and the IQ3_S on two cards (63.4) is above Almond's on one card (59.5). Almond's
two-card split is by layers only, so one card waits while the other runs. With MTP 2, llama.cpp accepted no draft at
123K in both runs (MTP 3 did), and its `-sm layer` run stalled: neither is looked into here. Prefill at 123K is the
same (~525 tok/s against Almond's ~555); llama.cpp's `-sm layer` prefill was faster before it stalled (2,000 tok/s
falling to 1,080 by 88K).

### Tensor parallel over the two cards (`--tp`, opt-in, 2026-11-16)

`--tp` (with `--gpus 0,1`) runs the 27B on ggml's meta backend over both cards instead of a layer split: each card
holds half of every layer's heads and FFN width, with the rules llama.cpp uses for this architecture (full attention:
Q with its gate, K, V, the KV cache and attn_output's rows by head; DeltaNet: the q/k/v channels, gate, decay terms,
conv window and recurrent state per key-head group, the value heads as Hv/Hk segments; FFN by its inner width; norms,
embeddings and the head mirrored), and the meta backend adds the partial outputs after attn_output, ssm_out and
ffn_down (two reductions a layer). What it took in Almond: the DeltaNet conv window stored time-inner
(`[CONV_W, C]`, read and written through views, its offset part of the graph key) instead of channel-inner rows read
with `get_rows` (that path gave wrong windows under the split); the vendored meta backend's view handling brought up
to llama.cpp's; and per-graph view containers in the meta backend (`ggml_backend_meta_set_view_owner` /
`release_views`), since Almond keeps several graphs allocated (trunk, MTP catch-up, MTP step) where the meta backend
kept only the last one's views (`tools/v41/ggml/meta-tp.patch`). (Disk sessions were refused under `--tp` at first; they work now, below.)

`--bench 512`, greedy, the 26-token story and the 123,595-token text, `GGML_CUDA_FA_Q8_SPLITK=1`:

| two cards, tok/s | layer split, story | `--tp`, story | layer split, 123K | `--tp`, 123K |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 39.60 | 42.46 | 30.41 | 45.12 (+48%) |
| IQ3_S | 44.10 | 50.99 | 32.22 | 50.15 (+56%) |

(IQ3_S on one card: 60.1 short, 44.6 at 123K; llama.cpp's `-sm tensor`: Q8_0 50.4 / 48.0, IQ3_S 63.4 short.) The
output is the same: the IQ3_S with `--tp` gives the one-card tokens, the Q8_0 the layer split's. At 123K `--tp` is
close to llama.cpp; at short context it is not: a profile of the Q8_0's decode has each card busy ~60% of the time,
the reductions (128 a verify window, ~40 KB each over the x2 link) take card 0 585 ms in 3 s against 242 ms for card 1,
and the rest is the host. HIP graphs make no difference (43.4 against 43.3 tok/s with `GGML_CUDA_DISABLE_GRAPHS=1`),
nor does reusing the graphs; llama.cpp's newer ggml-cuda and meta backend are the likely difference, not ported here.
Not measured yet with DSpine or through the server; the configurations do not set it.

Porting the newer ggml would not close the short-context gap: without MTP (one token a step, Q8_0, two cards,
`--bench 128` against `llama-bench -n 128 -fa 1 -ctk q8_0 -ctv q8_0`) Almond and llama.cpp are the same, 32.2
against 32.9 tok/s split by tensors and 23.4 against 23.6 by layers, and `GGML_CUDA_P2P=1` is slower (39.3 against
41.2 tok/s with MTP). The difference is in the MTP rounds: under `--tp` the 4-token verify window costs 45.8 ms against
~31 ms for one token (+15 ms; +8.6 ms with the layer split), a round with the MTP steps ~53 ms against ~45 ms for
llama.cpp. In the profile card 0 spends 96 us on average in each of the window's 128 reductions against 40 us for card
1, i.e. it waits for card 1 to reach them; the MTP steps (catch-up 3.3 ms, two draft steps of 2.9 ms) add the rest.

Most of it was HIP graphs not being replayed: with `GGML_CUDA_GRAPH_STATS=1` only 39% of the cards' sub-graph runs
were launched as captured graphs, 15% captured again and 45% run directly, with the number of graph keys growing.
The cause was the conv window's offset in the trunk graph's key: it moves with every verify that drops tokens, so the
trunk graph was rebuilt most rounds. Now a verify that drops tokens copies the window's 3 live columns back to its end
(one small cached graph per offset) and the trunk keeps one graph: 97% of the runs launch a captured graph, the
verify window takes 39.0 ms (45.8 before).

| `--tp`, two cards, tok/s | story, before | story, now | 123K, before | 123K, now |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 42.46 | 46.71 | 45.12 | 47.83 |
| IQ3_S | 50.99 | 57.33 | 50.15 | 54.46 |

(llama.cpp `-sm tensor`: Q8_0 50.4 short, 48.0 at 123K; IQ3_S 63.4 short.) The Q8_0's 512 tokens are the layer
split's up to token 338, where the two part at a near tie ("Elias had smiled" against "Elias had listened"), both
coherent to the end. Left: the head (248,320 rows) is mirrored, so both cards compute all of it at every trunk and MTP
call (1.6 ms each on the Q8_0).

`--tp` in the two-card configurations (2026-11-16): `qwen27-local-256k`, `qwen27-q8-local` and `qwen27-q8-local-dspine`
set it now. Disk sessions work under it (the chunks are whole rows of the split tensors, the conv window's columns go
through the host): a conversation saved, the server restarted, the next turn resumed 105 tokens from disk in 72 ms.
DSpine works under it with the drafter mirrored on both cards (its tensors reuse the trunk's names, so they are kept
out of the split rules). Engine `--bench 512`, Q8_0 + DSpine (`--spec auto`, 7 drafts), greedy: a Python module with
tests 81.23 -> 116.32 tok/s (layer split -> `--tp`, the code correct), the story 38.85 -> 45.46. Through the server
(the short tests at the top of this section, streamed, temperature 1.0; one run each), all correct:

| configuration | chat, layer split -> `--tp` | code | tool call |
| --- | ---: | ---: | ---: |
| `qwen27-local-256k` | 48.6 -> 60.5 | 68.4 -> 81.7 | 81.2 -> 78.0 |
| `qwen27-q8-local` | 43.4 -> 48.5 | 57.8 -> 67.7 | 76.8 -> 70.3 |
| `qwen27-q8-local-dspine` | 38.6 -> 43.0 | 57.4 -> 65.9 | 52.6 -> 59.5 |

(The tool call is 27 tokens: too short to compare.)

### `--spec auto` inside code: no give-up by default (2026-11-16)

The rule that gave the block drafter up inside code (its rate below 0.9x MTP's) compared its rate in the code with
MTP's on the prose before, and judged it on its first round after the block opened, which is slow. In four requests
with a paragraph of prose before the code it gave DSpine up after its first round every time, and MTP wrote the rest
of the block. `--bench 1024`, greedy, IQ3_S on card 0 (two values: two runs):

| prose, then code | give-up 0.9 (before) | give-up after 4 rounds | no give-up |
| --- | ---: | ---: | ---: |
| token bucket in Python | 80.32 | 82.67 | 82.85, 84.53 |
| single-file HTML/JS todo app | 80.76 | 81.90 | 87.70, 87.74 |
| trie in TypeScript | 88.99 | 88.05 | 98.09, 98.76 |
| Bash backup script | 78.90 | 77.77 | 82.13, 81.90 |
| code only (the module with tests) | 107.77 | 108.47 | (no give-up happened) |
| the story (no code) | 58.52 | 57.06 | (no code) |

Q8_0 + DSpine with `--tp`: the HTML app 66.88 -> 84.34 tok/s, the trie 70.33 -> 94.54. Waiting for 4 or 8 DSpine
rounds only gave it up later: DSpine in code is below MTP's rate on prose and still above MTP in the same code, so the
comparison itself was wrong. An MTP probe inside code (both rates on the same block, the better one drafting, the
other tried every 8 rounds) was slower too (90.5 against 107.8 tok/s on the module): one round's rate is too noisy.
So `ALMOND_Q35_AUTO_GIVEUP` is 0 by default: inside ``` blocks and tool calls the block drafter drafts.


### The server against the engine (2026-11-16)

The 56.6 tok/s of the story through the server above against the 73.3 tok/s of the table at the top of this section
was not server overhead: 73.3 is the harmonic mean of three prompts, the story alone is 57.6 there. The same story
request (thinking off, 512 tokens, `ignore_eos`) on `qwen27-local`: engine `--bench` 59.5 tok/s greedy; the server
greedy and streamed 57.8 tok/s from the first token to the last (57.9 in a second run), not streamed 56.6 tok/s
including the prompt; streamed at temperature 1.0 (the configuration's sampling) 53.4 and 53.2. So the server
(HTTP, detokenizer, streaming) costs ~3%, and sampling at temperature 1.0 ~8% through MTP accepting fewer drafts;
nothing to fix in the server.

Qwen3.8-Flash-Next's resident expert arena is checked against the RAM before it is pinned (2026-11-11): pinning an
arena of 119.5 GiB with 88.9 GiB available stalled the whole machine (the unsloth Q8_0, since removed). The engine
refuses an arena that does not leave `ALMOND_ARENA_HEADROOM_GIB` (default 8) free and names a `--mmap-experts
--resident-budget-gib` that fits; `ALMOND_ARENA_FORCE=1` loads it anyway. The IQ3_S (a 46 GiB arena) loads as before.

## Qwen3.8-Flash-Next UD-Q4_K_XL (Unsloth Dynamic 3.1, 2026-11-23)

`unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL (four shards, 111.3 GB; experts 63.47 GiB, the figure the engine
printed on an RX 6900 XT 16 GB), `local/fetch.sh qwen-q4` (its pack with `--compat-bf16`: the routers, SSM gates and
injections are F32 and the hyper-connection projections Q8_0 in this GGUF; the engine reads them as BF16),
(Since 2026-11-25 this is the only Qwen3.8-Flash configuration, `local/qwen-local.json`, `./almond -m qwen`; the IQ3_S
configurations of the sections above were removed. First run through the server: a 68-token prompt, 85 tokens at
42.8 tok/s, 51 of 65 drafts accepted, the expert cache 84.6% hit.)

`local/qwen-q4-local.json`, `./almond -m qwen-q4` (router name `qwen3.8-flash-next-q4`): card 0's expert slots, card 1
as the peer expert tier, the rest read through the page cache with up to 56 GiB kept in RAM (`--mmap-experts
--resident-budget-gib 56`), the MTP block of `models/qwen38-mtp`. `--ple-gguf` is left to the engine, which finds the
shard holding the PLE table.

Published against BF16, not measured here: mean KLD 0.047, top-1 92.3% (Unsloth's Dynamic 3.1 release). Speed and
KL against the IQ3_S on this machine: not measured yet.

## SenseNova-Loop-9B (Q8_0, `--qwen35`, 2026-11-26)

SenseNova's looped dense model (GGUF architecture `sensenova_loop`): a Qwen3.5 body of 32 layers whose layers 4-27
run twice (56 layer runs a token), each run with its own KV rows or DeltaNet state, two residual streams mixed by
Sinkhorn routers (mHC) inside the loop, YaRN x2 RoPE; the architecture and how the GGUF lays it out in
docs/sensenova/NOTES.md. It runs on the dense path `src/qwen35`, which switches to it from the GGUF's architecture
(other GGUFs are read as before): repeated runs share the weight tensors and get their own state; the routers'
coefficients (sigmoids and the 20 Sinkhorn iterations, not converged, so not replaced by the closed form) are one
ggml op, `ggml_loop_mhc_coef` (`tools/v41/ggml/loop-mhc.patch`), the mixing DeepSeek's `hc_pre` / `hc_post` ops.
One card; no `--tp` or images yet (the DFlash2 drafter: below). Files: `local/fetch.sh sensenova` (the official
`sensenova/SenseNova-Loop-9B-GGUF` Q8_0, 9.8 GB, and its tokenizer exported to `models/sensenova-loop-9b/tokenizer`);
`./almond serve --model sensenova`, `./almond -m sensenova` or the router entry (`local/sensenova-local.json` (card 0, 256K, q8_0 KV, MTP 3).
The server shim is Qwen's: its thinking levels low / medium are passed to a template that only knows thinking on or
off.

Against the reference (2026-11-26): the checkpoint's own `modeling_sensenova_loop.py` under transformers 5.18 in
FP32 on the CPU, 768 tokens (256 each of Python code, English prose, an agent turn with tool calls in the chat
format) read in one pass; the engine read the same ids with `--ppl FILE --ppl-ctx 768 --dump-logits` (two windows of
512 + 256, so the second continues from the cached state), f16 KV, card 0. KL(ref || engine) per token:

| GGUF | | code | prose | agent | all 768 |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | top-1 agreement | 99.6% | 95.3% | 96.9% | 97.3% |
| | mean KL / p99 / max | 0.0004 / 0.006 / 0.017 | 0.0024 / 0.016 / 0.035 | 0.0081 / 0.131 / 0.338 | 0.0036 / 0.076 / 0.338 |
| Q4_K_M | top-1 agreement | 97.7% | 84.8% | 83.6% | 88.7% |
| | mean KL / p99 / max | 0.013 / 0.18 / 0.22 | 0.101 / 0.50 / 0.63 | 0.195 / 1.96 / 6.57 | 0.103 / 1.35 / 6.57 |

Perplexity of the 767 next tokens: reference 5.016, Q8_0 5.006, Q4_K_M 4.990 (a single short text: the KL is the
measure). The Q4_K_M loses ~30x the Q8_0's KL, so the configuration is the Q8_0 and the Q4_K_M has none.

Speed, card 0, q8_0 KV, `GGML_CUDA_FA_Q8_SPLITK=1`, `--bench 256 --prompt-ids` the first 512 of those ids, greedy
(second of two runs):

| GGUF | prompt 512 | decode, no MTP | decode, MTP 3 (drafts accepted) |
| --- | --- | --- | --- |
| Q8_0 (32K context) | 1605 tok/s | 40.4 tok/s | 72.7 tok/s (156/297) |
| Q4_K_M (32K context) | 1475 tok/s | 50.4 tok/s | 86.9 tok/s (163/276) |
| Q8_0, 256K context (the configuration) | 1566 tok/s | | 63.4 tok/s, 128 tokens (74/168) |

A 32,768-token prompt (the 768 ids repeated) read at 1401 tok/s (Q8_0, 256K context). Weights 9.12 GiB, KV and state
at 256K 8.65 GiB. Through the server (`serve_qwen.py` with this configuration) a short coding question answered
with thinking and a correct function.

### DFlash2 drafter (opt-in, 2026-11-26)

`sensenova/SenseNova-Loop-9B-DFlash2` (967M parameters, BF16): the DFlash2 frame the engine already runs for
`--dflash` (5 layers, block 8, conv kernels, a top-16 selector, a 2048 window) trained on this model. It reads the
outputs of the target's *executions* 3, 27, 51 and 54 (zero-based over the 56 runs: the plain layer before the loop,
the end of each pass, where the two streams are already averaged back to one, and the third of the last four layers)
and, as a fifth feature, the final hidden after the output norm (`final_postnorm`, what LightLLM's integration
appends; fc reads 5 x 4096). The engine takes the fifth feature from fc's width, so the 27B's drafters read as
before. The last drafter layer is `full_attention` in the config, but LightLLM's DFlash2 layer applies the 2048 window
to every layer, as the engine does. `local/fetch.sh sensenova` downloads it with the model and converts it with
`tools/sensenova/dflash2_gguf.py` (Q8_0, 1.03 GB; tensors and keys byte for byte those of llama.cpp's DFlash
converter, which needs a target architecture it knows). Since 2026-11-26 the configuration
(`local/sensenova-local.json`) runs MTP 3 and DFlash2 5 drafts with `--spec auto`, which keeps per round the drafter
whose recent drafts paid more). Greedy output with DFlash2 is the same token for token as without (256 tokens).

Card 0, 256K context, q8_0 KV, greedy, `--bench 512`, prompts (chat template, thinking off): a Python ISO-8601 parser
with tests (52 tokens) / Gated DeltaNet against softmax attention (46) / a 400-word story in Italian (42); tok/s and
drafts accepted:

| | code | technical | story | mean (harmonic) |
| --- | --- | --- | --- | --- |
| no drafts | 40.5 | 40.4 | 40.4 | 40.4 |
| MTP 3 (the default) | 74.5 (319/579) | 66.7 (297/648) | 39.6 (146/1098) | 55.9 |
| DFlash2, 3 drafts | 89.9 (346/495) | 71.2 (305/627) | 34.9 (81/1290) | 55.7 |
| DFlash2, 5 drafts | 106.2 (385/655) | 61.1 (284/1150) | 33.0 (83/2140) | 53.5 |
| DFlash2, 7 drafts | 103.6 (389/854) | 58.6 (294/1526) | 29.9 (83/3003) | 49.9 |
| MTP 3 + DFlash2 5, `--spec auto` | 89.9 | 65.9 | 39.1 | 57.9 |
| MTP 3 + DFlash2 3, `--spec auto` | 86.7 | 66.0 | 39.2 | 57.5 |

(`--spec auto` rounds, 5 drafts: code 13 MTP / 134 DFlash2, technical and story all MTP.) On the 512 mixed ids of
the accuracy check: MTP 3 72.6, DFlash2 5 96.5, DFlash2 7 99.4 tok/s.

Through the server at the configuration's sampling (temperature 1.0, top_p 0.95, top_k 20), the same three prompts,
up to 768 tokens, two requests each, decode tok/s after the first token:

| | code | technical | story |
| --- | --- | --- | --- |
| MTP 3 only (the configuration before) | 74.0, 72.8 | 49.0, 64.2 | 39.6, 40.5 |
| MTP 3 + DFlash2 5, `--spec auto` (the configuration) | 85.4, 77.1 | 57.2, 52.9 | 39.3, 39.5 |

DFlash2 is ahead on code and level elsewhere; the story in Italian gains nothing from any drafter (MTP accepts 13% of
its drafts). `--spec auto` measured the best harmonic mean (57.9 against 55.9 for MTP 3 alone) and level or ahead on
every prompt through the server, so it is the configuration; `--dflash` off (MTP 3 alone) is the previous one.

## Hy4-flash (IQ2-AH, AngelSlim, `--hy4`, 2026-11-18)

Tencent's Hy4-flash (GGUF architecture `hy_v4_flash`: 42 trunk layers, MLA with sinks, 30 sliding-window layers and
12 sparse layers with a k-pool indexer, LatentMoE with 512 experts and 16 per token, four Engram modules, iHC with 4
streams) from `AngelSlim/Hy4-flash-GGUF`: routed experts IQ2_XXS / Q4_K, the Engram table FP8 (59 GiB, the second
shard). `local/fetch.sh hy4`, `./almond serve --model hy4` (`local/hy4-local.json`: cards 0 and 1, 64K context;
`local/hy4-local-vision.json`, which `./almond serve` picks when the encoder and `build-vision-hy4` are there). The
engine is `src/hy4/hy4_engine.cpp`; how it was built and checked against llama.cpp PR #32517, and every number, in
docs/hy4/GGUF_NOTES.md. Measured on 2026-11-18: 14.9 tok/s generating with a short prompt (the PR: 10.4 on the same
machine), 11.8 tok/s after a 15,000-token prompt read at 237 tok/s; perplexity within the PR's own run-to-run spread
(27.93 against 28.46 on `local/glm-ref/doc.txt`, 512-token chunks). With the configuration's `--mtp --fused-hc
--dense-q8 --reserve-mib 1400 --fused-proj` (2026-11-19): 21.4-23.0 tok/s greedy and 21.3-21.7 at temperature 0.9 on
chat prompts. Images through its encoder
(`local/fetch.sh hy4-vision`, `build-vision-hy4`), conversations on disk (`local/hy4-slots`).

## Inkling 2 Mini (Unsloth UD-Q4_K_XL, `almond-inkling`, 2026-11-23)

Since 2026-11-24 the default configuration runs Almond's own engine (`almond-inkling-native`, docs/inkling/ENGINE.md,
"Step 4"): F16 KV, prefill about 3x and decode about 2x the numbers below through the server, KL from Q8_0 0.00557
against 0.00573. What follows is the llama.cpp path, whose configuration was removed on 2026-11-24 as slower.

Card 0 alone, 128K context, q8_0 KV. It runs on llama.cpp's own `inkling2_loop` graph behind the engine's serve
protocol (`src/inkling`, built into `build-inkling/`). Quality against ggml-org's Q8_0 on 43,709 tokens of this
repository's docs: KL 0.00728, same top token 95.90%, PPL +0.13%. ggml-org's IQ4_NL of the same size: KL 0.01416,
94.31%. Speed (llama-bench): decode 30.6 tok/s at depth 0 and 29.0 at 32K; prefill 1,396 tok/s (512 tokens) and
388 at 32K depth. Through the server, a 21,725-token prompt prefilled at 444.7 tok/s and decoded at 29.3. The decode
is bound by ~7,400 small kernels a token, not by bytes. Details, memory and what is open: docs/inkling/NOTES.md.

DSpark (opt-in, `local/inkling-dspark-local.json`, `./almond -m inkling-dspark`): Unsloth's drafter, 3 greedy drafts
checked by the request's own sampler, on `almond-inkling-dspark` (llama.cpp PR #32218). Through the server, card 0,
600-token answers, reasoning off: 41.4-46.3 tok/s at temperature 0 and 1.0 against 29.2-29.4 without it.

### Qwen3.8-27B images on the card in lent memory (2026-11-24)

The one-card configurations (`local/qwen27-local-vision.json`, `local/qwen27-local-dspine.json`) run the image
encoder on card 0 only while a request's pictures are encoded, in memory the engine lends it (`VLEND` /
`VRECLAIM`, as GLM and Inkling): each layer's FFN weights sit in their own buffer, the last layers' buffers are freed
for the encoder (2.1 GB in 1 ms when 6 GB were asked) and allocated again and read back from the GGUF afterwards
(183 ms); generations before and after are identical, decode speed the same (300 tokens in 2,992-2,996 ms either
way). `QWEN35_NO_LEND=1` keeps one buffer. The encoder runs with `f32_matmul` and `no_flash_attn`: against a
reference made from the same mmproj converted to F32 and run on the CPU without flash attention, the CPU encoder
these configurations used (BF16 weights, which ggml's CPU path also multiplies in BF16) ended at cosine 0.970 -
0.993, the GPU at its defaults at 0.994 - 0.998, the GPU with both settings at 0.99999 (three pictures). A picture
now takes about 0.1 s to encode instead of 2.6 s; through the server the first token of a 321-token picture prompt
came after 1.34 s. At 300 image tokens the encoder needs 1.70 GB, which card 0 has free, so nothing is lent unless
the card is fuller; card 0 settles at the same use after every request. The two-card configurations (`--tp`) keep
the CPU encoder: the split weights are not lendable yet.

# Models: thinking levels and sampling

What each model gets, where it comes from, and what was measured.  Everything below is the server's behaviour with the
shipped configurations (`local/ds41-local*.json`, `local/glm-local.json`, `local/qwen-local.json`); a client sees the levels in `/health`,
`/v1/models` (the router) and `./almond`'s `/models`.  Reviewed 2026-10-29.

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
  after (`local/glm-local-vision.json`, which `./run.sh --model glm` picks when the encoder is there). Text decode, one
  run each: 29.2, 32.2 before, 29.8, 33.3 after the first picture, 29.2 after the second (cached: no encoder start).

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
at 176K-181K taking back all but 144-5,216 tokens (3-16 s). The skip is not measured yet (built, not run).
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
slots went to the trash, and `local/glm-pack.sh`, `local/fetch.sh glm`, `./run.sh --model glm` take the 3.5bit
(the expert profile it learned in these runs kept). On the two 7900 XTX, power-tune on: 35.5 GB of experts on the cards as with 3.0, 84.6 GB
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

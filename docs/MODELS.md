# Models: thinking levels and sampling

What each model gets, where it comes from, and what was measured.  Everything below is the server's behaviour with the
shipped configurations (`local/ds41-local*.json`, `local/glm-local.json`); a client sees the levels in `/health`,
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

## GLM-5.3-Flash (GSQ-RCO 3.0bit, the engine)

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

The same within the engine's run-to-run spread. Copying numbers (40 random sizes, five copies at temperature 1.0):
0 wrong of 200 with MTP off, 0 of 200 with MTP on on a fresh server and 0 of 200 after replaying an agent session;
one earlier copy on a long-running server wrote "O" for "0" six times (6 of 520 with MTP on in all). The digits GLM
garbled in agent sessions (47G read as 447G, 185G written as 1.8 TB) are the model's, not the engine's; the agent's
system prompt (`AGENT_PROMPT`) asks it to keep figures as the tools print them.

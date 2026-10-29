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

## GLM-5.3-Flash (GSQ-RCO 3.0bit, project-maya's engine)

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
| 1.0 / 0.95, top_k 20 (project-maya's engine default) | 30.0 | 32.4 | 35.3 | 36.0 | 32.1 | 26.8 | 32.1 | 79.5% |

Sampling at the API's values costs the drafts nothing measurable (one run an arm: the means are within 4%).

## Fixed in this review

- ds41_generate kept a request's repetition / frequency / presence penalty for every request after it (one global
  state): each request now starts from the command line's.  Checked: greedy "apple" x10, then a request with
  repetition_penalty 3.0 (2 words), then the first again - the same 10 words.
- The server sent top_k 0 ("off") to DeepSeek as 64; DeepSeek's engine without one takes its 256 best candidates,
  which is what it gets now.

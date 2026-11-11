# Handover

Where Almond stands and what is open. How it runs and every measured number: [README](README.md),
[docs/MODELS.md](docs/MODELS.md), [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md),
[docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md), [docs/ENGINE_PLAN.md](docs/ENGINE_PLAN.md),
[docs/ENGINE_REVIEW.md](docs/ENGINE_REVIEW.md).

## State (2026-11-08)

- **Machine:** two RX 7900 XTX (card 1 on a PCIe 4.0 x2 uplink), Ryzen 9 9900X, 122 GiB RAM, PCIe 5.0 NVMe, ROCm 7.14.
  The cards' 305 W cap, -200 mV and stock memory clock are deliberate; power-tune (NVMe APST and CPU C3 off, the
  `performance` governor) runs with a server (`/usr/local/sbin/almond-power-tune`).
- **One engine:** `build-hip-fused/almond-engine`, DeepSeek's path (`src/ds41`), GLM's (`src/glm`, `--glm-pack`) or
  Qwen's (`src/qwen`, `--qwen`), on one ggml (`third_party/llama.cpp`). The router (`./almond serve --router`) serves
  them at
  `127.0.0.1:8150`, loading the one a request names; `./almond` is the terminal chat and agent.
- **DeepSeek-V4.1-Flash-1017:** the ISTA-DASLab GSQ-RCO IQ2_XXS pack (138.2 GB, four shards of the 2026-10-25
  revision, read through `DS41_META`), the checkpoint's FP8 Engram rows, exact routing. Decode ~21-23 tok/s short,
  ~18-19 at 17K, 16-20 in agent sessions at 120-190K; Lucebox ppl 10.37. Levels off / low / high / max.
- **GLM-5.3-Flash:** the GSQ-RCO 3.5bit GGUF (137.1 GB) with the MTP block from an IQ2_S quant; light misses left out
  of layers 23+ (`ALMOND_GLM_MISS_SKIP_W` 0.16): 33.8 tok/s, PPL 13.51 over 4,080 tokens (exact: 25.9, 13.44; the
  3.0bit build it replaced: 31.8, 14.01). Its engine matches llama.cpp: PPL 7.02-7.14 against 7.20 at 512 tokens,
  5.97-6.01 against 5.95 at 8K. Levels low / high / max.
- **Qwen3.8-Flash-Next:** the GSQ-RCO IQ3_S GGUF (83.6 GB, two shards) with its pack and the MTP block from the BF16
  checkpoint, on card 0: 87.4 tok/s through the server (79-100 per prompt), the same as the build it came from (88.4,
  and the same 64 greedy ids one-shot). Levels none / low / medium / xhigh. Images on the CPU (`almond-vision`).
- **Qwen3.8-27B (dense):** the GSQ-RCO IQ3_S GGUF (11.28 GiB, MTP block included) on its own path (`src/qwen35`,
  `--qwen35`, ggml graphs), `./almond serve --model qwen27`: one card, 192K, q8_0 KV, MTP 3 by default. 73.3 tok/s
  mean greedy over three prompts (llama.cpp 75.5), PPL 6.1552 (llama.cpp 6.1542); 256K on two cards
  (`local/qwen27-local-256k.json`); the Q8_0 on two cards (`qwen27-q8`, 64.8 tok/s with MTP 3 against 101.9 for the
  IQ3_S); Tencent's DSpine drafter inside ``` blocks and tool calls, MTP elsewhere (`--spec auto`, `qwen27-dspine`,
  `qwen27-q8-dspine`: +31% on code, prose as MTP). z-lab's DFlash2 drafter was removed on 2026-11-11 (its
  configuration and download; the engine still reads the format, the frame DSpine shares). Its own chat template (the GGUF's); levels off / low / medium / xhigh, each checked
  through the server. The quantization of every model is chosen in `local/fetch.sh` (separate items, a suggestion
  from the machine's cards and RAM): GLM 3.5 or 3.0 bit, the 27B IQ3_S or Q8_0.
- **Images:** DeepSeek and GLM, each encoder in VRAM only for a request with pictures (DeepSeek's loaded and freed; GLM's
  in 2.78 GB the engine lends it, at up to 1,024 image tokens since its memory is measured without flash attention).
- **Conversations:** each kept in RAM and on disk and resumed from the longest prefix (`local/conv-cache-rco`,
  `local/glm-slots`, `local/qwen-slots`); `./almond --sessions` / `--clear-sessions`.

## Fixed in the last days

- GLM in long agent sessions with pictures (docs/MODELS.md, 2026-11-05): picture keys instead of embedding hashes,
  a checkpoint kept at the last user turn, the 2 GB slot save skipped when only dropped reasoning would be lost (that
  skip measured: 48 saves skipped, 16 still made; `ALMOND_GLM_SLOT_SKIP_SAME` for those, built, not measured yet).
- DeepSeek's and GLM's MMQ libraries made `ggml_backend_cuda_context` 64 bytes smaller than ggml-hip's own code reads
  it (`GGML_HIP_GRAPHS` missing): GLM's engine could stop at load with a corrupted heap; all three now use ggml's
  layout (docs/ENGINE_REVIEW.md).
- `stop.sh` finds the engine by its executable (Qwen's path names its main thread `qwen-host`, which `pgrep -x
  almond-engine` missed).

- GLM's tokenizer split digits one by one (qwen35's pre-split) where the model reads them in threes (glm4): figures
  copied wrong at 10K+ (21-41 of 200); now llama.cpp's ids on ~300K tokens, 0 of 200.
- DeepSeek's VRAM cache opens again after a batch prompt into what ggml gave back (2.5 GiB were left unused).
- The CLI sends the reasoning back (an agent's tool chain had lost its plan at every step); the agent has an English
  system prompt; tool calls closing a code fence on the same line are calls.
- Streams send events while a long prompt is read or a call written inside the thinking is held; dsh's own idle
  timeout (300 s, pi-ai) must still be raised for a local provider (`streamIdleTimeoutMs`).
- `fit_max_tokens` on: a client's max_tokens beside a long prompt gets the room left, not a 400.

## Open

- **Qwen3.8-27B:** images (CPU encoder, 300 image tokens: a large screenshot's small text is not legible, as on
  Flash-Next) and conversations on disk work; measured on a few pictures and one restart, not in long agent sessions. The 4-token verify window costs 1.4 decode tokens (ggml's MMVQ is
  latency-bound on RDNA3; table, decode-once and unroll changes did not help); past ~100K the q8_0 KV's f16
  conversion in the verify window halves the decode. Its files are not in `local/fetch.sh`.

- **Qwen3.8-Flash-Next:** no perplexity yet (no `LOGP` on its path, no prompt logits on the native one), no
  long-context or agent-session measurement; one card only (the engine has a layer split for two, not tried here).

- **GLM's DFlash2 drafter** measured: 26.6 tok/s against MTP's 34.5 (docs/MODELS.md); MTP stays.
- **An 8K-token `LOGP` on the 3.5bit GLM** stopped in "returning the expert pool's tail" (the 180 s watchdog); 12K
  prompts through the server read fine.
- **Speed:** where the time goes in the sessions and the levers left, with what was measured on 2026-11-10 (GLM's
  shared CPU pool, DeepSeek's readahead, arena 88, 24 threads, the late-layer skip: none adopted): docs/SPEED_LEVERS.md.
  Built 2026-11-11, on in the DeepSeek configurations since (`ALMOND_SERVE_BATCH_KEEP=1024`): it lends the VRAM cache's tail to a
  small prompt batch instead of releasing it (a ~500-token agent turn 10.5-18.5 s -> 6.8-8.3 s over 7 turns, measured there; turns of
  600-1,024 tokens and long agent sessions not measured yet).
- **DeepSeek:** `--skip-miss 0.15` +7% for +0.6-0.8% decode PPL (off); `--skip-miss-from 21` +6-8% at 17K, its PPL
  not measured (docs/SPEED_LEVERS.md).
  Prompt chunks of 4K instead of 8K (project-maya's RDNA3 finding) not measured. The CPU tier is at the RAM's
  bandwidth; speculation (lookup, DSpark) and the conversation-seeded cache measured slower and stay off.
- **Long agent sessions:** a client that rewrites the start of its prompt (dsh's plan mode in the system prompt,
  DeepSeek's reasoning level at token 8 when compaction runs at another level) reads the whole conversation again.

## Constraints

- Never expose a server beyond `127.0.0.1` without `--api-key`.
- A new feature is opt-in; a default changes only where it measured better, with the numbers written down.
- The engine is not bit-reproducible run to run: compare quality with PPL on fixed documents, Lucebox or repeated
  runs, not with a diff.
- Commit messages in English.

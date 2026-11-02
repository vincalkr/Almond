# Handover

Where Almond stands and what is open. How it runs and every measured number: [README](README.md),
[docs/MODELS.md](docs/MODELS.md), [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md),
[docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md), [docs/ENGINE_PLAN.md](docs/ENGINE_PLAN.md),
[docs/ENGINE_REVIEW.md](docs/ENGINE_REVIEW.md).

## State (2026-11-02)

- **Machine:** two RX 7900 XTX (card 1 on a PCIe 4.0 x2 uplink), Ryzen 9 9900X, 122 GiB RAM, PCIe 5.0 NVMe, ROCm 7.14.
  The cards' 305 W cap, -200 mV and stock memory clock are deliberate; power-tune (NVMe APST and CPU C3 off, the
  `performance` governor) runs with a server (`/usr/local/sbin/almond-power-tune`).
- **One engine:** `build-hip-fused/almond-engine`, DeepSeek's path (`src/ds41`), GLM's (`src/glm`, `--glm-pack`) or
  Qwen's (`src/qwen`, `--qwen`), on one ggml (`third_party/llama.cpp`). The router (`./run.sh --router`) serves them at
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
- **Images:** DeepSeek and GLM, each encoder in VRAM only for a request with pictures (DeepSeek's loaded and freed; GLM's
  in 2.78 GB the engine lends it, at up to 1,024 image tokens since its memory is measured without flash attention).
- **Conversations:** each kept in RAM and on disk and resumed from the longest prefix (`local/conv-cache-rco`,
  `local/glm-slots`); `./almond --sessions` / `--clear-sessions`. Qwen's in RAM only (up to 4, 8 GiB).

## Fixed in the last days

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

- **Qwen3.8-Flash-Next:** no perplexity yet (no `LOGP` on its path, no prompt logits on the native one), no
  long-context or agent-session measurement; one card only (the engine has a layer split for two, not tried here).

- **GLM's DFlash2 drafter** measured: 26.6 tok/s against MTP's 34.5 (docs/MODELS.md); MTP stays.
- **An 8K-token `LOGP` on the 3.5bit GLM** stopped in "returning the expert pool's tail" (the 180 s watchdog); 12K
  prompts through the server read fine.
- **DeepSeek:** `--skip-miss 0.15` +7% for +0.6-0.8% decode PPL (off); a from-layer option as GLM's is not built.
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

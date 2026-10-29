# Almond: the plan

Almond is ds41e (../ds41e) made multi-model. Today it runs what ds41e runs, renamed (`strata` -> `almond` in the code,
`STRATA_*` -> `ALMOND_*` in the environment, `include/almond/`); it builds with `./build.sh` and the server tests pass.
`models/` is a symlink to `../ds41e/models`. Nothing below is built yet; nothing below has a measured number yet.

## Rules that stay

- The default path computes every expert the router picks (no route bias, no dropped experts, no lossy skip). A lossy
  option may exist, opt-in, with its KL against the full model written down.
- A new feature is opt-in; a default changes only where it measured better, with the numbers.
- The box: two RX 7900 XTX (ROCm/HIP, gfx1100), Ryzen 9 9900X, 122 GiB RAM, NVMe. NVIDIA-only or Metal-only code is a
  reference to port, not something to link.

## Models, in order

1. **DeepSeek-V4.1-Flash-1017** (what ds41e runs, GSQ-RCO IQ2_XXS pack): unchanged, the reference.
2. **GLM-5.3-Flash** (321 B, 18 B active): first new model.
3. **Qwen3.8-Flash-Next** (125 B MoE): second new model. Smaller than the two above: on 48 GB of VRAM plus RAM most of
   it is in VRAM, so it should be far faster.

## Where each piece comes from

All five sources are MIT. Their code is read and ported by hand; nothing downloaded is run.

| source | what it has | what Almond takes |
| --- | --- | --- |
| ds41e (ours) | DeepSeek-V4.1 on HIP: VRAM / peer card / RAM arena / NVMe tiers, exact routing, Engram, vision, server | the base: tiers, kernels, server |
| Strata (../Strata) | DeepSeek-V4.1 engine (`src/dsv41`), Qwen 3.5/3.6 (`src/qwen35`), many kernels, CUDA/HIP/SYCL/Metal | kernels and pack tools where they beat ours; Qwen's layer code as a reference |
| project-maya (../project-maya) | GLM-5.3 on HIP (`src/core/glm_*`: fast path, prefill, DFlash, MTP, CPU lane), MiniMax M3.1, Qwen-3.8-Flash-Next (GGUF, text) | GLM's model and attention code (it already runs on this box, Maya-M / Maya-S24); its `TRUNK_KEEP` / `MISS_SKIP` only as opt-in lossy options |
| Xolotl (github.com/Indras-Mirror/Xolotl, a Strata fork) | DeepSeek-V4.1, GLM-5.3, Qwen3.8-Flash-Next, Qwen3.8-27B; NVIDIA only | Qwen3.8-Flash-Next's model code and its numbers as a target; not its GLM defaults (`--route-bias 0.25` + soft prune: lossy) |
| ds4 (github.com/antirez/ds4) | DeepSeek V4 Flash/PRO, GLM 5.2/5.3, Qwen3.8 Flash Next; Metal and CUDA, ROCm on Strix Halo; exact routing | a small, readable reference for each architecture (attention, router, MTP) to check ours against |
| llama.cpp (`third_party/llama.cpp/ggml`) | ggml and its HIP kernels (MMQ, MMVQ, flash attention) | stays the kernel base; GGUF readers for the new architectures |

Which piece is "the best" for a model is decided by a measurement on this box, not by the source: each candidate is
built behind a flag and compared (tok/s, prompt tok/s, KL / top-1 against the reference) before it becomes a default.

## Phases

1. **Rename** (done): Strata -> Almond, builds, server tests.
2. **A model interface in the engine.** Split `tools/v41/engine` into what every MoE model shares (the expert tiers:
   VRAM cache, peer card, RAM arena, NVMe reads; the pool; the sampler; the serve loop) and what is DeepSeek's (MLA /
   compressed attention, HC mixes, Engram, router). A `Model` with `load`, `prefill`, `decode_step`, `route(layer)`,
   `expert_layout`; DeepSeek becomes the first implementation. Gate: the same tok/s as ds41e (3 alternated runs,
   Promessi Sposi and 17K) and the same forced-token argmax.
3. **GLM-5.3-Flash.** Port project-maya's GLM layers onto the interface, experts through Almond's tiers. Read GGUF
   (GSQ-RCO 3.0bit, 117.5 GB; Maya-M for comparison). Gate: KL / top-1 against project-maya on the same pack, then speed
   against project-maya on this box.
4. **Qwen3.8-Flash-Next**, the same way (Xolotl, Strata and ds4 as references).
5. **The server stays in Python** (`serve/`, ~9,900 lines: OpenAI / Anthropic API, streaming, chat templates, tool
   calls, web app, MCP, the engine's process). A token's time is all in the engine; the server only moves text and
   JSON, so rewriting it would change no tok/s. Each new model adds its chat template and tool-call format here. If one
   day Almond is to ship as a single binary with no `.venv`, a Go server can replace it endpoint by endpoint over the
   same stdin/stdout protocol (Go over C++: far less code for HTTP, SSE, JSON and templates), with the Python tests
   turned into Go tests on the same fixtures.

## Open questions

- Engine binary: one `almond` with `--model deepseek41|glm53|qwen38`, or one binary per model sharing libraries.
- GLM pack: GSQ-RCO 3.0bit GGUF directly, or project-maya's pack format.

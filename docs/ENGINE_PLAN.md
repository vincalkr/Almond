# One engine

Today Almond runs two engines: `ds41_generate` (DeepSeek-V4.1-Flash, `src/ds41` + `src/`, from ds41e) and
project-maya's engine for GLM-5.3-Flash (`engines/maya`, vendored, ~160K lines with DeepSeek and MiniMax paths Almond
does not use, its own pinned ggml). The goal: one engine of Almond's own, no vendored part, one ggml, one build, one
executable that runs every model, and the parts both need written once.

Every step keeps the models' numbers: PPL on the same documents (GLM: `local/glm-ref/doc512.i32`, 510 positions;
DeepSeek: the Lucebox set, `local/eval-lucebox/`) and decode tok/s through the server, before and after. A step that
measures worse is not merged.

## Steps

1. **One ggml.** GLM's engine built against Almond's ggml (`third_party/llama.cpp`, 0.26 with ds41e's patches)
   instead of its pinned 0.24 copy. `src/prefill/moe_mmq.cu` already followed ggml's MMQ API change (`J_best`,
   the src1 precision as a template argument); GLM's copy gets the same change. Then
   `engines/maya/third_party/llama.cpp` goes.
2. **The code in `src/`.** GLM's model, its expert pools and RAM tier, its prompt path, MTP drafting and the
   conversation slots move from `engines/maya/src` into `src/` beside DeepSeek's; DeepSeek's engine moves from
   `src/ds41` into `src/` too. What only project-maya's DeepSeek (`src/ds4`) and MiniMax (`src/mm3`) paths
   use is left out. One CMake project (`build.sh` builds one tree).
3. **One executable** (`almond-engine`): the model's GGUF / pack says which path runs; the server protocol (GEN,
   GENI, STOP, LOGP, VLEND ...) is one.
4. **Our names.** `STRATA_*` / `MAYA_*` environment variables, macros, namespaces and file names become `ALMOND_*` /
   `almond::`; the configurations already say `ALMOND_*`.
5. **Shared parts, one at a time, each measured:** the CPU worker pool and its pinning, the expert tiers (VRAM pool,
   pinned RAM, file reads; the expert profile both can learn from), conversations on disk (DeepSeek's
   `--conversation-disk` and GLM's slots), the vision encoder's memory lending (GLM's VLEND for DeepSeek's encoder:
   the ~0.6 GiB it leaves behind today), power and timing tools.
6. **Other repositories:** Strata, project-maya, Xolotl, llama.cpp, antirez's ds4 - what each does better for
   these models, measured here before it goes in.

## Measured

**Step 1, one ggml (2026-10-30).** GLM's engine on Almond's ggml 0.26 (one change: `engines/maya/src/prefill/moe_mmq.cu`
picks MMQ's tile width `J_best` and passes the src1 precision, as `src/prefill/moe_mmq.cu` does). GLM-5.3-Flash
GSQ-RCO 3.0bit, `LOGP` on `local/glm-ref/doc512.i32` (510 positions), three runs each (the engine is not
bit-reproducible run to run):

| ggml | PPL runs | mean |
| --- | --- | ---: |
| project-maya's pinned 0.24 | 10.4473, 10.3724, 10.4727 | 10.431 |
| Almond's 0.26 | 10.3646, 10.4876, 10.4328 (10.3262 after the rebuild) | 10.429 |

Server, thinking low, temperature 0, 600 tokens, three prompts (code, Italian, English), two rounds alternated, tok/s:
0.24 32.2 31.0 30.8 / 32.1 32.7 30.7 (mean 31.6); 0.26 32.4 32.4 30.6 / 31.8 32.0 30.5 (mean 31.6).
`engines/maya/third_party/llama.cpp` is gone; gguf-py moved to `third_party/llama.cpp/gguf-py`.

**Step 2, the code in `src/` (2026-10-30).** GLM's model, kernels, prompt path, MTP and slots: `src/glm/` (with
project-maya's LICENSE), its own CMakeLists.txt added by the top one, on the top level's prebuilt ggml; the executable
is `build-hip-fused/almond-glm`. project-maya's DeepSeek (`src/ds4`) and MiniMax (`src/mm3`) paths, its tests and
benches went to the trash. Its Python tools (the pack builder, the hipBLASLt tuner, the image encoder's sources):
`tools/glm/`. DeepSeek's engine: `tools/v41/engine` -> `src/ds41` (its Python tools in `tools/v41/`). GLM, `almond-glm`:
PPL 10.4159, 10.4583, 10.2627 (mean 10.379); server tok/s 32.1 33.2 31.5 / 32.1 31.9 30.1 (mean 31.8). DeepSeek's code
only moved (the binary differs in its source paths).

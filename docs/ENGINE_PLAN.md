# One engine

Today Almond runs two engines: `ds41_generate` (DeepSeek-V4.1-Flash, `src/ds41` + `src/`, from Almond) and
the engine for GLM-5.3-Flash (`engines/glm`, vendored, ~160K lines with DeepSeek and MiniMax paths Almond
does not use, its own pinned ggml). The goal: one engine of Almond's own, no vendored part, one ggml, one build, one
executable that runs every model, and the parts both need written once.

Every step keeps the models' numbers: PPL on the same documents (GLM: `local/glm-ref/doc512.i32`, 510 positions;
DeepSeek: the Lucebox set, `local/eval-lucebox/`) and decode tok/s through the server, before and after. A step that
measures worse is not merged.

## Steps

1. **One ggml.** GLM's engine built against Almond's ggml (`third_party/llama.cpp`, 0.26 with Almond's patches)
   instead of its pinned 0.24 copy. `src/prefill/moe_mmq.cu` already followed ggml's MMQ API change (`J_best`,
   the src1 precision as a template argument); GLM's copy gets the same change. Then
   `engines/glm/third_party/llama.cpp` goes.
2. **The code in `src/`.** GLM's model, its expert pools and RAM tier, its prompt path, MTP drafting and the
   conversation slots move from `engines/glm/src` into `src/` beside DeepSeek's; DeepSeek's engine moves from
   `src/ds41` into `src/` too. What only the DeepSeek (`src/ds4`) and MiniMax (`src/mm3`) paths
   use is left out. One CMake project (`build.sh` builds one tree).
3. **One executable** (`almond-engine`): the model's GGUF / pack says which path runs; the server protocol (GEN,
   GENI, STOP, LOGP, VLEND ...) is one.
4. **Our names.** `STRATA_*` / `MAYA_*` environment variables, macros, namespaces and file names become `ALMOND_*` /
   `almond::`; the configurations already say `ALMOND_*`.
5. **Shared parts, one at a time, each measured:** the CPU worker pool and its pinning, the expert tiers (VRAM pool,
   pinned RAM, file reads; the expert profile both can learn from), conversations on disk (DeepSeek's
   `--conversation-disk` and GLM's slots), the vision encoder's memory lending (GLM's VLEND for DeepSeek's encoder:
   the ~0.6 GiB it leaves behind today), GLM's image tower inside the engine on Almond's ggml as DeepSeek's is (today
   `almond-vision`, a separate process on llama.cpp's mtmd at another commit, `build-vision`), power and timing tools.
6. **Other repositories:** other engines - what each does better for
   these models, measured here before it goes in.

## Measured

**Step 1, one ggml (2026-10-30).** GLM's engine on Almond's ggml 0.26 (one change: `engines/glm/src/prefill/moe_mmq.cu`
picks MMQ's tile width `J_best` and passes the src1 precision, as `src/prefill/moe_mmq.cu` does). GLM-5.3-Flash
GSQ-RCO 3.0bit, `LOGP` on `local/glm-ref/doc512.i32` (510 positions), three runs each (the engine is not
bit-reproducible run to run):

| ggml | PPL runs | mean |
| --- | --- | ---: |
| the pinned 0.24 | 10.4473, 10.3724, 10.4727 | 10.431 |
| Almond's 0.26 | 10.3646, 10.4876, 10.4328 (10.3262 after the rebuild) | 10.429 |

Server, thinking low, temperature 0, 600 tokens, three prompts (code, Italian, English), two rounds alternated, tok/s:
0.24 32.2 31.0 30.8 / 32.1 32.7 30.7 (mean 31.6); 0.26 32.4 32.4 30.6 / 31.8 32.0 30.5 (mean 31.6).
`engines/glm/third_party/llama.cpp` is gone; gguf-py moved to `third_party/llama.cpp/gguf-py`.

**Step 2, the code in `src/` (2026-10-30).** GLM's model, kernels, prompt path, MTP and slots: `src/glm/` (with
the LICENSE), its own CMakeLists.txt added by the top one, on the top level's prebuilt ggml; the executable
is `build-hip-fused/almond-glm`. the DeepSeek (`src/ds4`) and MiniMax (`src/mm3`) paths, its tests and
benches went to the trash. Its Python tools (the pack builder, the hipBLASLt tuner, the image encoder's sources):
`tools/glm/`. DeepSeek's engine: `tools/v41/engine` -> `src/ds41` (its Python tools in `tools/v41/`). GLM, `almond-glm`:
PPL 10.4159, 10.4583, 10.2627 (mean 10.379); server tok/s 32.1 33.2 31.5 / 32.1 31.9 30.1 (mean 31.8). DeepSeek's code
only moved (the binary differs in its source paths).

**Step 3, one executable (2026-10-30).** `build-hip-fused/almond-engine`: DeepSeek's engine target with GLM's path
linked in (`glm_program`, link-only so each tree keeps its own forced `hip_compat/cuda_runtime.h`); `src/engine_main.cpp`
runs GLM's path when the arguments name `--glm-pack`, DeepSeek's otherwise. GLM's ggml-cuda host shim is weak, as
DeepSeek's: ggml-hip's own definitions win. GLM: PPL 10.3034, 10.4309, 10.4771 (mean 10.404); server tok/s 32.1 32.1
30.7 / 32.4 33.0 31.1 (mean 31.9). DeepSeek: text 20.9-23.9 tok/s, the two pictures read as before.

**Step 4, our names (2026-10-30).** GLM's code: namespace `strata::` -> `almond::glm::`, headers `strata/...` ->
`almond/glm/...`, macros and environment variables `STRATA_*` / `MAYA_*` -> `ALMOND_*` (`ALMOND_GLM_*` for GLM's),
CMake targets `strata_*` -> `glm_*`, the image encoder `maya-vision` -> `almond-vision` (`build-vision`); the server no
longer maps the configurations' names. Prose that says where code came from keeps the and the names, and
`src/glm/LICENSE` their copyright. GLM: PPL 10.2178, 10.3842, 10.4711 (mean 10.358); server tok/s 31.4 32.2 30.1 /
32.8 33.4 31.0 (mean 31.8); pictures read, the terzine question closed and answered (2,795 tokens). DeepSeek: text
19.0-23.8 tok/s, pictures as before.

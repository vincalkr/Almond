# src/glm

GLM-5.3-Flash's path of `almond-engine` (MIT: `LICENSE`): the model, its expert pools and RAM tier, the prompt path,
MTP drafting and the conversation slots, in namespace `almond::glm`. `CMakeLists.txt` builds its `glm_*` libraries on
the top level's prebuilt ggml (`third_party/llama.cpp`); the top `CMakeLists.txt` links them into `almond-engine`,
which runs this path when its arguments name `--glm-pack` (`src/engine_main.cpp`). The server: `tools/glm/serve/serve_glm.py`,
`./run.sh --model glm`. How it came here and the measurements: docs/ENGINE_PLAN.md.

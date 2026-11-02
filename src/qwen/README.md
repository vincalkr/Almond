# src/qwen

Qwen3.8-Flash-Next's path of `almond-engine` (MIT: `LICENSE`): the hybrid model (`qwen4exp`: 48 layers, Gated
DeltaNet with a sparse-attention layer every fourth, hyper-connections, the PLE n-gram table, 512 experts with 10
routed), its expert pools and RAM tier, the prompt path, MTP drafting and the conversation cache, in namespace
`almond::qwen`. `CMakeLists.txt` builds its `qwen_*` libraries on the top level's prebuilt ggml
(`third_party/llama.cpp`); the top `CMakeLists.txt` links `qwen_program` into `almond-engine`, which runs this path
when its arguments name `--qwen` (`src/engine_main.cpp`, which takes the flag out). The server:
`tools/qwen/serve/serve_qwen.py`, `./run.sh --model qwen`; the pack and the MTP block: `tools/qwen/`. How it came here
and the measurements: docs/MODELS.md, "Qwen3.8-Flash-Next".

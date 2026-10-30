# engines/glm

the GLM-5.3-Flash engine (MIT: `LICENSE`), as
Almond's GLM-5.3-Flash engine: `CMakeLists.txt`, `cmake/`, `include/`, `src/`, `tests/`, `tools/hip/`, the pack builder
(`tools/iq_pack.py` and what it imports), `serve/glm5_chat_template.jinja` and its pinned ggml
(`third_party/llama.cpp/{ggml,gguf-py}`, llama.cpp 3cf03257, as the GLM engine builds it). Unchanged: Almond builds
only the `glm` target (`./build.sh` -> `build-maya-hip/glm`) and runs it behind its own server
(`tools/glm/serve/serve_glm.py`, `./run.sh --model glm`). Why and the measurements: docs/ALMOND_PLAN.md, phase 3.

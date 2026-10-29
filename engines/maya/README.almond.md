# engines/maya

project-maya's engine (https://github.com/mw00/project-maya, v1.1.37, commit 7ee8f27, MIT: `LICENSE`), vendored as
Almond's GLM-5.3-Flash engine: `CMakeLists.txt`, `cmake/`, `include/`, `src/`, `tests/`, `tools/hip/`, the pack builder
(`tools/iq_pack.py` and what it imports), `serve/glm5_chat_template.jinja` and its pinned ggml
(`third_party/llama.cpp/{ggml,gguf-py}`, llama.cpp 3cf03257, as project-maya builds it). Unchanged: Almond builds
only the `maya` target (`./build.sh` -> `build-maya-hip/maya`) and runs it behind its own server
(`tools/glm/serve/serve_glm.py`, `./run.sh --model glm`). Why and the measurements: docs/ALMOND_PLAN.md, phase 3.

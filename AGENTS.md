# AGENTS.md

Almond is a multi-model MoE engine for AMD RX 7900 XTX cards (ROCm/HIP): DeepSeek-V4.1-Flash-1017 (the published
GSQ-RCO IQ2_XXS pack), GLM-5.3-Flash (GSQ-RCO 3.5bit) and Qwen3.8-Flash-Next (GSQ-RCO IQ3_S), their experts across
card 0, a second 24 GB card, pinned RAM and NVMe. One C++/HIP executable, `build-hip-fused/almond-engine` (DeepSeek's
path in `src/ds41`, GLM's in `src/glm` when its arguments name `--glm-pack`, Qwen's in `src/qwen` when they name
`--qwen`, the shared libraries in `src/` and `include/`); an OpenAI- and Anthropic-compatible server (`serve/`,
started per model by `tools/v41/serve/serve_ds41.py`, `tools/glm/serve/serve_glm.py` or
`tools/qwen/serve/serve_qwen.py` through `./almond serve`); the router (`tools/router/almond_router.py`, `./almond
serve --router`);
the terminal chat and agent (`./almond`, `tools/cli/almond_cli.py`); configurations, tools and measurements in
`local/`, `tools/` and `docs/`.

## Running it

- `./almond` starts the router on `127.0.0.1:8150` if nothing answers and chats with the default model (GLM);
  `./almond serve` one model's server (DeepSeek; `--model glm` GLM, `--model qwen` Qwen), `./almond serve --router` all
  of them
  behind one address.
  Never expose a server beyond `127.0.0.1` without `--api-key`. `./stop.sh` stops this folder's server.
- DeepSeek: `local/ds41-local.json` on cards 0 and 1 (`--gpu 0` / `--gpu 1`: `local/ds41-local-1gpu.json`); with the
  image encoder in `models/vision-1017`, the `-vision` configurations (the pack lacks the image routing bias, which
  `DS41_VL_BIAS` adds from `local/rco-iq2xxs-meta/vl-bias.gguf`). The pack (`models/rco-iq2xxs`, four shards, 138.2
  GB) is read through the geometry overlay `local/rco-iq2xxs-meta/meta.gguf` (`DS41_META`, built by
  `tools/v41/ds41_pack_meta.py`). Engram rows: the checkpoint's FP8 rows (`models/engram-1017`, 203 GB,
  `local/fetch-model.sh`) or the pack's iq4_nl table (110.6 GB, `local/fetch-rco-1017.sh`), which the engine falls
  back to; both measured (docs/ds4/V41_EXACT_FAST.md).
- GLM: `local/glm-local.json` (`local/glm-local-vision.json` with its encoder in `models/glm53-vision`).
- Qwen: `local/qwen-local.json` on card 0 (`local/qwen-local-vision.json` with its encoder,
  `models/qwen38-iq3s/mmproj-Qwen3.8-Flash-Next-BF16.gguf`, read on the CPU): the GGUF's two shards in
  `models/qwen38-iq3s`, its pack in `models/qwen38-iq3s-pack` (`tools/qwen/iq_pack.py`), the MTP block in
  `models/qwen38-mtp` (`tools/qwen/mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`).
- The model files live in `models/`, inside this folder (gitignored); `local/fetch.sh` downloads and prepares them,
  `./setup.sh` walks a new installation.

## Working on the code

- `./build.sh`: ggml from the vendored `third_party/llama.cpp/ggml` (its changes also kept as patches in
  `tools/v41/ggml/`) into `third_party/llama.cpp/build-hip`, the engine into `build-hip-fused`, GLM's image encoder
  into `build-vision`. HIP only.
- Server tests: `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .` (no GPU or model needed).
- How the engines work and every measured number: `docs/ds4/V41_EXACT_FAST.md`, `docs/ds4/AMD_HIP_NOTES.md`,
  `docs/MODELS.md`, `docs/ENGINE_PLAN.md`, `docs/ENGINE_REVIEW.md`; the state and what is open: `HANDOVER.md`.
- The engine is not bit-reproducible run to run (an expert's kernel depends on where it is served from, which
  depends on timing): compare quality with repeated runs, PPL on fixed documents or the Lucebox set in
  `local/eval-lucebox/`, not with a diff.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
- A new feature is opt-in and the default path stays as it was; change a default only where it measured better, and
  write down the numbers.
- Commit messages in English.

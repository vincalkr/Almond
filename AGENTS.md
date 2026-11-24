# AGENTS.md

Almond is a multi-model engine for AMD RX 7900 XTX cards (ROCm/HIP), for MoE and dense models: the MoE models
DeepSeek-V4.1-Flash-1017 (the published GSQ-RCO IQ2_XXS pack), GLM-5.3-Flash (GSQ-RCO 3.5bit) and Qwen3.8-Flash-Next
(GSQ-RCO IQ3_S), their experts across card 0, a second 24 GB card, pinned RAM and NVMe, and the dense Qwen3.8-27B
(GSQ-RCO IQ3_S) on one card. One C++/HIP executable, `build-hip-fused/almond-engine` (DeepSeek's path in `src/ds41`,
GLM's in `src/glm` when its arguments name `--glm-pack`, Qwen's in `src/qwen` when they name `--qwen`, the dense
Qwen3.8-27B's in `src/qwen35` when they name `--qwen35`, the shared libraries in `src/` and `include/`); an OpenAI- and Anthropic-compatible server (`serve/`,
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
  `./almond telegram` the same chat on a Telegram bot (`tools/telegram/almond_telegram.py`, long polling, token and
  allowed user ids in `local/telegram.json`, which git ignores).
- DeepSeek: `local/ds41-local.json` on cards 0 and 1 (`--gpu 0` / `--gpu 1`: `local/ds41-local-1gpu.json`); with the
  image encoder in `models/vision-1017`, the `-vision` configurations (the pack lacks the image routing bias, which
  `DS41_VL_BIAS` adds from `local/rco-iq2xxs-meta/vl-bias.gguf`). The pack (`models/rco-iq2xxs`, four shards, 138.2
  GB) is read through the geometry overlay `local/rco-iq2xxs-meta/meta.gguf` (`DS41_META`, built by
  `tools/v41/ds41_pack_meta.py`). Engram rows: the checkpoint's FP8 rows (`models/engram-1017`, 203 GB,
  `local/fetch-model.sh`) or the pack's iq4_nl table (110.6 GB, `local/fetch-rco-1017.sh`), which the engine falls
  back to; both measured (docs/ds4/V41_EXACT_FAST.md).
- GLM: `local/glm-local.json` (`local/glm-local-vision.json` with its encoder in `models/glm53-vision`).
- The quantization of a model is chosen in `local/fetch.sh` (separate items, each with its configuration and router
  entry); the router leaves out an entry whose files are not downloaded.
- Qwen3.8-27B (dense): `local/qwen27-local.json` (`./almond serve --model qwen27`, `src/qwen35`, `--qwen35`: one
  card, 192K, q8_0 KV, MTP 3; `-vision` with its encoder); `local/qwen27-local-256k.json` on two cards; `local/qwen27-q8-local.json` the Q8_0 on two cards (the two-card ones with `--tp`, tensor parallel, docs/MODELS.md) (`./almond -m qwen27-q8`); `-dspine` configurations with Tencent's DSpine drafting inside code and MTP elsewhere (`--spec auto`, `./almond -m qwen27-dspine`). Files in `models/qwen38-27b` (docs/MODELS.md, "Qwen3.8-27B").
- Hy4-flash: `local/hy4-local.json` (`-vision` with its encoder and `build-vision-hy4`) (`./almond serve --model hy4`, `src/hy4`, `--hy4`: cards 0 and 1, the experts that
  do not fit in VRAM read from RAM, the Engram table from the SSD or RAM, `--engram-place auto`); files in
  `models/hy4-iq2ah` with the tokenizer extracted by `tools/almond_tokenizer.py`; images with
  `build-vision-hy4/bin/almond-vision` (built against the PR's mtmd); docs/hy4/GGUF_NOTES.md.
- Inkling 2 Mini: `local/inkling-local.json` (`./almond serve --model inkling`, `./almond -m inkling`): Unsloth's
  UD-Q4_K_XL (17.6 GB) on card 0, 128K, F16 KV, on Almond's own engine `build-inkling/almond-inkling-native`
  (`src/inkling/native`, HIP, no ggml; docs/inkling/ENGINE.md): through the server 1,500 tok/s prefill at 5.7K and
  1,047 at 48K, 55-64 tok/s decode, against 553 / 337 and 29 on llama.cpp, KL from Q8_0 0.00557 against 0.00573.
  The llama.cpp path (`build-inkling/almond-inkling`, `src/inkling`) is no longer served (removed as slower; its
  numbers stay in docs/inkling/); files in `models/inkling2-mini` (`local/fetch.sh inkling`), measurements in `local/inkling-measure.sh`
  and docs/inkling/.
  `local/inkling-dspark-local.json` (`./almond -m inkling-dspark`, opt-in) adds Unsloth's DSpark drafter
  (`local/fetch.sh inkling-dspark`) to the native engine (`--dspark`, docs/inkling/ENGINE.md step 6): through the
  server 123-144 tok/s at temperature 0 and 98-135 at 0.6 against 77-78, the same distribution; requests above
  temperature 0.8 are not drafted (no gain at 1.0). The llama.cpp build of it (`build-inkling/almond-inkling-dspark`,
  PR #32218 in `build-inkling/llama.cpp-dspark`) is still built but no configuration names it.
- Qwen: `local/qwen-local.json` on card 0 (`local/qwen-local-vision.json` with its encoder,
  `models/qwen38-iq3s/mmproj-Qwen3.8-Flash-Next-BF16.gguf`, read on the CPU): the GGUF's two shards in
  `models/qwen38-iq3s`, its pack in `models/qwen38-iq3s-pack` (`tools/qwen/iq_pack.py`), the MTP block in
  `models/qwen38-mtp` (`tools/qwen/mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`); `local/qwen-q4-local.json` Unsloth's UD-Q4_K_XL (Dynamic 3.1) on two cards
  with at most 56 GiB of experts in RAM (`./almond -m qwen-q4`, `local/fetch.sh qwen-q4`; not yet measured here).
- `./almond-bench`: quality benchmarks against the router (MMLU-Pro, GPQA, MATH-500, HumanEval+/MBPP+, IFEval,
  BFCL graded without a judge, Arena-Hard with a judge model, KL divergence from logit dumps), Textual UI;
  `./almond-bench fetch` downloads the missing datasets into `local/bench/data` (`--update` newer revisions; a
  manifest keeps each one's revision and sha256; not committed), `run` works without the UI; docs/BENCH.md.
- The model files live in `models/`, inside this folder (gitignored); `local/fetch.sh` downloads and prepares them,
  `./setup.sh` walks a new installation.

## Working on the code

- `./build.sh`: ggml from the vendored `third_party/llama.cpp/ggml` (its changes also kept as patches in
  `tools/v41/ggml/`) into `third_party/llama.cpp/build-hip`, the engine into `build-hip-fused`, GLM's image encoder
  into `build-vision`. HIP only.
- Server tests: `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .` (no GPU or model needed).
- How the engines work and every measured number: `docs/ds4/V41_EXACT_FAST.md`, `docs/ds4/AMD_HIP_NOTES.md`,
  `docs/MODELS.md`, `docs/ENGINE_PLAN.md`, `docs/ENGINE_REVIEW.md`; the state and what is open: `HANDOVER.md`.
- Reproducibility: **a hy4 run repeats exactly** run to run since 2026-11-22 - the indexer's pool pick sorts instead of
  taking `ggml_top_k`'s tied keys as they come (`docs/hy4/SPEED_WORK.md`, "The cause is the pick, not the score
  kernel"). The other engines do not: an expert's kernel depends on where it is served from, which depends on timing,
  and `src/ds41/ds41_dense.cpp:741,744` and `src/qwen35/qwen35.cpp:1057` still pick through `ggml_top_k`. Compare
  quality with repeated runs, PPL on fixed documents or the Lucebox set in `local/eval-lucebox/`, not with a diff.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
- A new feature is opt-in and the default path stays as it was; change a default only where it measured better, and
  write down the numbers.
- Commit messages in English.

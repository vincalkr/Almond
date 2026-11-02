<h1 align="center">Almond - MoE models on RX 7900 XTX</h1>

<p align="center"><b>DeepSeek-V4.1-Flash-1017 and GLM-5.3-Flash on one or two RX 7900 XTX, their experts across VRAM, pinned RAM and NVMe</b><br>
Linux - ROCm/HIP - MIT</p>

A C++/HIP inference engine for large MoE models (one executable, `build-hip-fused/almond-engine`), an OpenAI- and
Anthropic-compatible server, a router that serves every model at one address, and a terminal chat with an agent mode.

- **DeepSeek-V4.1-Flash-1017** as the [ISTA-DASLab GSQ-RCO pack](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
  (IQ2_XXS/IQ2_XS/IQ1_M experts, four shards, 138.2 GB), read through the geometry overlay `DS41_META` because its
  GGUF carries llama.cpp metadata instead of this engine's `deepseek41.*` keys. Every expert the model routes to is
  computed (no route bias), wherever it is: card 0's VRAM, the second card, an 80 GiB pinned RAM arena, or the NVMe.
- **GLM-5.3-Flash** as the GSQ-RCO 3.5bit pack (137 GB) with its MTP draft block (`src/glm`).

How each model is set up and measured: [docs/MODELS.md](docs/MODELS.md); the one-engine work:
[docs/ENGINE_PLAN.md](docs/ENGINE_PLAN.md).

## Performance

DeepSeek-V4.1-Flash, two RX 7900 XTX, Ryzen 9 9900X, 122 GiB RAM, PCIe 5.0 NVMe, ROCm 7.14, the default configuration's flags with
power-tune on, the GSQ-RCO pack (2026-10-26, one-shot runs, 3 each):

| | |
| --- | ---: |
| decode, Promessi Sposi (300 tokens) | 23.2-23.7 tok/s |
| decode at 17K context (128 tokens) | 18.4-19.2 tok/s |
| prompt from scratch, 8K / 17K / 25.5K tokens | 371-414 / 433-459 / 458-484 tok/s |

Through the server, in a coding-agent session at 120-190K context: 16-20 tok/s decode, and a turn's new tokens resumed
on the saved conversation in 0.6-10.5 s.

Quality on Lucebox's held-out set (8 x 1024 tokens, exact routing): ppl 10.37 with the checkpoint's FP8 Engram rows,
10.33 with the pack's own iq4_nl table (KL 0.026 between the two); the FP8 1017 checkpoint itself reads 8.56. Every
number and how it was measured: [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md).

GLM-5.3-Flash (GSQ-RCO 3.5bit), the same machine, through the server (thinking low, 600 tokens, three prompts, two
rounds): 27.5 tok/s decode; PPL 13.38-13.43 over 4,080 tokens of `local/glm-ref/long-docs.txt`, 4.3% under the 3.0bit
build it replaced (docs/MODELS.md).

## Requirements

| | |
| --- | --- |
| OS, GPU stack | Linux, ROCm (7.14 here) with `hipcc`, hipBLAS and `/opt/rocm/lib/llvm/bin/clang++` |
| GPU | RX 7900 XT / XTX (`gfx1100`); a second 24 GB card is used as an expert tier when present |
| RAM | 128 GB class (the engine holds ~94 GB with the 80 GiB arena) |
| CPU | x86-64 with AVX2 (AVX-512 used when present) |
| disk | on an NVMe for `models/`: DeepSeek ~250 GB (138 GB pack in four shards + 111 GB Engram GGUF + 0.9 GiB image encoder; the checkpoint's own FP8 Engram rows instead of that GGUF are 203 GB), GLM ~142 GB (the 3.5bit GGUF, MTP block, image encoder) |
| tools | `cmake` >= 3.24, gcc, Python 3, curl, `aria2c` (faster downloads) |

## Quick start

```sh
./setup.sh                                   # tools, .venv, the models to download, the build, power-tune, each model's context
./almond                                     # the chat (it starts the router on 127.0.0.1:8150)
```

or by hand:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # the server's Python packages
local/fetch.sh                                                        # the models: what is in models/, which to download
./build.sh                                                            # ggml, the engine, GLM's image encoder (tens of minutes the first time)
./run.sh                                                              # DeepSeek's server on 127.0.0.1:8150 (--model glm: GLM's)
```

- `./setup.sh --check` only says what is there. `local/fetch.sh` lists the models with their state in `models/` and
  asks which to download (DeepSeek's pack and FP8 Engram rows through their own scripts, GLM, its MTP block, both
  image encoders); `local/fetch.sh --list` only lists.
- `local/fetch-rco-1017.sh` downloads the four shards and the pack's Engram IQ4_NL GGUF with `aria2c` (4 connections,
  `CONN`), checks each sha256, and writes the geometry overlay `local/rco-iq2xxs-meta/meta.gguf` with
  `tools/v41/ds41_pack_meta.py` (its reference geometry is `local/rco-iq2xxs-meta/geometry-keys.json`, 8 KB, in the
  repo) plus the rank byte map.
- The Engram tables come in two forms and the engine reads either. `local/fetch-rco-1017.sh` brings the pack's own
  (iq4_nl, 144 bytes a row, 111 GB), and that is all a new machine needs. The checkpoint's FP8 rows (264 bytes a row,
  203 GB in `models/engram-1017`) measured the same on Lucebox (ppl 10.37 against 10.33, KL 0.026);
  `local/fetch-model.sh` extracts them (it downloads a 366 GB source pack, which can be deleted afterwards). The
  default configuration names the FP8 rows; where they are missing the engine uses the iq4_nl table beside the pack
  and prints that it does.
- `./build.sh` builds the vendored ggml (`third_party/llama.cpp/ggml`) into `third_party/llama.cpp/build-hip`, the
  engine into `build-hip-fused`, and GLM's image encoder into `build-vision`; `ROCM`, `GFX`, `JOBS`, `REBUILD_GGML=1`
  from the environment.
- `./run.sh` uses cards 0 and 1; `./run.sh --gpu 0` (or `--gpu 1`) one card, `--gpu 0,1` both. The number of cards
  picks the configuration: `local/ds41-local.json` (two cards) or `local/ds41-local-1gpu.json` (one);
  `--config FILE` uses that file. Other arguments go to the server (`./run.sh --port 8151`). `./stop.sh` stops a
  server started in the background (as Ctrl+C would; `--force` kills it if it does not end in 60 s). Loading takes about a minute; the API is at
  `http://127.0.0.1:8150/v1` (OpenAI and Anthropic), the chat is `./almond` (below). Never expose the server beyond
  `127.0.0.1` without `--api-key`.

## Models, the router and the chat

Almond serves DeepSeek-V4.1-Flash (`./run.sh`, the default) and GLM-5.3-Flash (`./run.sh --model glm`, its path of
the same executable in `src/glm`, MIT: `src/glm/LICENSE`; `local/fetch.sh` downloads and prepares its files).
`./run.sh --router` puts every model of `local/router.json` behind one address, as llama-server's
router mode does: `GET /v1/models` lists them with their status (loaded / loading / unloaded), context and thinking
levels (`reasoning_levels`, `reasoning_default`); `POST /models/load` / `/models/unload {"model": NAME}`; a request that
names another model loads it, stopping the one running once its requests are done (the cards hold one at a time:
GLM loaded in 32 s, DeepSeek in 38 s from the file cache). The others go to the loaded model; `--no-autoload` refuses
instead.

`./almond` is the chat in the terminal (it starts the router when nothing answers, and stops it on leaving):

```sh
./almond                                   # a conversation; /help lists the commands
./almond -m glm --think-level low -p "..." # one prompt, the answer on stdout (the thinking on stderr)
./almond --monitor                         # the server's state, refreshed every second (no web pages)
./almond --session-log                     # the servers it starts log every request and answer to local/session-log/<date>.jsonl
```

- thinking: each model's own levels, from the client's words. GLM-5.3-Flash: low / high / max (its default; an
  Anthropic budget under 2K low, under 8K high, else max). DeepSeek-V4.1: off / low (50) /
  high (75, default) / max (100) or a number 1-100 (`none` is taken as off). `--nothink`, `--think`, `--think-max`, `--think-level L`; `/think L`,
  `/nothink` in the chat. `/health` and `/v1/models` say which levels a model has.
- DeepSeek's sampling follows its cloud API (deepseek-flash): with thinking the temperature has no effect (1.0) and
  top_p is 0.95-1.0 (default 1.0); without thinking temperature 0-2 (default 1.0) and top_p 1.0; no top_k, min_p or
  penalties; an unset max_tokens is 8K without thinking, 64K with it, 128K at max effort.
  `ALMOND_DS41_FREE_SAMPLING=1` passes a request's values as they are.
- GLM-5.3-Flash follows Z.ai's API: it always thinks, at low / high / max (default max; none, minimal and
  Anthropic's disabled are low, medium high, xhigh max); temperature 0-1 (default 1.0), top_p 0.01-1 (0.95), no
  penalties, top_k or min_p, at most 128K tokens out. `ALMOND_GLM_FREE_SAMPLING=1` and `ALMOND_GLM_THINK_OFF=1`
  (an Off level that closes the thinking at once: not Z.ai's, not measured) step outside it.
- sampling: `--temp`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--presence-penalty`, `--seed`,
  `--max-tokens`; `/set KEY VALUE` in the chat, `/set` alone shows them next to the model's defaults (`/props`).
- `/stats` (or `--monitor`): `GET /metrics` as text - state, speed, context, experts in VRAM, each GPU's load, VRAM,
  temperature and power, PCIe, CPU, RAM, disk, the conversation cache, the last requests and the totals.
  `/forget` clears the conversation cache.
- on a terminal the last line is a status bar (a scroll region the chat scrolls above): reading the
  prompt a bar with tokens read / total and tok/s, writing the tokens, tok/s, the time and thinking or answer, idle the
  last request's prompt / reused / output and tok/s, the context in use, the thinking level and the model (from
  `/metrics`); `-p` and piped output have none.
- `--agent` (or `/agent on`): the model gets tools - `read_file`, `write_file`, `edit_file`, `list_dir`,
  `run_command` (bash in the working directory, the output cut to 8 KB) - and the chat runs them until an answer
  without calls (at most 50 steps; Ctrl+C stops). Reads and listings inside the working directory go ahead; a write,
  an edit (shown as a diff first), a command (shown first) or any path outside asks y / n / a (always, for that
  tool, this session); `--yes` asks only for paths outside, which always ask. Without `--system` the agent gets a
  short English system prompt (`AGENT_PROMPT` in `tools/cli/almond_cli.py`): the machine and directory, how to use the
  tools, and to report figures as the tools printed them.
- Conversations: `/new` (the current one saved first), `/save [NAME]`, `/sessions`, `/load N|NAME`, `/rename`,
  `/delete` (to the trash); they live in `local/chats/`, leaving saves too, `./almond -c` resumes the last one. A
  conversation the engine still keeps is not read again.
- `/retry`, `/undo`, `/edit` (the last exchange: again, dropped, your message back to edit), `/export [FILE]`
  (Markdown), `/status`, `/cd DIR` (the agent's directory).
- `/models`, `/model NAME` (switches through the router), `/read FILE`, `/system TEXT`, `/ctx`, `/hide`, `/show`;
  `/` alone lists the commands, Tab completes them (with what each does) and their arguments; a line ending in `\`
  goes on, `"""` opens a block; Ctrl+C stops an answer.

Through the API the level is `reasoning_effort`, `reasoning.effort` (OpenAI) or `output_config.effort` / `thinking`
(Anthropic). DeepSeek's goes into its prompt (`Reasoning Effort: N`), with no cap on the thinking, as DeepSeek's API
(a request's `reasoning_budget_tokens` sets one). Thinking that loops is caught (`serve/loop_guard.py`) and closed so
the model answers (docs/MODELS.md, "Loops in the thinking").

The API for another program (an agent harness, an IDE): `./run.sh --router` (or `./almond --keep`) and point it at
`http://127.0.0.1:8150/v1`; `GET /v1/models` lists `deepseek-v4.1-flash-1017` and `glm-5.3-flash`. `--session-log DIR`
(or `ALMOND_SESSION_LOG`) writes every request and its answer to `DIR/<date>.jsonl`, for debugging. A request whose `max_tokens` does not fit beside its prompt (a harness sending the
model's 128K) gets the room left instead of a 400 (`"fit_max_tokens": true` in the configurations).

## Images

Both models read images; each one's encoder is in VRAM only for a request with images.

**GLM-5.3-Flash:** with its encoder in `models/glm53-vision`, `./run.sh --model glm` picks
`local/glm-local-vision.json`; the encoder (`build-vision/build/bin/almond-vision`) runs in 2.7 GB of VRAM the engine
lends it for the request and takes back after (docs/MODELS.md, "Images").

**DeepSeek-V4.1-Flash:** images need the model's image routing bias, `blk.N.exp_probs_b_vl.bias`, which the GSQ-RCO pack does not carry. It is
in the repository as `local/rco-iq2xxs-meta/vl-bias.gguf` (61 KB): the checkpoint's `ffn.gate.bias_vl`, as
ISTA-DASLab publish it (`exp_probs_b_vl.f32`), written by `tools/v41/ds41_vl_bias.py`; the configurations add it with
`DS41_VL_BIAS`. The pack's text bias is bit-identical to the checkpoint's in all 40 layers, so the pack's routing was
not recalibrated and the image bias fits it. With the image encoder in `models/vision-1017` (`local/fetch-rco-1017.sh`
builds it from the checkpoint), `./run.sh` picks `local/ds41-local-vision.json` (`-1gpu-vision` for one card).

The encoder (0.9 GiB) is loaded on card 0 only for a request with images, in the VRAM the expert cache gives up for
the prompt, and freed before the cache re-opens; text requests never load it. A 22x70-patch image is loaded in 0.6 s
and encoded in 0.2 s; encoded images are kept (8 by default, `ALMOND_VISION_CACHE`), so a conversation re-sends them
free. After any batch prompt (an image's, or text over ~128 new tokens) the next request opens the expert cache again
once into the VRAM ggml gave back meanwhile: free VRAM 0.59-0.60 GiB after it as before, decode 20.7-21.9 tok/s
(docs/ENGINE_REVIEW.md; `ALMOND_VRAM_REGROW=0` turns it off).

## Conversations

DeepSeek's engine keeps every conversation a request does not continue in a 4 GiB pool of pinned RAM
(`--conversation-cache-mib`, ~340 MB for a 190K-token session) and resumes the one a request continues, exactly (a
re-rendered tail a few tokens back included). They are also saved in `local/conv-cache-rco` (`--conversation-disk`,
up to 32 GiB), so a restarted server resumes a session instead of reading it again
([docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md), "The conversation pool", "Conversations on disk"). GLM's
keeps its conversations in `local/glm-slots` (`ALMOND_GLM_SLOT_KEEP`, docs/MODELS.md, "Sessions on disk").
`./almond --sessions` shows both folders and their size, `./almond --clear-sessions` moves them to the trash (with no
engine running), and `--monitor` shows them too.

## Power-tune (optional, root)

Decode reads an expert from the NVMe every 10-20 ms; between reads the drive (APST) and the CPU cores (C3) fall
asleep and each read costs more. `local/power-tune.sh on` keeps them awake and sets the `performance`
governor; it also raises the RAM's streaming bandwidth from 51-54 to 61-62 GB/s, which the CPU's experts read at.
Install it once and `./run.sh` switches
it on at start and off when the server ends:

```sh
sudo local/install-power-tune.sh             # a root-owned copy and a sudo rule for it alone
```

`DS41_POWER_TUNE=0 ./run.sh` leaves the settings alone; `sudo local/install-power-tune.sh uninstall` removes it;
`sudo local/power-tune.sh on|off|show` by hand.

## Expert rank (DeepSeek)

Which experts live in VRAM and in the RAM arena follows a rank of the most used ones (`local/rank.bin`). The server
counts what the router picks (`local/routes-rco`); `local/update-rank.sh` builds a new rank from those counts (experts
ranked per byte with the pack's formats, `local/rco-iq2xxs-map.json`), measures it against the one in use on a code
review and a documentation prompt (`local/prompts/dev-code`, `dev-docs`; ~30 minutes, the server stopped meanwhile)
and applies it only if it is faster.

## Another machine (DeepSeek's configuration)

`local/cfg.py` edits `local/ds41-local.json` (`python3 local/cfg.py show` prints it):

```sh
python3 local/cfg.py arg --threads 8         # the CPU's physical cores (12 here)
python3 local/cfg.py arg --arena-gib 50      # pinned RAM for experts: 80 GiB on 122 GiB here, keep ~40 GiB free beside it
```

The arena is the biggest lever after the cards: an expert that is in neither VRAM nor the arena is read from the
NVMe. `--vram-margin` (0.5 GiB) is the VRAM left free beside the expert cache; raise it if another program uses the
card. With one card use `./run.sh --gpu 0` (the two-card default stops at startup: "peer device 1 does not exist").

## Development

- Server tests (no GPU or model): `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .`
- Measurement scripts: `local/bench-v41/` (one-shot sweeps, profiles, the server benchmark `serve-bench.py`);
  quality: `local/eval-lucebox/`; `local/ram-bench.sh TAG [--decode]` compares RAM bandwidth and decode across BIOS
  settings (no arguments: the results so far).
- [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md) - DeepSeek's configuration and every measurement behind
  it; [docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md) - the HIP port; [docs/MODELS.md](docs/MODELS.md) - each
  model; [docs/ENGINE_PLAN.md](docs/ENGINE_PLAN.md) - one engine for every model; [docs/ENGINE_REVIEW.md](docs/ENGINE_REVIEW.md)
  - the engine's review; [HANDOVER.md](HANDOVER.md) - the state and what is open; [docs/ALMOND_PLAN.md](docs/ALMOND_PLAN.md)
  - the first multi-model work, kept for its measurements.

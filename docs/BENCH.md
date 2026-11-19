# almond-bench

Quality benchmarks for the local models and their quantizations: public benchmarks run through the router's
OpenAI API, graded by code wherever the answer can be checked, and the KL divergence of a quantization against a
reference build of the same model. `./almond-bench` (the files in `tools/bench/`).

## What it measures and how it grades

| Suite | Measures | Tasks (quick / full) | Graded by | Judge model |
|---|---|---|---|---|
| `mmlu-pro` | knowledge, 14 subjects, 10 options | 300 / 12,032 | the letter after "The answer is" | no |
| `gpqa` | expert knowledge (GPQA Diamond) | 198 / 198 | the letter | no |
| `math500` | competition maths (MATH-500) | 200 / 500 | the last `\boxed{}`, normalized LaTeX | no |
| `humaneval+` | code (EvalPlus HumanEval+) | 164 / 164 | the code run against EvalPlus's hidden tests | no |
| `mbpp+` | code (EvalPlus MBPP+) | 150 / 378 | the same | no |
| `ifeval` | instruction following (IFEval) | 200 / 541 | the 25 verifiable rules (`ifeval.py`) | no |
| `bfcl` | tool calling (BFCL v3: simple, multiple, parallel, parallel_multiple, irrelevance) | 200 / 1,240 | structure of the calls: names, required arguments, accepted values (`bfcl.py`) | no |
| `arena-hard` | open answers (Arena-Hard v0.1) | 50 / 500 | win rate against GPT-4-0314's answers, judged in both orders | yes |
| KLD | what a quantization lost | any token stream | KL(reference ‖ quantization) from logit dumps (`kld.py`) | no |

The quick preset is a fixed sample (seed 0, the same proportion of every category as the full set), so two quick
runs see the same questions. A task scores 1 or 0 (arena-hard: 0, 0.25, 0.5, 0.75, 1); a suite's score is the mean,
reported with its standard error. Over n tasks that is about sqrt(p(1-p)/n): ±3.2 points at n = 200, ±1.4 at
n = 1,000. Two quantizations closer than about twice the larger error are not told apart by one run; the engine is
not bit-reproducible run to run either (AGENTS.md), so a difference worth writing down wants the full set or
repeated runs.

The answer the model gives is kept after its reasoning (`<think>...</think>` removed). Every suite has its own
`max_tokens` (8,192 for every suite but bfcl, 4,096; `--max-tokens` overrides them all): a model that reasons past the
limit leaves no answer and the task scores 0. Such a task is marked "TRUNCATED at max_tokens" and counted per
suite ("N truncated" beside the score): many of them mean the limit, not the model, set the score. Measured on
qwen3.8-27b (2026-11-19): 2,048 tokens left IFEval's first prompt (a 300-word summary under three rules) with
reasoning only, and 4,096 cut one MMLU-Pro answer in 14.

Graders checked against the reference answers (2026-11-19): math500 500/500 with the dataset's own answers;
humaneval+ 163/164 and mbpp+ 377/378 with the canonical solutions (the one failure in each is that solution failing
EvalPlus's own tests); bfcl 1,226/1,240 with the ground truth turned into calls (the 14 are nested arguments the
test built wrongly, not the checker). Where they differ from the reference implementations:

- IFEval: sentences are split by a regular expression instead of nltk's punkt, and the response language is told
  by its script (Kannada, Punjabi, Tamil, Korean... have one each) and, within a shared script, by stop-word counts,
  instead of langdetect. A first version used stop words only and failed a correct short Kannada answer
  (qwen3.8-27b-q8, ifeval-1108, 2026-11-19). Both concern a handful of the 541 prompts. Strict (the score) and
  loose (the reference's 8 variants of the response) are both recorded.
- BFCL: the AST check for the Python categories, simplified from gorilla's `ast_checker` (strings compared after
  dropping case, spaces and punctuation; int and float equal). Function names with dots are sent with `_`.
- MATH: string equality after the reference normalization plus fractions and decimals compared as numbers; no
  symbolic equivalence (no sympy), so an equal but differently written expression scores 0.
- Arena-Hard: the v0.1 judge prompt; the score is the mean verdict, not the Bradley-Terry fit of the leaderboard.

The generated code runs in a subprocess of this Python, in a temporary directory, with a 60 s limit and no
environment; it is not a sandbox. Run the code suites only on models you trust with your user account.

## The data

```bash
./almond-bench fetch                  # the suites that are missing; present ones are left alone
./almond-bench fetch --update         # also re-download a suite whose dataset has a newer revision
./almond-bench fetch mmlu-pro ifeval  # only these
./almond-bench fetch --list           # the local files against the manifest
```

The datasets go to `local/bench/data/<suite>.jsonl` (about 27 MB without GPQA), which git ignores: they are
downloaded, not committed. Committing them would put benchmark questions in plain text where future models can
train on them (the reason several carry canary strings), GPQA's authors ask not to republish it, Arena-Hard's
reference answers are GPT-4 output under OpenAI's terms, and every update would stay in the history.

`local/bench/data/manifest.json` records for each suite the Hugging Face dataset, its revision (commit), licence,
task count, download time and the sha256 of the local file; every run records the sha256 of the files it used,
and the Results tab says when a run's data has changed since. Every file is read from the dataset repository at that revision (`resolve/<commit>/...`): a full download of the
seven public suites took 5 s (2026-11-19, against about 10 minutes through Hugging Face's datasets-server, which
pages 100 rows a request and answers 429 when asked too fast). MMLU-Pro, MBPP+ and HumanEval+ are Parquet files,
read with pyarrow (`requirements.txt`, 170 MB in the venv); without pyarrow those three fall back to the
datasets-server, which serves only the latest revision. HumanEval+ also has a `test.jsonl` whose tests compare
floats with `np.testing.assert_allclose` instead of `np.allclose`; the Parquet is used, the same tests the graders
were checked against. `fetch --update` re-downloads a suite only when its revision changed or its file no longer
matches the manifest.

GPQA is gated: accept its terms on huggingface.co/datasets/Idavidrein/gpqa, then
`HF_TOKEN=... ./almond-bench fetch gpqa` (or `huggingface-cli login`, which writes `~/.cache/huggingface/token`).

| Suite | Dataset | Licence |
|---|---|---|
| mmlu-pro | TIGER-Lab/MMLU-Pro | MIT |
| gpqa | Idavidrein/gpqa (diamond) | CC-BY-4.0, gated, not to be republished in plain text |
| math500 | HuggingFaceH4/MATH-500 | MIT (from MATH) |
| humaneval+ | evalplus/humanevalplus | Apache-2.0 |
| mbpp+ | evalplus/mbppplus | Apache-2.0 |
| ifeval | google/IFEval | Apache-2.0 |
| bfcl | gorilla-llm/Berkeley-Function-Calling-Leaderboard (v3) | Apache-2.0 |
| arena-hard | lmarena-ai/arena-hard-auto (v0.1) | Apache-2.0; the GPT-4 answers under OpenAI's terms |

## Running

```bash
./almond-bench                                           # the interface
./almond-bench run --model qwen3.8-27b --preset quick    # without it
./almond-bench run --model glm-5.3-flash --suites ifeval,bfcl --preset full
./almond-bench run --model qwen3.8-27b --suites arena-hard --judge-model qwen3.8-27b-q8
./almond-bench run --resume local/bench/runs/<file>.json # continue a stopped run
```

The model names are the router's (`/v1/models`). The interface starts the router when nothing answers at a local
endpoint, as `./almond` does, and stops it on quit if it started it; `run` expects a running server
(`./almond serve --router`). A model is loaded on its first request, so the first task waits for it.

Each run is written to `local/bench/runs/<time>-<model>-<preset>.json` after every task, so a stopped or crashed
run loses at most the task in flight; `c` in the interface (or `--resume`) continues it with the same tasks.

The interface:

- Run: endpoint, model, judge, preset, tasks per suite, suites (the missing ones greyed out; Data or `d`
  downloads them). The running score per suite with its error band, and every task as it finishes: score, seconds,
  tokens, and the grader's note. "got X, expected Y" is the letter read from the answer and the right one;
  "got None" means no answer was found (often: the reasoning reached `max_tokens`).
- Results: the saved runs; for the selected one the score of every suite with its error band, every task, and
  the model's whole answer with the reason of its grade.
- Compare: every run side by side (the best score of each column underlined, and tok/s), and the KL divergences.
- Delete: `Del` the selected run (Results) or the selected KL result (Compare, its table focused), `D` every run
  and KL result; both ask first. Nothing is deleted: the files go to `local/bench/trash/` with the time in front
  of the name, and moving one back to `local/bench/runs/` (or `kld/`) restores it. Empty the trash by hand.

Time: the quick preset is about 1,260 requests. With answers of 500-2,000 tokens at 20-30 tok/s that is many
hours per model; the full preset is days. Run it overnight, or pick the suites and `--limit`.

## Arena-Hard and the judge

The only suite that needs a model to grade. The judge sees the question, GPT-4-0314's answer and the model's, in
both orders (to cancel the judge's preference for one position), and gives one of five verdicts; the task's score is
the mean, 0.5 meaning as good as GPT-4-0314. The judge should be the strongest model available and lightly
quantized (`qwen3.8-27b-q8` here); a judge of the same family as the model under test favours it, and the scores
are only comparable between runs with the same judge (the run records it).

## KL divergence

The most sensitive measure of what a quantization lost, with no answers and no judge: on the same token stream,
how far the quantization's next-token distribution is from the reference build's (Q8_0 or BF16).

```bash
./almond-bench kld ref.f32 cand.f32 --vocab V --name qwen27-iq3s --ref-name qwen27-q8 --text "long-docs.txt, 4096 tokens"
```

Input: raw little-endian float32 logits, one row of V per position, as the engines' `--dump-logits` writes them;
both files over the same tokens. Output (as `llama-perplexity --kl-divergence`): mean KLD with its standard error,
median, 99th percentile, maximum, and the share of positions where both put the same token first, written to
`local/bench/kld/` and shown in Compare. Checked on synthetic logits only: producing the dumps is per engine for
now, with each engine's own `--dump-logits`.

## Files

| File | |
|---|---|
| `almond-bench` | the command |
| `tools/bench/bench_core.py` | suites, prompts, graders, runner (`run`) |
| `tools/bench/datasets.py` | download, manifest (`fetch`) |
| `tools/bench/ifeval.py` | IFEval's rules |
| `tools/bench/bfcl.py` | BFCL's schemas and call checker |
| `tools/bench/kld.py` | KL divergence (`kld`) |
| `tools/bench/bench_ui.py` | the interface (Textual) |
| `local/bench/data/`, `runs/`, `kld/`, `trash/` | datasets and manifest, runs, KL results, deleted results (not in git) |

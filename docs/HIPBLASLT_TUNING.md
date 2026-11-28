# hipBLASLt tuning files

GLM's and Qwen3.8-Flash-Next's prompt read multiply in hipBLASLt. Its heuristic does not always pick the fastest kernel
for a shape, so a tuning file tells the engine which solution to use for each shape it runs. The engine reads it
from `ALMOND_HIPBLASLT_TUNING` (in the configuration's `env`), for example
`local/qwen-local.json` -> `local/hipblaslt/gfx1100-qwen-hipblaslt-100500-local.txt`.

## When it has to be made again

A solution id only means something for one hipBLASLt version on one GPU architecture. The file's header names both:

```
# solution IDs are scoped to this hipBLASLt version and device architecture
ALMOND_HIPBLASLT_TUNING_V1 gfx1100 100500
```

After a ROCm update with a new hipBLASLt, the engine refuses the old file and writes in its log:

```
prefill gemm: hipBLASLt version mismatch: file=100401 runtime=100500; using hipBLASEx
```

The prompt read then runs on hipBLASEx's own choice. Generate a new file, as below.

## The tool

`tools/glm/hip/tune_hipblaslt.cpp`. `./build.sh` builds it into `build-hip-fused/src/glm/tune_hipblaslt` against the
installed ROCm. For each shape it times hipBLASLt's candidate solutions and keeps the fastest. It only keeps a
solution whose result matches hipBLAS within tolerance: relative L2 error 1e-4, largest absolute error 1e-2, no
non-finite values, no writes into the padding. It writes one line per shape it could tune. A shape with no candidate
inside the tolerance gets no line, and the engine uses the default for it.

```
tune_hipblaslt [--workspace-mib N] [--case dtype,T,N,K,ldy]... [--shape T,N,K]...
               [--shapes-file PATH [--tokens T1,T2,...]] [--tuning-out PATH]
```

- `--case dtype,T,N,K,ldy`: one product. `dtype` is `f16` or `bf16`, `T` the tokens (rows of the activation), `N` the
  output columns, `K` the inner size, `ldy` the output's leading dimension (at least `N`).
- `--workspace-mib`: hipBLASLt's workspace, 32 MiB by default, the same size the engine gives it.
- `--tuning-out`: the file to write. Without it the tool only prints its measurements.

Each line in the file is `dtype N K ldy T solution_id`. Note the order: `T` comes fifth there, but second in `--case`.

## Making a new file from the old one

The shapes do not change with ROCm, so the old file lists exactly what to tune. Run on one card, with nothing else
using it:

```bash
OLD=local/hipblaslt/gfx1100-qwen-hipblaslt-100401-local.txt
NEW=local/hipblaslt/gfx1100-qwen-hipblaslt-100500-local.txt
HIP_VISIBLE_DEVICES=0 build-hip-fused/src/glm/tune_hipblaslt \
  $(awk '!/^#/ && !/^ALMOND/ {printf "--case %s,%s,%s,%s,%s ", $1, $5, $2, $3, $4}' "$OLD") \
  --tuning-out "$NEW"
```

Name the new file after the version in its header, point the configuration's `ALMOND_HIPBLASLT_TUNING` at it, and
check that the engine log no longer says `version mismatch`. Then measure the prompt read with the old and the new
file (a long prompt through the server, alternated). Change the configuration only where the new file measured better.

The Qwen file for ROCm 10.2 (hipBLASLt 100500, 2026-11-28) took about 10 minutes on one 7900 XTX. It kept 37 of the
old file's 43 shapes. The other 6 are bf16 with K = 10240 or N = 10240: T 128, 1024, 2048 and 6144 at N 2560, T 4096
at N 320, and T 8192 at N 4. Every candidate hipBLASLt 100500 offered for them gave non-finite results
(`lt_reject ... finite=false relative_l2=inf`), so those six stay on the default path.

Measured (2026-11-28, `local/qwen-local.json` without vision, a 20,324-token prompt through the server, 16 tokens out,
arms alternated): with the 100401 file (refused, hipBLASEx) 31.2 / 36.1 s, with the 100500 file 23.0 / 25.4 s, the
same answer text. `local/qwen-local.json` now names the 100500 file. GLM's configurations still name
`gfx1100-glm-hipblaslt-100401-local.txt` and so run on hipBLASEx under ROCm 10.2 (not regenerated yet).

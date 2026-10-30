#!/bin/sh
# The GLM engine's pack (the index of a glm5-next GGUF: the small float weights, about 1 GB, the tensor
# index and the tokenizer with the model's chat template) for models/glm53-gsq-rco: written into
# models/glm53-gsq-rco/pack, next to the GGUF the engine reads the experts from.  A few seconds.
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
PY=${ALMOND_PY:-$R/.venv/bin/python}
G=$R/models/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf
[ -s "$G" ] || { echo "glm-pack.sh: $G is missing" >&2; exit 1; }
"$PY" "$R/tools/glm/iq_pack.py" --gguf "$G" --out "$R/models/glm53-gsq-rco/pack" --compat-bf16

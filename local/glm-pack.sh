#!/bin/sh
# The GLM engine's pack (the index of a glm5-next GGUF: the small float weights, about 1 GB, the tensor
# index and the tokenizer with the model's chat template), next to the GSQ-RCO GGUF the engine reads the experts
# from (the 3.5-bit GGUF -> models/glm53-gsq-rco/pack, local/glm-local.json).  A few seconds.
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
PY=${ALMOND_PY:-$R/.venv/bin/python}
BITS=${1:-3.5}
G=$R/models/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-${BITS}bit.gguf
OUT=$R/models/glm53-gsq-rco/pack; [ "$BITS" = 3.5 ] || OUT=$OUT-$BITS
[ -s "$G" ] || { echo "glm-pack.sh: $G is missing" >&2; exit 1; }
"$PY" "$R/tools/glm/iq_pack.py" --gguf "$G" --out "$OUT" --compat-bf16

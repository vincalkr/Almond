#!/bin/sh
# The GLM engine's pack (the index of a glm5-next GGUF: the small float weights, about 1 GB, the tensor
# index and the tokenizer with the model's chat template) for models/glm53-gsq-rco, next to the GGUF the engine
# reads the experts from.  A few seconds.
#   local/glm-pack.sh          the 3.0-bit GGUF -> models/glm53-gsq-rco/pack
#   local/glm-pack.sh 3.5      the 3.5-bit GGUF -> models/glm53-gsq-rco/pack-3.5
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
PY=${ALMOND_PY:-$R/.venv/bin/python}
BITS=${1:-3.0}
G=$R/models/glm53-gsq-rco/GLM-5.3-Flash-GSQ-RCO-${BITS}bit.gguf
OUT=$R/models/glm53-gsq-rco/pack; [ "$BITS" = 3.0 ] || OUT=$OUT-$BITS
[ -s "$G" ] || { echo "glm-pack.sh: $G is missing" >&2; exit 1; }
"$PY" "$R/tools/glm/iq_pack.py" --gguf "$G" --out "$OUT" --compat-bf16

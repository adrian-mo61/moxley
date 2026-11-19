#!/usr/bin/env bash
# Moxley-64M supervised fine-tuning launcher for a single RTX 4090 (24 GB).
#
# Prerequisites:
#   1. data/sft_t2t.jsonl downloaded
#   2. data/tokenizer/ from the pretraining stage
#   3. a pretrained checkpoint, by default out/moxley-64m-4090/final
#
# First run (converts SFT JSONL to binary shards automatically):
#   SETUP=1 bash scripts/train_4090_sft.sh
#
# Later runs (auto-resume if latest.pt exists):
#   bash scripts/train_4090_sft.sh
#
# Force a fresh SFT run:
#   RESUME=0 bash scripts/train_4090_sft.sh
#
# Disable the progress bar:
#   PROGRESS=0 bash scripts/train_4090_sft.sh
#
# If you removed SETUP locally because the cloud image already has PyTorch:
#   bash scripts/train_4090_sft.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

if [[ "${SETUP:-0}" == "1" ]]; then
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "[setup] creating .venv"
    "$PYTHON" -m venv "$ROOT/.venv"
  fi
  PYTHON="$ROOT/.venv/bin/python"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  "$PYTHON" -m pip install -r "$ROOT/requirements.txt"
fi

if ! "$PYTHON" -c 'import numpy, torch' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: PyTorch or NumPy is missing.
Run:
  python3 -m pip install -r requirements.txt
EOF
  exit 1
fi

"$PYTHON" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: PyTorch does not detect CUDA.", file=sys.stderr)
    raise SystemExit(1)

props = torch.cuda.get_device_properties(0)
print(
    f"[gpu] {props.name}, {props.total_memory / 1024**3:.1f} GiB, "
    f"CUDA {torch.version.cuda}, torch {torch.__version__}"
)
PY

SFT_JSONL="${SFT_JSONL:-$ROOT/data/sft_t2t.jsonl}"
DATA_DIR="${DATA_DIR:-$ROOT/data/sft_bin}"
OUT_DIR="${OUT_DIR:-$ROOT/out/moxley-64m-sft}"
INIT_FROM="${INIT_FROM:-$ROOT/out/moxley-64m-4090/final}"
TOKENIZER="${TOKENIZER:-$ROOT/data/tokenizer}"

# ---------------------------------------------------------------------------
# Build SFT shards once. JSONL parsing + tokenization is expensive; doing it
# outside the training loop avoids repeating it every epoch.
# ---------------------------------------------------------------------------
if [[ ! -f "$DATA_DIR/tokens.bin" ]]; then
  if [[ ! -f "$SFT_JSONL" ]]; then
    cat >&2 <<EOF
ERROR: cannot find $SFT_JSONL

Upload it from the local machine, for example:
  scp data/sft_t2t.jsonl USER@HOST:~/moxley/data/
EOF
    exit 1
  fi
  if [[ ! -d "$TOKENIZER" ]]; then
    echo "ERROR: tokenizer not found: $TOKENIZER" >&2
    exit 1
  fi
  if ! "$PYTHON" -c 'import tokenizers' >/dev/null 2>&1; then
    echo "ERROR: tokenizers is required to convert SFT JSONL." >&2
    echo "Run: $PYTHON -m pip install -r requirements.txt" >&2
    exit 1
  fi
  echo "[prepare] building SFT shards from $SFT_JSONL ..."
  "$PYTHON" -u scripts/prepare_sft.py \
    --input "$SFT_JSONL" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$DATA_DIR" \
    --max-len "${MAX_LEN:-1024}" \
    --val-ratio "${VAL_RATIO:-0.005}" \
    --max-samples "${PREP_MAX_SAMPLES:-0}" \
    --overwrite
fi

if [[ ! -e "$INIT_FROM" ]]; then
  cat >&2 <<EOF
ERROR: pretrained checkpoint not found: $INIT_FROM

Run pretraining first, or override:
  INIT_FROM=/path/to/out/moxley-64m-4090/final bash scripts/train_4090_sft.sh
EOF
  exit 1
fi

mkdir -p "$OUT_DIR"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
DTYPE="${DTYPE:-bf16}"
RESUME="${RESUME:-auto}"
PROGRESS="${PROGRESS:-1}"
COMPILE="${COMPILE:-0}"
FROM_SCRATCH="${FROM_SCRATCH:-0}"

if [[ "$DTYPE" == "bf16" ]]; then
  if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)' >/dev/null 2>&1; then
    echo "[warning] bf16 unsupported; falling back to fp16" >&2
    DTYPE="fp16"
  fi
fi

ARGS=(
  --data-dir "$DATA_DIR"
  --out-dir "$OUT_DIR"
  --init-from "$INIT_FROM"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --warmup-steps "$WARMUP_STEPS"
  --num-workers "$NUM_WORKERS"
  --eval-interval "$EVAL_INTERVAL"
  --eval-batches "$EVAL_BATCHES"
  --save-interval "$SAVE_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --device cuda:0
  --dtype "$DTYPE"
)

case "$RESUME" in
  1|true|yes|always) ARGS+=(--resume) ;;
  0|false|no|never) ARGS+=(--fresh) ;;
  auto) ;;
  *) echo "[warning] unknown RESUME=$RESUME; treating it as auto" >&2 ;;
esac

[[ "$PROGRESS" == "0" ]] && ARGS+=(--no-progress)
[[ "$COMPILE" == "1" ]] && ARGS+=(--compile)
[[ "$FROM_SCRATCH" == "1" ]] && ARGS+=(--from-scratch)

cat <<EOF
============================================================
Moxley-64M supervised fine-tuning on a single 4090
  repository  : $ROOT
  sft data    : $DATA_DIR
  init from   : $INIT_FROM
  output      : $OUT_DIR
  batch       : $BATCH_SIZE x $GRAD_ACCUM (variable-length samples)
  epochs      : $EPOCHS (max_steps=$MAX_STEPS)
  precision   : $DTYPE
  resume      : $RESUME
  progress    : $PROGRESS
  log         : $OUT_DIR/train.log
============================================================
EOF

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set +e
"$PYTHON" -u train_sft.py "${ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/train.log"
status="${PIPESTATUS[0]}"
set -e

if (( status != 0 )); then
  cat >&2 <<EOF
ERROR: SFT exited with code $status.

If this is CUDA OOM, retry with the same accumulation budget:
  BATCH_SIZE=4 GRAD_ACCUM=8 bash scripts/train_4090_sft.sh
  BATCH_SIZE=2 GRAD_ACCUM=16 bash scripts/train_4090_sft.sh

To resume after an interruption:
  bash scripts/train_4090_sft.sh
  # or require an existing checkpoint:
  RESUME=1 bash scripts/train_4090_sft.sh
EOF
  exit "$status"
fi

echo "[done] final model: $OUT_DIR/final"
echo "[done] log:         $OUT_DIR/train.log"

#!/usr/bin/env bash
# Moxley-64M pretraining launcher for a single NVIDIA RTX 4090 (24 GB) on Ubuntu.
#
# Quick start:
#   bash scripts/train_4090.sh
#
# First-time environment setup (creates .venv, installs CUDA PyTorch):
#   SETUP=1 bash scripts/train_4090.sh
#
# Continue after an interruption (also automatic if latest.pt exists):
#   RESUME=1 bash scripts/train_4090.sh
# Force a fresh run:
#   RESUME=0 bash scripts/train_4090.sh
# Disable the progress bar:
#   PROGRESS=0 bash scripts/train_4090.sh
#
# If CUDA runs out of memory, halve the batch and double accumulation:
#   BATCH_SIZE=16 GRAD_ACCUM=16 bash scripts/train_4090.sh
#
# More aggressive fallback:
#   BATCH_SIZE=8 GRAD_ACCUM=32 bash scripts/train_4090.sh
#   BATCH_SIZE=4 GRAD_ACCUM=64 bash scripts/train_4090.sh
#
# All paths are resolved from the repository root, so the script can be launched
# from any directory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

# ---------------------------------------------------------------------------
# Optional environment bootstrap
# ---------------------------------------------------------------------------
if [[ "${SETUP:-0}" == "1" ]]; then
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "[setup] creating .venv"
    "$PYTHON" -m venv "$ROOT/.venv"
  fi
  PYTHON="$ROOT/.venv/bin/python"
  "$PYTHON" -m pip install --upgrade pip
  # CUDA 12.1 wheels work on Ada / RTX 4090 with a recent NVIDIA driver.
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  "$PYTHON" -m pip install -r "$ROOT/requirements.txt"
fi

if ! "$PYTHON" -c 'import numpy, torch' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: PyTorch or NumPy is not installed in the selected Python environment.

Run:
  python3 -m pip install -r requirements.txt
or, for a clean virtual environment:
  SETUP=1 bash scripts/train_4090.sh

If PyTorch is missing, install a CUDA build first:
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
EOF
  exit 1
fi

# Verify CUDA and print the exact device before spending time on data checks.
"$PYTHON" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print(
        "ERROR: PyTorch does not detect CUDA. Install a CUDA build of PyTorch "
        "and make sure nvidia-smi works.",
        file=sys.stderr,
    )
    raise SystemExit(1)

props = torch.cuda.get_device_properties(0)
print(
    f"[gpu] {props.name}, {props.total_memory / 1024**3:.1f} GiB, "
    f"CUDA {torch.version.cuda}, torch {torch.__version__}"
)
if props.total_memory < 10 * 1024**3:
    print(
        "[warning] less than 10 GiB VRAM detected; lower BATCH_SIZE if OOM.",
        file=sys.stderr,
    )
PY

DATA_DIR="${DATA_DIR:-$ROOT/data/bin}"
OUT_DIR="${OUT_DIR:-$ROOT/out/moxley-64m-4090}"

# If the shards are missing but the raw corpus + tokenizer are on the machine,
# build them automatically. Otherwise fail with an explicit upload hint.
if [[ ! -f "$DATA_DIR/train.bin" ]]; then
  if [[ -f "$ROOT/data/pretrain_t2t.jsonl" && -d "$ROOT/data/tokenizer" ]]; then
    if ! "$PYTHON" -c 'import tokenizers' >/dev/null 2>&1; then
      echo "ERROR: tokenizers is required to convert JSONL -> .bin." >&2
      echo "Run: $PYTHON -m pip install -r requirements.txt" >&2
      exit 1
    fi
    echo "[prepare] $DATA_DIR/train.bin not found; converting the JSONL corpus..."
    mkdir -p "$DATA_DIR"
    "$PYTHON" -u scripts/prepare_data.py \
      --input "$ROOT/data/pretrain_t2t.jsonl" \
      --tokenizer "$ROOT/data/tokenizer" \
      --output-dir "$DATA_DIR" \
      --val-ratio "${VAL_RATIO:-0.005}" \
      --overwrite
  else
    cat >&2 <<EOF
ERROR: cannot find $DATA_DIR/train.bin

Upload the three prepared shard files from the machine where you converted them:
  data/bin/train.bin
  data/bin/val.bin
  data/bin/meta.json

For example, from Windows PowerShell:
  scp data/bin/train.bin data/bin/val.bin data/bin/meta.json USER@HOST:~/moxley/data/bin/

Or from Linux/macOS:
  rsync -avP data/bin/ USER@HOST:~/moxley/data/bin/
EOF
    exit 1
  fi
fi

TRAIN_BYTES="$(stat -c%s "$DATA_DIR/train.bin")"
if (( TRAIN_BYTES < 100 * 1024 * 1024 )); then
  echo "[warning] train.bin is only $((TRAIN_BYTES / 1024 / 1024)) MiB; is this the full shard?" >&2
fi
if [[ ! -f "$DATA_DIR/val.bin" ]]; then
  echo "[warning] val.bin not found; validation will fall back to train.bin" >&2
fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Defaults tuned for 24 GiB VRAM. Effective batch = 32 * 8 * 512 = 131,072 tokens.
# Override any value with an environment variable.
# ---------------------------------------------------------------------------
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
DTYPE="${DTYPE:-bf16}"
RESUME="${RESUME:-auto}"
PROGRESS="${PROGRESS:-1}"
COMPILE="${COMPILE:-0}"

if [[ "$DTYPE" == "bf16" ]]; then
  if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)' >/dev/null 2>&1; then
    echo "[warning] bf16 is not supported on this GPU; falling back to fp16" >&2
    DTYPE="fp16"
  fi
fi

ARGS=(
  --data-dir "$DATA_DIR"
  --out-dir "$OUT_DIR"
  --seq-len "$SEQ_LEN"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --warmup-steps "$WARMUP_STEPS"
  --num-workers "$NUM_WORKERS"
  --save-interval "$SAVE_INTERVAL"
  --eval-interval "$EVAL_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --device cuda:0
  --dtype "$DTYPE"
)

case "$RESUME" in
  1|true|yes|always)
    ARGS+=(--resume)
    ;;
  0|false|no|never)
    ARGS+=(--fresh)
    ;;
  auto)
    # train_pretrain.py auto-detects out-dir/latest.pt.
    ;;
  *)
    echo "[warning] unknown RESUME=$RESUME; treating it as auto" >&2
    ;;
esac

if [[ "$PROGRESS" == "0" ]]; then
  ARGS+=(--no-progress)
fi

if [[ "$COMPILE" == "1" ]]; then
  ARGS+=(--compile)
fi

TOKENS_PER_STEP=$((BATCH_SIZE * GRAD_ACCUM * SEQ_LEN))

cat <<EOF
============================================================
Moxley-64M pretraining on a single 4090
  repository  : $ROOT
  data        : $DATA_DIR
  train.bin   : $((TRAIN_BYTES / 1024 / 1024)) MiB
  output      : $OUT_DIR
  seq_len     : $SEQ_LEN
  batch       : $BATCH_SIZE x $GRAD_ACCUM x $SEQ_LEN = $TOKENS_PER_STEP tokens/step
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
"$PYTHON" -u train_pretrain.py "${ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/train.log"
status="${PIPESTATUS[0]}"
set -e

if (( status != 0 )); then
  cat >&2 <<EOF
ERROR: training exited with code $status.

If this is CUDA OOM, retry with the same effective batch:
  BATCH_SIZE=16 GRAD_ACCUM=16 bash scripts/train_4090.sh
  BATCH_SIZE=8 GRAD_ACCUM=32 bash scripts/train_4090.sh
  BATCH_SIZE=4 GRAD_ACCUM=64 bash scripts/train_4090.sh

To resume after a stopped job (automatic when latest.pt exists):
  bash scripts/train_4090.sh
  # or require an existing checkpoint:
  RESUME=1 bash scripts/train_4090.sh
EOF
  exit "$status"
fi

echo "[done] final model: $OUT_DIR/final"
echo "[done] log:         $OUT_DIR/train.log"

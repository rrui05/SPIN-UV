#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME:-python310_torch270_cu128}"
  fi
fi

CONFIG="e1_models/segformer_b5_imagenet1k/config.json"
SMOKE_DIR="${SMOKE_DIR:-e1_models/segformer_b5_imagenet1k/runs/smoke}"
FULL_DIR="${FULL_DIR:-e1_models/segformer_b5_imagenet1k/runs/full_8k}"
LOG_DIR="${LOG_DIR:-e1_models/segformer_b5_imagenet1k/logs}"
DATA_ROOT="${DATA_ROOT:-data/processed/spinuv_semantic}"
mkdir -p "${LOG_DIR}"

if ! python - <<'PY' >/dev/null 2>&1
import torch
import transformers
import matplotlib
import pandas
from transformers import SegformerForSemanticSegmentation
PY
then
  bash setup_autodl_env.sh 2>&1 | tee "${LOG_DIR}/setup_stdout.log"
fi

RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_FULL="${RUN_FULL:-1}"
SMOKE_ITERS="${SMOKE_ITERS:-20}"
MAX_ITERS="${MAX_ITERS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAMPLE_COUNT="${SAMPLE_COUNT:-8}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RESUME_MODE="${RESUME_MODE:-auto}"

echo "model=segformer_b5_imagenet1k"
echo "model_id=nvidia/mit-b5"
echo "data_root=${DATA_ROOT}"
echo "full_dir=${FULL_DIR}"
echo "autodl_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "batch_size=${BATCH_SIZE}"
echo "grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "train_crop_exposures=$((MAX_ITERS * BATCH_SIZE * GRAD_ACCUM_STEPS))"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  python scripts/train_spinuv_smp.py \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --run-dir "${SMOKE_DIR}" \
    --max-iters "${SMOKE_ITERS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --val-interval "${SMOKE_ITERS}" \
    --save-interval "${SMOKE_ITERS}" \
    --log-interval 5 \
    --sample-count 4 \
    2>&1 | tee "${LOG_DIR}/smoke_stdout.log"
fi

if [[ "${RUN_FULL}" == "1" ]]; then
  if [[ "${FORCE_TRAIN}" != "1" && -f "${FULL_DIR}/checkpoints/best_miou.pt" ]]; then
    echo "existing_best_checkpoint=${FULL_DIR}/checkpoints/best_miou.pt"
    echo "skip_training=1"
  else
    train_args=()
    if [[ "${RESUME_MODE}" == "auto" && -f "${FULL_DIR}/checkpoints/latest.pt" ]]; then
      train_args+=(--resume "${FULL_DIR}/checkpoints/latest.pt")
    fi
    python scripts/train_spinuv_smp.py \
      --config "${CONFIG}" \
      --data-root "${DATA_ROOT}" \
      --run-dir "${FULL_DIR}" \
      --max-iters "${MAX_ITERS}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
      --val-interval "${VAL_INTERVAL}" \
      --save-interval "${SAVE_INTERVAL}" \
      --log-interval "${LOG_INTERVAL}" \
      --sample-count "${SAMPLE_COUNT}" \
      "${train_args[@]}" \
      2>&1 | tee "${LOG_DIR}/full_stdout.log"
  fi
fi

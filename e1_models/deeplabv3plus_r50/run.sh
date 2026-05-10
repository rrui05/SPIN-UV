#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

CONFIG="e1_models/deeplabv3plus_r50/config.json"
SMOKE_DIR="e1_models/deeplabv3plus_r50/runs/smoke"
FULL_DIR="e1_models/deeplabv3plus_r50/runs/full_8k"
LOG_DIR="e1_models/deeplabv3plus_r50/logs"
mkdir -p "${LOG_DIR}"

if ! python - <<'PY' >/dev/null 2>&1
import segmentation_models_pytorch
import transformers
import matplotlib
import pandas
PY
then
  bash setup_autodl_env.sh 2>&1 | tee "${LOG_DIR}/setup_stdout.log"
fi

RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"
SMOKE_ITERS="${SMOKE_ITERS:-20}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  python scripts/train_spinuv_smp.py \
    --config "${CONFIG}" \
    --run-dir "${SMOKE_DIR}" \
    --max-iters "${SMOKE_ITERS}" \
    --val-interval "${SMOKE_ITERS}" \
    --save-interval "${SMOKE_ITERS}" \
    --log-interval 5 \
    --sample-count 4 \
    2>&1 | tee "${LOG_DIR}/smoke_stdout.log"
fi

if [[ "${RUN_FULL}" == "1" ]]; then
  python scripts/train_spinuv_smp.py \
    --config "${CONFIG}" \
    --run-dir "${FULL_DIR}" \
    2>&1 | tee "${LOG_DIR}/full_stdout.log"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

CONFIG="${CONFIG:-e1_rgbd_baselines/rgbd_dual_encoder_unet/config.json}"
RUN_DIR="${RUN_DIR:-e1_rgbd_baselines/rgbd_dual_encoder_unet/runs/rgbd_16k}"
LOG_DIR="${LOG_DIR:-e1_rgbd_baselines/rgbd_dual_encoder_unet/logs}"

mkdir -p "${LOG_DIR}"

latest_ckpt="${RUN_DIR}/checkpoints/latest.pt"
train_args=()
if [[ "${FORCE_TRAIN:-0}" != "1" && -f "${latest_ckpt}" ]]; then
  train_args+=(--resume "${latest_ckpt}")
fi

if [[ "${VERIFY_DEPTH_ONLY:-0}" == "1" ]]; then
  python scripts/train_spinuv_rgbd.py \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT:-data/processed/spinuv_semantic}" \
    --raw-dataset-root "${RAW_DATASET_ROOT:-data/raw/dataset1}" \
    --dry-run-depth-check "${DEPTH_CHECK_COUNT:-8}"
  exit 0
fi

python scripts/train_spinuv_rgbd.py \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT:-data/processed/spinuv_semantic}" \
  --raw-dataset-root "${RAW_DATASET_ROOT:-data/raw/dataset1}" \
  --run-dir "${RUN_DIR}" \
  --max-iters "${MAX_ITERS:-16000}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
  --crop-height "${CROP_HEIGHT:-512}" \
  --crop-width "${CROP_WIDTH:-1024}" \
  --base-lr "${BASE_LR:-0.001}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --val-interval "${VAL_INTERVAL:-1000}" \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --sample-count "${SAMPLE_COUNT:-8}" \
  "${train_args[@]}" \
  2>&1 | tee "${LOG_DIR}/full_stdout.log"

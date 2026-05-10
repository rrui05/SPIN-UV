#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

resolve_raw_root() {
  if [ -n "${RAW_DATASET_ROOT:-}" ] && [ -d "${RAW_DATASET_ROOT}/dataset_sequences" ]; then
    echo "${RAW_DATASET_ROOT}"
    return
  fi
  for candidate in \
    data/raw/dataset1 \
    dataset1 \
    ../dataset1 \
    data/dataset1; do
    if [ -d "${candidate}/dataset_sequences" ]; then
      echo "${candidate}"
      return
    fi
  done
  echo "${RAW_DATASET_ROOT:-data/raw/dataset1}"
}

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-rgbd_16k}"
DATA_ROOT="${DATA_ROOT:-data/processed/spinuv_semantic}"
RAW_DATASET_ROOT="$(resolve_raw_root)"
MODEL_LIST="${MODEL_LIST:-rgbd_early_fusion_unet rgbd_dual_encoder_unet rgbd_se_fusion_unet}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-experiment_01_rgbd_baselines}"
MAX_ITERS="${MAX_ITERS:-16000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
CROP_HEIGHT="${CROP_HEIGHT:-512}"
CROP_WIDTH="${CROP_WIDTH:-1024}"
BASE_LR="${BASE_LR:-0.001}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAMPLE_COUNT="${SAMPLE_COUNT:-8}"

mkdir -p logs tables figures e1_rgbd_baselines/logs e1_rgbd_baselines/tables e1_rgbd_baselines/figures

cat > "e1_rgbd_baselines/logs/autodl_e1_rgbd_plan_${RUN_TAG}.txt" <<EOF
SPIN-UV RGB-D baseline run
run_tag=${RUN_TAG}
conda_env=${CONDA_DEFAULT_ENV:-unknown}
data_root=${DATA_ROOT}
raw_dataset_root=${RAW_DATASET_ROOT}
model_list=${MODEL_LIST}
run_name=${RUN_NAME}
output_prefix=${OUTPUT_PREFIX}
max_iters=${MAX_ITERS}
batch_size=${BATCH_SIZE}
grad_accum_steps=${GRAD_ACCUM_STEPS}
crop_size=${CROP_HEIGHT}x${CROP_WIDTH}
optimizer=${OPTIMIZER}
base_lr=${BASE_LR}
weight_decay=${WEIGHT_DECAY}
train_crop_exposures=$((MAX_ITERS * BATCH_SIZE * GRAD_ACCUM_STEPS))
EOF
cat "e1_rgbd_baselines/logs/autodl_e1_rgbd_plan_${RUN_TAG}.txt"

if ! python - <<'PY' >/dev/null 2>&1
import matplotlib
import numpy
import PIL
import torch
import tqdm
PY
then
  python -m pip install --quiet pillow tqdm matplotlib
fi
python - <<'PY'
import torch
print("rgbd_env_ok torch=" + torch.__version__)
PY

export DATA_ROOT RAW_DATASET_ROOT MAX_ITERS BATCH_SIZE GRAD_ACCUM_STEPS CROP_HEIGHT CROP_WIDTH
export BASE_LR OPTIMIZER WEIGHT_DECAY VAL_INTERVAL SAVE_INTERVAL LOG_INTERVAL SAMPLE_COUNT

if [[ "${VERIFY_DEPTH_ONLY:-0}" == "1" ]]; then
  first_model="${MODEL_LIST%% *}"
  CONFIG="e1_rgbd_baselines/${first_model}/config.json" \
  RUN_DIR="e1_rgbd_baselines/${first_model}/runs/${RUN_NAME}" \
  LOG_DIR="e1_rgbd_baselines/${first_model}/logs" \
    bash "e1_rgbd_baselines/${first_model}/run.sh" \
    2>&1 | tee "e1_rgbd_baselines/logs/autodl_e1_rgbd_depth_check_${RUN_TAG}.log"
  exit 0
fi

for model_slug in ${MODEL_LIST}; do
  CONFIG="e1_rgbd_baselines/${model_slug}/config.json" \
  RUN_DIR="e1_rgbd_baselines/${model_slug}/runs/${RUN_NAME}" \
  LOG_DIR="e1_rgbd_baselines/${model_slug}/logs" \
    bash "e1_rgbd_baselines/${model_slug}/run.sh" \
    2>&1 | tee "e1_rgbd_baselines/logs/autodl_e1_rgbd_${model_slug}_${RUN_TAG}.log"
done

python scripts/experiment_01_collect_model_results.py \
  --models-root e1_rgbd_baselines \
  --model-list "${MODEL_LIST}" \
  --run-name "${RUN_NAME}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  2>&1 | tee "e1_rgbd_baselines/logs/autodl_e1_rgbd_collect_${RUN_TAG}.log"

echo "rgbd_run_tag=${RUN_TAG}"
echo "rgbd_summary=tables/${OUTPUT_PREFIX}_summary.csv"
echo "rgbd_per_class=tables/${OUTPUT_PREFIX}_per_class_iou.csv"

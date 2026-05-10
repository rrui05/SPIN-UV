#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

mkdir -p logs tables figures

MODEL_LIST="${MODEL_LIST:-deeplabv3plus_r50 segformer_b2 upernet_convnext_tiny}"
TRAIN_MODEL_LIST="${TRAIN_MODEL_LIST:-${MODEL_LIST}}"
COLLECT_MODEL_LIST="${COLLECT_MODEL_LIST:-${MODEL_LIST}}"
RUN_NAME="${RUN_NAME:-controlled_16k_crops}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-experiment_01_model_controlled_16k_crops}"
MAX_ITERS="${MAX_ITERS:-8000}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-2}"
CROP_HEIGHT="${CROP_HEIGHT:-512}"
CROP_WIDTH="${CROP_WIDTH:-1024}"
BASE_LR="${BASE_LR:-0.00006}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
POLY_POWER="${POLY_POWER:-0.9}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SEED="${SEED:-42}"
RESUME_MODE="${RESUME_MODE:-auto}"
RUN_SMOKE="${RUN_SMOKE:-0}"
SMOKE_ITERS="${SMOKE_ITERS:-20}"
SAMPLE_COUNT="${SAMPLE_COUNT:-8}"
PARALLEL_MODE="${PARALLEL_MODE:-0}"

PLAN_PATH="logs/autodl_e1_controlled_plan_${RUN_NAME}.txt"
cat > "${PLAN_PATH}" <<EOF
Experiment 01 controlled model run plan
model_list=${MODEL_LIST}
train_model_list=${TRAIN_MODEL_LIST}
collect_model_list=${COLLECT_MODEL_LIST}
run_name=${RUN_NAME}
output_prefix=${OUTPUT_PREFIX}
max_iters=${MAX_ITERS}
effective_batch_size=${EFFECTIVE_BATCH_SIZE}
target_train_crops=$((MAX_ITERS * EFFECTIVE_BATCH_SIZE))
crop_size=${CROP_HEIGHT}x${CROP_WIDTH}
optimizer=${OPTIMIZER}
base_lr=${BASE_LR}
weight_decay=${WEIGHT_DECAY}
poly_power=${POLY_POWER}
val_interval=${VAL_INTERVAL}
save_interval=${SAVE_INTERVAL}
log_interval=${LOG_INTERVAL}
seed=${SEED}
parallel_mode=${PARALLEL_MODE}
parallel_policy=launch_all_models_then_collect_after_all_finish
train_split=full_spinuv_train
val_split=full_spinuv_val
test_split=full_spinuv_test
metric_scope=analysis_classes
control_policy=same_optimizer_updates_same_effective_batch_same_train_crop_count
EOF
cat "${PLAN_PATH}"

if ! python - <<'PY' >/dev/null 2>&1
import transformers
import segmentation_models_pytorch
import matplotlib
import pandas
PY
then
  bash setup_autodl_env.sh 2>&1 | tee "logs/autodl_e1_controlled_setup_${RUN_NAME}.log"
fi

python scripts/autodl_verify_env.py 2>&1 | tee "logs/autodl_e1_controlled_verify_${RUN_NAME}.log"

run_one_model() {
  local model_slug="$1"
  config_path="e1_models/${model_slug}/config.json"
  if [[ ! -f "${config_path}" ]]; then
    echo "missing_config=${config_path}"
    exit 1
  fi

  micro_batch="${EFFECTIVE_BATCH_SIZE}"
  grad_accum_steps="1"
  if [[ "${model_slug}" == "segformer_b5_imagenet1k" ]]; then
    micro_batch="1"
    grad_accum_steps="${EFFECTIVE_BATCH_SIZE}"
  fi

  log_dir="e1_models/${model_slug}/logs/${RUN_NAME}"
  smoke_dir="e1_models/${model_slug}/runs/${RUN_NAME}_smoke"
  full_dir="e1_models/${model_slug}/runs/${RUN_NAME}"
  latest_ckpt="${full_dir}/checkpoints/latest.pt"
  mkdir -p "${log_dir}"

  echo "running_model=${model_slug}"
  echo "config_path=${config_path}"
  echo "full_dir=${full_dir}"
  echo "micro_batch=${micro_batch}"
  echo "grad_accum_steps=${grad_accum_steps}"

  if [[ "${RUN_SMOKE}" == "1" ]]; then
    python scripts/train_spinuv_smp.py \
      --config "${config_path}" \
      --run-dir "${smoke_dir}" \
      --max-iters "${SMOKE_ITERS}" \
      --batch-size "${micro_batch}" \
      --grad-accum-steps "${grad_accum_steps}" \
      --crop-height "${CROP_HEIGHT}" \
      --crop-width "${CROP_WIDTH}" \
      --optimizer "${OPTIMIZER}" \
      --base-lr "${BASE_LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --poly-power "${POLY_POWER}" \
      --val-interval "${SMOKE_ITERS}" \
      --save-interval "${SMOKE_ITERS}" \
      --log-interval 5 \
      --seed "${SEED}" \
      --sample-count 4 \
      2>&1 | tee "${log_dir}/smoke_stdout.log"
  fi

  train_args=()
  if [[ "${RESUME_MODE}" == "auto" && -f "${latest_ckpt}" ]]; then
    train_args+=(--resume "${latest_ckpt}")
  fi

  python scripts/train_spinuv_smp.py \
    --config "${config_path}" \
    --run-dir "${full_dir}" \
    --max-iters "${MAX_ITERS}" \
    --batch-size "${micro_batch}" \
    --grad-accum-steps "${grad_accum_steps}" \
    --crop-height "${CROP_HEIGHT}" \
    --crop-width "${CROP_WIDTH}" \
    --optimizer "${OPTIMIZER}" \
    --base-lr "${BASE_LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --poly-power "${POLY_POWER}" \
    --val-interval "${VAL_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --log-interval "${LOG_INTERVAL}" \
    --seed "${SEED}" \
    --sample-count "${SAMPLE_COUNT}" \
    "${train_args[@]}" \
    2>&1 | tee "${log_dir}/full_stdout.log"
}

if [[ "${PARALLEL_MODE}" == "1" ]]; then
  pids=()
  slugs=()
  for model_slug in ${TRAIN_MODEL_LIST}; do
    echo "launching_model=${model_slug}"
    ( run_one_model "${model_slug}" ) &
    pids+=("$!")
    slugs+=("${model_slug}")
  done

  failed=0
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    slug="${slugs[$idx]}"
    if wait "${pid}"; then
      echo "model_complete=${slug}"
    else
      status="$?"
      echo "model_failed=${slug} exit_status=${status}"
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "parallel_training_failed=1"
    exit 1
  fi
else
  for model_slug in ${TRAIN_MODEL_LIST}; do
    run_one_model "${model_slug}"
  done
fi

python scripts/experiment_01_collect_model_results.py \
  --models-root e1_models \
  --model-list "${COLLECT_MODEL_LIST}" \
  --run-name "${RUN_NAME}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  2>&1 | tee "logs/autodl_e1_controlled_collect_${RUN_NAME}.log"

echo "experiment_01_controlled_summary=tables/${OUTPUT_PREFIX}_summary.csv"
echo "experiment_01_controlled_bar=figures/${OUTPUT_PREFIX}_test_miou_bar.png"

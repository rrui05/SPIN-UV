#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-data/processed/spinuv_semantic}"
ANALYSIS_CLASSES="${ANALYSIS_CLASSES:-building,road,car,wall,sky,sidewalk,vegetation,pole,person}"
MODEL_CONFIGS="${MODEL_CONFIGS:-configs/segformer_b5_cityscapes_source_only.json configs/segformer_b5_ade20k_source_only.json configs/segformer_b5_idd_source_only.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-experiment_02_imagenet1k_b5}"
RUN_EXTERNAL="${RUN_EXTERNAL:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
MAKE_PANELS="${MAKE_PANELS:-0}"
MAKE_PUTINPAPER="${MAKE_PUTINPAPER:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_ITERS="${MAX_ITERS:-8000}"

mkdir -p logs tables figures e2_domain_gap/logs e2_domain_gap/tables e2_domain_gap/figures

cat > "e2_domain_gap/logs/autodl_e2_imagenet1k_b5_plan_${RUN_TAG}.txt" <<EOF
Experiment 02 strict SegFormer-B5 ImageNet-1K-init comparison
run_tag=${RUN_TAG}
conda_env=${CONDA_DEFAULT_ENV:-unknown}
data_root=${DATA_ROOT}
spinuv_model=segformer_b5_imagenet1k
spinuv_init=nvidia/mit-b5
external_configs=${MODEL_CONFIGS}
analysis_classes=${ANALYSIS_CLASSES}
output_prefix=${OUTPUT_PREFIX}
run_external=${RUN_EXTERNAL}
run_train=${RUN_TRAIN}
make_panels=${MAKE_PANELS}
make_putinpaper=${MAKE_PUTINPAPER}
batch_size=${BATCH_SIZE}
grad_accum_steps=${GRAD_ACCUM_STEPS}
max_iters=${MAX_ITERS}
train_crop_exposures=$((MAX_ITERS * BATCH_SIZE * GRAD_ACCUM_STEPS))
EOF
cat "e2_domain_gap/logs/autodl_e2_imagenet1k_b5_plan_${RUN_TAG}.txt"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  DATA_ROOT="${DATA_ROOT}" RUN_SMOKE=0 RUN_FULL=1 \
    BATCH_SIZE="${BATCH_SIZE}" GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" MAX_ITERS="${MAX_ITERS}" \
    bash e1_models/segformer_b5_imagenet1k/run.sh \
    2>&1 | tee "e2_domain_gap/logs/autodl_e2_train_spinuv_segformer_b5_imagenet1k_${RUN_TAG}.log"
fi

cd e2_domain_gap

bash setup_autodl_e2_env.sh 2>&1 | tee "logs/autodl_e2_setup_${RUN_TAG}.log"
python scripts/verify_e2_env.py 2>&1 | tee "logs/autodl_e2_verify_${RUN_TAG}.log"
python scripts/write_common_taxonomy.py 2>&1 | tee "logs/autodl_e2_taxonomy_${RUN_TAG}.log"
python scripts/write_external_checkpoint_registry.py 2>&1 | tee "logs/autodl_e2_checkpoint_registry_${RUN_TAG}.log"

MAX_IMAGES_ARG=()
if [ -n "${MAX_IMAGES:-}" ]; then
  MAX_IMAGES_ARG=(--max-images "${MAX_IMAGES}")
fi

if [[ "${RUN_EXTERNAL}" == "1" ]]; then
  for config_path in ${MODEL_CONFIGS}; do
    echo "running_source_config=${config_path}"
    python scripts/evaluate_hf_source_model.py \
      --config "${config_path}" \
      --run-tag "${RUN_TAG}" \
      "${MAX_IMAGES_ARG[@]}" \
      2>&1 | tee "logs/autodl_e2_eval_$(basename "${config_path}" .json)_${RUN_TAG}.log"
  done
fi

python scripts/generate_e1_predictions.py \
  --run-tag "${RUN_TAG}" \
  --analysis-classes "${ANALYSIS_CLASSES}" \
  --model-list "segformer_b5_imagenet1k" \
  "${MAX_IMAGES_ARG[@]}" \
  2>&1 | tee "logs/autodl_e2_e1_predictions_segformer_b5_imagenet1k_${RUN_TAG}.log"

python scripts/collect_e2_results.py \
  --run-tag "${RUN_TAG}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  2>&1 | tee "logs/autodl_e2_collect_${OUTPUT_PREFIX}_${RUN_TAG}.log"

if [[ "${MAKE_PUTINPAPER}" == "1" ]]; then
  python scripts/make_putinpaper_with_indomain.py \
    --indomain-slug "e1_segformer_b5_imagenet1k" \
    --indomain-run-tag "${RUN_TAG}" \
    --source-run-tag "${RUN_TAG}" \
    2>&1 | tee "logs/autodl_e2_putinpaper_${OUTPUT_PREFIX}_${RUN_TAG}.log"
fi

if [[ "${MAKE_PANELS}" == "1" ]]; then
  python scripts/make_comparison_panels.py \
    --run-tag "${RUN_TAG}" \
    --indomain-slug "e1_segformer_b5_imagenet1k" \
    --analysis-classes "${ANALYSIS_CLASSES}" \
    2>&1 | tee "logs/autodl_e2_comparison_panels_${OUTPUT_PREFIX}_${RUN_TAG}.log"
fi

echo "strict_e2_run_tag=${RUN_TAG}"
echo "strict_e2_transfer_matrix=e2_domain_gap/tables/${OUTPUT_PREFIX}_transfer_matrix.csv"
echo "strict_e2_putinpaper=e2_domain_gap/figures/experiment_02_comparison_put_in_paper.png"

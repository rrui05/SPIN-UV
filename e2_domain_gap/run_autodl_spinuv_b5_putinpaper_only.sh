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
SOURCE_RUN_TAG="${SOURCE_RUN_TAG:-}"
E1_RUN_DIR="${E1_RUN_DIR:-}"
DATA_ROOT="${DATA_ROOT:-data/processed/spinuv_semantic}"
ANALYSIS_CLASSES="${ANALYSIS_CLASSES:-building,road,car,wall,sky,sidewalk,vegetation,pole,person}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_ITERS="${MAX_ITERS:-8000}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RESUME_MODE="${RESUME_MODE:-auto}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_PREDICT="${RUN_PREDICT:-1}"
MAKE_FIGURE="${MAKE_FIGURE:-1}"
FIGURE_PATH="${FIGURE_PATH:-figures/experiment_02_comparison_put_in_paper_imagenet1k.png}"
BASE_FIGURE="${BASE_FIGURE:-}"

mkdir -p e2_domain_gap/logs e2_domain_gap/tables e2_domain_gap/figures

cat > "e2_domain_gap/logs/autodl_spinuv_b5_putinpaper_only_plan_${RUN_TAG}.txt" <<EOF
SPIN-UV ImageNet-1K-init SegFormer-B5 fine-tune and put-in-paper figure only
run_tag=${RUN_TAG}
source_run_tag=${SOURCE_RUN_TAG:-latest}
e1_run_dir=${E1_RUN_DIR:-auto}
conda_env=${CONDA_DEFAULT_ENV:-unknown}
data_root=${DATA_ROOT}
analysis_classes=${ANALYSIS_CLASSES}
batch_size=${BATCH_SIZE}
grad_accum_steps=${GRAD_ACCUM_STEPS}
max_iters=${MAX_ITERS}
force_train=${FORCE_TRAIN}
resume_mode=${RESUME_MODE}
run_train=${RUN_TRAIN}
run_predict=${RUN_PREDICT}
make_figure=${MAKE_FIGURE}
figure_path=e2_domain_gap/${FIGURE_PATH}
base_figure=${BASE_FIGURE:-auto}
train_crop_exposures=$((MAX_ITERS * BATCH_SIZE * GRAD_ACCUM_STEPS))
EOF
cat "e2_domain_gap/logs/autodl_spinuv_b5_putinpaper_only_plan_${RUN_TAG}.txt"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  DATA_ROOT="${DATA_ROOT}" RUN_SMOKE=0 RUN_FULL=1 \
    BATCH_SIZE="${BATCH_SIZE}" GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" MAX_ITERS="${MAX_ITERS}" \
    FORCE_TRAIN="${FORCE_TRAIN}" RESUME_MODE="${RESUME_MODE}" \
    bash e1_models/segformer_b5_imagenet1k/run.sh \
    2>&1 | tee "e2_domain_gap/logs/autodl_train_spinuv_segformer_b5_imagenet1k_only_${RUN_TAG}.log"
fi

cd e2_domain_gap

MAX_IMAGES_ARG=()
if [ -n "${MAX_IMAGES:-}" ]; then
  MAX_IMAGES_ARG=(--max-images "${MAX_IMAGES}")
fi

if [[ "${RUN_PREDICT}" == "1" ]]; then
  E1_RUN_DIR_ARG=()
  if [ -n "${E1_RUN_DIR}" ]; then
    E1_RUN_DIR_ARG=(--e1-run-dir "${E1_RUN_DIR}")
  fi

  python scripts/generate_e1_predictions.py \
    --run-tag "${RUN_TAG}" \
    --analysis-classes "${ANALYSIS_CLASSES}" \
    --model-list "segformer_b5_imagenet1k" \
    "${E1_RUN_DIR_ARG[@]}" \
    "${MAX_IMAGES_ARG[@]}" \
    2>&1 | tee "logs/autodl_e2_e1_predictions_segformer_b5_imagenet1k_only_${RUN_TAG}.log"
fi

if [[ "${MAKE_FIGURE}" == "1" ]]; then
  SOURCE_RUN_TAG_ARG=()
  if [ -n "${SOURCE_RUN_TAG}" ]; then
    SOURCE_RUN_TAG_ARG=(--source-run-tag "${SOURCE_RUN_TAG}")
  fi
  BASE_FIGURE_ARG=()
  if [ -n "${BASE_FIGURE}" ]; then
    BASE_FIGURE_ARG=(--base-figure "${BASE_FIGURE}")
  fi

  python scripts/make_putinpaper_with_indomain.py \
    --indomain-slug "e1_segformer_b5_imagenet1k" \
    --indomain-run-tag "${RUN_TAG}" \
    "${SOURCE_RUN_TAG_ARG[@]}" \
    "${BASE_FIGURE_ARG[@]}" \
    --output "${FIGURE_PATH}" \
    --paper-output "${FIGURE_PATH}" \
    --selection-output "tables/experiment_02_comparison_put_in_paper_imagenet1k_selection.csv" \
    --status-output "logs/experiment_02_comparison_put_in_paper_imagenet1k_status.json" \
    2>&1 | tee "logs/autodl_e2_putinpaper_imagenet1k_b5_only_${RUN_TAG}.log"
fi

echo "spinuv_b5_run_tag=${RUN_TAG}"
echo "spinuv_b5_e2_run=e2_domain_gap/runs/e1_segformer_b5_imagenet1k/${RUN_TAG}"
echo "spinuv_b5_putinpaper=e2_domain_gap/${FIGURE_PATH}"
echo "spinuv_b5_putinpaper_pdf=e2_domain_gap/${FIGURE_PATH%.*}.pdf"

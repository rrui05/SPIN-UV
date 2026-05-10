#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

python - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Expected Python 3.10 in the AutoDL image, got {platform.python_version()}."
    )

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch is not importable in the active environment: {exc}")

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

python -m pip install --upgrade pip
python -m pip install \
  "segmentation-models-pytorch==0.5.0" \
  "pandas>=2.0" \
  "matplotlib>=3.7" \
  "seaborn>=0.13" \
  "tqdm>=4.66" \
  "pillow>=10.0" \
  "transformers>=4.46,<5" \
  "safetensors>=0.4" \
  "huggingface_hub>=0.24"

python -m pip check || true
python scripts/autodl_verify_env.py

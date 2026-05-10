#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "transformers>=4.46,<5" \
  "huggingface_hub>=0.24" \
  "safetensors>=0.4" \
  "pandas>=2.0" \
  "matplotlib>=3.8" \
  "seaborn>=0.13" \
  "tqdm>=4.66" \
  "pillow>=10.0"

python -m pip check

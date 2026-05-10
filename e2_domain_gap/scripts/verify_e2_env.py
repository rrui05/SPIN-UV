#!/usr/bin/env python
"""Write an Experiment 02 environment and dataset report."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


E2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = E2_ROOT.parent


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def split_counts(data_root: Path) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {"images": {}, "masks": {}}
    for split in ["train", "val", "test"]:
        output["images"][split] = len(list((data_root / "images" / split).glob("*.png")))
        output["masks"][split] = len(list((data_root / "masks" / split).glob("*.png")))
    return output


def read_class_count(data_root: Path) -> int:
    path = data_root / "meta" / "classes.json"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return len(payload.get("classes", []))


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main() -> None:
    data_root = EXPERIMENT_ROOT / "data" / "processed" / "spinuv_semantic"
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        torch = None
        torch_error = str(exc)
    else:
        torch_error = ""

    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "not_imported") if torch is not None else "not_imported",
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", "") if torch is not None else "",
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
        "cuda_device_count": int(torch.cuda.device_count()) if torch is not None else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else "",
        "torch_import_error": torch_error,
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "matplotlib": package_version("matplotlib"),
        "seaborn": package_version("seaborn"),
        "tqdm": package_version("tqdm"),
        "pillow": package_version("pillow"),
        "transformers": package_version("transformers"),
        "safetensors": package_version("safetensors"),
        "huggingface_hub": package_version("huggingface-hub"),
        "dataset_root": data_root,
        "split_counts": split_counts(data_root) if data_root.exists() else {},
        "class_count": read_class_count(data_root) if data_root.exists() else 0,
    }
    output_path = E2_ROOT / "logs" / "experiment_02_environment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(clean(payload), f, indent=2)
        f.write("\n")
    print(json.dumps(clean(payload), indent=2))


if __name__ == "__main__":
    main()

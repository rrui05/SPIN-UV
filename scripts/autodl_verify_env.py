#!/usr/bin/env python
"""Verify the AutoDL runtime and SPIN-UV dataset contract."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "spinuv_semantic"
LOG_PATH = ROOT / "logs" / "autodl_env_verification.json"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def split_counts() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for family in ("images", "masks"):
        out[family] = {}
        for split in ("train", "val", "test"):
            split_dir = DATA_ROOT / family / split
            out[family][split] = len(list(split_dir.glob("*.png")))
    return out


def main() -> None:
    import numpy as np
    import torch
    import torchvision
    import segmentation_models_pytorch as smp

    classes_path = DATA_ROOT / "meta" / "classes.json"
    with classes_path.open("r", encoding="utf-8") as f:
        class_payload = json.load(f)
    class_count = len(class_payload["classes"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=class_count,
        activation=None,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 128, 256, device=device)
        y = model(x)
    if tuple(y.shape) != (1, class_count, 128, 256):
        raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")

    counts = split_counts()
    for split in ("train", "val", "test"):
        if counts["images"][split] != counts["masks"][split]:
            raise RuntimeError(f"Image/mask count mismatch for split={split}: {counts}")

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "segmentation_models_pytorch": package_version("segmentation-models-pytorch"),
        "pandas": package_version("pandas"),
        "matplotlib": package_version("matplotlib"),
        "seaborn": package_version("seaborn"),
        "tqdm": package_version("tqdm"),
        "transformers": package_version("transformers"),
        "safetensors": package_version("safetensors"),
        "huggingface_hub": package_version("huggingface-hub"),
        "dataset_root": str(DATA_ROOT.as_posix()),
        "split_counts": counts,
        "class_count": class_count,
        "model_name": "DeepLabV3Plus",
        "encoder_name": "resnet50",
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "forward_check_shape": list(y.shape),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

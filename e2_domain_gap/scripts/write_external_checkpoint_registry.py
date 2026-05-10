#!/usr/bin/env python
"""Write the public source checkpoint registry for Experiment 02."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


E2_ROOT = Path(__file__).resolve().parents[1]


ROWS: list[dict[str, Any]] = [
    {
        "source_dataset": "Cityscapes",
        "model_name": "SegFormer-B5 Cityscapes",
        "architecture_key": "segformer_b5",
        "framework": "hf_auto",
        "checkpoint_url": "https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "config_path": "configs/segformer_b5_cityscapes_source_only.json",
        "current_e2_status": "default_runnable",
        "current_runner_compatible": "yes",
        "control_variable_status": "matched_with_spinuv_segformer_b5",
        "notes": "Official NVIDIA Hugging Face checkpoint.",
    },
    {
        "source_dataset": "ADE20K",
        "model_name": "SegFormer-B5 ADE20K",
        "architecture_key": "segformer_b5",
        "framework": "hf_auto",
        "checkpoint_url": "https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640",
        "config_path": "configs/segformer_b5_ade20k_source_only.json",
        "current_e2_status": "default_runnable",
        "current_runner_compatible": "yes",
        "control_variable_status": "matched_with_spinuv_segformer_b5",
        "notes": "Official NVIDIA Hugging Face checkpoint.",
    },
    {
        "source_dataset": "IDD 20K",
        "model_name": "SegFormer-B5 IDD",
        "architecture_key": "segformer_b5",
        "framework": "hf_auto",
        "checkpoint_url": "https://huggingface.co/izzako/segformer-b5-finetuned-IDD-L2_v2",
        "config_path": "configs/segformer_b5_idd_source_only.json",
        "current_e2_status": "default_runnable",
        "current_runner_compatible": "yes",
        "control_variable_status": "matched_with_spinuv_segformer_b5",
        "notes": "Community Hugging Face checkpoint fine-tuned from nvidia/mit-b5 on IDD 20K; evaluated on the 7 exact labels exposed by the public L2 model card.",
    },
    {
        "source_dataset": "IDD 20K",
        "model_name": "SegFormer-B4 IDD",
        "architecture_key": "segformer_b4",
        "framework": "hf_auto",
        "checkpoint_url": "https://huggingface.co/izzako/segformer-b4-finetuned-IDD-L2_v2",
        "config_path": "",
        "current_e2_status": "public_weights_found_optional",
        "current_runner_compatible": "yes",
        "control_variable_status": "architecture_mismatch",
        "notes": "Community Hugging Face checkpoint; compatible with the HF runner but not used in the default SegFormer-B5 controlled comparison.",
    },
    {
        "source_dataset": "BDD100K sem_seg 10K",
        "model_name": "UPerNet ConvNeXt-B BDD100K",
        "architecture_key": "upernet_convnext_b",
        "framework": "mmsegmentation",
        "checkpoint_url": "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/upernet_convnext-b_fp16_512x1024_80k_sem_seg_bdd100k.pth",
        "config_path": "https://github.com/SysCV/bdd100k-models/blob/main/sem_seg/configs/sem_seg/upernet_convnext-b_fp16_512x1024_80k_sem_seg_bdd100k.py",
        "current_e2_status": "public_weights_found_not_default",
        "current_runner_compatible": "no",
        "control_variable_status": "architecture_mismatch",
        "notes": "Official BDD100K model-zoo checkpoint; requires an MMSegmentation runner or HF conversion before evaluation.",
    },
    {
        "source_dataset": "BDD100K sem_seg 10K",
        "model_name": "UPerNet Swin-B BDD100K",
        "architecture_key": "upernet_swin_b",
        "framework": "mmsegmentation",
        "checkpoint_url": "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/upernet_swin-b_fp16_512x1024_80k_sem_seg_bdd100k.pth",
        "config_path": "https://github.com/SysCV/bdd100k-models/blob/main/sem_seg/configs/sem_seg/upernet_swin-b_fp16_512x1024_80k_sem_seg_bdd100k.py",
        "current_e2_status": "public_weights_found_not_default",
        "current_runner_compatible": "no",
        "control_variable_status": "architecture_mismatch",
        "notes": "Official BDD100K model-zoo lists Swin Transformer semantic segmentation checkpoints; keep as optional until the runner supports MMSeg.",
    },
    {
        "source_dataset": "BDD100K sem_seg 10K",
        "model_name": "DeepLabv3+ R101 BDD100K",
        "architecture_key": "deeplabv3plus_r101",
        "framework": "mmsegmentation",
        "checkpoint_url": "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/deeplabv3%2B_r101-d8_512x1024_80k_sem_seg_bdd100k.pth",
        "config_path": "https://github.com/SysCV/bdd100k-models/blob/main/sem_seg/configs/sem_seg/deeplabv3%2B_r101-d8_512x1024_80k_sem_seg_bdd100k.py",
        "current_e2_status": "public_weights_found_not_default",
        "current_runner_compatible": "no",
        "control_variable_status": "architecture_mismatch",
        "notes": "Official BDD100K model-zoo lists DeepLabv3+ semantic segmentation checkpoints; not architecture-controlled against SegFormer-B5.",
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    tables_dir = E2_ROOT / "tables"
    logs_dir = E2_ROOT / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "experiment_02_external_checkpoint_registry.csv", ROWS)
    (tables_dir / "experiment_02_external_checkpoint_registry.json").write_text(
        json.dumps(ROWS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(ROWS), "output": (tables_dir / "experiment_02_external_checkpoint_registry.csv").as_posix()}, indent=2))


if __name__ == "__main__":
    main()

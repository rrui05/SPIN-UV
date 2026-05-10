#!/usr/bin/env python
"""Evaluate a public source-trained HF segmentation model on SPIN-UV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from spinuv_palette import colorize_train_ids

E2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = E2_ROOT.parent
DEFAULT_ANALYSIS_CLASSES = [
    "building",
    "scooter",
    "road",
    "car",
    "wall",
    "clutter",
    "sky",
    "sidewalk",
    "vegetation",
    "pole",
    "person",
    "tricycle",
    "roadblock",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [E2_ROOT / path, EXPERIMENT_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return E2_ROOT / path


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, E2_ROOT / path, EXPERIMENT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return E2_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_for_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean_for_json(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class EventLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_path = run_dir / "logs" / "eval_events.jsonl"
        self.global_path = E2_ROOT / "logs" / "experiment_02_events.jsonl"
        self.run_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        row = {
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            **payload,
        }
        line = json.dumps(clean_for_json(row), ensure_ascii=False)
        with self.run_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        with self.global_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def normalize_label(label: Any) -> str:
    text = str(label).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_spinuv_classes(data_root: Path) -> tuple[list[str], int]:
    payload = read_json(data_root / "meta" / "classes.json")
    ordered = sorted(payload["classes"], key=lambda item: int(item["train_id"]))
    class_names = [str(item["name"]) for item in ordered]
    ignore_index = int(payload.get("ignore_value", 255))
    return class_names, ignore_index


def resolve_analysis_class_ids(class_names: list[str], selected: Iterable[str] | None) -> list[int]:
    names = list(selected or DEFAULT_ANALYSIS_CLASSES)
    label2id = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in names if name not in label2id]
    if missing:
        raise ValueError(f"Unknown SPIN-UV analysis classes: {missing}")
    return [label2id[name] for name in names]


def list_split_items(data_root: Path, split: str, max_images: int | None) -> list[tuple[Path, Path]]:
    image_dir = data_root / "images" / split
    mask_dir = data_root / "masks" / split
    image_paths = sorted(image_dir.glob("*.png"))
    if max_images is not None:
        image_paths = image_paths[: max(0, int(max_images))]
    pairs = [(image_path, mask_dir / image_path.name) for image_path in image_paths]
    missing = [mask_path for _, mask_path in pairs if not mask_path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks for split={split}; first={missing[0]}")
    if not pairs:
        raise FileNotFoundError(f"No image-mask pairs found for split={split} under {data_root}")
    return pairs


def get_id2label(model: torch.nn.Module) -> dict[int, str]:
    raw = getattr(getattr(model, "config", None), "id2label", None) or {}
    id2label: dict[int, str] = {}
    for key, value in raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            idx = int(str(key))
        id2label[idx] = str(value)
    if not id2label:
        out_channels = int(getattr(getattr(model, "config", None), "num_labels", 0) or 0)
        id2label = {idx: f"label_{idx}" for idx in range(out_channels)}
    return dict(sorted(id2label.items()))


def build_source_mapping(
    id2label: dict[int, str],
    source_to_spinuv: dict[str, str],
    spinuv_class_names: list[str],
    analysis_class_ids: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    spinuv_label2id = {name: idx for idx, name in enumerate(spinuv_class_names)}
    normalized_config = {normalize_label(src): (src, dst) for src, dst in source_to_spinuv.items()}
    source_norm_to_ids: dict[str, list[int]] = {}
    for source_id, source_label in id2label.items():
        source_norm_to_ids.setdefault(normalize_label(source_label), []).append(source_id)

    max_source_id = max(id2label) if id2label else -1
    mapping = np.full(max_source_id + 1, 255, dtype=np.int64)
    matched_rows: list[dict[str, Any]] = []
    missing_source_labels: list[dict[str, str]] = []
    target_to_sources: dict[str, list[str]] = {}

    for source_norm, (configured_source, target_label) in normalized_config.items():
        if target_label not in spinuv_label2id:
            raise ValueError(f"Unknown target SPIN-UV label in source mapping: {target_label}")
        source_ids = source_norm_to_ids.get(source_norm, [])
        if not source_ids:
            missing_source_labels.append(
                {
                    "configured_source_label": configured_source,
                    "target_spinuv_label": target_label,
                }
            )
            continue
        target_id = spinuv_label2id[target_label]
        for source_id in source_ids:
            mapping[source_id] = target_id
            matched_rows.append(
                {
                    "source_id": source_id,
                    "source_label": id2label[source_id],
                    "spinuv_id": target_id,
                    "spinuv_label": target_label,
                }
            )
            target_to_sources.setdefault(target_label, []).append(id2label[source_id])

    analysis_labels = [spinuv_class_names[idx] for idx in analysis_class_ids]
    matched_analysis = sorted(set(target_to_sources).intersection(analysis_labels))
    unsupported_analysis = [name for name in analysis_labels if name not in matched_analysis]
    report = {
        "source_label_count": len(id2label),
        "source_id2label": {str(k): v for k, v in id2label.items()},
        "matched_rows": sorted(matched_rows, key=lambda row: (row["spinuv_id"], row["source_id"])),
        "missing_configured_source_labels": missing_source_labels,
        "matched_analysis_classes": matched_analysis,
        "unsupported_analysis_classes": unsupported_analysis,
        "target_to_source_labels": target_to_sources,
    }
    return mapping, report


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def predict_source_map(
    image: Image.Image,
    model: torch.nn.Module,
    processor: Any,
    framework: str,
    device: torch.device,
    output_size: tuple[int, int],
) -> np.ndarray:
    if framework == "hf_oneformer":
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = move_inputs_to_device(inputs, device)
        outputs = model(**inputs)
        pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[output_size])[0]
        return pred.detach().cpu().numpy().astype(np.int64)

    inputs = processor(images=image, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, device)
    outputs = model(**inputs)
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, dict):
        logits = outputs.get("logits")
    if logits is None:
        raise RuntimeError("Model output does not expose logits for semantic segmentation.")
    logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)


def map_source_to_spinuv(source_pred: np.ndarray, source_mapping: np.ndarray, ignore_index: int) -> np.ndarray:
    mapped = np.full(source_pred.shape, ignore_index, dtype=np.int64)
    valid = (source_pred >= 0) & (source_pred < len(source_mapping))
    mapped[valid] = source_mapping[source_pred[valid]]
    return mapped


def init_totals(class_ids: list[int]) -> dict[int, dict[str, int]]:
    return {
        idx: {
            "intersection_pixels": 0,
            "union_pixels": 0,
            "support_pixels": 0,
            "pred_pixels": 0,
        }
        for idx in class_ids
    }


def update_totals(
    totals: dict[int, dict[str, int]],
    pred: np.ndarray,
    target: np.ndarray,
    class_ids: list[int],
    ignore_index: int,
) -> dict[str, Any]:
    valid_gt = target != ignore_index
    image_intersection = 0
    image_union_values: list[float] = []
    image_acc_values: list[float] = []
    for class_id in class_ids:
        gt_c = valid_gt & (target == class_id)
        pred_c = valid_gt & (pred == class_id)
        intersection = int(np.logical_and(gt_c, pred_c).sum())
        union = int(np.logical_or(gt_c, pred_c).sum())
        support = int(gt_c.sum())
        pred_pixels = int(pred_c.sum())
        totals[class_id]["intersection_pixels"] += intersection
        totals[class_id]["union_pixels"] += union
        totals[class_id]["support_pixels"] += support
        totals[class_id]["pred_pixels"] += pred_pixels
        image_intersection += intersection
        if union > 0:
            image_union_values.append(intersection / union)
        if support > 0:
            image_acc_values.append(intersection / support)

    eval_mask = valid_gt & np.isin(target, np.array(class_ids, dtype=np.int64))
    eval_pixels = int(eval_mask.sum())
    correct_pixels = int(((pred == target) & eval_mask).sum())
    return {
        "image_miou": float(np.mean(image_union_values)) if image_union_values else float("nan"),
        "image_mean_accuracy": float(np.mean(image_acc_values)) if image_acc_values else float("nan"),
        "image_pixel_accuracy": float(correct_pixels / eval_pixels) if eval_pixels else float("nan"),
        "eval_pixels": eval_pixels,
        "correct_pixels": correct_pixels,
    }


def totals_to_rows(
    totals: dict[int, dict[str, int]],
    class_names: list[str],
    class_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ious: list[float] = []
    accuracies: list[float] = []
    total_intersection = 0
    total_support = 0
    for class_id in class_ids:
        item = totals[class_id]
        intersection = item["intersection_pixels"]
        union = item["union_pixels"]
        support = item["support_pixels"]
        pred_pixels = item["pred_pixels"]
        iou = intersection / union if union > 0 else float("nan")
        accuracy = intersection / support if support > 0 else float("nan")
        if math.isfinite(iou):
            ious.append(iou)
        if math.isfinite(accuracy):
            accuracies.append(accuracy)
        total_intersection += intersection
        total_support += support
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "iou": iou if math.isfinite(iou) else "",
                "accuracy": accuracy if math.isfinite(accuracy) else "",
                "support_pixels": support,
                "pred_pixels": pred_pixels,
                "intersection_pixels": intersection,
                "union_pixels": union,
            }
        )

    rows = sorted(
        rows,
        key=lambda row: float(row["iou"]) if row["iou"] not in ("", None) and math.isfinite(float(row["iou"])) else -1.0,
        reverse=True,
    )
    summary = {
        "mIoU": float(np.mean(ious)) if ious else float("nan"),
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "pixel_accuracy": float(total_intersection / total_support) if total_support else float("nan"),
        "analysis_pixels": total_support,
    }
    return rows, summary


def make_sample(
    image: Image.Image,
    target: np.ndarray,
    pred: np.ndarray,
    image_name: str,
    image_miou: float,
) -> dict[str, Any]:
    return {
        "image": np.asarray(image.convert("RGB"), dtype=np.uint8),
        "gt": target.astype(np.int64),
        "pred": pred.astype(np.int64),
        "image_name": image_name,
        "image_miou": image_miou,
    }


def update_worst_samples(samples: list[dict[str, Any]], candidate: dict[str, Any], sample_count: int) -> list[dict[str, Any]]:
    samples.append(candidate)
    samples = sorted(
        samples,
        key=lambda item: item["image_miou"] if math.isfinite(float(item["image_miou"])) else 1e9,
    )
    return samples[:sample_count]


def save_visual_grid(samples: list[dict[str, Any]], path: Path, title: str, class_names: list[str], ignore_index: int, class_ids: list[int]) -> None:
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.2 * rows), squeeze=False)
    selected = np.array(class_ids, dtype=np.int64)
    for row_idx, sample in enumerate(samples):
        image = sample["image"]
        gt = sample["gt"]
        pred = sample["pred"]
        gt_color = colorize_train_ids(gt, class_names, ignore_index)
        pred_color = colorize_train_ids(pred, class_names, ignore_index)
        eval_mask = (gt != ignore_index) & np.isin(gt, selected)
        error = np.zeros_like(image)
        error[eval_mask & (gt == pred)] = np.array([35, 120, 60], dtype=np.uint8)
        error[eval_mask & (gt != pred)] = np.array([210, 45, 45], dtype=np.uint8)
        error[~eval_mask] = np.array([40, 40, 40], dtype=np.uint8)
        panels = [image, gt_color, pred_color, error]
        titles = ["Image", "Ground truth", "Mapped prediction", "Analysis error"]
        for col_idx, (panel, panel_title) in enumerate(zip(panels, titles)):
            axes[row_idx, col_idx].imshow(panel)
            axes[row_idx, col_idx].set_title(panel_title, fontsize=10)
            axes[row_idx, col_idx].axis("off")
        ylabel = f"{sample['image_name']}\nmIoU={sample['image_miou']:.3f}"
        axes[row_idx, 0].set_ylabel(ylabel, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_per_class_iou_plot(rows: list[dict[str, Any]], path: Path, display_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["class_name"]) for row in rows]
    values: list[float] = []
    labels: list[str] = []
    for row in rows:
        raw = row["iou"]
        if raw == "" or raw is None:
            values.append(0.0)
            labels.append("n/a")
        else:
            value = float(raw)
            values.append(value if math.isfinite(value) else 0.0)
            labels.append(f"{value:.3f}" if math.isfinite(value) else "n/a")

    fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(names))))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color="#4f7cac")
    for bar, label, value in zip(bars, labels, values):
        ax.text(min(value + 0.015, 0.96), bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("IoU")
    ax.set_title(f"Per-class IoU: {display_name}")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_hf_model(config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, Any]:
    framework = str(config.get("framework", "hf_auto"))
    model_id = str(config["model_id"])
    if framework == "hf_oneformer":
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        processor = OneFormerProcessor.from_pretrained(model_id)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id)
    else:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = read_json(config_path)
    model_slug = str(config["model_slug"])
    display_name = str(config.get("display_name", model_slug))
    split = args.split or str(config.get("split", "test"))
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = E2_ROOT / "runs" / model_slug / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(run_dir)

    data_root = resolve_path(config.get("data_root", "data/processed/spinuv_semantic"))
    class_names, dataset_ignore = load_spinuv_classes(data_root)
    ignore_index = int(config.get("ignore_index", dataset_ignore))
    analysis_class_ids = resolve_analysis_class_ids(class_names, config.get("analysis_classes"))
    pairs = list_split_items(data_root, split, args.max_images)
    sample_count = int(args.sample_count if args.sample_count is not None else config.get("sample_count", 8))
    save_predictions = bool(config.get("save_predictions", True)) and not args.no_save_predictions

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "args.json",
        {
            "config": config_path.as_posix(),
            "run_tag": run_tag,
            "split": split,
            "max_images": args.max_images,
            "device": str(device),
            "sample_count": sample_count,
            "save_predictions": save_predictions,
        },
    )
    logger.log(
        "start",
        model_slug=model_slug,
        display_name=display_name,
        model_id=config["model_id"],
        split=split,
        image_count=len(pairs),
        device=str(device),
    )

    load_start = time.time()
    model, processor = load_hf_model(config, device)
    total_params = int(sum(param.numel() for param in model.parameters()))
    trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    id2label = get_id2label(model)
    source_mapping, mapping_report = build_source_mapping(
        id2label,
        config.get("source_to_spinuv", {}),
        class_names,
        analysis_class_ids,
    )
    mapping_report.update(
        {
            "model_slug": model_slug,
            "display_name": display_name,
            "model_id": config["model_id"],
            "analysis_classes": [class_names[idx] for idx in analysis_class_ids],
            "mapping_notes": config.get("mapping_notes", []),
        }
    )
    write_json(run_dir / "label_mapping_report.json", mapping_report)
    write_json(
        run_dir / "model_info.json",
        {
            "model_slug": model_slug,
            "display_name": display_name,
            "model_id": config["model_id"],
            "framework": config.get("framework", "hf_auto"),
            "source_dataset": config.get("source_dataset", ""),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "source_label_count": len(id2label),
        },
    )
    logger.log(
        "model_loaded",
        load_seconds=time.time() - load_start,
        total_params=total_params,
        trainable_params=trainable_params,
        matched_analysis_classes=mapping_report["matched_analysis_classes"],
        unsupported_analysis_classes=mapping_report["unsupported_analysis_classes"],
    )

    totals = init_totals(analysis_class_ids)
    per_image_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    first_samples: list[dict[str, Any]] = []
    worst_samples: list[dict[str, Any]] = []
    pred_dir = run_dir / "predictions"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    eval_start = time.time()
    framework = str(config.get("framework", "hf_auto"))
    for index, (image_path, mask_path) in enumerate(tqdm(pairs, desc=f"eval:{model_slug}:{split}"), start=1):
        image_start = time.time()
        image = Image.open(image_path).convert("RGB")
        target = np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)
        source_pred = predict_source_map(
            image=image,
            model=model,
            processor=processor,
            framework=framework,
            device=device,
            output_size=target.shape,
        )
        mapped_pred = map_source_to_spinuv(source_pred, source_mapping, ignore_index)
        metrics = update_totals(totals, mapped_pred, target, analysis_class_ids, ignore_index)
        row = {
            "image_name": image_path.name,
            "image_path": image_path.as_posix(),
            "mask_path": mask_path.as_posix(),
            "image_miou": metrics["image_miou"] if math.isfinite(float(metrics["image_miou"])) else "",
            "image_mean_accuracy": metrics["image_mean_accuracy"] if math.isfinite(float(metrics["image_mean_accuracy"])) else "",
            "image_pixel_accuracy": metrics["image_pixel_accuracy"] if math.isfinite(float(metrics["image_pixel_accuracy"])) else "",
            "eval_pixels": metrics["eval_pixels"],
            "correct_pixels": metrics["correct_pixels"],
            "seconds": time.time() - image_start,
        }
        per_image_rows.append(row)
        logger.log("image_evaluated", index=index, **row)

        if save_predictions:
            Image.fromarray(mapped_pred.astype(np.uint8), mode="L").save(pred_dir / image_path.name)
        sample = make_sample(image, target, mapped_pred, image_path.name, float(metrics["image_miou"]))
        if len(first_samples) < sample_count:
            first_samples.append(sample)
        worst_samples = update_worst_samples(worst_samples, sample, sample_count)
        sample_rows.append(
            {
                "image_name": image_path.name,
                "image_miou": row["image_miou"],
                "saved_prediction_path": (pred_dir / image_path.name).as_posix() if save_predictions else "",
            }
        )

    per_class_rows, summary_metrics = totals_to_rows(totals, class_names, analysis_class_ids)
    summary = {
        "status": "complete",
        "experiment": "experiment_02_domain_gap",
        "experiment_role": "source_only_public_checkpoint",
        "training_type": config.get("training_type", "public_source_only_checkpoint"),
        "model_slug": model_slug,
        "display_name": display_name,
        "model_id": config["model_id"],
        "model_url": config.get("model_url", ""),
        "framework": framework,
        "source_dataset": config.get("source_dataset", ""),
        "split": split,
        "run_tag": run_tag,
        "run_dir": run_dir.as_posix(),
        "image_count": len(pairs),
        "analysis_class_count": len(analysis_class_ids),
        "matched_analysis_class_count": len(mapping_report["matched_analysis_classes"]),
        "matched_analysis_classes": mapping_report["matched_analysis_classes"],
        "unsupported_analysis_classes": mapping_report["unsupported_analysis_classes"],
        "mIoU": summary_metrics["mIoU"],
        "mean_accuracy": summary_metrics["mean_accuracy"],
        "pixel_accuracy": summary_metrics["pixel_accuracy"],
        "analysis_pixels": summary_metrics["analysis_pixels"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1_000_000,
        "trainable_params_m": trainable_params / 1_000_000,
        "eval_seconds": time.time() - eval_start,
    }

    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    write_json(tables_dir / f"{split}_summary.json", summary)
    write_csv(tables_dir / f"{split}_per_class_iou.csv", per_class_rows)
    write_csv(tables_dir / f"{split}_per_image_iou.csv", per_image_rows)
    write_csv(tables_dir / f"{split}_sample_index.csv", sample_rows)
    save_per_class_iou_plot(per_class_rows, figures_dir / f"{split}_per_class_iou.png", display_name)
    save_visual_grid(
        first_samples,
        figures_dir / f"{split}_samples.png",
        f"Source-only samples: {display_name}",
        class_names,
        ignore_index,
        analysis_class_ids,
    )
    save_visual_grid(
        worst_samples,
        figures_dir / f"{split}_worst_predictions.png",
        f"Worst source-only predictions: {display_name}",
        class_names,
        ignore_index,
        analysis_class_ids,
    )

    logger.log("complete", **summary)
    print(json.dumps(clean_for_json(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate E2-compatible predictions from completed Experiment 01 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


E2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = E2_ROOT.parent
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-models-root", default="../e1_models")
    parser.add_argument("--data-root", default="../data/processed/spinuv_semantic")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--analysis-classes", default="building,road,car,wall,sky,sidewalk,vegetation,pole,person")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--model-list", default=None)
    parser.add_argument("--e1-run-dir", default=None, help="Explicit E1 run dir, checkpoints dir, or checkpoint file for the selected model.")
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [base / path, E2_ROOT / path, EXPERIMENT_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base / path


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
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
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
    def __init__(self) -> None:
        self.path = E2_ROOT / "logs" / "experiment_02_events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        row = {
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean_for_json(row), ensure_ascii=False) + "\n")


def load_spinuv_classes(data_root: Path) -> tuple[list[str], int]:
    payload = read_json(data_root / "meta" / "classes.json")
    ordered = sorted(payload["classes"], key=lambda item: int(item["train_id"]))
    return [str(item["name"]) for item in ordered], int(payload.get("ignore_value", 255))


def resolve_analysis_ids(class_names: list[str], analysis_classes: str | Iterable[str]) -> list[int]:
    if isinstance(analysis_classes, str):
        names = [item.strip() for item in analysis_classes.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in analysis_classes if str(item).strip()]
    label2id = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in names if name not in label2id]
    if missing:
        raise ValueError(f"Unknown analysis classes: {missing}")
    return [label2id[name] for name in names]


def list_split_items(data_root: Path, split: str, max_images: int | None) -> list[tuple[Path, Path]]:
    image_paths = sorted((data_root / "images" / split).glob("*.png"))
    if max_images is not None:
        image_paths = image_paths[: max(0, int(max_images))]
    pairs = [(image_path, data_root / "masks" / split / image_path.name) for image_path in image_paths]
    missing = [mask_path for _, mask_path in pairs if not mask_path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks for split={split}; first={missing[0]}")
    return pairs


def find_e1_run_dir(model_dir: Path) -> Path | None:
    preferred = model_dir / "runs" / "full_8k"
    if (preferred / "checkpoints" / "best_miou.pt").exists():
        return preferred
    candidates = sorted(
        [path for path in model_dir.glob("runs/full_8k*") if (path / "checkpoints" / "best_miou.pt").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def normalize_explicit_run_path(path: Path) -> tuple[Path, Path]:
    if path.suffix == ".pt":
        run_dir = path.parent.parent
        checkpoint_path = path
    elif path.name == "checkpoints":
        run_dir = path.parent
        checkpoint_path = path / "best_miou.pt"
    else:
        run_dir = path
        checkpoint_path = path / "checkpoints" / "best_miou.pt"
    if not checkpoint_path.exists():
        latest_path = run_dir / "checkpoints" / "latest.pt"
        if latest_path.exists():
            checkpoint_path = latest_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found for explicit E1 run path: {path}")
    return run_dir, checkpoint_path


def find_e1_checkpoint(model_dir: Path, explicit_run_path: Path | None = None) -> tuple[Path, Path] | None:
    if explicit_run_path is not None:
        return normalize_explicit_run_path(explicit_run_path)
    run_dir = find_e1_run_dir(model_dir)
    if run_dir is None:
        return None
    return run_dir, run_dir / "checkpoints" / "best_miou.pt"


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous().unsqueeze(0)


def label_maps(class_names: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    id2label = {idx: name for idx, name in enumerate(class_names)}
    label2id = {name: idx for idx, name in id2label.items()}
    return id2label, label2id


def normalize_weights_arg(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).lower() in {"none", "null", "false", "0"}:
        return None
    return value


def build_model(config: dict[str, Any], class_names: list[str], ignore_index: int) -> torch.nn.Module:
    framework = str(config.get("framework", "smp")).lower()
    class_count = len(class_names)
    if framework == "smp":
        import segmentation_models_pytorch as smp

        model_name = str(config.get("model_name", "DeepLabV3Plus"))
        model_cls = getattr(smp, model_name)
        return model_cls(
            encoder_name=config.get("encoder_name", "resnet50"),
            encoder_weights=normalize_weights_arg(config.get("encoder_weights", None)),
            in_channels=3,
            classes=class_count,
            activation=None,
        )

    id2label, label2id = label_maps(class_names)
    model_id = config.get("model_id") or config.get("model_name")
    common_kwargs = {
        "num_labels": class_count,
        "id2label": id2label,
        "label2id": label2id,
        "ignore_mismatched_sizes": bool(config.get("ignore_mismatched_sizes", True)),
    }
    if framework == "hf_segformer":
        from transformers import SegformerForSemanticSegmentation

        model = SegformerForSemanticSegmentation.from_pretrained(model_id, **common_kwargs)
    elif framework == "hf_upernet":
        from transformers import UperNetForSemanticSegmentation

        model = UperNetForSemanticSegmentation.from_pretrained(model_id, **common_kwargs)
    elif framework == "hf_auto":
        from transformers import AutoModelForSemanticSegmentation

        model = AutoModelForSemanticSegmentation.from_pretrained(model_id, **common_kwargs)
    else:
        raise ValueError(f"Unsupported E1 framework: {framework}")

    if hasattr(model.config, "semantic_loss_ignore_index"):
        model.config.semantic_loss_ignore_index = ignore_index
    if hasattr(model.config, "ignore_index"):
        model.config.ignore_index = ignore_index
    return model


def load_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {}


@torch.no_grad()
def forward_pred(model: torch.nn.Module, image_tensor: torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    output = model(image_tensor)
    if torch.is_tensor(output):
        logits = output
    elif hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, dict) and "logits" in output:
        logits = output["logits"]
    elif isinstance(output, dict) and "out" in output:
        logits = output["out"]
    else:
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    if logits.shape[-2:] != output_size:
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)


def init_totals(class_ids: list[int]) -> dict[int, dict[str, int]]:
    return {
        class_id: {
            "intersection_pixels": 0,
            "union_pixels": 0,
            "support_pixels": 0,
            "pred_pixels": 0,
        }
        for class_id in class_ids
    }


def update_totals(
    totals: dict[int, dict[str, int]],
    pred: np.ndarray,
    target: np.ndarray,
    class_ids: list[int],
    ignore_index: int,
) -> dict[str, Any]:
    valid_gt = target != ignore_index
    image_ious: list[float] = []
    image_accs: list[float] = []
    correct = 0
    eval_pixels = 0
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
        correct += intersection
        eval_pixels += support
        if union > 0:
            image_ious.append(intersection / union)
        if support > 0:
            image_accs.append(intersection / support)
    return {
        "image_miou": float(np.mean(image_ious)) if image_ious else float("nan"),
        "image_mean_accuracy": float(np.mean(image_accs)) if image_accs else float("nan"),
        "image_pixel_accuracy": float(correct / eval_pixels) if eval_pixels else float("nan"),
        "eval_pixels": eval_pixels,
        "correct_pixels": correct,
    }


def totals_to_rows(
    totals: dict[int, dict[str, int]],
    class_names: list[str],
    class_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ious: list[float] = []
    accs: list[float] = []
    total_intersection = 0
    total_support = 0
    for class_id in class_ids:
        item = totals[class_id]
        intersection = item["intersection_pixels"]
        union = item["union_pixels"]
        support = item["support_pixels"]
        iou = intersection / union if union else float("nan")
        acc = intersection / support if support else float("nan")
        if math.isfinite(iou):
            ious.append(iou)
        if math.isfinite(acc):
            accs.append(acc)
        total_intersection += intersection
        total_support += support
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "iou": iou if math.isfinite(iou) else "",
                "accuracy": acc if math.isfinite(acc) else "",
                "support_pixels": support,
                "pred_pixels": item["pred_pixels"],
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
        "mean_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "pixel_accuracy": float(total_intersection / total_support) if total_support else float("nan"),
        "analysis_pixels": total_support,
    }
    return rows, summary


def evaluate_model(
    model_dir: Path,
    run_dir: Path,
    data_pairs: list[tuple[Path, Path]],
    class_names: list[str],
    ignore_index: int,
    class_ids: list[int],
    split: str,
    device: torch.device,
    logger: EventLogger,
    explicit_e1_run_dir: Path | None = None,
) -> dict[str, Any]:
    config = read_json(model_dir / "config.json")
    checkpoint_info = find_e1_checkpoint(model_dir, explicit_e1_run_dir)
    if checkpoint_info is None:
        raise FileNotFoundError(f"No completed E1 checkpoint found under {model_dir}")
    e1_run_dir, checkpoint_path = checkpoint_info
    model = build_model(config, class_names, ignore_index)
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()
    total_params = int(sum(param.numel() for param in model.parameters()))
    trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))

    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "model_info.json",
        {
            "source_e1_model_dir": model_dir,
            "source_e1_run_dir": e1_run_dir,
            "checkpoint_path": checkpoint_path,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
    )

    totals = init_totals(class_ids)
    per_image_rows: list[dict[str, Any]] = []
    start = time.time()
    for image_path, mask_path in tqdm(data_pairs, desc=f"eval:e1:{model_dir.name}:{split}"):
        image_start = time.time()
        image = Image.open(image_path).convert("RGB")
        target = np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)
        image_tensor = image_to_tensor(image).to(device)
        pred = forward_pred(model, image_tensor, target.shape)
        metrics = update_totals(totals, pred, target, class_ids, ignore_index)
        Image.fromarray(pred.astype(np.uint8), mode="L").save(pred_dir / image_path.name)
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

    per_class_rows, summary_metrics = totals_to_rows(totals, class_names, class_ids)
    model_slug = f"e1_{config.get('model_slug', model_dir.name)}"
    display_name = f"SPIN-UV trained {config.get('display_name', model_dir.name)}"
    summary = {
        "status": "complete",
        "experiment": "experiment_02_domain_gap",
        "experiment_role": "spinuv_in_domain_trained",
        "model_slug": model_slug,
        "display_name": display_name,
        "model_id": config.get("model_id", ""),
        "model_url": "",
        "framework": config.get("framework", ""),
        "source_dataset": "SPIN-UV",
        "split": split,
        "run_tag": run_dir.name,
        "run_dir": run_dir.as_posix(),
        "image_count": len(data_pairs),
        "analysis_class_count": len(class_ids),
        "matched_analysis_class_count": len(class_ids),
        "matched_analysis_classes": [class_names[idx] for idx in class_ids],
        "unsupported_analysis_classes": [],
        "mIoU": summary_metrics["mIoU"],
        "mean_accuracy": summary_metrics["mean_accuracy"],
        "pixel_accuracy": summary_metrics["pixel_accuracy"],
        "analysis_pixels": summary_metrics["analysis_pixels"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1_000_000,
        "trainable_params_m": trainable_params / 1_000_000,
        "eval_seconds": time.time() - start,
        "source_e1_run_dir": e1_run_dir.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
    }
    write_json(tables_dir / f"{split}_summary.json", summary)
    write_csv(tables_dir / f"{split}_per_class_iou.csv", per_class_rows)
    write_csv(tables_dir / f"{split}_per_image_iou.csv", per_image_rows)
    logger.log("e1_predictions_complete", **summary)
    return summary


def main() -> None:
    args = parse_args()
    logger = EventLogger()
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    data_root = resolve_path(args.data_root, E2_ROOT)
    e1_root = resolve_path(args.e1_models_root, E2_ROOT)
    explicit_e1_run_dir = resolve_path(args.e1_run_dir, E2_ROOT) if args.e1_run_dir else None
    class_names, ignore_index = load_spinuv_classes(data_root)
    analysis_ids = resolve_analysis_ids(class_names, args.analysis_classes)
    data_pairs = list_split_items(data_root, args.split, args.max_images)

    wanted = None
    if args.model_list:
        wanted = {item.strip() for item in args.model_list.split() if item.strip()}
    model_dirs = sorted([path for path in e1_root.iterdir() if path.is_dir()]) if e1_root.exists() else []
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for model_dir in model_dirs:
        if wanted is not None and model_dir.name not in wanted:
            continue
        checkpoint_info = find_e1_checkpoint(model_dir, explicit_e1_run_dir)
        if checkpoint_info is None:
            missing.append({"model_dir": model_dir.as_posix(), "reason": "missing_best_checkpoint"})
            logger.log("e1_predictions_skipped", model_dir=model_dir.as_posix(), reason="missing_best_checkpoint")
            continue
        out_dir = E2_ROOT / "runs" / f"e1_{model_dir.name}" / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            summaries.append(evaluate_model(model_dir, out_dir, data_pairs, class_names, ignore_index, analysis_ids, args.split, device, logger, explicit_e1_run_dir))
        except Exception as exc:
            missing.append({"model_dir": model_dir.as_posix(), "reason": f"prediction_error: {exc}"})
            logger.log("e1_predictions_error", model_dir=model_dir.as_posix(), error=str(exc))

    status = {
        "run_tag": run_tag,
        "completed": len(summaries),
        "missing": missing,
        "analysis_classes": [class_names[idx] for idx in analysis_ids],
    }
    write_json(E2_ROOT / "logs" / "experiment_02_e1_prediction_status.json", status)
    print(json.dumps(clean_for_json(status), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

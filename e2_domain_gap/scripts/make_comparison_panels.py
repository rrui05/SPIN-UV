#!/usr/bin/env python
"""Create external-vs-SPIN-UV qualitative comparison panels for E2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spinuv_palette import colorize_train_ids

E2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = E2_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--source-slug", default=None)
    parser.add_argument("--indomain-slug", default=None)
    parser.add_argument("--allow-architecture-mismatch", action="store_true")
    parser.add_argument("--analysis-classes", default="building,road,car,wall,sky,sidewalk,vegetation,pole,person")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def clean_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def load_class_names() -> tuple[list[str], int]:
    payload = read_json(EXPERIMENT_ROOT / "data" / "processed" / "spinuv_semantic" / "meta" / "classes.json")
    ordered = sorted(payload["classes"], key=lambda item: int(item["train_id"]))
    return [str(item["name"]) for item in ordered], int(payload.get("ignore_value", 255))


def analysis_ids(class_names: list[str], names: str) -> list[int]:
    selected = [item.strip() for item in names.split(",") if item.strip()]
    label2id = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in selected if name not in label2id]
    if missing:
        raise ValueError(f"Unknown analysis classes: {missing}")
    return [label2id[name] for name in selected]


def latest_run_dirs(run_tag: str | None) -> list[Path]:
    runs_root = E2_ROOT / "runs"
    if not runs_root.exists():
        return []
    output: list[Path] = []
    for model_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if run_tag:
            candidate = model_dir / run_tag
            if (candidate / "tables" / "test_summary.json").exists():
                output.append(candidate)
            continue
        candidates = sorted(
            [path for path in model_dir.iterdir() if path.is_dir() and (path / "tables" / "test_summary.json").exists()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            output.append(candidates[0])
    return output


def choose_runs(
    run_dirs: list[Path],
    source_slug: str | None,
    indomain_slug: str | None,
    allow_architecture_mismatch: bool,
) -> tuple[list[tuple[Path, Path]], list[dict[str, Any]]]:
    source_runs: list[Path] = []
    indomain_runs: list[Path] = []
    for run_dir in run_dirs:
        summary = read_json(run_dir / "tables" / "test_summary.json")
        role = summary.get("experiment_role", "")
        slug = summary.get("model_slug", run_dir.parent.name)
        if role == "source_only_public_checkpoint":
            if source_slug is None or slug == source_slug or run_dir.parent.name == source_slug:
                source_runs.append(run_dir)
        elif role == "spinuv_in_domain_trained":
            if indomain_slug is None or slug == indomain_slug or run_dir.parent.name == indomain_slug:
                indomain_runs.append(run_dir)
    pairs: list[tuple[Path, Path]] = []
    skipped: list[dict[str, Any]] = []
    for source_run in source_runs:
        source_summary = read_json(source_run / "tables" / "test_summary.json")
        source_arch = source_summary.get("architecture_key", source_summary.get("model_slug", source_run.parent.name))
        candidates: list[Path] = []
        for indomain_run in indomain_runs:
            indomain_summary = read_json(indomain_run / "tables" / "test_summary.json")
            indomain_arch = indomain_summary.get("architecture_key", indomain_summary.get("model_slug", indomain_run.parent.name))
            if allow_architecture_mismatch or indomain_arch == source_arch:
                candidates.append(indomain_run)
        candidates = sorted(candidates, key=lambda path: clean_float(read_json(path / "tables" / "test_summary.json").get("mIoU")), reverse=True)
        if candidates:
            pairs.append((source_run, candidates[0]))
        else:
            skipped.append(
                {
                    "source_model_slug": source_summary.get("model_slug", source_run.parent.name),
                    "source_architecture_key": source_arch,
                    "reason": "no_matching_indomain_architecture",
                }
            )
    return pairs, skipped


def row_by_image(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["image_name"]): row for row in rows}


def save_panel(source_run: Path, indomain_run: Path, selected_rows: list[dict[str, Any]], keep_ids: list[int], class_names: list[str], ignore_index: int, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_summary = read_json(source_run / "tables" / "test_summary.json")
    indomain_summary = read_json(indomain_run / "tables" / "test_summary.json")
    rows = len(selected_rows)
    fig, axes = plt.subplots(rows, 4, figsize=(17, 3.5 * rows), squeeze=False)
    for row_idx, row in enumerate(selected_rows):
        image_name = row["image_name"]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8)
        gt = np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.int64)
        source_pred = np.asarray(Image.open(source_run / "predictions" / image_name).convert("L"), dtype=np.int64)
        indomain_pred = np.asarray(Image.open(indomain_run / "predictions" / image_name).convert("L"), dtype=np.int64)
        panels = [
            image,
            colorize_train_ids(gt, class_names, ignore_index, keep_ids),
            colorize_train_ids(source_pred, class_names, ignore_index, keep_ids),
            colorize_train_ids(indomain_pred, class_names, ignore_index, keep_ids),
        ]
        titles = [
            "Image",
            "Ground truth",
            f"{source_summary.get('source_dataset', 'External')} source-only",
            "SPIN-UV trained",
        ]
        for col_idx, (panel, title) in enumerate(zip(panels, titles)):
            axes[row_idx, col_idx].imshow(panel)
            axes[row_idx, col_idx].set_title(title, fontsize=10)
            axes[row_idx, col_idx].axis("off")
        label = f"{image_name}\ngain={clean_float(row['delta_miou']):.3f}"
        axes[row_idx, 0].set_ylabel(label, fontsize=8)
    title = (
        f"External source-only vs SPIN-UV trained "
        f"({source_summary.get('display_name', source_run.parent.name)} vs {indomain_summary.get('display_name', indomain_run.parent.name)})"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def ids_from_names(class_names: list[str], names: list[str], fallback_ids: list[int]) -> list[int]:
    label2id = {name: idx for idx, name in enumerate(class_names)}
    ids = [label2id[name] for name in names if name in label2id]
    return ids if ids else fallback_ids


def image_miou_for_ids(pred: np.ndarray, target: np.ndarray, keep_ids: list[int], ignore_index: int) -> float:
    valid_gt = target != ignore_index
    values: list[float] = []
    for class_id in keep_ids:
        gt_c = valid_gt & (target == class_id)
        pred_c = valid_gt & (pred == class_id)
        intersection = int(np.logical_and(gt_c, pred_c).sum())
        union = int(np.logical_or(gt_c, pred_c).sum())
        if union > 0:
            values.append(intersection / union)
    return float(np.mean(values)) if values else float("nan")


def compare_one_source(
    source_run: Path,
    indomain_run: Path,
    split: str,
    fallback_keep_ids: list[int],
    class_names: list[str],
    ignore_index: int,
    top_k: int,
) -> dict[str, Any]:
    source_summary = read_json(source_run / "tables" / f"{split}_summary.json")
    indomain_summary = read_json(indomain_run / "tables" / f"{split}_summary.json")
    source_rows = row_by_image(read_csv(source_run / "tables" / f"{split}_per_image_iou.csv"))
    indomain_rows = row_by_image(read_csv(indomain_run / "tables" / f"{split}_per_image_iou.csv"))
    source_analysis_names = source_summary.get("matched_analysis_classes") or source_summary.get("analysis_classes") or []
    if not isinstance(source_analysis_names, list):
        source_analysis_names = []
    keep_ids = ids_from_names(class_names, [str(name) for name in source_analysis_names], fallback_keep_ids)
    keep_names = [class_names[idx] for idx in keep_ids]
    selected: list[dict[str, Any]] = []
    for image_name, source_row in source_rows.items():
        indomain_row = indomain_rows.get(image_name)
        if not indomain_row:
            continue
        target = np.asarray(Image.open(source_row["mask_path"]).convert("L"), dtype=np.int64)
        source_pred = np.asarray(Image.open(source_run / "predictions" / image_name).convert("L"), dtype=np.int64)
        indomain_pred = np.asarray(Image.open(indomain_run / "predictions" / image_name).convert("L"), dtype=np.int64)
        source_miou = image_miou_for_ids(source_pred, target, keep_ids, ignore_index)
        indomain_miou = image_miou_for_ids(indomain_pred, target, keep_ids, ignore_index)
        if not math.isfinite(source_miou) or not math.isfinite(indomain_miou):
            continue
        selected.append(
            {
                "image_name": image_name,
                "image_path": source_row["image_path"],
                "mask_path": source_row["mask_path"],
                "source_model_slug": source_summary.get("model_slug", source_run.parent.name),
                "source_dataset": source_summary.get("source_dataset", ""),
                "source_image_miou": source_miou,
                "indomain_model_slug": indomain_summary.get("model_slug", indomain_run.parent.name),
                "indomain_image_miou": indomain_miou,
                "delta_miou": indomain_miou - source_miou,
                "analysis_classes": ",".join(keep_names),
                "selection_rule": "largest_positive_indomain_minus_source_miou_on_source_matched_labels",
            }
        )
    selected = sorted(selected, key=lambda row: clean_float(row["delta_miou"]), reverse=True)[:top_k]
    safe_slug = str(source_summary.get("model_slug", source_run.parent.name)).replace("/", "_")
    selection_path = E2_ROOT / "tables" / f"experiment_02_comparison_selection_{safe_slug}.csv"
    output_path = E2_ROOT / "figures" / f"experiment_02_comparison_panel_{safe_slug}.png"
    write_csv(selection_path, selected)
    if selected:
        save_panel(source_run, indomain_run, selected, keep_ids, class_names, ignore_index, output_path)
    return {
        "source_model_slug": source_summary.get("model_slug", source_run.parent.name),
        "source_dataset": source_summary.get("source_dataset", ""),
        "indomain_model_slug": indomain_summary.get("model_slug", indomain_run.parent.name),
        "selected_images": len(selected),
        "analysis_classes": keep_names,
        "selection_csv": selection_path.as_posix(),
        "figure": output_path.as_posix() if selected else "",
    }


def main() -> None:
    args = parse_args()
    class_names, ignore_index = load_class_names()
    keep_ids = analysis_ids(class_names, args.analysis_classes)
    run_dirs = latest_run_dirs(args.run_tag)
    source_indomain_pairs, skipped = choose_runs(run_dirs, args.source_slug, args.indomain_slug, args.allow_architecture_mismatch)
    results: list[dict[str, Any]] = []
    for source_run, indomain_run in source_indomain_pairs:
        results.append(compare_one_source(source_run, indomain_run, args.split, keep_ids, class_names, ignore_index, args.top_k))
    status = {
        "source_runs": len(source_indomain_pairs) + len(skipped),
        "matched_pairs": len(source_indomain_pairs),
        "skipped_sources": skipped,
        "comparison_panels": results,
        "analysis_classes": [class_names[idx] for idx in keep_ids],
    }
    (E2_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with (E2_ROOT / "logs" / "experiment_02_comparison_panel_status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
        f.write("\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

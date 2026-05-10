#!/usr/bin/env python
"""Collect Experiment 01 multi-model results into sorted tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", default="e1_models")
    parser.add_argument("--output-prefix", default="experiment_01_model")
    parser.add_argument(
        "--model-list",
        default="",
        help="Optional whitespace-separated model slugs to collect.",
    )
    parser.add_argument(
        "--run-name",
        default="full_8k",
        help="Run directory name under each model's runs/ folder.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def model_dirs(models_root: Path, model_list: str = "") -> list[Path]:
    requested = [item for item in model_list.split() if item]
    if requested:
        return [models_root / slug for slug in requested if (models_root / slug).is_dir()]
    return sorted([path for path in models_root.iterdir() if path.is_dir()])


def find_run_dir(model_dir: Path, run_name: str) -> Path | None:
    full_dir = model_dir / "runs" / run_name
    if (full_dir / "tables" / "test_summary.json").exists():
        return full_dir
    candidates = sorted(model_dir.glob(f"runs/{run_name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "tables" / "test_summary.json").exists():
            return candidate
    return None


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if config_path.exists():
        return read_json(config_path)
    return {}


def best_val_from_log(log_rows: list[dict[str, Any]]) -> tuple[float | None, int | None]:
    best_value = None
    best_iter = None
    for row in log_rows:
        if row.get("event") == "val" and row.get("mIoU") is not None:
            value = float(row["mIoU"])
            if best_value is None or value > best_value:
                best_value = value
                best_iter = int(row.get("iteration", -1))
    return best_value, best_iter


def collect_model(model_dir: Path, run_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config(model_dir)
    run_dir = find_run_dir(model_dir, run_name)
    slug = config.get("model_slug", model_dir.name)
    display = config.get("display_name", slug)
    if run_dir is None:
        summary = {
            "rank": "",
            "model_slug": slug,
            "display_name": display,
            "run_dir": "",
            "framework": config.get("framework", ""),
            "model_name": config.get("model_name", ""),
            "model_id": config.get("model_id", ""),
            "initialization": config.get("initialization", ""),
            "hf_from_config": config.get("hf_from_config", ""),
            "encoder_name": config.get("encoder_name", ""),
            "encoder_weights": config.get("encoder_weights", ""),
            "best_val_miou": "",
            "best_val_iteration": "",
            "test_miou": "",
            "test_mean_accuracy": "",
            "test_pixel_accuracy": "",
            "total_params_m": "",
            "trainable_params_m": "",
            "max_iters": config.get("max_iters", ""),
            "crop_size": f"{config.get('crop_height', '')}x{config.get('crop_width', '')}",
            "batch_size": config.get("batch_size", ""),
            "grad_accum_steps": config.get("grad_accum_steps", config.get("gradient_accumulation_steps", 1)),
            "effective_batch_size": "",
            "train_crop_exposures": "",
            "optimizer": config.get("optimizer", ""),
            "base_lr": config.get("base_lr", ""),
            "weight_decay": config.get("weight_decay", ""),
            "seed": config.get("seed", ""),
            "amp": config.get("amp", ""),
            "status": "missing",
        }
        return summary, [], [], []

    test_summary = read_json(run_dir / "tables" / "test_summary.json")
    model_info = read_json(run_dir / "model_info.json") if (run_dir / "model_info.json").exists() else {}
    args = read_json(run_dir / "args.json") if (run_dir / "args.json").exists() else {}
    logs = read_jsonl(run_dir / "logs" / "train_log.jsonl")
    best_val_miou, best_val_iteration = best_val_from_log(logs)

    total_params = int(test_summary.get("total_params", model_info.get("total_params", 0) or 0))
    trainable_params = int(test_summary.get("trainable_params", model_info.get("trainable_params", 0) or 0))
    batch_size = int(args.get("batch_size", config.get("batch_size", 0)) or 0)
    grad_accum_steps = int(args.get("grad_accum_steps", config.get("grad_accum_steps", config.get("gradient_accumulation_steps", 1))) or 1)
    max_iters = int(args.get("max_iters", config.get("max_iters", 0)) or 0)
    summary = {
        "rank": "",
        "model_slug": slug,
        "display_name": display,
        "run_dir": run_dir.as_posix(),
        "framework": args.get("framework", config.get("framework", "")),
        "model_name": args.get("model_name", config.get("model_name", "")),
        "model_id": args.get("model_id", config.get("model_id", "")),
        "initialization": args.get("initialization", config.get("initialization", "")),
        "hf_from_config": args.get("hf_from_config", config.get("hf_from_config", "")),
        "encoder_name": args.get("encoder_name", config.get("encoder_name", "")),
        "encoder_weights": args.get("encoder_weights", config.get("encoder_weights", "")),
        "best_val_miou": best_val_miou if best_val_miou is not None else "",
        "best_val_iteration": best_val_iteration if best_val_iteration is not None else "",
        "test_miou": float(test_summary.get("mIoU", float("nan"))),
        "test_mean_accuracy": float(test_summary.get("mean_accuracy", float("nan"))),
        "test_pixel_accuracy": float(test_summary.get("pixel_accuracy", float("nan"))),
        "total_params_m": total_params / 1_000_000,
        "trainable_params_m": trainable_params / 1_000_000,
        "max_iters": max_iters,
        "crop_size": f"{args.get('crop_height', config.get('crop_height', ''))}x{args.get('crop_width', config.get('crop_width', ''))}",
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": int(batch_size * grad_accum_steps),
        "train_crop_exposures": int(max_iters * batch_size * grad_accum_steps),
        "optimizer": args.get("optimizer", config.get("optimizer", "")),
        "base_lr": args.get("base_lr", config.get("base_lr", "")),
        "weight_decay": args.get("weight_decay", config.get("weight_decay", "")),
        "seed": args.get("seed", config.get("seed", "")),
        "amp": args.get("amp", config.get("amp", "")),
        "status": "complete",
    }

    per_class_rows: list[dict[str, Any]] = []
    class_path = run_dir / "tables" / "test_per_class_iou.csv"
    if class_path.exists():
        with class_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row_out = dict(row)
                row_out.update(
                    {
                        "model_slug": slug,
                        "display_name": display,
                        "run_dir": run_dir.as_posix(),
                        "split": "test",
                    }
                )
                per_class_rows.append(row_out)

    per_image_rows: list[dict[str, Any]] = []
    image_path = run_dir / "tables" / "test_per_image_iou.csv"
    if image_path.exists():
        with image_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row_out = dict(row)
                row_out.update({"model_slug": slug, "display_name": display, "run_dir": run_dir.as_posix()})
                per_image_rows.append(row_out)

    curve_rows = []
    for row in logs:
        event = row.get("event")
        if event in {"train", "val", "final_val", "final_test"}:
            row_out = {
                "model_slug": slug,
                "display_name": display,
                "event": event,
                "iteration": row.get("iteration", ""),
                "loss": row.get("loss", ""),
                "mIoU": row.get("mIoU", ""),
                "lr": row.get("lr", ""),
            }
            curve_rows.append(row_out)

    return summary, per_class_rows, per_image_rows, curve_rows


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figures(summary_df: pd.DataFrame, class_df: pd.DataFrame, curve_df: pd.DataFrame, output_prefix: str) -> None:
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    complete = summary_df[summary_df["status"] == "complete"].copy()
    complete["test_miou"] = pd.to_numeric(complete["test_miou"], errors="coerce")
    complete["total_params_m"] = pd.to_numeric(complete["total_params_m"], errors="coerce")
    complete = complete.sort_values("test_miou", ascending=False)

    if not complete.empty:
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(complete))))
        ax.barh(complete["display_name"], complete["test_miou"], color="#4f7cac")
        ax.invert_yaxis()
        ax.set_xlabel("Test mIoU")
        ax.set_title("Experiment 01 model test mIoU")
        ax.set_xlim(0, max(1.0, complete["test_miou"].max() * 1.1))
        ax.grid(axis="x", alpha=0.25)
        for y_idx, value in enumerate(complete["test_miou"]):
            ax.text(value + 0.01, y_idx, f"{value:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{output_prefix}_test_miou_bar.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(complete["total_params_m"], complete["test_miou"], color="#4f7cac", s=60)
        for _, row in complete.iterrows():
            ax.annotate(row["display_name"], (row["total_params_m"], row["test_miou"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel("Test mIoU")
        ax.set_title("Parameters vs test mIoU")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{output_prefix}_params_vs_miou_scatter.png", dpi=180)
        plt.close(fig)

    if not class_df.empty:
        heat = class_df.copy()
        heat["iou"] = pd.to_numeric(heat["iou"], errors="coerce")
        pivot = heat.pivot_table(index="display_name", columns="class_name", values="iou", aggfunc="mean")
        if not complete.empty:
            order = complete["display_name"].tolist()
            pivot = pivot.reindex([name for name in order if name in pivot.index])
        fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(pivot.columns)), max(3, 0.55 * len(pivot.index))))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
        ax.set_title("Experiment 01 per-class IoU")
        ax.set_xlabel("Class")
        ax.set_ylabel("Model")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{output_prefix}_per_class_iou_heatmap.png", dpi=180)
        plt.close(fig)

    if not curve_df.empty:
        curves = curve_df.copy()
        curves["iteration"] = pd.to_numeric(curves["iteration"], errors="coerce")
        curves["loss"] = pd.to_numeric(curves["loss"], errors="coerce")
        curves["mIoU"] = pd.to_numeric(curves["mIoU"], errors="coerce")
        train_curves = curves[curves["event"] == "train"].dropna(subset=["iteration", "loss"])
        if not train_curves.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            for name, group in train_curves.groupby("display_name"):
                group = group.sort_values("iteration")
                ax.plot(group["iteration"], group["loss"], label=name)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Training loss")
            ax.set_title("Experiment 01 training loss")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"{output_prefix}_train_loss_curves.png", dpi=180)
            plt.close(fig)

        val_curves = curves[curves["event"] == "val"].dropna(subset=["iteration", "mIoU"])
        if not val_curves.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            for name, group in val_curves.groupby("display_name"):
                group = group.sort_values("iteration")
                ax.plot(group["iteration"], group["mIoU"], marker="o", label=name)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Validation mIoU")
            ax.set_title("Experiment 01 validation mIoU")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"{output_prefix}_val_miou_curves.png", dpi=180)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    import pandas as pd

    models_root = Path(args.models_root)
    if not models_root.is_absolute():
        models_root = ROOT / models_root

    summaries = []
    per_class = []
    per_image = []
    curves = []
    artifact_index = []
    for model_dir in model_dirs(models_root, args.model_list):
        summary, class_rows, image_rows, curve_rows = collect_model(model_dir, args.run_name)
        summaries.append(summary)
        per_class.extend(class_rows)
        per_image.extend(image_rows)
        curves.extend(curve_rows)
        if summary["run_dir"]:
            artifact_index.extend(
                [
                    {"model_slug": summary["model_slug"], "artifact": "test_summary", "path": f"{summary['run_dir']}/tables/test_summary.json"},
                    {"model_slug": summary["model_slug"], "artifact": "test_per_class_iou", "path": f"{summary['run_dir']}/tables/test_per_class_iou.csv"},
                    {"model_slug": summary["model_slug"], "artifact": "model_info", "path": f"{summary['run_dir']}/model_info.json"},
                    {"model_slug": summary["model_slug"], "artifact": "best_checkpoint", "path": f"{summary['run_dir']}/checkpoints/best_miou.pt"},
                ]
            )

    summaries = sorted(
        summaries,
        key=lambda row: float(row["test_miou"]) if row["test_miou"] not in ("", None) and math.isfinite(float(row["test_miou"])) else -1.0,
        reverse=True,
    )
    rank = 1
    for row in summaries:
        if row["status"] == "complete":
            row["rank"] = rank
            rank += 1

    table_dir = ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    save_csv(summaries, table_dir / f"{args.output_prefix}_summary.csv")
    save_csv(per_class, table_dir / f"{args.output_prefix}_per_class_iou_long.csv")
    save_csv(per_image, table_dir / f"{args.output_prefix}_per_image_iou.csv")
    save_csv(curves, table_dir / f"{args.output_prefix}_training_curves.csv")
    save_csv(artifact_index, table_dir / f"{args.output_prefix}_artifact_index.csv")

    if per_class:
        class_df = pd.DataFrame(per_class)
        class_df["iou"] = pd.to_numeric(class_df["iou"], errors="coerce")
        wide = class_df.pivot_table(index=["class_id", "class_name"], columns="model_slug", values="iou", aggfunc="mean").reset_index()
        wide.to_csv(table_dir / f"{args.output_prefix}_per_class_iou_wide.csv", index=False)
    else:
        (table_dir / f"{args.output_prefix}_per_class_iou_wide.csv").write_text("", encoding="utf-8")

    summary_df = pd.DataFrame(summaries)
    class_df = pd.DataFrame(per_class)
    curve_df = pd.DataFrame(curves)
    save_figures(summary_df, class_df, curve_df, args.output_prefix)
    print(json.dumps({"models": len(summaries), "complete": int((summary_df["status"] == "complete").sum()) if not summary_df.empty else 0}, indent=2))


if __name__ == "__main__":
    main()

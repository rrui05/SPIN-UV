#!/usr/bin/env python
"""Collect Experiment 02 source-only runs and optional E1 references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


E2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = E2_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--include-e1-root", default="")
    parser.add_argument("--include-e1-original-reference", action="store_true")
    parser.add_argument("--output-prefix", default="experiment_02")
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
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
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


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [base / path, E2_ROOT / path, EXPERIMENT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base / path


def find_e2_run_dirs(runs_root: Path, run_tag: str | None, split: str) -> list[Path]:
    if not runs_root.exists():
        return []
    dirs: list[Path] = []
    for model_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if run_tag:
            candidate = model_dir / run_tag
            if (candidate / "tables" / f"{split}_summary.json").exists():
                dirs.append(candidate)
            continue
        candidates = sorted(
            [
                path
                for path in model_dir.iterdir()
                if path.is_dir() and (path / "tables" / f"{split}_summary.json").exists()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            dirs.append(candidates[0])
    return dirs


def collect_e2_rows(run_dirs: list[Path], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "tables" / f"{split}_summary.json"
        summary = read_json(summary_path)
        summary_rows.append(summary)
        model_slug = summary.get("model_slug", run_dir.parent.name)
        display_name = summary.get("display_name", model_slug)
        for row in read_csv(run_dir / "tables" / f"{split}_per_class_iou.csv"):
            row_out = dict(row)
            row_out.update(
                {
                    "model_slug": model_slug,
                    "display_name": display_name,
                    "experiment_role": summary.get("experiment_role", "source_only_public_checkpoint"),
                    "source_dataset": summary.get("source_dataset", ""),
                    "run_dir": run_dir.as_posix(),
                    "split": split,
                }
            )
            class_rows.append(row_out)
        artifact_rows.extend(
            [
                {"model_slug": model_slug, "artifact": "summary", "path": summary_path.as_posix()},
                {"model_slug": model_slug, "artifact": "per_class_iou", "path": (run_dir / "tables" / f"{split}_per_class_iou.csv").as_posix()},
                {"model_slug": model_slug, "artifact": "per_image_iou", "path": (run_dir / "tables" / f"{split}_per_image_iou.csv").as_posix()},
                {"model_slug": model_slug, "artifact": "label_mapping_report", "path": (run_dir / "label_mapping_report.json").as_posix()},
                {"model_slug": model_slug, "artifact": "worst_predictions", "path": (run_dir / "figures" / f"{split}_worst_predictions.png").as_posix()},
            ]
        )
    return summary_rows, class_rows, artifact_rows


def find_e1_run_dir(model_dir: Path) -> Path | None:
    preferred = model_dir / "runs" / "full_8k"
    if (preferred / "tables" / "test_summary.json").exists():
        return preferred
    candidates = sorted(
        [path for path in model_dir.glob("runs/full_8k*") if (path / "tables" / "test_summary.json").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def collect_e1_reference_rows(e1_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not e1_root.exists():
        return [], [], []
    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in e1_root.iterdir() if path.is_dir()):
        config = read_json(model_dir / "config.json") if (model_dir / "config.json").exists() else {}
        run_dir = find_e1_run_dir(model_dir)
        if run_dir is None:
            continue
        test_summary = read_json(run_dir / "tables" / "test_summary.json")
        args_path = run_dir / "args.json"
        run_args = read_json(args_path) if args_path.exists() else {}
        model_slug = config.get("model_slug", model_dir.name)
        display_name = config.get("display_name", model_slug)
        total_params = int(test_summary.get("total_params", 0) or 0)
        trainable_params = int(test_summary.get("trainable_params", 0) or 0)
        summary = {
            "status": "complete",
            "experiment": "experiment_01_spinuv_baseline",
            "experiment_role": "spinuv_in_domain_trained",
            "model_slug": f"e1_{model_slug}",
            "display_name": f"E1 {display_name}",
            "model_id": run_args.get("model_id", config.get("model_id", "")),
            "model_url": "",
            "framework": run_args.get("framework", config.get("framework", "")),
            "source_dataset": "SPIN-UV",
            "split": "test",
            "run_tag": "",
            "run_dir": run_dir.as_posix(),
            "image_count": "",
            "analysis_class_count": "",
            "matched_analysis_class_count": "",
            "matched_analysis_classes": "",
            "unsupported_analysis_classes": "",
            "mIoU": test_summary.get("mIoU", ""),
            "mean_accuracy": test_summary.get("mean_accuracy", ""),
            "pixel_accuracy": test_summary.get("pixel_accuracy", ""),
            "analysis_pixels": "",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_m": total_params / 1_000_000 if total_params else "",
            "trainable_params_m": trainable_params / 1_000_000 if trainable_params else "",
            "eval_seconds": "",
        }
        summary_rows.append(summary)
        for row in read_csv(run_dir / "tables" / "test_per_class_iou.csv"):
            row_out = dict(row)
            row_out.update(
                {
                    "model_slug": summary["model_slug"],
                    "display_name": summary["display_name"],
                    "experiment_role": "spinuv_in_domain_trained",
                    "source_dataset": "SPIN-UV",
                    "run_dir": run_dir.as_posix(),
                    "split": "test",
                }
            )
            class_rows.append(row_out)
        artifact_rows.extend(
            [
                {"model_slug": summary["model_slug"], "artifact": "summary", "path": (run_dir / "tables" / "test_summary.json").as_posix()},
                {"model_slug": summary["model_slug"], "artifact": "per_class_iou", "path": (run_dir / "tables" / "test_per_class_iou.csv").as_posix()},
                {"model_slug": summary["model_slug"], "artifact": "best_checkpoint", "path": (run_dir / "checkpoints" / "best_miou.pt").as_posix()},
            ]
        )
    return summary_rows, class_rows, artifact_rows


def sort_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: clean_float(row.get("mIoU")), reverse=True)
    rank = 1
    for row in rows:
        if math.isfinite(clean_float(row.get("mIoU"))):
            row["rank"] = rank
            rank += 1
        else:
            row["rank"] = ""
    return rows


def save_figures(summary_rows: list[dict[str, Any]], class_rows: list[dict[str, Any]], prefix: str) -> None:
    if not summary_rows and not class_rows:
        return
    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        log_path = E2_ROOT / "logs" / f"{prefix}_figure_generation_error.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(exc), encoding="utf-8")
        return

    fig_dir = E2_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["mIoU_num"] = pd.to_numeric(summary_df["mIoU"], errors="coerce")
        summary_df["total_params_m_num"] = pd.to_numeric(summary_df["total_params_m"], errors="coerce")
        plot_df = summary_df.dropna(subset=["mIoU_num"]).sort_values("mIoU_num", ascending=False)
        if not plot_df.empty:
            colors = [
                "#2a9d8f" if role == "spinuv_in_domain_trained" else "#4f7cac"
                for role in plot_df["experiment_role"].tolist()
            ]
            fig, ax = plt.subplots(figsize=(9, max(4, 0.52 * len(plot_df))))
            bars = ax.barh(plot_df["display_name"], plot_df["mIoU_num"], color=colors)
            ax.invert_yaxis()
            ax.set_xlabel("Test mIoU")
            ax.set_title("Experiment 02 transfer comparison on SPIN-UV test")
            ax.set_xlim(0, max(1.0, float(plot_df["mIoU_num"].max()) * 1.1))
            ax.grid(axis="x", alpha=0.25)
            for bar, value in zip(bars, plot_df["mIoU_num"]):
                ax.text(min(float(value) + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{prefix}_transfer_miou_bar.png", dpi=180)
            plt.close(fig)

            scatter_df = plot_df.dropna(subset=["total_params_m_num"])
            if not scatter_df.empty:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.scatter(scatter_df["total_params_m_num"], scatter_df["mIoU_num"], color="#4f7cac", s=60)
                for _, row in scatter_df.iterrows():
                    ax.annotate(row["display_name"], (row["total_params_m_num"], row["mIoU_num"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
                ax.set_xlabel("Parameters (M)")
                ax.set_ylabel("Test mIoU")
                ax.set_title("Parameters vs transfer mIoU")
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(fig_dir / f"{prefix}_params_vs_miou.png", dpi=180)
                plt.close(fig)

    class_df = pd.DataFrame(class_rows)
    if not class_df.empty:
        class_df["iou_num"] = pd.to_numeric(class_df["iou"], errors="coerce")
        pivot = class_df.pivot_table(index="display_name", columns="class_name", values="iou_num", aggfunc="mean")
        if summary_rows:
            order = [row["display_name"] for row in summary_rows]
            pivot = pivot.reindex([name for name in order if name in pivot.index])
        fig, ax = plt.subplots(figsize=(max(9, 0.65 * len(pivot.columns)), max(3.5, 0.55 * len(pivot.index))))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
        ax.set_title("Experiment 02 per-class IoU")
        ax.set_xlabel("Class")
        ax.set_ylabel("Model")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{prefix}_per_class_iou_heatmap.png", dpi=180)
        plt.close(fig)


def write_wide_iou(path: Path, class_rows: list[dict[str, Any]]) -> None:
    if not class_rows:
        path.write_text("", encoding="utf-8")
        return
    class_keys: dict[tuple[str, str], dict[str, Any]] = {}
    model_names: list[str] = []
    for row in class_rows:
        model_name = str(row.get("display_name", row.get("model_slug", "")))
        if model_name not in model_names:
            model_names.append(model_name)
        key = (str(row.get("class_id", "")), str(row.get("class_name", "")))
        class_keys.setdefault(key, {"class_id": key[0], "class_name": key[1]})
        class_keys[key][model_name] = row.get("iou", "")
    rows = list(class_keys.values())
    rows = sorted(rows, key=lambda row: int(row["class_id"]) if str(row["class_id"]).isdigit() else 10**9)
    fieldnames = ["class_id", "class_name", *model_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs_root = resolve_path(args.runs_root, E2_ROOT)
    e2_run_dirs = find_e2_run_dirs(runs_root, args.run_tag, args.split)
    e2_summaries, e2_class_rows, e2_artifacts = collect_e2_rows(e2_run_dirs, args.split)
    if args.include_e1_original_reference and args.include_e1_root:
        e1_root = resolve_path(args.include_e1_root, E2_ROOT)
        e1_summaries, e1_class_rows, e1_artifacts = collect_e1_reference_rows(e1_root)
    else:
        e1_summaries, e1_class_rows, e1_artifacts = [], [], []

    summaries = sort_summaries(e2_summaries + e1_summaries)
    class_rows = e2_class_rows + e1_class_rows
    artifact_rows = e2_artifacts + e1_artifacts

    tables_dir = E2_ROOT / "tables"
    write_csv(tables_dir / f"{args.output_prefix}_transfer_matrix.csv", summaries)
    write_csv(tables_dir / f"{args.output_prefix}_per_class_iou_long.csv", class_rows)
    write_csv(tables_dir / f"{args.output_prefix}_artifact_index.csv", artifact_rows)
    write_csv(tables_dir / f"{args.output_prefix}_e1_reference_rows.csv", e1_summaries)

    write_wide_iou(tables_dir / f"{args.output_prefix}_per_class_iou_wide.csv", class_rows)

    save_figures(summaries, class_rows, args.output_prefix)
    status = {
        "e2_complete_runs": len(e2_summaries),
        "e1_reference_runs": len(e1_summaries),
        "rows": len(summaries),
        "output": (tables_dir / f"{args.output_prefix}_transfer_matrix.csv").as_posix(),
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

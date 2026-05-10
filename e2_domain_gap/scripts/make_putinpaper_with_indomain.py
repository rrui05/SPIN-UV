#!/usr/bin/env python
"""Rebuild the paper qualitative E2 panel with a chosen SPIN-UV in-domain row.

This script keeps the existing source-only rows and selected images, but swaps
the final SPIN-UV prediction column to another completed in-domain run, such as
`e1_segformer_b5_imagenet1k`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spinuv_palette import colorize_train_ids

E2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = E2_ROOT.parent
DEFAULT_SOURCES = [
    "segformer_b5_ade20k_source_only",
    "segformer_b5_cityscapes_source_only",
    "segformer_b5_idd_source_only",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indomain-slug", default="e1_segformer_b5_imagenet1k")
    parser.add_argument("--indomain-run-tag", default=None)
    parser.add_argument("--source-run-tag", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--source-slugs", default=" ".join(DEFAULT_SOURCES))
    parser.add_argument("--selection-prefix", default="tables/experiment_02_comparison_selection")
    parser.add_argument("--output", default="figures/experiment_02_comparison_put_in_paper.png")
    parser.add_argument("--paper-output", default="figures/experiment_02_comparison_put_in_paper.png")
    parser.add_argument("--base-figure", default=None)
    parser.add_argument("--selection-output", default="tables/experiment_02_comparison_put_in_paper_imagenet1k_selection.csv")
    parser.add_argument("--status-output", default="logs/experiment_02_comparison_put_in_paper_imagenet1k_status.json")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--with-text", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def e2_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else E2_ROOT / path


def resolve_existing_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if not path.is_absolute():
        for candidate in (E2_ROOT / path, EXPERIMENT_ROOT / path):
            if candidate.exists():
                return candidate
    return path


def find_run(model_slug: str, run_tag: str | None, split: str) -> Path:
    root = E2_ROOT / "runs" / model_slug
    summary = Path("tables") / f"{split}_summary.json"
    if run_tag:
        candidate = root / run_tag
        if (candidate / summary).exists():
            return candidate
        raise FileNotFoundError(f"No completed {model_slug} run for tag {run_tag}")
    candidates = sorted(
        [path for path in root.glob("*") if path.is_dir() and (path / summary).exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No completed {model_slug} run")
    return candidates[0]


def load_class_meta() -> tuple[list[str], int]:
    candidates = [
        EXPERIMENT_ROOT / "data" / "processed" / "spinuv_semantic" / "meta" / "classes.json",
    ]
    for path in candidates:
        if path.exists():
            payload = read_json(path)
            break
    else:
        raise FileNotFoundError("Could not find SPIN-UV classes.json")
    ordered = sorted(payload["classes"], key=lambda row: int(row["train_id"]))
    return [str(row["name"]) for row in ordered], int(payload.get("ignore_value", 255))


def pred_path(run_dir: Path, image_name: str) -> Path:
    path = run_dir / "predictions" / image_name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def first_selection_row(source_slug: str, selection_prefix: str) -> dict[str, str]:
    path = e2_path(f"{selection_prefix}_{source_slug}.csv")
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"Selection CSV is empty: {path}")
    return rows[0]


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_panel(array: np.ndarray, width: int, height: int, nearest: bool = False) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8))
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return image.resize((width, height), resampling)


def save_canvas(canvas: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    canvas.convert("RGB").save(output.with_suffix(".pdf"), resolution=300.0)


def find_segments(values: np.ndarray, threshold: float, min_length: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(values > threshold):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_length:
                segments.append((start, idx))
            start = None
    if start is not None and len(values) - start >= min_length:
        segments.append((start, len(values)))
    return segments


def detect_grid_boxes(base: Image.Image, row_count: int, col_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    arr = np.asarray(base.convert("RGB"), dtype=np.uint8)
    nonwhite = np.any(arr < 245, axis=2)
    row_counts = nonwhite.sum(axis=1)
    row_segments = find_segments(row_counts, base.width * 0.2, max(40, base.height // 20))
    row_segments = sorted(row_segments, key=lambda item: item[1] - item[0], reverse=True)[:row_count]
    row_segments = sorted(row_segments)
    if len(row_segments) != row_count:
        raise RuntimeError(f"Could not detect {row_count} content rows in {base.size}")

    row_mask = np.zeros(nonwhite.shape[0], dtype=bool)
    for start, end in row_segments:
        row_mask[start:end] = True
    col_counts = nonwhite[row_mask].sum(axis=0)
    col_segments = find_segments(col_counts, int(row_mask.sum() * 0.2), max(40, base.width // 40))
    col_segments = sorted(col_segments, key=lambda item: item[1] - item[0], reverse=True)[:col_count]
    col_segments = sorted(col_segments)
    if len(col_segments) != col_count:
        raise RuntimeError(f"Could not detect {col_count} content columns in {base.size}")
    return row_segments, col_segments


def save_spliced_panel(
    panel_rows: list[dict[str, Any]],
    indomain_run: Path,
    class_names: list[str],
    ignore_index: int,
    base_figure: Path,
    output: Path,
    paper_output: Path,
) -> None:
    base = Image.open(base_figure).convert("RGB")
    row_boxes, col_boxes = detect_grid_boxes(base, len(panel_rows), 4)
    target_col = col_boxes[-1]
    for row_idx, row in enumerate(panel_rows):
        image_name = row["image_name"]
        pred = np.asarray(Image.open(pred_path(indomain_run, image_name)).convert("L"), dtype=np.int64)
        color = colorize_train_ids(pred, class_names, ignore_index)
        x0, x1 = target_col
        y0, y1 = row_boxes[row_idx]
        tile = resize_panel(color, x1 - x0, y1 - y0, nearest=True)
        base.paste(tile, (x0, y0))
    save_canvas(base, output)
    if paper_output.resolve() != output.resolve():
        save_canvas(base, paper_output)


def choose_base_figure(base_arg: str | None, output: Path, paper_output: Path) -> Path | None:
    candidates: list[Path] = []
    if base_arg:
        candidates.append(e2_path(base_arg))
    else:
        candidates.extend(
            [
                output,
                paper_output,
                E2_ROOT / "figures" / "experiment_02_comparison_put_in_paper.png",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def save_panel(
    panel_rows: list[dict[str, Any]],
    indomain_run: Path,
    class_names: list[str],
    ignore_index: int,
    output: Path,
    paper_output: Path,
    panel_width: int,
    gap: int,
    with_text: bool,
) -> None:
    if not panel_rows:
        raise RuntimeError("No panel rows to render")
    cols = 4
    first_image = Image.open(panel_rows[0]["image_path"]).convert("RGB")
    aspect = first_image.height / first_image.width
    panel_height = int(round(panel_width * aspect))
    title_height = 32 if with_text else 0
    canvas_width = cols * panel_width + (cols - 1) * gap
    canvas_height = title_height + len(panel_rows) * panel_height + (len(panel_rows) - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    titles = ["Image", "Ground truth", "Source-only", "SPIN-UV trained"]
    if with_text:
        for col_idx, title in enumerate(titles):
            x = col_idx * (panel_width + gap)
            bbox = draw.textbbox((0, 0), title, font=font)
            text_x = x + (panel_width - (bbox[2] - bbox[0])) // 2
            draw.text((text_x, 5), title, fill=(0, 0, 0), font=font)
    for row_idx, row in enumerate(panel_rows):
        image_name = row["image_name"]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8)
        target = np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.int64)
        source_pred = np.asarray(Image.open(pred_path(row["source_run"], image_name)).convert("L"), dtype=np.int64)
        indomain_pred = np.asarray(Image.open(pred_path(indomain_run, image_name)).convert("L"), dtype=np.int64)
        panels = [
            image,
            colorize_train_ids(target, class_names, ignore_index),
            colorize_train_ids(source_pred, class_names, ignore_index),
            colorize_train_ids(indomain_pred, class_names, ignore_index),
        ]
        for col_idx, panel in enumerate(panels):
            nearest = col_idx > 0
            tile = resize_panel(panel, panel_width, panel_height, nearest=nearest)
            x = col_idx * (panel_width + gap)
            y = title_height + row_idx * (panel_height + gap)
            canvas.paste(tile, (x, y))
    save_canvas(canvas, output)
    if paper_output.resolve() != output.resolve():
        save_canvas(canvas, paper_output)


def main() -> None:
    args = parse_args()
    class_names, ignore_index = load_class_meta()
    source_slugs = [slug.strip() for slug in args.source_slugs.split() if slug.strip()]
    indomain_run = find_run(args.indomain_slug, args.indomain_run_tag, args.split)
    panel_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for source_slug in source_slugs:
        source_run = find_run(source_slug, args.source_run_tag, args.split)
        source_summary = read_json(source_run / "tables" / f"{args.split}_summary.json")
        selected = first_selection_row(source_slug, args.selection_prefix)
        image_name = selected["image_name"]
        image_path = resolve_existing_path(selected["image_path"])
        mask_path = resolve_existing_path(selected["mask_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        panel_rows.append(
            {
                "image_name": image_name,
                "image_path": image_path,
                "mask_path": mask_path,
                "source_slug": source_slug,
                "source_dataset": source_summary.get("source_dataset", source_slug),
                "source_run": source_run,
            }
        )
        selection_rows.append(
            {
                "source_slug": source_slug,
                "source_dataset": source_summary.get("source_dataset", source_slug),
                "image_name": image_name,
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "source_run": source_run.as_posix(),
                "indomain_run": indomain_run.as_posix(),
                "selection_source_csv": e2_path(f"{args.selection_prefix}_{source_slug}.csv").as_posix(),
            }
        )

    output = e2_path(args.output)
    paper_output = e2_path(args.paper_output)
    base_figure = choose_base_figure(args.base_figure, output, paper_output)
    if base_figure is not None:
        save_spliced_panel(panel_rows, indomain_run, class_names, ignore_index, base_figure, output, paper_output)
        render_mode = "spliced_existing_putinpaper_figure"
    else:
        save_panel(panel_rows, indomain_run, class_names, ignore_index, output, paper_output, args.panel_width, args.gap, args.with_text)
        render_mode = "rebuilt_full_panel"
    write_csv(e2_path(args.selection_output), selection_rows)
    status = {
        "status": "complete",
        "render_mode": render_mode,
        "indomain_slug": args.indomain_slug,
        "indomain_run": indomain_run.as_posix(),
        "source_slugs": source_slugs,
        "rows": len(panel_rows),
        "figure": output.as_posix(),
        "figure_pdf": output.with_suffix(".pdf").as_posix(),
        "paper_figure": paper_output.as_posix(),
        "paper_figure_pdf": paper_output.with_suffix(".pdf").as_posix(),
        "selection_csv": e2_path(args.selection_output).as_posix(),
    }
    write_json(e2_path(args.status_output), status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

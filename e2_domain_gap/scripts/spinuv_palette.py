"""Shared SPIN-UV paper palette utilities.

The paper figures must keep a stable label-name to RGB mapping.  Using raw
train-id indexing alone is fragile when a model or export changes class order.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


PAPER_DEFAULT_PALETTE = np.array(
    [
        [120, 120, 120],
        [220, 20, 60],
        [70, 70, 70],
        [0, 0, 142],
        [153, 153, 153],
        [119, 11, 32],
        [220, 20, 60],
        [153, 153, 153],
        [255, 0, 0],
        [128, 64, 128],
        [190, 153, 153],
        [0, 0, 230],
        [244, 35, 232],
        [70, 130, 180],
        [220, 220, 0],
        [0, 0, 90],
        [0, 0, 70],
        [107, 142, 35],
        [102, 102, 156],
    ],
    dtype=np.uint8,
)


PAPER_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "building": (70, 70, 70),
    "car": (0, 0, 142),
    "clutter": (153, 153, 153),
    "person": (220, 20, 60),
    "pole": (153, 153, 153),
    "road": (128, 64, 128),
    "roadblock": (190, 153, 153),
    "scooter": (0, 0, 230),
    "sidewalk": (244, 35, 232),
    "sky": (70, 130, 180),
    "tricycle": (0, 0, 90),
    "vegetation": (107, 142, 35),
    "wall": (102, 102, 156),
}

IGNORE_COLOR = np.array([42, 42, 42], dtype=np.uint8)


def paper_palette_for_classes(class_names: Iterable[str]) -> np.ndarray:
    names = [str(name) for name in class_names]
    fallback = np.resize(PAPER_DEFAULT_PALETTE, (len(names), 3)).astype(np.uint8)
    for idx, name in enumerate(names):
        color = PAPER_LABEL_COLORS.get(name)
        if color is not None:
            fallback[idx] = np.array(color, dtype=np.uint8)
    return fallback


def colorize_train_ids(
    mask: np.ndarray,
    class_names: Iterable[str],
    ignore_index: int,
    keep_ids: Iterable[int] | None = None,
) -> np.ndarray:
    palette = paper_palette_for_classes(class_names)
    out = np.full((*mask.shape, 3), IGNORE_COLOR, dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    if keep_ids is not None:
        valid &= np.isin(mask, np.array(list(keep_ids), dtype=np.int64))
    out[valid] = palette[mask[valid]]
    out[mask == ignore_index] = IGNORE_COLOR
    return out


def class_area_rows(mask: np.ndarray, class_names: Iterable[str], ignore_index: int) -> list[dict[str, object]]:
    names = [str(name) for name in class_names]
    valid = mask != ignore_index
    total = int(valid.sum())
    palette = paper_palette_for_classes(names)
    rows: list[dict[str, object]] = []
    if total == 0:
        return rows
    for class_id, name in enumerate(names):
        pixels = int((valid & (mask == class_id)).sum())
        if pixels == 0:
            continue
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": name,
                "pixels": pixels,
                "area_percent": float(pixels / total * 100.0),
                "rgb": [int(v) for v in palette[class_id].tolist()],
            }
        )
    return sorted(rows, key=lambda row: float(row["area_percent"]), reverse=True)

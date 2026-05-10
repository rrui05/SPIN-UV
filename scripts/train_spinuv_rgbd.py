#!/usr/bin/env python
"""Train lightweight RGB-D semantic-segmentation baselines on SPIN-UV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_spinuv_smp import (
    DEFAULT_PALETTE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ROOT,
    anchor_crop_boxes,
    autocast_context,
    build_optimizer,
    color_jitter,
    load_checkpoint,
    load_classes,
    load_config,
    make_grad_scaler,
    make_run_dir,
    metrics_from_confusion,
    poly_lr,
    preprocess_augment,
    resolve_analysis_class_ids,
    resolve_optional_class_ids,
    sample_crop_box,
    save_checkpoint,
    save_visual_grid,
    set_optimizer_lr,
    set_seed,
    update_confusion,
    write_jsonl,
    write_metrics,
)


SPINUV_NAME_RE = re.compile(r"^seq(?P<date>\d{8})_(?P<time>\d{6})_f(?P<frame>\d+)\.[^.]+$")


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    known, _ = pre_parser.parse_known_args()
    defaults = load_config(known.config)
    default_analysis_classes = defaults.get("analysis_classes", [])
    if isinstance(default_analysis_classes, list):
        default_analysis_classes = ",".join(default_analysis_classes)
    default_hard_classes = defaults.get("hard_example_classes", "all")
    if isinstance(default_hard_classes, list):
        default_hard_classes = ",".join(default_hard_classes)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--data-root", default=defaults.get("data_root", "data/processed/spinuv_semantic"))
    parser.add_argument("--raw-dataset-root", default=defaults.get("raw_dataset_root", "data/raw/dataset1"))
    parser.add_argument("--run-name", default=defaults.get("run_name", "rgbd_baseline_16k"))
    parser.add_argument("--run-dir", default=defaults.get("run_dir", None))
    parser.add_argument(
        "--framework",
        default=defaults.get("framework", "spinuv_rgbd"),
        help="Kept for result collectors; only spinuv_rgbd is implemented here.",
    )
    parser.add_argument(
        "--model-name",
        choices=["rgbd_early_fusion_unet", "rgbd_dual_encoder_unet", "rgbd_se_fusion_unet"],
        default=defaults.get("model_name", "rgbd_early_fusion_unet"),
    )
    parser.add_argument("--model-id", default=defaults.get("model_id", None))
    parser.add_argument("--initialization", default=defaults.get("initialization", "random_scratch_no_external_pretraining"))
    parser.add_argument("--encoder-name", default=defaults.get("encoder_name", "compact_cnn"))
    parser.add_argument("--encoder-weights", default=defaults.get("encoder_weights", None))
    parser.add_argument("--model-width", type=int, default=int(defaults.get("model_width", 32)))
    parser.add_argument("--depth-stream", choices=["depth", "depth_native"], default=defaults.get("depth_stream", "depth"))
    parser.add_argument("--depth-clip-m", type=float, default=float(defaults.get("depth_clip_m", 10.0)))
    parser.add_argument("--depth-scale", default=str(defaults.get("depth_scale", "auto")))
    parser.add_argument("--depth-fill", choices=["zero", "median"], default=defaults.get("depth_fill", "median"))
    parser.add_argument("--max-iters", type=int, default=int(defaults.get("max_iters", 16000)))
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 4)))
    parser.add_argument("--grad-accum-steps", type=int, default=int(defaults.get("grad_accum_steps", 1)))
    parser.add_argument("--crop-height", type=int, default=int(defaults.get("crop_height", 512)))
    parser.add_argument("--crop-width", type=int, default=int(defaults.get("crop_width", 1024)))
    parser.add_argument("--crop-repeat-factor", type=int, default=int(defaults.get("crop_repeat_factor", 1)))
    parser.add_argument("--anchor-crop-prob", type=float, default=float(defaults.get("anchor_crop_prob", 0.0)))
    parser.add_argument("--anchor-crop-jitter", type=float, default=float(defaults.get("anchor_crop_jitter", 0.08)))
    parser.add_argument("--preprocess-aug-prob", type=float, default=float(defaults.get("preprocess_aug_prob", 0.0)))
    parser.add_argument("--base-lr", type=float, default=float(defaults.get("base_lr", 0.001)))
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=defaults.get("optimizer", "adamw"))
    parser.add_argument("--momentum", type=float, default=float(defaults.get("momentum", 0.9)))
    parser.add_argument("--weight-decay", type=float, default=float(defaults.get("weight_decay", 0.0001)))
    parser.add_argument("--poly-power", type=float, default=float(defaults.get("poly_power", 0.9)))
    parser.add_argument("--val-interval", type=int, default=int(defaults.get("val_interval", 1000)))
    parser.add_argument("--save-interval", type=int, default=int(defaults.get("save_interval", 1000)))
    parser.add_argument("--log-interval", type=int, default=int(defaults.get("log_interval", 50)))
    parser.add_argument("--num-workers", type=int, default=int(defaults.get("num_workers", 4)))
    parser.add_argument("--ignore-index", type=int, default=int(defaults.get("ignore_index", 255)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=bool(defaults.get("amp", True)))
    parser.add_argument("--hard-example-sampling", action=argparse.BooleanOptionalAction, default=bool(defaults.get("hard_example_sampling", False)))
    parser.add_argument("--hard-example-prob", type=float, default=float(defaults.get("hard_example_prob", 0.0)))
    parser.add_argument("--hard-example-mode", choices=["class_pixels", "mixed_boundary"], default=defaults.get("hard_example_mode", "mixed_boundary"))
    parser.add_argument("--hard-example-classes", default=default_hard_classes)
    parser.add_argument("--hard-example-crop-candidates", type=int, default=int(defaults.get("hard_example_crop_candidates", 8)))
    parser.add_argument("--hard-example-min-pixels", type=int, default=int(defaults.get("hard_example_min_pixels", 512)))
    parser.add_argument("--sample-count", type=int, default=int(defaults.get("sample_count", 8)))
    parser.add_argument("--analysis-classes", default=default_analysis_classes)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--dry-run-depth-check",
        type=int,
        default=int(defaults.get("dry_run_depth_check", 0)),
        help="Resolve this many RGB-D pairs per split and exit. Use -1 to check all.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path, root: Path = ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def candidate_raw_roots(value: str | Path) -> list[Path]:
    base = Path(value)
    candidates = [base]
    candidates.extend([base / "dataset1", base / "statistics" / "dataset1", base / "stats" / "dataset1"])
    candidates.extend(
        [
            ROOT / "data" / "raw" / "dataset1",
            ROOT / "dataset1",
            ROOT.parent / "dataset1",
            ROOT / "data" / "dataset1",
        ]
    )
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_raw_dataset_root(value: str | Path) -> Path:
    for candidate in candidate_raw_roots(value):
        path = candidate if candidate.is_absolute() else ROOT / candidate
        if path.name == "dataset_sequences" and path.is_dir():
            return path.parent
        if (path / "dataset_sequences").is_dir():
            return path
    attempted = [str((p if p.is_absolute() else ROOT / p)) for p in candidate_raw_roots(value)]
    raise FileNotFoundError("Could not find dataset1/dataset_sequences. Tried: " + "; ".join(attempted))


def parse_spinuv_image_name(name: str) -> tuple[str, int] | None:
    match = SPINUV_NAME_RE.match(name)
    if not match:
        return None
    sequence = f"sequence_{match.group('date')}_{match.group('time')}"
    frame_id = int(match.group("frame"))
    return sequence, frame_id


def depth_path_from_rgb(rgb_path: Path, stream: str) -> Path:
    parts = list(rgb_path.parts)
    try:
        idx = parts.index("rgb")
    except ValueError:
        return rgb_path
    parts[idx] = stream
    return Path(*parts)


class DepthResolver:
    def __init__(self, raw_root: Path, stream: str) -> None:
        self.raw_root = raw_root
        self.stream = stream
        self.sequence_root = raw_root / "dataset_sequences"
        self.manifest_depth_by_name = self._load_manifest_depth_paths()

    def _load_manifest_depth_paths(self) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        for manifest in sorted(self.raw_root.glob("dataset_ann*/manifests/sampling_manifest.csv")):
            try:
                with manifest.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        new_name = (row.get("new_name") or "").strip()
                        original_rgb = (row.get("original_rgb") or "").strip()
                        sequence = (row.get("sequence") or "").strip()
                        frame_id = (row.get("frame_id") or "").strip()
                        if not new_name:
                            continue
                        if sequence and frame_id:
                            depth_path = self.sequence_root / sequence / "frames" / self.stream / f"{int(frame_id):06d}.png"
                        elif original_rgb:
                            depth_path = depth_path_from_rgb(Path(original_rgb), self.stream)
                        else:
                            continue
                        if not depth_path.is_absolute():
                            depth_path = self.raw_root / depth_path
                        mapping[new_name] = depth_path
            except OSError:
                continue
        return mapping

    def resolve(self, image_path: Path) -> Path:
        parsed = parse_spinuv_image_name(image_path.name)
        if parsed is not None:
            sequence, frame_id = parsed
            candidate = self.sequence_root / sequence / "frames" / self.stream / f"{frame_id:06d}.png"
            if candidate.exists():
                return candidate
            fallback_stream = "depth_native" if self.stream == "depth" else "depth"
            fallback = self.sequence_root / sequence / "frames" / fallback_stream / f"{frame_id:06d}.png"
            if fallback.exists():
                return fallback
        manifest_candidate = self.manifest_depth_by_name.get(image_path.name)
        if manifest_candidate is not None and manifest_candidate.exists():
            return manifest_candidate
        raise FileNotFoundError(f"Depth file not found for {image_path.name} under {self.raw_root}")


def pad_rgbd_pair(
    img: Image.Image,
    depth: Image.Image,
    mask: Image.Image,
    min_height: int,
    min_width: int,
    ignore_index: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    width, height = img.size
    new_width = max(width, min_width)
    new_height = max(height, min_height)
    if new_width == width and new_height == height:
        return img, depth, mask
    right = new_width - width
    bottom = new_height - height
    img = ImageOps.expand(img, border=(0, 0, right, bottom), fill=(0, 0, 0))
    depth = ImageOps.expand(depth, border=(0, 0, right, bottom), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, right, bottom), fill=ignore_index)
    return img, depth, mask


def random_resized_rgbd_pair(
    img: Image.Image,
    depth: Image.Image,
    mask: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    width, height = img.size
    ratio = rng.uniform(0.5, 2.0)
    new_width = max(1, int(round(width * ratio)))
    new_height = max(1, int(round(height * ratio)))
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    depth = depth.resize((new_width, new_height), Image.Resampling.BILINEAR)
    mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
    return img, depth, mask


def crop_rgbd_pair(
    img: Image.Image,
    depth: Image.Image,
    mask: Image.Image,
    left: int,
    top: int,
    crop_height: int,
    crop_width: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    box = (left, top, left + crop_width, top + crop_height)
    return img.crop(box), depth.crop(box), mask.crop(box)


def random_crop_rgbd_pair(
    img: Image.Image,
    depth: Image.Image,
    mask: Image.Image,
    crop_height: int,
    crop_width: int,
    rng: random.Random,
    ignore_index: int,
    anchor_prob: float,
    anchor_jitter: float,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    img, depth, mask = pad_rgbd_pair(img, depth, mask, crop_height, crop_width, ignore_index)
    width, height = img.size
    left, top = sample_crop_box(width, height, crop_height, crop_width, rng, anchor_prob, anchor_jitter)
    return crop_rgbd_pair(img, depth, mask, left, top, crop_height, crop_width)


def hard_random_crop_rgbd_pair(
    img: Image.Image,
    depth: Image.Image,
    mask: Image.Image,
    crop_height: int,
    crop_width: int,
    rng: random.Random,
    ignore_index: int,
    hard_class_ids: Sequence[int],
    candidates: int,
    min_pixels: int,
    mode: str,
    anchor_prob: float,
    anchor_jitter: float,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    if not hard_class_ids or candidates <= 1:
        return random_crop_rgbd_pair(img, depth, mask, crop_height, crop_width, rng, ignore_index, anchor_prob, anchor_jitter)

    img, depth, mask = pad_rgbd_pair(img, depth, mask, crop_height, crop_width, ignore_index)
    width, height = img.size
    hard_ids = np.asarray(list(hard_class_ids), dtype=np.int64)
    mask_arr = np.asarray(mask, dtype=np.int64)
    best_left = rng.randint(0, width - crop_width)
    best_top = rng.randint(0, height - crop_height)
    best_score = -1
    crop_points: list[tuple[int, int]] = []
    if anchor_prob > 0 and rng.random() < anchor_prob:
        crop_points.extend(anchor_crop_boxes(width, height, crop_height, crop_width, rng, anchor_jitter))
    while len(crop_points) < candidates:
        crop_points.append((rng.randint(0, width - crop_width), rng.randint(0, height - crop_height)))
    for left, top in crop_points[:candidates]:
        crop = mask_arr[top : top + crop_height, left : left + crop_width]
        hard_mask = np.isin(crop, hard_ids)
        valid_mask = crop != ignore_index
        hard_pixels = int(hard_mask.sum())
        valid_pixels = int(valid_mask.sum())
        if mode == "mixed_boundary":
            context_mask = valid_mask & ~hard_mask
            context_pixels = int(context_mask.sum())
            horizontal = (
                (valid_mask[:, 1:] & valid_mask[:, :-1])
                & (crop[:, 1:] != crop[:, :-1])
                & (hard_mask[:, 1:] | hard_mask[:, :-1])
            )
            vertical = (
                (valid_mask[1:, :] & valid_mask[:-1, :])
                & (crop[1:, :] != crop[:-1, :])
                & (hard_mask[1:, :] | hard_mask[:-1, :])
            )
            boundary_pixels = int(horizontal.sum() + vertical.sum())
            balance_pixels = min(hard_pixels, context_pixels) if context_pixels > 0 else 0
            score = boundary_pixels * 1_000_000 + balance_pixels * 1000 + valid_pixels
            if hard_pixels >= min_pixels and balance_pixels >= min_pixels and boundary_pixels > 0:
                best_left, best_top, best_score = left, top, score
                break
        else:
            score = hard_pixels * 1000 + valid_pixels
            if hard_pixels >= min_pixels:
                best_left, best_top, best_score = left, top, score
                break
        if score > best_score:
            best_left, best_top, best_score = left, top, score
    return crop_rgbd_pair(img, depth, mask, best_left, best_top, crop_height, crop_width)


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def depth_scale_factor(arr: np.ndarray, scale: str) -> float:
    if scale.lower() == "auto":
        valid = arr[np.isfinite(arr) & (arr > 0)]
        if valid.size == 0:
            return 1.0
        return 0.001 if float(np.nanpercentile(valid, 95)) > 100.0 else 1.0
    return float(scale)


def depth_to_tensor(depth: Image.Image, clip_m: float, scale: str, fill: str) -> torch.Tensor:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr * depth_scale_factor(arr, scale)
    valid = np.isfinite(arr) & (arr > 0.0)
    if fill == "median" and valid.any():
        arr[~valid] = float(np.median(arr[valid]))
    else:
        arr[~valid] = 0.0
    arr = np.clip(arr, 0.0, clip_m) / max(clip_m, 1e-6)
    arr = (arr - 0.5) / 0.25
    return torch.from_numpy(arr[None, :, :].astype(np.float32)).contiguous()


def denormalize_rgb(tensor: torch.Tensor) -> np.ndarray:
    rgb = tensor.detach().cpu()[:3].numpy().transpose(1, 2, 0)
    rgb = (rgb * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


class SpinUVRGBDDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        raw_root: Path,
        split: str,
        crop_size: tuple[int, int],
        ignore_index: int,
        train: bool,
        seed: int,
        depth_stream: str,
        depth_clip_m: float,
        depth_scale: str,
        depth_fill: str,
        crop_repeat_factor: int,
        anchor_crop_prob: float,
        anchor_crop_jitter: float,
        preprocess_aug_prob: float,
        hard_class_ids: Sequence[int] | None,
        hard_example_sampling: bool,
        hard_example_prob: float,
        hard_example_mode: str,
        hard_example_crop_candidates: int,
        hard_example_min_pixels: int,
    ) -> None:
        self.data_root = data_root
        self.raw_root = raw_root
        self.split = split
        self.crop_size = crop_size
        self.ignore_index = ignore_index
        self.train = train
        self.seed = seed
        self.depth_clip_m = depth_clip_m
        self.depth_scale = depth_scale
        self.depth_fill = depth_fill
        self.crop_repeat_factor = max(1, int(crop_repeat_factor))
        self.anchor_crop_prob = max(0.0, min(1.0, float(anchor_crop_prob)))
        self.anchor_crop_jitter = max(0.0, float(anchor_crop_jitter))
        self.preprocess_aug_prob = max(0.0, min(1.0, float(preprocess_aug_prob)))
        self.hard_class_ids = list(hard_class_ids or [])
        self.hard_example_sampling = bool(hard_example_sampling and self.hard_class_ids)
        self.hard_example_prob = max(0.0, min(1.0, float(hard_example_prob)))
        self.hard_example_mode = str(hard_example_mode)
        self.hard_example_crop_candidates = max(1, int(hard_example_crop_candidates))
        self.hard_example_min_pixels = max(1, int(hard_example_min_pixels))
        image_paths = sorted((data_root / "images" / split).glob("*.png"))
        mask_paths = [data_root / "masks" / split / p.name for p in image_paths]
        missing_masks = [p for p in mask_paths if not p.exists()]
        if missing_masks:
            raise FileNotFoundError(f"Missing {len(missing_masks)} masks for split={split}; first={missing_masks[0]}")
        if not image_paths:
            raise FileNotFoundError(f"No images found for split={split} under {data_root}")
        resolver = DepthResolver(raw_root, depth_stream)
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []
        self.depth_paths: list[Path] = []
        self.skipped_missing_depth: list[str] = []
        self.original_image_count = len(image_paths)
        for image_path, mask_path in zip(image_paths, mask_paths):
            try:
                depth_path = resolver.resolve(image_path)
            except FileNotFoundError:
                self.skipped_missing_depth.append(str(image_path))
                continue
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)
            self.depth_paths.append(depth_path)
        if self.skipped_missing_depth:
            examples = ", ".join(Path(p).name for p in self.skipped_missing_depth[:5])
            warnings.warn(
                f"Skipping {len(self.skipped_missing_depth)} {split} samples with missing {depth_stream} depth. "
                f"First skipped: {examples}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No RGB-D pairs remain for split={split}; "
                f"all {len(image_paths)} images are missing {depth_stream} depth under {raw_root}"
            )

    def __len__(self) -> int:
        if self.train:
            return len(self.image_paths) * self.crop_repeat_factor
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = index % len(self.image_paths)
        image_path = self.image_paths[base_index]
        mask_path = self.mask_paths[base_index]
        depth_path = self.depth_paths[base_index]
        img = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("F")
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        rng = random if self.train else random.Random(self.seed + index)
        if self.train:
            img = preprocess_augment(img, rng, self.preprocess_aug_prob)
            img, depth, mask = random_resized_rgbd_pair(img, depth, mask, rng)
            if self.hard_example_sampling and rng.random() < self.hard_example_prob:
                img, depth, mask = hard_random_crop_rgbd_pair(
                    img,
                    depth,
                    mask,
                    self.crop_size[0],
                    self.crop_size[1],
                    rng,
                    self.ignore_index,
                    self.hard_class_ids,
                    self.hard_example_crop_candidates,
                    self.hard_example_min_pixels,
                    self.hard_example_mode,
                    self.anchor_crop_prob,
                    self.anchor_crop_jitter,
                )
            else:
                img, depth, mask = random_crop_rgbd_pair(
                    img,
                    depth,
                    mask,
                    self.crop_size[0],
                    self.crop_size[1],
                    rng,
                    self.ignore_index,
                    self.anchor_crop_prob,
                    self.anchor_crop_jitter,
                )
            if rng.random() < 0.5:
                img = ImageOps.mirror(img)
                depth = ImageOps.mirror(depth)
                mask = ImageOps.mirror(mask)
            img = color_jitter(img, rng)

        image_tensor = image_to_tensor(img)
        depth_tensor = depth_to_tensor(depth, self.depth_clip_m, self.depth_scale, self.depth_fill)
        rgbd_tensor = torch.cat([image_tensor, depth_tensor], dim=0)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
        return {
            "image": rgbd_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "depth_path": str(depth_path),
        }


def make_rgbd_loader(
    data_root: Path,
    raw_root: Path,
    split: str,
    crop_size: tuple[int, int],
    args: argparse.Namespace,
    train: bool,
    hard_class_ids: Sequence[int] | None = None,
    hard_example_enabled: bool = False,
) -> DataLoader:
    dataset = SpinUVRGBDDataset(
        data_root=data_root,
        raw_root=raw_root,
        split=split,
        crop_size=crop_size,
        ignore_index=args.ignore_index,
        train=train,
        seed=args.seed,
        depth_stream=args.depth_stream,
        depth_clip_m=args.depth_clip_m,
        depth_scale=args.depth_scale,
        depth_fill=args.depth_fill,
        crop_repeat_factor=args.crop_repeat_factor,
        anchor_crop_prob=args.anchor_crop_prob,
        anchor_crop_jitter=args.anchor_crop_jitter,
        preprocess_aug_prob=args.preprocess_aug_prob,
        hard_class_ids=hard_class_ids,
        hard_example_sampling=hard_example_enabled,
        hard_example_prob=args.hard_example_prob,
        hard_example_mode=args.hard_example_mode,
        hard_example_crop_candidates=args.hard_example_crop_candidates,
        hard_example_min_pixels=args.hard_example_min_pixels,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else 1,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=args.num_workers > 0,
    )


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvBNAct(channels, channels), ConvBNAct(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CompactEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int) -> None:
        super().__init__()
        channels = [width, width * 2, width * 4, width * 6]
        self.channels = channels
        self.stem = nn.Sequential(ConvBNAct(in_ch, channels[0]), ResidualBlock(channels[0]))
        self.down1 = nn.Sequential(ConvBNAct(channels[0], channels[1], stride=2), ResidualBlock(channels[1]))
        self.down2 = nn.Sequential(ConvBNAct(channels[1], channels[2], stride=2), ResidualBlock(channels[2]))
        self.down3 = nn.Sequential(ConvBNAct(channels[2], channels[3], stride=2), ResidualBlock(channels[3]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stem(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        return [f0, f1, f2, f3]


class CompactDecoder(nn.Module):
    def __init__(self, channels: Sequence[int], num_classes: int) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.up2 = ConvBNAct(c3 + c2, c2)
        self.up1 = ConvBNAct(c2 + c1, c1)
        self.up0 = ConvBNAct(c1 + c0, c0)
        self.head = nn.Conv2d(c0, num_classes, kernel_size=1)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3 = feats
        x = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))
        x = F.interpolate(x, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up0(torch.cat([x, f0], dim=1))
        return self.head(x)


class RGBDEarlyFusionUNet(nn.Module):
    def __init__(self, num_classes: int, width: int) -> None:
        super().__init__()
        self.encoder = CompactEncoder(4, width)
        self.decoder = CompactDecoder(self.encoder.channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class RGBDDualEncoderUNet(nn.Module):
    def __init__(self, num_classes: int, width: int) -> None:
        super().__init__()
        self.rgb_encoder = CompactEncoder(3, width)
        self.depth_encoder = CompactEncoder(1, width)
        self.channels = self.rgb_encoder.channels
        self.fuse = nn.ModuleList([ConvBNAct(ch * 2, ch, kernel_size=1) for ch in self.channels])
        self.decoder = CompactDecoder(self.channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_feats = self.rgb_encoder(x[:, :3])
        depth_feats = self.depth_encoder(x[:, 3:4])
        fused = [fuse(torch.cat([rgb, depth], dim=1)) for fuse, rgb, depth in zip(self.fuse, rgb_feats, depth_feats)]
        return self.decoder(fused)


class SEFusionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pre = ConvBNAct(channels * 2, channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        gates = self.gate(torch.cat([rgb, depth], dim=1))
        rgb_gate, depth_gate = torch.chunk(gates, 2, dim=1)
        gated = torch.cat([rgb * rgb_gate, depth * depth_gate], dim=1)
        return self.pre(gated)


class RGBDSEFusionUNet(nn.Module):
    def __init__(self, num_classes: int, width: int) -> None:
        super().__init__()
        self.rgb_encoder = CompactEncoder(3, width)
        self.depth_encoder = CompactEncoder(1, width)
        self.channels = self.rgb_encoder.channels
        self.fuse = nn.ModuleList([SEFusionBlock(ch) for ch in self.channels])
        self.decoder = CompactDecoder(self.channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_feats = self.rgb_encoder(x[:, :3])
        depth_feats = self.depth_encoder(x[:, 3:4])
        fused = [fuse(rgb, depth) for fuse, rgb, depth in zip(self.fuse, rgb_feats, depth_feats)]
        return self.decoder(fused)


def build_rgbd_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.framework != "spinuv_rgbd":
        raise ValueError(f"Unsupported framework={args.framework}; train_spinuv_rgbd.py only supports spinuv_rgbd")
    if args.encoder_weights not in (None, "", "none", "null", "false", "0"):
        raise ValueError("RGB-D scratch baselines do not load external encoder weights.")
    if args.model_name == "rgbd_early_fusion_unet":
        return RGBDEarlyFusionUNet(num_classes, args.model_width)
    if args.model_name == "rgbd_dual_encoder_unet":
        return RGBDDualEncoderUNet(num_classes, args.model_width)
    if args.model_name == "rgbd_se_fusion_unet":
        return RGBDSEFusionUNet(num_classes, args.model_width)
    raise ValueError(f"Unsupported model_name={args.model_name}")


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    return images, masks


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits = model(images)
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    analysis_class_ids: Sequence[int],
    ignore_index: int,
    split: str,
    run_dir: Path,
    sample_count: int,
    amp: bool,
    total_params: int,
    trainable_params: int,
) -> dict[str, Any]:
    model.eval()
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    per_image_rows: list[dict[str, Any]] = []
    visual_samples: list[dict[str, Any]] = []
    worst_samples: list[dict[str, Any]] = []
    iterator = tqdm(loader, desc=f"eval:{split}", dynamic_ncols=True)
    for batch in iterator:
        images, masks = batch_to_device(batch, device)
        with autocast_context(device, amp):
            logits = forward_logits(model, images)
        pred = logits.argmax(dim=1)
        for item_idx in range(pred.shape[0]):
            item_conf = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
            update_confusion(item_conf, pred[item_idx], masks[item_idx], len(class_names), ignore_index)
            update_confusion(confusion, pred[item_idx], masks[item_idx], len(class_names), ignore_index)
            item_metrics = metrics_from_confusion(item_conf, analysis_class_ids)
            image_iou = item_metrics["mIoU"]
            row = {
                "image_path": batch["image_path"][item_idx],
                "image_iou": image_iou,
                "valid_pixels": int(item_conf.sum()),
            }
            per_image_rows.append(row)
            sample = {
                "image": denormalize_rgb(images[item_idx]),
                "gt": masks[item_idx].detach().cpu().numpy(),
                "pred": pred[item_idx].detach().cpu().numpy(),
                "image_iou": image_iou,
            }
            if len(visual_samples) < sample_count:
                visual_samples.append(sample)
            worst_samples.append(sample)
            worst_samples = sorted(
                worst_samples,
                key=lambda x: x["image_iou"] if math.isfinite(x["image_iou"]) else -1.0,
            )[:sample_count]

    summary = write_metrics(run_dir, split, class_names, analysis_class_ids, confusion, per_image_rows, total_params, trainable_params)
    save_visual_grid(visual_samples, run_dir / "figures" / f"{split}_sample_predictions.png", split, ignore_index)
    save_visual_grid(worst_samples, run_dir / "figures" / f"{split}_worst_predictions.png", split, ignore_index)
    return summary


def verify_depth_pairs(data_root: Path, raw_root: Path, args: argparse.Namespace) -> None:
    crop_size = (args.crop_height, args.crop_width)
    counts: dict[str, dict[str, int]] = {}
    examples: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        dataset = SpinUVRGBDDataset(
            data_root=data_root,
            raw_root=raw_root,
            split=split,
            crop_size=crop_size,
            ignore_index=args.ignore_index,
            train=False,
            seed=args.seed,
            depth_stream=args.depth_stream,
            depth_clip_m=args.depth_clip_m,
            depth_scale=args.depth_scale,
            depth_fill=args.depth_fill,
            crop_repeat_factor=1,
            anchor_crop_prob=args.anchor_crop_prob,
            anchor_crop_jitter=args.anchor_crop_jitter,
            preprocess_aug_prob=0.0,
            hard_class_ids=[],
            hard_example_sampling=False,
            hard_example_prob=0.0,
            hard_example_mode=args.hard_example_mode,
            hard_example_crop_candidates=1,
            hard_example_min_pixels=args.hard_example_min_pixels,
        )
        limit = len(dataset) if args.dry_run_depth_check < 0 else min(len(dataset), args.dry_run_depth_check)
        counts[split] = {
            "original_images": int(dataset.original_image_count),
            "rgbd_pairs": int(len(dataset.image_paths)),
            "skipped_missing_depth": int(len(dataset.skipped_missing_depth)),
        }
        examples[split] = [
            {"image": str(dataset.image_paths[idx]), "depth": str(dataset.depth_paths[idx])}
            for idx in range(limit)
        ]
    print(json.dumps({"event": "depth_pair_check", "raw_root": str(raw_root), "counts": counts, "examples": examples}, indent=2))


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data_root = resolve_path(args.data_root)
    raw_root = resolve_raw_dataset_root(args.raw_dataset_root)
    class_names, ignore_from_data = load_classes(data_root)
    if args.ignore_index != ignore_from_data:
        args.ignore_index = ignore_from_data
    analysis_class_ids = resolve_analysis_class_ids(class_names, args.analysis_classes)
    hard_class_ids = resolve_optional_class_ids(class_names, args.hard_example_classes)
    hard_example_enabled = bool(args.hard_example_sampling and hard_class_ids and args.hard_example_prob > 0)
    crop_size = (args.crop_height, args.crop_width)

    if args.dry_run_depth_check != 0:
        verify_depth_pairs(data_root, raw_root, args)
        return

    run_dir = make_run_dir(args)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if args.config:
        config_src = resolve_path(args.config)
        if config_src.exists():
            shutil.copyfile(config_src, run_dir / "config_snapshot.json")
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_rgbd_model(args, len(class_names)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with (run_dir / "model_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "framework": args.framework,
                "model_id": args.model_id,
                "initialization": args.initialization,
                "encoder_name": args.encoder_name,
                "encoder_weights": args.encoder_weights,
                "class_count": len(class_names),
                "analysis_class_count": len(analysis_class_ids),
                "analysis_class_names": [class_names[idx] for idx in analysis_class_ids],
                "input_modalities": ["rgb", args.depth_stream],
                "raw_dataset_root": raw_root.as_posix(),
                "depth_stream": args.depth_stream,
                "depth_clip_m": float(args.depth_clip_m),
                "depth_scale": args.depth_scale,
                "depth_fill": args.depth_fill,
                "model_width": int(args.model_width),
                "hard_example_sampling": bool(hard_example_enabled),
                "hard_example_prob": float(args.hard_example_prob),
                "hard_example_mode": args.hard_example_mode,
                "hard_example_class_ids": [int(idx) for idx in hard_class_ids],
                "hard_example_class_names": [class_names[idx] for idx in hard_class_ids],
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "micro_batch_size": int(args.batch_size),
                "grad_accum_steps": int(args.grad_accum_steps),
                "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
                "optimizer_updates": int(args.max_iters),
                "train_crop_exposures": int(args.max_iters * args.batch_size * args.grad_accum_steps),
                "crop_repeat_factor": int(args.crop_repeat_factor),
                "anchor_crop_prob": float(args.anchor_crop_prob),
                "anchor_crop_jitter": float(args.anchor_crop_jitter),
                "preprocess_aug_prob": float(args.preprocess_aug_prob),
            },
            f,
            indent=2,
        )

    val_loader = make_rgbd_loader(data_root, raw_root, "val", crop_size, args, False)
    test_loader = make_rgbd_loader(data_root, raw_root, "test", crop_size, args, False)

    if args.eval_only:
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best_miou.pt"
        load_checkpoint(checkpoint, model)
        evaluate(model, val_loader, device, class_names, analysis_class_ids, args.ignore_index, "val", run_dir, args.sample_count, args.amp, total_params, trainable_params)
        if not args.skip_test:
            evaluate(model, test_loader, device, class_names, analysis_class_ids, args.ignore_index, "test", run_dir, args.sample_count, args.amp, total_params, trainable_params)
        print(f"eval_complete run_dir={run_dir.as_posix()}")
        return

    train_loader = make_rgbd_loader(data_root, raw_root, "train", crop_size, args, True, hard_class_ids, hard_example_enabled)
    train_iter = iter(train_loader)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    optimizer = build_optimizer(args, model)
    scaler = make_grad_scaler(device, args.amp)
    start_iter = 0
    best_miou = -1.0
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), model, optimizer, scaler)
        start_iter = int(checkpoint.get("iteration", 0))
        best_miou = float(checkpoint.get("best_miou", -1.0))

    log_path = run_dir / "logs" / "train_log.jsonl"
    write_jsonl(
        log_path,
        {
            "event": "start",
            "run_dir": run_dir.as_posix(),
            "device": str(device),
            "train_images": len(train_loader.dataset),
            "train_original_images": len(train_loader.dataset.image_paths),
            "train_source_images": int(train_loader.dataset.original_image_count),
            "train_skipped_missing_depth": int(len(train_loader.dataset.skipped_missing_depth)),
            "val_images": len(val_loader.dataset),
            "val_source_images": int(val_loader.dataset.original_image_count),
            "val_skipped_missing_depth": int(len(val_loader.dataset.skipped_missing_depth)),
            "test_images": len(test_loader.dataset),
            "test_source_images": int(test_loader.dataset.original_image_count),
            "test_skipped_missing_depth": int(len(test_loader.dataset.skipped_missing_depth)),
            "framework": args.framework,
            "model_name": args.model_name,
            "model_id": args.model_id,
            "initialization": args.initialization,
            "optimizer": args.optimizer,
            "micro_batch_size": int(args.batch_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
            "optimizer_updates": int(args.max_iters),
            "train_crop_exposures": int(args.max_iters * args.batch_size * args.grad_accum_steps),
            "depth_stream": args.depth_stream,
            "depth_clip_m": float(args.depth_clip_m),
            "depth_scale": args.depth_scale,
            "depth_fill": args.depth_fill,
            "analysis_class_count": len(analysis_class_ids),
            "analysis_class_names": [class_names[idx] for idx in analysis_class_ids],
            "hard_example_sampling": bool(hard_example_enabled),
            "hard_example_prob": float(args.hard_example_prob),
            "hard_example_mode": args.hard_example_mode,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
        },
    )

    model.train()
    last_log_time = time.time()
    running_loss = 0.0
    running_loss_count = 0
    iterator = tqdm(range(start_iter + 1, args.max_iters + 1), desc="train", dynamic_ncols=True)
    for iteration in iterator:
        lr = poly_lr(args.base_lr, iteration, args.max_iters, args.poly_power)
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            images, masks = batch_to_device(batch, device)
            with autocast_context(device, args.amp):
                logits = forward_logits(model, images)
                loss = criterion(logits, masks)
                scaled_loss = loss / float(args.grad_accum_steps)
            scaler.scale(scaled_loss).backward()
            loss_accum += float(loss.detach().cpu())
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss_accum / float(args.grad_accum_steps)
        running_loss += loss_value
        running_loss_count += 1
        iterator.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.6f}")

        if iteration % args.log_interval == 0 or iteration == 1:
            now = time.time()
            avg_loss = running_loss / max(1, running_loss_count)
            running_loss = 0.0
            running_loss_count = 0
            payload = {"event": "train", "iteration": iteration, "loss": avg_loss, "lr": lr, "seconds_since_last_log": now - last_log_time}
            write_jsonl(log_path, payload)
            print(json.dumps(payload, sort_keys=True))
            last_log_time = now

        if iteration % args.val_interval == 0 or iteration == args.max_iters:
            summary = evaluate(model, val_loader, device, class_names, analysis_class_ids, args.ignore_index, "val", run_dir, args.sample_count, args.amp, total_params, trainable_params)
            payload = {"event": "val", "iteration": iteration, **summary}
            write_jsonl(log_path, payload)
            print(json.dumps(payload, sort_keys=True))
            if summary["mIoU"] > best_miou:
                best_miou = float(summary["mIoU"])
                save_checkpoint(run_dir / "checkpoints" / "best_miou.pt", model, optimizer, scaler, iteration, best_miou, args, class_names)
            model.train()

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, scaler, iteration, best_miou, args, class_names)

    best_path = run_dir / "checkpoints" / "best_miou.pt"
    latest_path = run_dir / "checkpoints" / "latest.pt"
    load_checkpoint(best_path if best_path.exists() else latest_path, model)
    val_summary = evaluate(model, val_loader, device, class_names, analysis_class_ids, args.ignore_index, "val_best", run_dir, args.sample_count, args.amp, total_params, trainable_params)
    final_payload = {"event": "final_val", **val_summary}
    write_jsonl(log_path, final_payload)
    print(json.dumps(final_payload, sort_keys=True))
    if not args.skip_test:
        test_summary = evaluate(model, test_loader, device, class_names, analysis_class_ids, args.ignore_index, "test", run_dir, args.sample_count, args.amp, total_params, trainable_params)
        final_test_payload = {"event": "final_test", **test_summary}
        write_jsonl(log_path, final_test_payload)
        print(json.dumps(final_test_payload, sort_keys=True))
    print(f"train_complete run_dir={run_dir.as_posix()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train and evaluate a DeepLabV3+ R50 baseline on SPIN-UV semantic masks."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_PALETTE = np.array(
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


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    known, _ = pre_parser.parse_known_args()
    defaults = load_config(known.config)
    default_analysis_classes = defaults.get("analysis_classes", [])
    if isinstance(default_analysis_classes, list):
        default_analysis_classes = ",".join(default_analysis_classes)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--data-root", default=defaults.get("data_root", "data/processed/spinuv_semantic"))
    parser.add_argument("--run-name", default=defaults.get("run_name", "experiment_01_deeplabv3plus_r50_8k"))
    parser.add_argument("--run-dir", default=defaults.get("run_dir", None))
    parser.add_argument("--framework", default=defaults.get("framework", "smp"))
    parser.add_argument("--model-name", default=defaults.get("model_name", "DeepLabV3Plus"))
    parser.add_argument("--model-id", default=defaults.get("model_id", None))
    parser.add_argument("--initialization", default=defaults.get("initialization", ""))
    parser.add_argument("--hf-from-config", action=argparse.BooleanOptionalAction, default=bool(defaults.get("hf_from_config", False)))
    parser.add_argument("--encoder-name", default=defaults.get("encoder_name", "resnet50"))
    parser.add_argument("--encoder-weights", default=defaults.get("encoder_weights", "imagenet"))
    parser.add_argument("--ignore-mismatched-sizes", action=argparse.BooleanOptionalAction, default=bool(defaults.get("ignore_mismatched_sizes", True)))
    parser.add_argument("--reset-classifier", action=argparse.BooleanOptionalAction, default=bool(defaults.get("reset_classifier", False)))
    parser.add_argument("--max-iters", type=int, default=int(defaults.get("max_iters", 8000)))
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 4)))
    parser.add_argument("--grad-accum-steps", type=int, default=int(defaults.get("grad_accum_steps", defaults.get("gradient_accumulation_steps", 1))))
    parser.add_argument("--crop-height", type=int, default=int(defaults.get("crop_height", 512)))
    parser.add_argument("--crop-width", type=int, default=int(defaults.get("crop_width", 1024)))
    parser.add_argument("--crop-repeat-factor", type=int, default=int(defaults.get("crop_repeat_factor", 1)))
    parser.add_argument("--anchor-crop-prob", type=float, default=float(defaults.get("anchor_crop_prob", 0.0)))
    parser.add_argument("--anchor-crop-jitter", type=float, default=float(defaults.get("anchor_crop_jitter", 0.08)))
    parser.add_argument("--preprocess-aug-prob", type=float, default=float(defaults.get("preprocess_aug_prob", 0.0)))
    parser.add_argument("--base-lr", type=float, default=float(defaults.get("base_lr", 0.005)))
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=defaults.get("optimizer", "sgd"))
    parser.add_argument("--momentum", type=float, default=float(defaults.get("momentum", 0.9)))
    parser.add_argument("--weight-decay", type=float, default=float(defaults.get("weight_decay", 0.0005)))
    parser.add_argument("--poly-power", type=float, default=float(defaults.get("poly_power", 0.9)))
    parser.add_argument("--val-interval", type=int, default=int(defaults.get("val_interval", 1000)))
    parser.add_argument("--save-interval", type=int, default=int(defaults.get("save_interval", 1000)))
    parser.add_argument("--log-interval", type=int, default=int(defaults.get("log_interval", 50)))
    parser.add_argument("--num-workers", type=int, default=int(defaults.get("num_workers", 4)))
    parser.add_argument("--ignore-index", type=int, default=int(defaults.get("ignore_index", 255)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=bool(defaults.get("amp", True)))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=bool(defaults.get("gradient_checkpointing", False)))
    default_hard_classes = defaults.get("hard_example_classes", "all")
    if isinstance(default_hard_classes, list):
        default_hard_classes = ",".join(default_hard_classes)
    parser.add_argument("--hard-example-sampling", action=argparse.BooleanOptionalAction, default=bool(defaults.get("hard_example_sampling", False)))
    parser.add_argument("--hard-example-prob", type=float, default=float(defaults.get("hard_example_prob", 0.0)))
    parser.add_argument("--hard-example-mode", choices=["class_pixels", "mixed_boundary"], default=defaults.get("hard_example_mode", "mixed_boundary"))
    parser.add_argument("--hard-example-classes", default=default_hard_classes)
    parser.add_argument("--hard-example-crop-candidates", type=int, default=int(defaults.get("hard_example_crop_candidates", 8)))
    parser.add_argument("--hard-example-min-pixels", type=int, default=int(defaults.get("hard_example_min_pixels", 512)))
    parser.add_argument("--sample-count", type=int, default=int(defaults.get("sample_count", 8)))
    parser.add_argument("--analysis-classes", default=default_analysis_classes)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Load model weights before training without restoring optimizer or iteration.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def normalize_weights_arg(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).lower() in {"none", "null", "false", "0"}:
        return None
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_classes(data_root: Path) -> tuple[list[str], int]:
    classes_path = data_root / "meta" / "classes.json"
    with classes_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    ordered = sorted(payload["classes"], key=lambda item: item["train_id"])
    return [item["name"] for item in ordered], int(payload.get("ignore_value", 255))


def resolve_analysis_class_ids(class_names: list[str], analysis_classes: str | Sequence[str] | None) -> list[int]:
    if analysis_classes is None:
        return list(range(len(class_names)))
    if isinstance(analysis_classes, str):
        selected_names = [name.strip() for name in analysis_classes.split(",") if name.strip()]
    else:
        selected_names = [str(name).strip() for name in analysis_classes if str(name).strip()]
    if not selected_names:
        return list(range(len(class_names)))
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in selected_names if name not in name_to_id]
    if missing:
        raise ValueError(f"Unknown analysis classes: {missing}. Available classes: {class_names}")
    return [name_to_id[name] for name in selected_names]


def resolve_optional_class_ids(class_names: list[str], selected_classes: str | Sequence[str] | None) -> list[int]:
    if selected_classes is None:
        return []
    if isinstance(selected_classes, str):
        selected_names = [name.strip() for name in selected_classes.split(",") if name.strip()]
    else:
        selected_names = [str(name).strip() for name in selected_classes if str(name).strip()]
    if not selected_names:
        return []
    if len(selected_names) == 1 and selected_names[0].lower() in {"all", "all_valid", "*"}:
        return list(range(len(class_names)))
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in selected_names if name not in name_to_id]
    if missing:
        raise ValueError(f"Unknown hard-example classes: {missing}. Available classes: {class_names}")
    return [name_to_id[name] for name in selected_names]


def pad_pair(img: Image.Image, mask: Image.Image, min_height: int, min_width: int, ignore_index: int) -> tuple[Image.Image, Image.Image]:
    width, height = img.size
    new_width = max(width, min_width)
    new_height = max(height, min_height)
    if new_width == width and new_height == height:
        return img, mask
    right = new_width - width
    bottom = new_height - height
    img = ImageOps.expand(img, border=(0, 0, right, bottom), fill=(0, 0, 0))
    mask = ImageOps.expand(mask, border=(0, 0, right, bottom), fill=ignore_index)
    return img, mask


def random_resized_pair(img: Image.Image, mask: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    width, height = img.size
    ratio = rng.uniform(0.5, 2.0)
    new_width = max(1, int(round(width * ratio)))
    new_height = max(1, int(round(height * ratio)))
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
    return img, mask


def anchor_crop_boxes(
    width: int,
    height: int,
    crop_height: int,
    crop_width: int,
    rng: random.Random,
    jitter: float,
) -> list[tuple[int, int]]:
    max_left = max(0, width - crop_width)
    max_top = max(0, height - crop_height)
    x_fracs = [0.0, 0.25, 0.5, 0.75, 1.0] if max_left > 0 else [0.0]
    y_fracs = [0.0, 0.5, 1.0] if max_top > 0 else [0.0]
    x_jitter = int(round(max_left * max(0.0, float(jitter))))
    y_jitter = int(round(max_top * max(0.0, float(jitter))))
    boxes: list[tuple[int, int]] = []
    for y_frac in y_fracs:
        for x_frac in x_fracs:
            left = int(round(max_left * x_frac))
            top = int(round(max_top * y_frac))
            if x_jitter > 0:
                left += rng.randint(-x_jitter, x_jitter)
            if y_jitter > 0:
                top += rng.randint(-y_jitter, y_jitter)
            boxes.append((min(max(left, 0), max_left), min(max(top, 0), max_top)))
    rng.shuffle(boxes)
    return boxes


def sample_crop_box(
    width: int,
    height: int,
    crop_height: int,
    crop_width: int,
    rng: random.Random,
    anchor_prob: float,
    anchor_jitter: float,
) -> tuple[int, int]:
    if anchor_prob > 0 and rng.random() < anchor_prob:
        return anchor_crop_boxes(width, height, crop_height, crop_width, rng, anchor_jitter)[0]
    return rng.randint(0, width - crop_width), rng.randint(0, height - crop_height)


def random_crop_pair(
    img: Image.Image,
    mask: Image.Image,
    crop_height: int,
    crop_width: int,
    rng: random.Random,
    ignore_index: int,
    anchor_prob: float = 0.0,
    anchor_jitter: float = 0.08,
) -> tuple[Image.Image, Image.Image]:
    img, mask = pad_pair(img, mask, crop_height, crop_width, ignore_index)
    width, height = img.size
    left, top = sample_crop_box(width, height, crop_height, crop_width, rng, anchor_prob, anchor_jitter)
    box = (left, top, left + crop_width, top + crop_height)
    return img.crop(box), mask.crop(box)


def hard_random_crop_pair(
    img: Image.Image,
    mask: Image.Image,
    crop_height: int,
    crop_width: int,
    rng: random.Random,
    ignore_index: int,
    hard_class_ids: Sequence[int],
    candidates: int,
    min_pixels: int,
    mode: str,
    anchor_prob: float = 0.0,
    anchor_jitter: float = 0.08,
) -> tuple[Image.Image, Image.Image]:
    if not hard_class_ids or candidates <= 1:
        return random_crop_pair(img, mask, crop_height, crop_width, rng, ignore_index, anchor_prob, anchor_jitter)

    img, mask = pad_pair(img, mask, crop_height, crop_width, ignore_index)
    width, height = img.size
    hard_ids = np.asarray(list(hard_class_ids), dtype=np.int64)
    mask_arr = np.asarray(mask, dtype=np.int64)
    best_left = rng.randint(0, width - crop_width)
    best_top = rng.randint(0, height - crop_height)
    best_score = -1
    candidate_count = max(1, int(candidates))
    crop_points: list[tuple[int, int]] = []
    if anchor_prob > 0 and rng.random() < anchor_prob:
        crop_points.extend(anchor_crop_boxes(width, height, crop_height, crop_width, rng, anchor_jitter))
    while len(crop_points) < candidate_count:
        crop_points.append((rng.randint(0, width - crop_width), rng.randint(0, height - crop_height)))
    for left, top in crop_points[:candidate_count]:
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
            if context_pixels > 0:
                balance_pixels = min(hard_pixels, context_pixels)
            elif hard_pixels > 0:
                _, selected_counts = np.unique(crop[hard_mask], return_counts=True)
                balance_pixels = int(hard_pixels - selected_counts.max())
            else:
                balance_pixels = 0
            score = boundary_pixels * 1_000_000 + balance_pixels * 1000 + valid_pixels
        else:
            score = hard_pixels * 1000 + valid_pixels
        if score > best_score:
            best_left = left
            best_top = top
            best_score = score
        if mode == "mixed_boundary":
            if hard_pixels >= min_pixels and balance_pixels >= min_pixels and boundary_pixels > 0:
                break
        elif hard_pixels >= min_pixels:
            break
    box = (best_left, best_top, best_left + crop_width, best_top + crop_height)
    return img.crop(box), mask.crop(box)


def color_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() >= 0.8:
        return img
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = rng.uniform(0.8, 1.2)
        img = enhancer_cls(img).enhance(factor)
    return img


def preprocess_augment(img: Image.Image, rng: random.Random, prob: float) -> Image.Image:
    if prob <= 0 or rng.random() >= prob:
        return img
    if rng.random() < 0.7:
        img = ImageOps.autocontrast(img, cutoff=rng.uniform(0.0, 1.5))
    if rng.random() < 0.6:
        img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.85, 1.7))
    if rng.random() < 0.35:
        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=rng.uniform(0.5, 1.2),
                percent=int(rng.uniform(60, 140)),
                threshold=3,
            )
        )
    return img


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


class SpinUVDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        crop_size: tuple[int, int],
        ignore_index: int,
        train: bool,
        seed: int,
        crop_repeat_factor: int = 1,
        anchor_crop_prob: float = 0.0,
        anchor_crop_jitter: float = 0.08,
        preprocess_aug_prob: float = 0.0,
        hard_class_ids: Sequence[int] | None = None,
        hard_example_sampling: bool = False,
        hard_example_prob: float = 0.0,
        hard_example_mode: str = "mixed_boundary",
        hard_example_crop_candidates: int = 8,
        hard_example_min_pixels: int = 512,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size
        self.ignore_index = ignore_index
        self.train = train
        self.seed = seed
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
        self.image_paths = sorted((data_root / "images" / split).glob("*.png"))
        self.mask_paths = [data_root / "masks" / split / p.name for p in self.image_paths]
        missing = [p for p in self.mask_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} masks for split={split}; first={missing[0]}")
        if not self.image_paths:
            raise FileNotFoundError(f"No images found for split={split} under {data_root}")

    def __len__(self) -> int:
        if self.train:
            return len(self.image_paths) * self.crop_repeat_factor
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = index % len(self.image_paths)
        image_path = self.image_paths[base_index]
        mask_path = self.mask_paths[base_index]
        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        rng = random if self.train else random.Random(self.seed + index)
        if self.train:
            img = preprocess_augment(img, rng, self.preprocess_aug_prob)
            img, mask = random_resized_pair(img, mask, rng)
            if self.hard_example_sampling and rng.random() < self.hard_example_prob:
                img, mask = hard_random_crop_pair(
                    img,
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
                img, mask = random_crop_pair(
                    img,
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
                mask = ImageOps.mirror(mask)
            img = color_jitter(img, rng)

        image_tensor = image_to_tensor(img)
        mask_arr = np.asarray(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_arr).long()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def make_loader(
    data_root: Path,
    split: str,
    crop_size: tuple[int, int],
    ignore_index: int,
    train: bool,
    batch_size: int,
    num_workers: int,
    seed: int,
    crop_repeat_factor: int = 1,
    anchor_crop_prob: float = 0.0,
    anchor_crop_jitter: float = 0.08,
    preprocess_aug_prob: float = 0.0,
    hard_class_ids: Sequence[int] | None = None,
    hard_example_sampling: bool = False,
    hard_example_prob: float = 0.0,
    hard_example_mode: str = "mixed_boundary",
    hard_example_crop_candidates: int = 8,
    hard_example_min_pixels: int = 512,
) -> DataLoader:
    dataset = SpinUVDataset(
        data_root=data_root,
        split=split,
        crop_size=crop_size,
        ignore_index=ignore_index,
        train=train,
        seed=seed,
        crop_repeat_factor=crop_repeat_factor,
        anchor_crop_prob=anchor_crop_prob,
        anchor_crop_jitter=anchor_crop_jitter,
        preprocess_aug_prob=preprocess_aug_prob,
        hard_class_ids=hard_class_ids,
        hard_example_sampling=hard_example_sampling,
        hard_example_prob=hard_example_prob,
        hard_example_mode=hard_example_mode,
        hard_example_crop_candidates=hard_example_crop_candidates,
        hard_example_min_pixels=hard_example_min_pixels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size if train else 1,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=num_workers > 0,
    )


def label_maps(class_names: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    id2label = {idx: name for idx, name in enumerate(class_names)}
    label2id = {name: idx for idx, name in id2label.items()}
    return id2label, label2id


def build_model(args: argparse.Namespace, class_names: list[str], ignore_index: int) -> torch.nn.Module:
    class_count = len(class_names)
    framework = args.framework.lower()
    if framework == "smp":
        import segmentation_models_pytorch as smp

        if not hasattr(smp, args.model_name):
            raise ValueError(f"Unsupported SMP model_name={args.model_name}")
        model_cls = getattr(smp, args.model_name)
        return model_cls(
            encoder_name=args.encoder_name,
            encoder_weights=normalize_weights_arg(args.encoder_weights),
            in_channels=3,
            classes=class_count,
            activation=None,
        )

    id2label, label2id = label_maps(class_names)
    model_id = args.model_id or args.model_name
    common_kwargs = dict(
        num_labels=class_count,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    if framework == "hf_segformer":
        from transformers import AutoConfig, SegformerForSemanticSegmentation

        if args.hf_from_config:
            config = AutoConfig.from_pretrained(
                model_id,
                num_labels=class_count,
                id2label=id2label,
                label2id=label2id,
            )
            model = SegformerForSemanticSegmentation(config)
        else:
            model = SegformerForSemanticSegmentation.from_pretrained(model_id, **common_kwargs)
    elif framework == "hf_upernet":
        from transformers import AutoConfig, UperNetForSemanticSegmentation

        if args.hf_from_config:
            config = AutoConfig.from_pretrained(
                model_id,
                num_labels=class_count,
                id2label=id2label,
                label2id=label2id,
            )
            model = UperNetForSemanticSegmentation(config)
        else:
            model = UperNetForSemanticSegmentation.from_pretrained(model_id, **common_kwargs)
    elif framework == "hf_auto":
        from transformers import AutoConfig, AutoModelForSemanticSegmentation

        if args.hf_from_config:
            config = AutoConfig.from_pretrained(
                model_id,
                num_labels=class_count,
                id2label=id2label,
                label2id=label2id,
            )
            model = AutoModelForSemanticSegmentation.from_config(config)
        else:
            model = AutoModelForSemanticSegmentation.from_pretrained(model_id, **common_kwargs)
    else:
        raise ValueError(f"Unsupported framework={args.framework}")

    if hasattr(model.config, "semantic_loss_ignore_index"):
        model.config.semantic_loss_ignore_index = ignore_index
    if hasattr(model.config, "ignore_index"):
        model.config.ignore_index = ignore_index
    if args.reset_classifier:
        reset_paths = reset_semantic_classifiers(model)
        setattr(model, "_spinuv_reset_classifier_paths", reset_paths)
    return model


def new_like_module(module: torch.nn.Module) -> torch.nn.Module:
    if isinstance(module, torch.nn.Conv2d):
        return torch.nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
    if isinstance(module, torch.nn.Linear):
        return torch.nn.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
    raise TypeError(f"Unsupported classifier module type for reset: {type(module).__name__}")


def reset_child_module(parent: torch.nn.Module, attr_name: str, path: str) -> str | None:
    child = getattr(parent, attr_name, None)
    if not isinstance(child, (torch.nn.Conv2d, torch.nn.Linear)):
        return None
    replacement = new_like_module(child)
    setattr(parent, attr_name, replacement)
    return path


def reset_semantic_classifiers(model: torch.nn.Module) -> list[str]:
    reset_paths: list[str] = []
    for holder_name in ("decode_head", "auxiliary_head", "auxiliary_head.0"):
        holder: Any = model
        parts = holder_name.split(".")
        for part in parts:
            if isinstance(holder, (list, torch.nn.ModuleList)) and part.isdigit():
                idx = int(part)
                holder = holder[idx] if idx < len(holder) else None
            else:
                holder = getattr(holder, part, None)
            if holder is None:
                break
        if holder is None:
            continue
        reset_path = reset_child_module(holder, "classifier", f"{holder_name}.classifier")
        if reset_path:
            reset_paths.append(reset_path)
    reset_path = reset_child_module(model, "classifier", "classifier")
    if reset_path:
        reset_paths.append(reset_path)
    if not reset_paths:
        raise ValueError("reset_classifier was requested, but no supported semantic classifier was found.")
    return reset_paths


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer={args.optimizer}")


def poly_lr(base_lr: float, current_iter: int, max_iters: int, power: float) -> float:
    progress = min(float(current_iter) / float(max_iters), 1.0)
    return base_lr * ((1.0 - progress) ** power)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def update_confusion(confusion: np.ndarray, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> None:
    pred_cpu = pred.detach().to(torch.int64).cpu().view(-1)
    target_cpu = target.detach().to(torch.int64).cpu().view(-1)
    valid = (target_cpu != ignore_index) & (target_cpu >= 0) & (target_cpu < num_classes) & (pred_cpu >= 0) & (pred_cpu < num_classes)
    if valid.sum().item() == 0:
        return
    encoded = target_cpu[valid] * num_classes + pred_cpu[valid]
    hist = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += hist.reshape(num_classes, num_classes).numpy()


def metrics_from_confusion(confusion: np.ndarray, analysis_class_ids: Sequence[int] | None = None) -> dict[str, Any]:
    intersection = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(axis=1).astype(np.float64)
    pred = confusion.sum(axis=0).astype(np.float64)
    union = gt + pred - intersection
    iou = np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0)
    acc = np.divide(intersection, gt, out=np.full_like(intersection, np.nan), where=gt > 0)
    ids = np.array(list(analysis_class_ids) if analysis_class_ids is not None else list(range(len(iou))), dtype=np.int64)
    selected_iou = iou[ids]
    selected_acc = acc[ids]
    selected_intersection = intersection[ids].sum()
    selected_gt = gt[ids].sum()
    pixel_accuracy = float(selected_intersection / max(selected_gt, 1.0))
    valid_iou = selected_iou[~np.isnan(selected_iou)]
    valid_acc = selected_acc[~np.isnan(selected_acc)]
    return {
        "per_class_iou": iou,
        "per_class_accuracy": acc,
        "support_pixels": gt,
        "mIoU": float(valid_iou.mean()) if valid_iou.size else float("nan"),
        "mean_accuracy": float(valid_acc.mean()) if valid_acc.size else float("nan"),
        "pixel_accuracy": pixel_accuracy,
        "analysis_class_ids": ids.tolist(),
    }


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray, palette: np.ndarray, ignore_index: int) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    out[valid] = palette[mask[valid]]
    out[mask == ignore_index] = np.array([40, 40, 40], dtype=np.uint8)
    return out


def save_visual_grid(samples: list[dict[str, Any]], output_path: Path, split: str, ignore_index: int) -> None:
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.2 * rows), squeeze=False)
    for row, sample in enumerate(samples):
        image = sample["image"]
        gt = sample["gt"]
        pred = sample["pred"]
        gt_color = colorize_mask(gt, DEFAULT_PALETTE, ignore_index)
        pred_color = colorize_mask(pred, DEFAULT_PALETTE, ignore_index)
        error = np.zeros_like(image)
        valid = gt != ignore_index
        error[valid & (gt == pred)] = np.array([35, 120, 60], dtype=np.uint8)
        error[valid & (gt != pred)] = np.array([210, 45, 45], dtype=np.uint8)
        error[~valid] = np.array([40, 40, 40], dtype=np.uint8)
        panels = [image, gt_color, pred_color, error]
        titles = ["Image", "Ground truth", "Prediction", "Error map"]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, col].imshow(panel)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"{split} sample {row + 1}", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_iou_plot(per_class_rows: list[dict[str, Any]], output_path: Path, split: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_rows = sorted(
        per_class_rows,
        key=lambda row: float(row["iou"]) if row["iou"] not in ("", None) and math.isfinite(float(row["iou"])) else -1.0,
        reverse=True,
    )
    names = [row["class_name"] for row in ordered_rows]
    values = []
    value_labels = []
    for row in ordered_rows:
        raw_value = row["iou"]
        if raw_value == "" or raw_value is None:
            values.append(0.0)
            value_labels.append("n/a")
        else:
            value = float(raw_value)
            if math.isfinite(value):
                values.append(value)
                value_labels.append(f"{value:.3f}")
            else:
                values.append(0.0)
                value_labels.append("n/a")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(names))))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color="#4f7cac")
    for bar, label, value in zip(bars, value_labels, values):
        x_pos = min(value + 0.015, 0.96)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("IoU")
    ax.set_title(f"Per-class IoU on SPIN-UV {split}")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def autocast_context(device: torch.device, enabled: bool):
    use_amp = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def write_metrics(
    run_dir: Path,
    split: str,
    class_names: list[str],
    analysis_class_ids: Sequence[int],
    confusion: np.ndarray,
    per_image_rows: list[dict[str, Any]],
    total_params: int,
    trainable_params: int,
) -> dict[str, Any]:
    metrics = metrics_from_confusion(confusion, analysis_class_ids)
    per_class_rows = []
    for idx in analysis_class_ids:
        name = class_names[idx]
        per_class_rows.append(
            {
                "class_id": idx,
                "class_name": name,
                "iou": float(metrics["per_class_iou"][idx]) if not math.isnan(metrics["per_class_iou"][idx]) else "",
                "accuracy": float(metrics["per_class_accuracy"][idx]) if not math.isnan(metrics["per_class_accuracy"][idx]) else "",
                "support_pixels": int(metrics["support_pixels"][idx]),
            }
        )
    per_class_rows = sorted(
        per_class_rows,
        key=lambda row: float(row["iou"]) if row["iou"] not in ("", None) and math.isfinite(float(row["iou"])) else -1.0,
        reverse=True,
    )

    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_class_csv = tables_dir / f"{split}_per_class_iou.csv"
    with per_class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "iou", "accuracy", "support_pixels"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    per_image_csv = tables_dir / f"{split}_per_image_iou.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "image_iou", "valid_pixels"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary = {
        "split": split,
        "mIoU": metrics["mIoU"],
        "mean_accuracy": metrics["mean_accuracy"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "analysis_class_count": len(analysis_class_ids),
        "analysis_class_names": "|".join(class_names[idx] for idx in analysis_class_ids),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "per_class_csv": str(per_class_csv.as_posix()),
        "per_image_csv": str(per_image_csv.as_posix()),
    }
    with (tables_dir / f"{split}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (tables_dir / f"{split}_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    save_iou_plot(per_class_rows, figures_dir / f"{split}_per_class_iou.png", split)
    return summary


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    return images, masks


def forward_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
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
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
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
            valid_pixels = int(item_conf.sum())
            row = {
                "image_path": batch["image_path"][item_idx],
                "image_iou": image_iou,
                "valid_pixels": valid_pixels,
            }
            per_image_rows.append(row)
            sample = {
                "image": denormalize_image(images[item_idx]),
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


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        base = Path(args.run_dir)
        if not base.is_absolute():
            base = ROOT / base
    else:
        base = ROOT / "runs" / args.run_name
    if args.resume or args.eval_only:
        base.mkdir(parents=True, exist_ok=True)
        return base
    if not base.exists() or not any(base.iterdir()):
        base.mkdir(parents=True, exist_ok=True)
        return base
    suffix = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base.with_name(f"{base.name}_{suffix}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    iteration: int,
    best_miou: float,
    args: argparse.Namespace,
    class_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "args": vars(args),
            "class_names": class_names,
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and isinstance(checkpoint, dict) and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint if isinstance(checkpoint, dict) else {}


def normalize_checkpoint_path(value: str | Path) -> Path:
    path = Path(value)
    if path.suffix == ".pt":
        checkpoint_path = path
    elif path.name == "checkpoints":
        checkpoint_path = path / "best_miou.pt"
        if not checkpoint_path.exists():
            checkpoint_path = path / "latest.pt"
    else:
        checkpoint_path = path / "checkpoints" / "best_miou.pt"
        if not checkpoint_path.exists():
            checkpoint_path = path / "checkpoints" / "latest.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    class_names, ignore_from_data = load_classes(data_root)
    analysis_class_ids = resolve_analysis_class_ids(class_names, args.analysis_classes)
    hard_class_ids = resolve_optional_class_ids(class_names, args.hard_example_classes)
    hard_example_enabled = bool(args.hard_example_sampling and hard_class_ids and args.hard_example_prob > 0)
    if args.ignore_index != ignore_from_data:
        args.ignore_index = ignore_from_data
    crop_size = (args.crop_height, args.crop_width)
    run_dir = make_run_dir(args)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if args.config:
        config_src = Path(args.config)
        if not config_src.is_absolute():
            config_src = ROOT / config_src
        if config_src.exists():
            shutil.copyfile(config_src, run_dir / "config_snapshot.json")
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args, class_names, args.ignore_index).to(device)
    gradient_checkpointing_enabled = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            gradient_checkpointing_enabled = True
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "gradient_checkpointing_skipped",
                        "reason": str(exc),
                    },
                    sort_keys=True,
                )
            )
    init_checkpoint_path = normalize_checkpoint_path(args.init_checkpoint) if args.init_checkpoint else None
    if init_checkpoint_path is not None:
        load_checkpoint(init_checkpoint_path, model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with (run_dir / "model_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "framework": args.framework,
                "model_id": args.model_id,
                "initialization": args.initialization,
                "hf_from_config": bool(args.hf_from_config),
                "encoder_name": args.encoder_name,
                "encoder_weights": args.encoder_weights,
                "reset_classifier": bool(args.reset_classifier),
                "gradient_checkpointing": bool(gradient_checkpointing_enabled),
                "reset_classifier_paths": getattr(model, "_spinuv_reset_classifier_paths", []),
                "class_count": len(class_names),
                "analysis_class_count": len(analysis_class_ids),
                "analysis_class_names": [class_names[idx] for idx in analysis_class_ids],
                "hard_example_sampling": bool(hard_example_enabled),
                "hard_example_prob": float(args.hard_example_prob),
                "hard_example_mode": args.hard_example_mode,
                "hard_example_class_ids": [int(idx) for idx in hard_class_ids],
                "hard_example_class_names": [class_names[idx] for idx in hard_class_ids],
                "hard_example_crop_candidates": int(args.hard_example_crop_candidates),
                "hard_example_min_pixels": int(args.hard_example_min_pixels),
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "init_checkpoint": init_checkpoint_path.as_posix() if init_checkpoint_path else "",
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

    val_loader = make_loader(data_root, "val", crop_size, args.ignore_index, False, 1, args.num_workers, args.seed)
    test_loader = make_loader(data_root, "test", crop_size, args.ignore_index, False, 1, args.num_workers, args.seed)

    if args.eval_only:
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best_miou.pt"
        load_checkpoint(checkpoint, model)
        evaluate(model, val_loader, device, class_names, analysis_class_ids, args.ignore_index, "val", run_dir, args.sample_count, args.amp, total_params, trainable_params)
        if not args.skip_test:
            evaluate(model, test_loader, device, class_names, analysis_class_ids, args.ignore_index, "test", run_dir, args.sample_count, args.amp, total_params, trainable_params)
        print(f"eval_complete run_dir={run_dir.as_posix()}")
        return

    train_loader = make_loader(
        data_root,
        "train",
        crop_size,
        args.ignore_index,
        True,
        args.batch_size,
        args.num_workers,
        args.seed,
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
    train_iter = iter(train_loader)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_index)
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
            "val_images": len(val_loader.dataset),
            "test_images": len(test_loader.dataset),
            "framework": args.framework,
            "model_name": args.model_name,
            "model_id": args.model_id,
            "initialization": args.initialization,
            "hf_from_config": bool(args.hf_from_config),
            "optimizer": args.optimizer,
            "micro_batch_size": int(args.batch_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
            "optimizer_updates": int(args.max_iters),
            "train_crop_exposures": int(args.max_iters * args.batch_size * args.grad_accum_steps),
            "crop_repeat_factor": int(args.crop_repeat_factor),
            "anchor_crop_prob": float(args.anchor_crop_prob),
            "anchor_crop_jitter": float(args.anchor_crop_jitter),
            "preprocess_aug_prob": float(args.preprocess_aug_prob),
            "reset_classifier": bool(args.reset_classifier),
            "gradient_checkpointing": bool(gradient_checkpointing_enabled),
            "reset_classifier_paths": getattr(model, "_spinuv_reset_classifier_paths", []),
            "analysis_class_count": len(analysis_class_ids),
            "analysis_class_names": [class_names[idx] for idx in analysis_class_ids],
            "hard_example_sampling": bool(hard_example_enabled),
            "hard_example_prob": float(args.hard_example_prob),
            "hard_example_mode": args.hard_example_mode,
            "hard_example_class_ids": [int(idx) for idx in hard_class_ids],
            "hard_example_class_names": [class_names[idx] for idx in hard_class_ids],
            "hard_example_crop_candidates": int(args.hard_example_crop_candidates),
            "hard_example_min_pixels": int(args.hard_example_min_pixels),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "init_checkpoint": init_checkpoint_path.as_posix() if init_checkpoint_path else "",
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
            payload = {
                "event": "train",
                "iteration": iteration,
                "loss": avg_loss,
                "lr": lr,
                "seconds_since_last_log": now - last_log_time,
            }
            write_jsonl(log_path, payload)
            print(json.dumps(payload, sort_keys=True))
            last_log_time = now

        if iteration % args.val_interval == 0 or iteration == args.max_iters:
            summary = evaluate(
                model,
                val_loader,
                device,
                class_names,
                analysis_class_ids,
                args.ignore_index,
                "val",
                run_dir,
                args.sample_count,
                args.amp,
                total_params,
                trainable_params,
            )
            payload = {"event": "val", "iteration": iteration, **summary}
            write_jsonl(log_path, payload)
            print(json.dumps(payload, sort_keys=True))
            if summary["mIoU"] > best_miou:
                best_miou = float(summary["mIoU"])
                save_checkpoint(
                    run_dir / "checkpoints" / "best_miou.pt",
                    model,
                    optimizer,
                    scaler,
                    iteration,
                    best_miou,
                    args,
                    class_names,
                )
            model.train()

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            save_checkpoint(
                run_dir / "checkpoints" / "latest.pt",
                model,
                optimizer,
                scaler,
                iteration,
                best_miou,
                args,
                class_names,
            )

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

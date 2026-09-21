from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASSES = ("EMPTY", "ACCEPTED", "REJECTED")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Config:
    data_root: Path
    artifacts: Path
    annotations: Path | None = None
    labels_csv: Path | None = None
    infer_dir: Path | None = None
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    dino_model: str = "IDEA-Research/grounding-dino-base"
    dino_box_threshold: float = 0.18
    dino_text_threshold: float = 0.16
    merge_iou: float = 0.32
    merge_ios: float = 0.72
    final_iou: float = 0.62
    final_ios: float = 0.88
    classifier_model: str = "efficientnet_b2"
    classifier_resolution: int = 320
    classifier_context: float = 0.05
    classifier_batch_size: int = 32
    classifier_workers: int = 4
    stage1_epochs: int = 3
    stage2_epochs: int = 12
    early_stopping_patience: int = 3
    head_lr: float = 2e-3
    finetune_lr: float = 2e-4
    weight_decay: float = 1e-4
    infer_batch_size: int = 32
    max_failure_images: int = 30


@dataclass
class PalletAnnotation:
    image_name: str
    image_path: str
    bbox: list[float]
    label: str | None
    group_id: str
    annotation_id: str


@dataclass
class DinoCandidate:
    box: list[float]
    score: float
    geometry_score: float
    view: str


class PadToSquare:
    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        size = max(w, h)
        left = (size - w) // 2
        top = (size - h) // 2
        canvas = Image.new("RGB", (size, size), (self.fill, self.fill, self.fill))
        canvas.paste(image, (left, top))
        return canvas


class CropDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], resolution: int, train: bool):
        self.rows = rows
        ops: list[Any] = [PadToSquare()]
        if train:
            ops += [
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.05),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.7))], p=0.08),
                transforms.RandomAffine(degrees=2.0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            ]
        ops += [
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASS_TO_ID[row["class"]], row["image_name"], int(row["pallet_index"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def device_for_runtime() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "EMPTY": "EMPTY",
        "EMPTY_PALLET": "EMPTY",
        "ACCEPTED": "ACCEPTED",
        "ACCEPT": "ACCEPTED",
        "GOOD": "ACCEPTED",
        "PASS": "ACCEPTED",
        "REJECTED": "REJECTED",
        "REJECT": "REJECTED",
        "BAD": "REJECTED",
        "FAIL": "REJECTED",
    }
    return aliases.get(text)


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def build_image_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    images = find_images(root)
    by_rel = {str(p.relative_to(root)).replace("\\", "/"): p for p in images}
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        by_name[path.name].append(path)
    return by_rel, by_name


def resolve_image_path(
    name: str,
    root: Path,
    by_rel: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> Path:
    clean = name.replace("\\", "/").lstrip("./")
    clean_lower = clean.lower()

    # 1. Exact relative path.
    if clean in by_rel:
        return by_rel[clean]

    # 2. Case-insensitive exact relative path.
    for rel_name, path in by_rel.items():
        if rel_name.lower() == clean_lower:
            return path

    # 3. Allow an extra dataset folder before annotation path.
    #
    # Example:
    # annotation:
    #     empty/original_0039.jpg
    #
    # possible dataset path:
    #     new-dataset/empty/original_0039.jpg
    suffix_matches = [
        path
        for rel_name, path in by_rel.items()
        if rel_name.lower().endswith("/" + clean_lower)
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # 4. Use parent folder + basename to disambiguate.
    annotation_path = Path(clean)
    basename = annotation_path.name
    parent_name = (
        annotation_path.parent.name.lower()
        if annotation_path.parent.name
        else ""
    )

    matches = by_name.get(
        basename,
        [],
    )

    if parent_name:
        parent_matches = [
            path
            for path in matches
            if path.parent.name.lower() == parent_name
        ]

        if len(parent_matches) == 1:
            return parent_matches[0]

    # 5. Basename fallback only when unique.
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Annotation references missing image: {name}"
        )

    candidate_text = "\n".join(
        str(path)
        for path in matches[:20]
    )

    raise RuntimeError(
        f"Ambiguous image path '{name}'.\n"
        f"Candidates:\n{candidate_text}"
    )

def infer_group_id(
    image_path: Path,
    root: Path,
) -> str:
    return str(
        image_path
        .relative_to(root)
        .with_suffix("")
    ).replace("\\", "/")

def box_from_item(item: Any) -> list[float] | None:
    if isinstance(item, (list, tuple)) and len(item) == 4:
        try:
            return [float(v) for v in item]
        except (TypeError, ValueError):
            return None
    if not isinstance(item, dict):
        return None
    for key in ("bbox_2d", "box", "bbox", "pallet_box", "xyxy"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return [float(v) for v in value]
            except (TypeError, ValueError):
                pass
    keys = ("x1", "y1", "x2", "y2")
    if all(k in item for k in keys):
        return [float(item[k]) for k in keys]
    if all(k in item for k in ("x", "y", "w", "h")):
        x, y, w, h = (float(item[k]) for k in ("x", "y", "w", "h"))
        return [x, y, x + w, y + h]
    return None


def item_label(item: dict[str, Any]) -> str | None:
    for key in ("class", "label", "status", "category", "category_name"):
        if key in item:
            label = normalize_label(item[key])
            if label:
                return label
    return None


def auto_find_annotations(root: Path) -> Path:
    preferred = sorted(root.rglob("*pallet*annotation*.json")) + sorted(root.rglob("*annotation*.json"))
    preferred = [p for p in preferred if "artifacts" not in p.parts]
    if not preferred:
        raise FileNotFoundError(
            "No annotation JSON was found. Provide --annotations PATH. "
            "The pipeline will not fabricate pallet boxes."
        )
    return preferred[0]


def parse_annotations(
    cfg: Config,
) -> tuple[list[PalletAnnotation], Path]:

    annotation_path = (
        cfg.annotations
        or auto_find_annotations(
            cfg.data_root
        )
    )

    by_rel, by_name = build_image_index(
        cfg.data_root
    )

    data = json.loads(
        annotation_path.read_text(
            encoding="utf-8"
        )
    )

    records: list[PalletAnnotation] = []

    def add_record(
        image_name: str,
        box: list[float],
        label: str | None,
        ann_id: str,
    ) -> None:

        path = resolve_image_path(
            image_name,
            cfg.data_root,
            by_rel,
            by_name,
        )

        records.append(
            PalletAnnotation(
                image_name=str(
                    path.relative_to(
                        cfg.data_root
                    )
                ).replace("\\", "/"),

                image_path=str(path),

                bbox=[
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ],

                label=label,

                group_id=infer_group_id(
                    path,
                    cfg.data_root,
                ),

                annotation_id=ann_id,
            )
        )

    # ========================================================
    # CREST LAB ANNOTATION FORMAT VERSION 2
    #
    # {
    #   "version": 2,
    #   "classes": ["EMPTY", "REJECTED"],
    #   "selected_images": [...],
    #   "images": {
    #       "empty/original_0039.jpg": {
    #           "pallets": [
    #               {
    #                   "bbox": [x1,y1,x2,y2],
    #                   "class": "EMPTY"
    #               }
    #           ]
    #       }
    #   }
    # }
    #
    # IMPORTANT:
    # bbox is ALREADY xyxy.
    # Do NOT convert it as COCO xywh.
    # ========================================================

    if (
        isinstance(data, dict)
        and isinstance(
            data.get("images"),
            dict,
        )
    ):

        selected_labels = {}

        selected_images = data.get(
            "selected_images",
            [],
        )

        if isinstance(
            selected_images,
            list,
        ):
            for selected in selected_images:

                if not isinstance(
                    selected,
                    dict,
                ):
                    continue

                image_name = selected.get(
                    "image"
                )

                if not image_name:
                    continue

                source_label = normalize_label(
                    selected.get(
                        "source_class"
                    )
                )

                if source_label:
                    selected_labels[
                        str(image_name)
                    ] = source_label

        sequence = 0

        for image_name, image_record in data[
            "images"
        ].items():

            if not isinstance(
                image_record,
                dict,
            ):
                continue

            parent_label = (
                item_label(
                    image_record
                )
                or selected_labels.get(
                    str(image_name)
                )
            )

            pallets = image_record.get(
                "pallets",
                [],
            )

            if isinstance(
                pallets,
                dict,
            ):
                pallets = list(
                    pallets.values()
                )

            if not isinstance(
                pallets,
                list,
            ):
                continue

            for pallet_index, pallet in enumerate(
                pallets
            ):

                if not isinstance(
                    pallet,
                    dict,
                ):
                    continue

                box = pallet.get(
                    "bbox"
                )

                if (
                    not isinstance(
                        box,
                        (list, tuple),
                    )
                    or len(box) != 4
                ):
                    continue

                try:
                    xyxy = [
                        float(value)
                        for value in box
                    ]
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                label = (
                    item_label(
                        pallet
                    )
                    or parent_label
                )

                add_record(
                    str(image_name),
                    xyxy,
                    label,
                    (
                        f"crest_v2:"
                        f"{sequence}:"
                        f"{pallet_index}"
                    ),
                )

            sequence += 1

        if not records:
            raise RuntimeError(
                "CREST v2 annotation file was detected, "
                "but no valid pallet boxes were found."
            )

        return (
            records,
            annotation_path,
        )

    # ========================================================
    # COCO FORMAT FALLBACK
    # ========================================================

    if (
        isinstance(data, dict)
        and all(
            key in data
            for key in (
                "images",
                "annotations",
            )
        )
        and isinstance(
            data["images"],
            list,
        )
    ):

        images = {
            item["id"]: item
            for item in data["images"]
            if (
                isinstance(
                    item,
                    dict,
                )
                and "id" in item
            )
        }

        categories = {
            item["id"]: normalize_label(
                item.get(
                    "name"
                )
            )
            for item in data.get(
                "categories",
                [],
            )
            if (
                isinstance(
                    item,
                    dict,
                )
                and "id" in item
            )
        }

        for index, annotation in enumerate(
            data["annotations"]
        ):

            if (
                not isinstance(
                    annotation,
                    dict,
                )
                or annotation.get(
                    "image_id"
                )
                not in images
            ):
                continue

            box = annotation.get(
                "bbox"
            )

            if (
                not isinstance(
                    box,
                    list,
                )
                or len(box) != 4
            ):
                continue

            x, y, width, height = map(
                float,
                box,
            )

            xyxy = [
                x,
                y,
                x + width,
                y + height,
            ]

            label = (
                item_label(
                    annotation
                )
                or categories.get(
                    annotation.get(
                        "category_id"
                    )
                )
            )

            add_record(
                images[
                    annotation["image_id"]
                ].get(
                    "file_name",
                    "",
                ),
                xyxy,
                label,
                str(
                    annotation.get(
                        "id",
                        index,
                    )
                ),
            )

        if records:
            return (
                records,
                annotation_path,
            )

    # ========================================================
    # GENERIC FORMAT FALLBACK
    # ========================================================

    def consume_image_record(
        image_record: dict[str, Any],
        sequence: int,
    ) -> None:

        image_name = next(
            (
                image_record.get(key)
                for key in (
                    "image_name",
                    "file_name",
                    "image",
                    "path",
                )
                if image_record.get(key)
            ),
            None,
        )

        if not image_name:
            return

        parent_label = item_label(
            image_record
        )

        items: Any = None

        for key in (
            "pallets",
            "detections",
            "annotations",
            "boxes",
        ):
            if key in image_record:
                items = image_record[key]
                break

        if items is None:

            single = box_from_item(
                image_record
            )

            if single:
                add_record(
                    str(image_name),
                    single,
                    parent_label,
                    str(
                        image_record.get(
                            "id",
                            sequence,
                        )
                    ),
                )

            return

        if isinstance(
            items,
            dict,
        ):
            items = list(
                items.values()
            )

        if not isinstance(
            items,
            list,
        ):
            return

        for item_index, item in enumerate(
            items
        ):

            box = box_from_item(
                item
            )

            if box is None:
                continue

            label = (
                item_label(
                    item
                )
                if isinstance(
                    item,
                    dict,
                )
                else None
            )

            add_record(
                str(image_name),
                box,
                label or parent_label,
                f"{sequence}:{item_index}",
            )

    if (
        isinstance(
            data,
            dict,
        )
        and isinstance(
            data.get(
                "images"
            ),
            list,
        )
    ):

        for index, image_record in enumerate(
            data["images"]
        ):

            if isinstance(
                image_record,
                dict,
            ):
                consume_image_record(
                    image_record,
                    index,
                )

    elif isinstance(
        data,
        list,
    ):

        for index, image_record in enumerate(
            data
        ):

            if isinstance(
                image_record,
                dict,
            ):
                consume_image_record(
                    image_record,
                    index,
                )

    elif isinstance(
        data,
        dict,
    ):

        for index, (
            image_name,
            value,
        ) in enumerate(
            data.items()
        ):

            if not isinstance(
                value,
                (
                    list,
                    dict,
                ),
            ):
                continue

            if (
                isinstance(
                    value,
                    dict,
                )
                and any(
                    key in value
                    for key in (
                        "pallets",
                        "detections",
                        "annotations",
                        "boxes",
                    )
                )
            ):

                record = dict(
                    value
                )

                record.setdefault(
                    "image_name",
                    image_name,
                )

                consume_image_record(
                    record,
                    index,
                )

                continue

            items = (
                value
                if isinstance(
                    value,
                    list,
                )
                else [value]
            )

            for item_index, item in enumerate(
                items
            ):

                box = box_from_item(
                    item
                )

                if box is None:
                    continue

                label = (
                    item_label(
                        item
                    )
                    if isinstance(
                        item,
                        dict,
                    )
                    else None
                )

                add_record(
                    image_name,
                    box,
                    label,
                    f"{index}:{item_index}",
                )

    if not records:
        raise RuntimeError(
            f"Could not parse any pallet boxes from "
            f"{annotation_path}"
        )

    return (
        records,
        annotation_path,
    )

def apply_labels_csv(records: list[PalletAnnotation], labels_csv: Path | None) -> None:
    if labels_csv is None or not labels_csv.exists():
        return
    rows = list(csv.DictReader(labels_csv.open("r", encoding="utf-8", newline="")))
    lookup: dict[tuple[str, int], str] = {}
    for row in rows:
        label = normalize_label(row.get("class"))
        if not label:
            continue
        lookup[(row["image_name"].replace("\\", "/"), int(row["pallet_index"]))] = label
    by_image: dict[str, list[PalletAnnotation]] = defaultdict(list)
    for rec in records:
        by_image[rec.image_name].append(rec)
    for image_name, items in by_image.items():
        items.sort(key=lambda r: ((r.bbox[1] + r.bbox[3]) / 2, (r.bbox[0] + r.bbox[2]) / 2))
        for index, rec in enumerate(items, start=1):
            rec.label = lookup.get((image_name, index), rec.label)


def validate_annotations(records: list[PalletAnnotation]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    size_cache: dict[str, tuple[int, int]] = {}
    for rec in records:
        if rec.annotation_id in seen_ids:
            # Duplicate IDs are only meaningful if they are globally explicit. Composite parser IDs may repeat across images.
            pass
        seen_ids.add(rec.annotation_id)
        if rec.image_path not in size_cache:
            image = cv2.imread(rec.image_path)
            if image is None:
                errors.append(f"Corrupt/missing image: {rec.image_path}")
                continue
            h, w = image.shape[:2]
            size_cache[rec.image_path] = (w, h)
        w, h = size_cache[rec.image_path]
        x1, y1, x2, y2 = rec.bbox
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            errors.append(f"Invalid box {rec.bbox} for {rec.image_name} ({w}x{h})")
        if rec.label is not None and rec.label not in CLASSES:
            errors.append(f"Unknown class {rec.label} for {rec.image_name}")
    return errors


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(cfg: Config, records: list[PalletAnnotation], annotation_path: Path) -> dict[str, Any]:
    images = find_images(cfg.data_root)
    corrupt: list[str] = []
    resolutions = Counter()
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            corrupt.append(str(path))
        else:
            h, w = image.shape[:2]
            resolutions[f"{w}x{h}"] += 1

    hash_groups: dict[str, list[str]] = defaultdict(list)
    for path in images:
        try:
            hash_groups[sha1_file(path)].append(str(path.relative_to(cfg.data_root)))
        except OSError:
            pass
    duplicates = [group for group in hash_groups.values() if len(group) > 1]
    label_counts = Counter(rec.label or "UNLABELED" for rec in records)
    annotated_images = {rec.image_name for rec in records}
    mixed_images = 0
    labels_by_image: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        if rec.label:
            labels_by_image[rec.image_name].add(rec.label)
    mixed_images = sum(1 for labels in labels_by_image.values() if len(labels) > 1)

    folder_counts = Counter(str(p.parent.relative_to(cfg.data_root)) for p in images)
    audit = {
        "data_root": str(cfg.data_root),
        "annotation_file": str(annotation_path),
        "images": len(images),
        "annotated_images": len(annotated_images),
        "annotated_pallets": len(records),
        "class_counts": dict(label_counts),
        "mixed_labeled_images": mixed_images,
        "corrupt_images": corrupt,
        "duplicate_exact_groups": duplicates,
        "resolutions": dict(resolutions),
        "image_folders": dict(folder_counts),
    }
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nDATASET AUDIT")
    print("Images:", len(images))
    print("Annotated source images:", len(annotated_images))
    print("Annotated pallets:", len(records))
    for name in (*CLASSES, "UNLABELED"):
        if label_counts[name]:
            print(f"{name}: {label_counts[name]}")
    print("Mixed labeled images:", mixed_images)
    print("Corrupt images:", len(corrupt))
    print("Exact duplicate groups:", len(duplicates))
    print("Annotation file:", annotation_path)
    return audit


def write_label_template(cfg: Config, records: list[PalletAnnotation]) -> Path:
    output = cfg.artifacts / "pallet_label_template.csv"
    by_image: dict[str, list[PalletAnnotation]] = defaultdict(list)
    for rec in records:
        by_image[rec.image_name].append(rec)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "pallet_index", "x1", "y1", "x2", "y2", "class"])
        writer.writeheader()
        for image_name in sorted(by_image):
            items = sorted(by_image[image_name], key=lambda r: ((r.bbox[1] + r.bbox[3]) / 2, (r.bbox[0] + r.bbox[2]) / 2))
            for index, rec in enumerate(items, start=1):
                writer.writerow({
                    "image_name": image_name,
                    "pallet_index": index,
                    "x1": int(rec.bbox[0]), "y1": int(rec.bbox[1]),
                    "x2": int(rec.bbox[2]), "y2": int(rec.bbox[3]),
                    "class": rec.label or "",
                })
    return output


def require_three_class_labels(cfg: Config, records: list[PalletAnnotation]) -> None:
    unlabeled = sum(rec.label is None for rec in records)
    counts = Counter(rec.label for rec in records if rec.label)
    missing = [name for name in CLASSES if counts[name] == 0]
    if unlabeled or missing:
        template = write_label_template(cfg, records)
        details = []
        if unlabeled:
            details.append(f"{unlabeled} pallet boxes have no class label")
        if missing:
            details.append("missing class supervision: " + ", ".join(missing))
        raise RuntimeError(
            "Cannot train a truthful 3-class classifier: " + "; ".join(details) + ". "
            f"Fill the per-pallet class column in {template} with EMPTY/ACCEPTED/REJECTED, then rerun with --labels-csv {template}. "
            "Image-folder labels are not silently copied to every pallet because mixed images may contain multiple classes."
        )


def split_records(cfg: Config, records: list[PalletAnnotation]) -> dict[str, list[str]]:
    split_path = cfg.artifacts / "split.json"
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))

    groups: dict[str, list[PalletAnnotation]] = defaultdict(list)
    for rec in records:
        groups[rec.group_id].append(rec)
    if len(groups) < 3:
        raise RuntimeError("Need at least 3 independent source groups for train/validation/test splitting.")

    targets = {"train": 1.0 - cfg.val_fraction - cfg.test_fraction, "val": cfg.val_fraction, "test": cfg.test_fraction}
    total_class = Counter(rec.label for rec in records if rec.label)
    total_items = len(records)
    rng = random.Random(cfg.seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_class: dict[str, Counter] = {k: Counter() for k in assigned}
    split_total = Counter()

    for group_id, items in group_items:
        item_counts = Counter(rec.label for rec in items if rec.label)
        best_split = None
        best_score = None
        for split_name in ("train", "val", "test"):
            target_total = max(1.0, total_items * targets[split_name])
            total_after = split_total[split_name] + len(items)
            total_penalty = abs(total_after - target_total) / target_total
            class_penalty = 0.0
            for cls in CLASSES:
                target_cls = max(1.0, total_class[cls] * targets[split_name])
                class_penalty += abs(split_class[split_name][cls] + item_counts[cls] - target_cls) / target_cls
            score = total_penalty + 0.5 * class_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assert best_split is not None
        assigned[best_split].append(group_id)
        split_total[best_split] += len(items)
        split_class[best_split].update(item_counts)

    # Ensure every split gets at least one group.
    if any(not assigned[name] for name in assigned):
        raise RuntimeError("Automatic grouped split produced an empty split. Add more independent source images/groups.")

    split = {
        name: sorted({rec.image_name for group in group_ids for rec in groups[group]})
        for name, group_ids in assigned.items()
    }
    split["seed"] = cfg.seed
    split["group_rule"] = "source image; trailing _NNN frame identifiers are grouped by prefix"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print("Split source images:", {k: len(v) for k, v in split.items() if isinstance(v, list)})
    return split


def robust_range(values: np.ndarray, low: float = 1.0, high: float = 99.0, expansion: float = 0.12) -> tuple[float, float]:
    lo, hi = np.percentile(values, [low, high])
    span = max(hi - lo, 1e-6)
    return max(0.0, float(lo - expansion * span)), float(hi + expansion * span)


def learn_geometry(cfg: Config, records: list[PalletAnnotation], split: dict[str, Any]) -> dict[str, Any]:
    path = cfg.artifacts / "geometry.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    train_names = set(split["train"])
    values: dict[str, list[float]] = defaultdict(list)
    extents = []
    for rec in records:
        if rec.image_name not in train_names:
            continue
        image = cv2.imread(rec.image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        x1, y1, x2, y2 = rec.bbox
        bw, bh = x2 - x1, y2 - y1
        values["rel_width"].append(bw / w)
        values["rel_height"].append(bh / h)
        values["rel_area"].append((bw * bh) / (w * h))
        values["aspect"].append(bw / max(bh, 1e-6))
        values["center_x"].append(((x1 + x2) / 2) / w)
        values["center_y"].append(((y1 + y2) / 2) / h)
        extents.append([x1 / w, y1 / h, x2 / w, y2 / h])
    if not values["rel_width"]:
        raise RuntimeError("No training annotations available for geometry learning.")

    geometry: dict[str, Any] = {"sample_count": len(values["rel_width"]), "full_image_coordinates": True}
    for name, seq in values.items():
        arr = np.asarray(seq, dtype=np.float64)
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        geometry[name] = {
            "median": float(median), "q1": float(q1), "q3": float(q3),
            "range": list(robust_range(arr)),
        }
    ext = np.asarray(extents, dtype=np.float64)
    roi = [
        max(0.0, float(np.percentile(ext[:, 0], 1) - 0.04)),
        max(0.0, float(np.percentile(ext[:, 1], 1) - 0.04)),
        min(1.0, float(np.percentile(ext[:, 2], 99) + 0.04)),
        min(1.0, float(np.percentile(ext[:, 3], 99) + 0.04)),
    ]
    geometry["conveyor_roi"] = roi
    path.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    print("Learned geometry:")
    for key in ("rel_width", "rel_height", "rel_area", "aspect"):
        print(f"  {key}: median={geometry[key]['median']:.4f}, range={geometry[key]['range']}")
    print("  conveyor_roi:", roi)
    return geometry


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ios(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def geometry_score_and_valid(box: list[float], width: int, height: int, geometry: dict[str, Any]) -> tuple[float, bool]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0, False
    features = {
        "rel_width": bw / width,
        "rel_height": bh / height,
        "rel_area": (bw * bh) / (width * height),
        "aspect": bw / bh,
    }
    valid = True
    scores = []
    for name, value in features.items():
        stats = geometry[name]
        lo, hi = stats["range"]
        if not (lo <= value <= hi):
            valid = False
        iqr = max(stats["q3"] - stats["q1"], 1e-6)
        z = abs(value - stats["median"]) / (2.5 * iqr)
        scores.append(max(0.0, 1.0 - z))
    cx, cy = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
    rx1, ry1, rx2, ry2 = geometry["conveyor_roi"]
    roi_margin = 0.04
    in_roi = (rx1 - roi_margin <= cx <= rx2 + roi_margin and ry1 - roi_margin <= cy <= ry2 + roi_margin)
    valid = valid and in_roi
    return float(np.mean(scores)), valid


def merge_candidates(candidates: list[DinoCandidate], cfg: Config) -> list[dict[str, Any]]:
    if not candidates:
        return []
    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            box_a, box_b = candidates[a].box, candidates[b].box
            ov_iou, ov_ios = iou(box_a, box_b), ios(box_a, box_b)
            area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
            area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
            area_ratio = min(area_a, area_b) / max(area_a, area_b)
            cax, cay = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
            cbx, cby = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
            scale = max(1.0, min(math.sqrt(area_a), math.sqrt(area_b)))
            center_norm = math.hypot(cax - cbx, cay - cby) / scale
            if ov_iou >= cfg.merge_iou or (ov_ios >= cfg.merge_ios and area_ratio >= 0.45 and center_norm <= 0.42):
                union(a, b)

    groups: dict[int, list[DinoCandidate]] = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        groups[find(idx)].append(candidate)

    merged: list[dict[str, Any]] = []
    for group in groups.values():
        weights = np.asarray([max(1e-4, c.score * (0.5 + 0.5 * c.geometry_score)) for c in group])
        boxes = np.asarray([c.box for c in group], dtype=np.float64)
        fused = np.average(boxes, axis=0, weights=weights).tolist()
        views = sorted({c.view for c in group})
        score = max(c.score for c in group)
        geom = float(np.average([c.geometry_score for c in group], weights=weights))
        if len(views) == 1 and score < 0.32:
            continue
        merged.append({
            "bbox": [round(v, 2) for v in fused],
            "localization_score": float(score),
            "geometry_score": geom,
            "support_count": len(views),
            "supporting_views": views,
        })

    merged.sort(key=lambda x: x["localization_score"] + 0.2 * x["geometry_score"], reverse=True)
    final: list[dict[str, Any]] = []
    for item in merged:
        duplicate = False
        for kept in final:
            if iou(item["bbox"], kept["bbox"]) >= cfg.final_iou or ios(item["bbox"], kept["bbox"]) >= cfg.final_ios:
                duplicate = True
                break
        if not duplicate:
            final.append(item)
    final.sort(key=lambda x: (((x["bbox"][1] + x["bbox"][3]) / 2), ((x["bbox"][0] + x["bbox"][2]) / 2)))
    return final


class GroundingDinoLocalizer:
    def __init__(self, cfg: Config, geometry: dict[str, Any], prompt: str, extra_views: bool = False):
        from transformers import AutoTokenizer, GroundingDinoForObjectDetection, GroundingDinoImageProcessor, GroundingDinoProcessor

        self.cfg = cfg
        self.geometry = geometry
        self.prompt = prompt if prompt.endswith(".") else prompt + "."
        self.extra_views = extra_views
        self.device = device_for_runtime()
        image_processor = GroundingDinoImageProcessor.from_pretrained(cfg.dino_model)
        tokenizer = AutoTokenizer.from_pretrained(cfg.dino_model)
        self.processor = GroundingDinoProcessor(image_processor=image_processor, tokenizer=tokenizer)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = GroundingDinoForObjectDetection.from_pretrained(cfg.dino_model, torch_dtype=dtype).to(self.device)
        self.model.eval()

    def _views(self, image: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
        h, w = image.shape[:2]
        views = [("FULL", (0, 0, w, h))]
        rx1, ry1, rx2, ry2 = self.geometry["conveyor_roi"]
        x1, y1, x2, y2 = int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 > 64 and y2 - y1 > 64 and (x1 > 8 or y1 > 8 or x2 < w - 8 or y2 < h - 8):
            views.append(("ROI", (x1, y1, x2, y2)))
        if self.extra_views and x2 - x1 > 200:
            roi_w = x2 - x1
            tile_w = int(roi_w * 0.62)
            views.append(("ROI_LEFT", (x1, y1, min(x2, x1 + tile_w), y2)))
            views.append(("ROI_RIGHT", (max(x1, x2 - tile_w), y1, x2, y2)))
        return views

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        h, w = image.shape[:2]
        views = self._views(image)
        pil_images, offsets, names, sizes = [], [], [], []
        for name, (x1, y1, x2, y2) in views:
            crop = image[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))
            offsets.append((x1, y1))
            names.append(name)
            sizes.append((crop.shape[0], crop.shape[1]))

        inputs = self.processor(images=pil_images, text=[self.prompt] * len(pil_images), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        amp_enabled = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.cfg.dino_box_threshold,
            text_threshold=self.cfg.dino_text_threshold,
            target_sizes=sizes,
        )

        candidates: list[DinoCandidate] = []
        for result, (ox, oy), name in zip(results, offsets, names):
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            for box, score in zip(boxes, scores):
                local = box.detach().float().cpu().tolist()
                full = [local[0] + ox, local[1] + oy, local[2] + ox, local[3] + oy]
                full[0] = max(0.0, min(w - 1.0, full[0]))
                full[1] = max(0.0, min(h - 1.0, full[1]))
                full[2] = max(full[0] + 1.0, min(float(w), full[2]))
                full[3] = max(full[1] + 1.0, min(float(h), full[3]))
                geom, valid = geometry_score_and_valid(full, w, h, self.geometry)
                if valid:
                    candidates.append(DinoCandidate(full, float(score.detach().cpu()), geom, name))
        return merge_candidates(candidates, self.cfg)


def match_boxes(pred_boxes: list[list[float]], gt_boxes: list[list[float]], threshold: float = 0.5) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not pred_boxes or not gt_boxes:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
    matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            matrix[i, j] = iou(pred, gt)
    rows, cols = linear_sum_assignment(1.0 - matrix)
    matches = [(int(r), int(c), float(matrix[r, c])) for r, c in zip(rows, cols) if matrix[r, c] >= threshold]
    matched_pred = {r for r, _, _ in matches}
    matched_gt = {c for _, c, _ in matches}
    return matches, [i for i in range(len(pred_boxes)) if i not in matched_pred], [j for j in range(len(gt_boxes)) if j not in matched_gt]


def gt_by_image(records: list[PalletAnnotation], names: set[str] | None = None) -> dict[str, list[PalletAnnotation]]:
    grouped: dict[str, list[PalletAnnotation]] = defaultdict(list)
    for rec in records:
        if names is None or rec.image_name in names:
            grouped[rec.image_name].append(rec)
    return grouped


def cache_key(prompt: str, extra_views: bool, cfg: Config) -> str:
    text = json.dumps({"prompt": prompt, "extra": extra_views, "box": cfg.dino_box_threshold, "text": cfg.dino_text_threshold, "model": cfg.dino_model}, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def localize_dataset(cfg: Config, records: list[PalletAnnotation], geometry: dict[str, Any], image_names: list[str], prompt: str, extra_views: bool) -> dict[str, list[dict[str, Any]]]:
    cache_dir = cfg.artifacts / "dino_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pred_{cache_key(prompt, extra_views, cfg)}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    lookup = {rec.image_name: rec.image_path for rec in records}
    missing = [name for name in image_names if name not in cache]
    localizer = None
    if missing:
        localizer = GroundingDinoLocalizer(cfg, geometry, prompt, extra_views)
    for idx, name in enumerate(missing, start=1):
        image = cv2.imread(lookup[name])
        if image is None:
            cache[name] = []
        else:
            cache[name] = localizer.detect(image) if localizer else []
        if idx % 5 == 0 or idx == len(missing):
            path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            print(f"DINO {idx}/{len(missing)} new validation images")
    return {name: cache.get(name, []) for name in image_names}


def localization_metrics(predictions: dict[str, list[dict[str, Any]]], grouped_gt: dict[str, list[PalletAnnotation]]) -> dict[str, float]:
    tp = fp = fn = 0
    ious = []
    total_images = max(1, len(grouped_gt))
    for name, gt_items in grouped_gt.items():
        pred_boxes = [item["bbox"] for item in predictions.get(name, [])]
        gt_boxes = [item.bbox for item in gt_items]
        matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes, 0.5)
        tp += len(matches)
        fp += len(unmatched_pred)
        fn += len(unmatched_gt)
        ious.extend(value for _, _, value in matches)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "recall_iou_0_50": recall,
        "false_positives_per_image": fp / total_images,
        "false_negatives_per_image": fn / total_images,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
    }


def save_localization_failures(cfg: Config, predictions: dict[str, list[dict[str, Any]]], grouped_gt: dict[str, list[PalletAnnotation]]) -> None:
    out = cfg.artifacts / "failures" / "localization"
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for name, gt_items in grouped_gt.items():
        pred_items = predictions.get(name, [])
        matches, unpred, ungt = match_boxes([p["bbox"] for p in pred_items], [g.bbox for g in gt_items], 0.5)
        if not unpred and not ungt and all(v >= 0.65 for _, _, v in matches):
            continue
        image = cv2.imread(gt_items[0].image_path)
        if image is None:
            continue
        for item in pred_items:
            x1, y1, x2, y2 = map(int, item["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for gt in gt_items:
            x1, y1, x2, y2 = map(int, gt.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        safe = name.replace("/", "__")
        cv2.imwrite(str(out / f"{safe}.jpg"), image)
        saved += 1
        if saved >= cfg.max_failure_images:
            break


def select_and_evaluate_localizer(cfg: Config, records: list[PalletAnnotation], geometry: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    config_path = cfg.artifacts / "localizer_config.json"
    metrics_path = cfg.artifacts / "localization_metrics.json"
    val_names = list(split["val"])
    grouped = gt_by_image(records, set(val_names))
    prompts = [
        "black conveyor pallet.",
        "black conveyor pallet. industrial workpiece carrier.",
    ]
    trials = []
    for prompt in prompts:
        preds = localize_dataset(cfg, records, geometry, val_names, prompt, False)
        metrics = localization_metrics(preds, grouped)
        trials.append({"prompt": prompt, "extra_views": False, "metrics": metrics})
        print("Localization trial:", prompt, metrics)
    best = max(trials, key=lambda t: (t["metrics"]["f1"], t["metrics"]["recall"], -t["metrics"]["false_positives_per_image"]))

    if best["metrics"]["recall"] < 0.97:
        preds = localize_dataset(cfg, records, geometry, val_names, best["prompt"], True)
        metrics = localization_metrics(preds, grouped)
        extra = {"prompt": best["prompt"], "extra_views": True, "metrics": metrics}
        trials.append(extra)
        if metrics["recall"] > best["metrics"]["recall"] + 0.005 and metrics["f1"] >= best["metrics"]["f1"] - 0.02:
            best = extra

    selected = {"prompt": best["prompt"], "extra_views": best["extra_views"], "validation_metrics": best["metrics"], "trials": trials}
    config_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    final_preds = localize_dataset(cfg, records, geometry, val_names, selected["prompt"], selected["extra_views"])
    save_localization_failures(cfg, final_preds, grouped)
    return selected


def expanded_box(box: list[float], width: int, height: int, context: float) -> list[int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * context, bh * context
    return [
        max(0, int(math.floor(x1 - px))), max(0, int(math.floor(y1 - py))),
        min(width, int(math.ceil(x2 + px))), min(height, int(math.ceil(y2 + py))),
    ]


def prepare_crops(cfg: Config, records: list[PalletAnnotation], split: dict[str, Any]) -> Path:
    manifest_path = cfg.artifacts / "crop_manifest.csv"
    if manifest_path.exists():
        return manifest_path
    require_three_class_labels(cfg, records)
    split_lookup = {name: split_name for split_name in ("train", "val", "test") for name in split[split_name]}
    by_image: dict[str, list[PalletAnnotation]] = defaultdict(list)
    for rec in records:
        by_image[rec.image_name].append(rec)
    rows = []
    for image_name, items in by_image.items():
        if image_name not in split_lookup:
            continue
        image = cv2.imread(items[0].image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        ordered = sorted(items, key=lambda r: ((r.bbox[1] + r.bbox[3]) / 2, (r.bbox[0] + r.bbox[2]) / 2))
        for index, rec in enumerate(ordered, start=1):
            assert rec.label in CLASSES
            crop_box = expanded_box(rec.bbox, w, h, cfg.classifier_context)
            x1, y1, x2, y2 = crop_box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise RuntimeError(f"Zero-size crop for {image_name} pallet {index}")
            split_name = split_lookup[image_name]
            crop_dir = cfg.artifacts / "crops" / split_name / rec.label
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{hashlib.sha1(image_name.encode()).hexdigest()[:10]}_{index:03d}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
            rows.append({
                "image_name": image_name, "pallet_index": index, "split": split_name, "class": rec.label,
                "x1": int(rec.bbox[0]), "y1": int(rec.bbox[1]), "x2": int(rec.bbox[2]), "y2": int(rec.bbox[3]),
                "crop_path": str(crop_path),
            })
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print("Prepared GT pallet crops:", len(rows))
    return manifest_path


def load_manifest(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))


def make_classifier(cfg: Config, pretrained: bool) -> nn.Module:
    import timm
    return timm.create_model(cfg.classifier_model, pretrained=pretrained, num_classes=len(CLASSES))


def set_head_only(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "blocks"):
        blocks = list(model.blocks.children())
        for block in blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
    for attr in ("conv_head", "bn2", "classifier"):
        module = getattr(model, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def class_weights(rows: list[dict[str, str]], device: torch.device) -> torch.Tensor | None:
    counts = Counter(row["class"] for row in rows)
    values = [counts[name] for name in CLASSES]
    if min(values) == 0:
        raise RuntimeError(f"Training split is missing a class: {counts}")
    if max(values) / min(values) < 1.5:
        return None
    total = sum(values)
    weights = [total / (len(CLASSES) * count) for count in values]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def eval_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, preds, probs_all = [], [], []
    with torch.inference_mode():
        for images, targets, _, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            labels.extend(targets.numpy().tolist())
            preds.extend(np.argmax(probs, axis=1).tolist())
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, len(CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f),
        "per_class": {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i in range(len(CLASSES))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
    }
    cm = np.asarray(metrics["confusion_matrix"])
    metrics["specific_errors"] = {
        "EMPTY_to_ACCEPTED": int(cm[0, 1]), "EMPTY_to_REJECTED": int(cm[0, 2]),
        "ACCEPTED_to_REJECTED": int(cm[1, 2]), "REJECTED_to_ACCEPTED": int(cm[2, 1]),
    }
    return metrics, y_true, y_pred, probs


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_loss = 0.0
    for images, targets, _, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * images.size(0)
    return total_loss / max(1, len(loader.dataset))


def train_classifier(cfg: Config, manifest_path: Path) -> dict[str, Any]:
    checkpoint_path = cfg.artifacts / "classifier_best.pt"
    config_path = cfg.artifacts / "classifier_config.json"
    if checkpoint_path.exists() and config_path.exists():
        print("Reusing existing classifier checkpoint:", checkpoint_path)
        return json.loads(config_path.read_text(encoding="utf-8"))

    rows = load_manifest(manifest_path)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    device = device_for_runtime()
    model = make_classifier(cfg, pretrained=True).to(device)
    train_ds = CropDataset(train_rows, cfg.classifier_resolution, train=True)
    val_ds = CropDataset(val_rows, cfg.classifier_resolution, train=False)
    loader_args = dict(num_workers=cfg.classifier_workers, pin_memory=device.type == "cuda", persistent_workers=cfg.classifier_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.classifier_batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=cfg.classifier_batch_size, shuffle=False, **loader_args)
    weights = class_weights(train_rows, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = -1.0
    best_epoch = 0
    history = []
    stage_specs = [("head", cfg.stage1_epochs, cfg.head_lr), ("finetune", cfg.stage2_epochs, cfg.finetune_lr)]
    global_epoch = 0
    patience = 0

    for stage_name, epochs, lr in stage_specs:
        if stage_name == "head":
            set_head_only(model)
        else:
            unfreeze_last_blocks(model)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=cfg.weight_decay)
        for _ in range(epochs):
            global_epoch += 1
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            metrics, _, _, probs = eval_classifier(model, val_loader, device)
            macro_f1 = metrics["macro_f1"]
            history.append({"epoch": global_epoch, "stage": stage_name, "loss": loss, "val_macro_f1": macro_f1, "val_accuracy": metrics["accuracy"]})
            print(f"Epoch {global_epoch:02d} {stage_name}: loss={loss:.4f} val_f1={macro_f1:.4f} val_acc={metrics['accuracy']:.4f}")
            if macro_f1 > best_f1 + 1e-4:
                best_f1, best_epoch, patience = macro_f1, global_epoch, 0
                torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "resolution": cfg.classifier_resolution, "model": cfg.classifier_model}, checkpoint_path)
            else:
                patience += 1
            if stage_name == "finetune" and patience >= cfg.early_stopping_patience:
                break
        if stage_name == "finetune" and patience >= cfg.early_stopping_patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    val_metrics, y_true, y_pred, probs = eval_classifier(model, val_loader, device)
    correct_conf = probs[np.arange(len(y_true)), y_pred][y_true == y_pred] if len(y_true) else np.array([])
    threshold = float(np.clip(np.percentile(correct_conf, 5), 0.50, 0.90)) if len(correct_conf) else 0.60
    classifier_config = {
        "model": cfg.classifier_model, "resolution": cfg.classifier_resolution, "classes": CLASSES,
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "context": cfg.classifier_context,
        "unknown_threshold": threshold, "best_validation_macro_f1": best_f1, "best_epoch": best_epoch,
        "seed": cfg.seed, "history": history, "validation_metrics": val_metrics,
    }
    config_path.write_text(json.dumps(classifier_config, indent=2), encoding="utf-8")
    return classifier_config


def load_classifier(cfg: Config) -> tuple[nn.Module, dict[str, Any], torch.device]:
    config = json.loads((cfg.artifacts / "classifier_config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(cfg.artifacts / "classifier_best.pt", map_location="cpu")
    device = device_for_runtime()
    model = make_classifier(cfg, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, config, device


def classification_eval(cfg: Config, manifest_path: Path, split_name: str = "test") -> dict[str, Any]:
    model, config, device = load_classifier(cfg)
    rows = [r for r in load_manifest(manifest_path) if r["split"] == split_name]
    dataset = CropDataset(rows, int(config["resolution"]), train=False)
    loader = DataLoader(dataset, batch_size=cfg.classifier_batch_size, shuffle=False, num_workers=cfg.classifier_workers, pin_memory=device.type == "cuda")
    metrics, y_true, y_pred, probs = eval_classifier(model, loader, device)
    confidence = probs.max(axis=1) if len(probs) else np.array([])
    unknown = confidence < float(config["unknown_threshold"])
    metrics["unknown_threshold"] = float(config["unknown_threshold"])
    metrics["unknown_rate"] = float(np.mean(unknown)) if len(unknown) else 0.0
    output = cfg.artifacts / "classification_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    failures = cfg.artifacts / "failures" / "classification"
    failures.mkdir(parents=True, exist_ok=True)
    wrong = np.where(y_true != y_pred)[0]
    order = wrong[np.argsort(-confidence[wrong])] if len(wrong) else []
    for idx in order[: cfg.max_failure_images]:
        src = Path(rows[int(idx)]["crop_path"])
        dst = failures / f"true_{CLASSES[y_true[idx]]}__pred_{CLASSES[y_pred[idx]]}__{confidence[idx]:.3f}__{src.name}"
        shutil.copy2(src, dst)
    print("GT-crop classification metrics:", json.dumps(metrics, indent=2))
    return metrics


def classifier_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def classify_boxes(cfg: Config, image: np.ndarray, detections: list[dict[str, Any]], model: nn.Module, classifier_config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    h, w = image.shape[:2]
    transform = classifier_transform(int(classifier_config["resolution"]))
    tensors, valid_indices = [], []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = expanded_box(det["bbox"], w, h, float(classifier_config["context"]))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(transform(pil))
        valid_indices.append(i)
    results = [dict(det) for det in detections]
    for det in results:
        det.update({"class": "UNKNOWN", "class_confidence": 0.0, "prob_empty": 0.0, "prob_accepted": 0.0, "prob_rejected": 0.0})
    if not tensors:
        return results
    threshold = float(classifier_config["unknown_threshold"])
    for start in range(0, len(tensors), cfg.infer_batch_size):
        batch = torch.stack(tensors[start:start + cfg.infer_batch_size]).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        for j, prob in enumerate(probs):
            idx = valid_indices[start + j]
            best = int(np.argmax(prob))
            confidence = float(prob[best])
            results[idx].update({
                "class": CLASSES[best] if confidence >= threshold else "UNKNOWN",
                "class_confidence": confidence,
                "prob_empty": float(prob[0]), "prob_accepted": float(prob[1]), "prob_rejected": float(prob[2]),
            })
    return results


def load_localizer_config(cfg: Config) -> dict[str, Any]:
    path = cfg.artifacts / "localizer_config.json"
    if not path.exists():
        raise FileNotFoundError("Missing localizer_config.json. Run --mode localize-eval first.")
    return json.loads(path.read_text(encoding="utf-8"))


def end_to_end_eval(cfg: Config, records: list[PalletAnnotation], geometry: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    require_three_class_labels(cfg, records)
    loc_cfg = load_localizer_config(cfg)
    test_names = list(split["test"])
    predictions = localize_dataset(cfg, records, geometry, test_names, loc_cfg["prompt"], bool(loc_cfg["extra_views"]))
    classifier, classifier_cfg, device = load_classifier(cfg)
    grouped = gt_by_image(records, set(test_names))
    tp = fp = fn = correct_class = matched_count = 0
    rows = []
    failure_dir = cfg.artifacts / "failures" / "end_to_end"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_saved = 0

    for name, gt_items in grouped.items():
        image = cv2.imread(gt_items[0].image_path)
        pred = classify_boxes(cfg, image, predictions.get(name, []), classifier, classifier_cfg, device)
        matches, unpred, ungt = match_boxes([p["bbox"] for p in pred], [g.bbox for g in gt_items], 0.5)
        tp += len(matches); fp += len(unpred); fn += len(ungt); matched_count += len(matches)
        wrong = False
        for pidx, gidx, match_iou in matches:
            gt_label = gt_items[gidx].label
            predicted_label = pred[pidx]["class"]
            class_ok = predicted_label == gt_label
            correct_class += int(class_ok)
            wrong |= not class_ok
            rows.append({"image_name": name, "iou": match_iou, "gt_class": gt_label, "pred_class": predicted_label, "class_correct": class_ok})
        if (unpred or ungt or wrong) and failure_saved < cfg.max_failure_images:
            debug = image.copy()
            for item in pred:
                x1, y1, x2, y2 = map(int, item["bbox"])
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug, f"{item['class']} {item['class_confidence']:.2f}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            safe = name.replace("/", "__")
            cv2.imwrite(str(failure_dir / f"{safe}.jpg"), debug)
            failure_saved += 1

    det_precision = tp / max(1, tp + fp)
    det_recall = tp / max(1, tp + fn)
    metrics = {
        "pallet_detection_precision": det_precision,
        "pallet_detection_recall": det_recall,
        "correct_class_accuracy_among_matched": correct_class / max(1, matched_count),
        "end_to_end_localized_and_correctly_classified_rate": correct_class / max(1, tp + fn),
        "matched_pallets": matched_count, "false_positive_pallets": fp, "missed_pallets": fn,
    }
    (cfg.artifacts / "end_to_end_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("End-to-end metrics:", json.dumps(metrics, indent=2))
    return metrics


def infer_directory(cfg: Config, geometry: dict[str, Any]) -> None:
    infer_dir = cfg.infer_dir
    if infer_dir is None:
        candidates = [p for p in cfg.data_root.rglob("mix") if p.is_dir() and any(x.suffix.lower() in IMAGE_EXTENSIONS for x in p.iterdir() if x.is_file())]
        if len(candidates) != 1:
            raise RuntimeError("Provide --infer-dir PATH. The code will not guess among multiple folders.")
        infer_dir = candidates[0]
    images = sorted(p for p in infer_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No inference images in {infer_dir}")
    loc_cfg = load_localizer_config(cfg)
    localizer = GroundingDinoLocalizer(cfg, geometry, loc_cfg["prompt"], bool(loc_cfg["extra_views"]))
    classifier, classifier_cfg, device = load_classifier(cfg)
    debug_dir = cfg.artifacts / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    csv_rows = []
    start = time.perf_counter()

    for number, path in enumerate(images, start=1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        detections = localizer.detect(image)
        classified = classify_boxes(cfg, image, detections, classifier, classifier_cfg, device)
        debug = image.copy()
        image_rows = []
        for index, item in enumerate(classified, start=1):
            x1, y1, x2, y2 = map(int, item["bbox"])
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
            text = f"PALLET_{index} {item['class']} {item['class_confidence']:.2f}"
            cv2.putText(debug, text, (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            record = {
                "image_name": path.name, "pallet_index": index,
                "crop_bbox": [x1, y1, x2, y2], "original_bbox": [x1, y1, x2, y2],
                "localization_score": item["localization_score"], "geometry_score": item["geometry_score"],
                "support_count": item["support_count"], "supporting_views": item["supporting_views"],
                "class": item["class"], "class_confidence": item["class_confidence"],
                "prob_empty": item["prob_empty"], "prob_accepted": item["prob_accepted"], "prob_rejected": item["prob_rejected"],
            }
            image_rows.append(record)
            csv_rows.append({**record, "crop_bbox": json.dumps(record["crop_bbox"]), "original_bbox": json.dumps(record["original_bbox"]), "supporting_views": json.dumps(record["supporting_views"])})
        debug_path = debug_dir / f"{path.stem}_pred.jpg"
        cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 94])
        output_records.append({"image_name": path.name, "pallet_count": len(image_rows), "pallets": image_rows, "debug_image": str(debug_path)})
        print(f"[{number}/{len(images)}] {path.name}: pallets={len(image_rows)}")

    (cfg.artifacts / "predictions.json").write_text(json.dumps({"top_crop": 0, "images": output_records}, indent=2), encoding="utf-8")
    csv_path = cfg.artifacts / "predictions.csv"
    if csv_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader(); writer.writerows(csv_rows)
    elapsed = time.perf_counter() - start
    print(f"Inference complete: {len(images)} images in {elapsed:.1f}s ({len(images) / max(elapsed, 1e-6):.3f} images/s)")
    print("Output:", cfg.artifacts)


def save_config(cfg: Config) -> None:
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    serializable = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
    (cfg.artifacts / "run_config.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_root=Path(args.data_root).resolve(), artifacts=Path(args.artifacts).resolve(),
        annotations=Path(args.annotations).resolve() if args.annotations else None,
        labels_csv=Path(args.labels_csv).resolve() if args.labels_csv else None,
        infer_dir=Path(args.infer_dir).resolve() if args.infer_dir else None,
        seed=args.seed, classifier_resolution=args.classifier_resolution,
        classifier_batch_size=args.classifier_batch_size, classifier_workers=args.workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CREST pallet localization + 3-class classification pipeline. Full images only; no top crop.")
    parser.add_argument("--mode", required=True, choices=["audit", "prepare", "train", "localize-eval", "classify-eval", "e2e-eval", "infer", "all"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--annotations")
    parser.add_argument("--labels-csv")
    parser.add_argument("--infer-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-resolution", type=int, default=320)
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = build_config(args)
    seed_everything(cfg.seed)
    save_config(cfg)
    print("Device:", device_for_runtime())
    print("Coordinate policy: FULL SOURCE IMAGE. No TOP_CROP is applied anywhere.")

    records, annotation_path = parse_annotations(cfg)
    apply_labels_csv(records, cfg.labels_csv)
    errors = validate_annotations(records)
    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"Annotation sanity checks failed ({len(errors)} errors):\n{preview}")
    audit_dataset(cfg, records, annotation_path)

    if args.mode == "audit":
        template = write_label_template(cfg, records)
        if any(rec.label is None for rec in records):
            print("Per-pallet label template:", template)
        return

    require_three_class_labels(cfg, records)
    split = split_records(cfg, records)
    geometry = learn_geometry(cfg, records, split)

    if args.mode == "prepare":
        prepare_crops(cfg, records, split)
        return

    if args.mode == "localize-eval":
        select_and_evaluate_localizer(cfg, records, geometry, split)
        return

    manifest = prepare_crops(cfg, records, split)

    if args.mode == "train":
        train_classifier(cfg, manifest)
        return

    if args.mode == "classify-eval":
        classification_eval(cfg, manifest, "test")
        return

    if args.mode == "e2e-eval":
        end_to_end_eval(cfg, records, geometry, split)
        return

    if args.mode == "infer":
        infer_directory(cfg, geometry)
        return

    if args.mode == "all":
        train_classifier(cfg, manifest)
        select_and_evaluate_localizer(cfg, records, geometry, split)
        classification_eval(cfg, manifest, "test")
        end_to_end_eval(cfg, records, geometry, split)
        if cfg.infer_dir:
            infer_directory(cfg, geometry)


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import random
import shutil

from ultralytics import YOLO


ROOT = Path(r"C:\Do_Not_Delete_PLC\original images")

WORK_DIR = (
    ROOT
    / "06_Empty_vs_Object_Classifier"
    / "04_Improved_Classifier_Training"
)

DATASET_DIR = WORK_DIR / "balanced_dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
RUNS_DIR = WORK_DIR / "training_output"

FINAL_MODEL = (
    WORK_DIR
    / "improved_empty_vs_object_yolo11n_cls_best.pt"
)

CONFIG_PATH = (
    WORK_DIR
    / "improved_classifier_config.json"
)

README_PATH = (
    WORK_DIR
    / "README_IMPROVED_CLASSIFIER.txt"
)


ORIGINAL_EMPTY = Path(
    r"C:\Do_Not_Delete_PLC\PLC\PLC\close_original\empty"
)

ORIGINAL_OBJECT = Path(
    r"C:\Do_Not_Delete_PLC\PLC\PLC\close_original\rejcted"
)

AUGMENTED_EMPTY = Path(
    r"C:\Do_Not_Delete_PLC\PLC\PLC\pre_augumented\empty"
)

AUGMENTED_OBJECT = Path(
    r"C:\Do_Not_Delete_PLC\PLC\PLC\pre_augumented\rejected\close_look_aug_rejected"
)


SEED = 42
BASE_MODEL = "yolo11n-cls.pt"
IMAGE_SIZE = 320
EPOCHS = 80
PATIENCE = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0005
MAX_AUGMENTED_PER_CLASS = 500
VAL_PER_CLASS = 40


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def collect_images(folder: Path):

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing folder:\n{folder}"
        )

    return sorted(
        [
            path
            for path in folder.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )


def reset_folder(folder: Path):

    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


def copy_image(
    source: Path,
    destination_folder: Path,
    prefix: str,
    index: int
):

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        destination_folder
        / f"{prefix}_{index:05d}{source.suffix.lower()}"
    )

    shutil.copy2(
        source,
        destination
    )


print("=" * 72)
print("UPDATED EMPTY VS OBJECT_PRESENT TRAINING")
print("=" * 72)

print("EMPTY = whole pallet with no object")
print("OBJECT_PRESENT = whole pallet with object")


original_empty = collect_images(
    ORIGINAL_EMPTY
)

original_object = collect_images(
    ORIGINAL_OBJECT
)

augmented_empty = collect_images(
    AUGMENTED_EMPTY
)

augmented_object = collect_images(
    AUGMENTED_OBJECT
)


print()
print("Original EMPTY:", len(original_empty))
print("Original OBJECT_PRESENT:", len(original_object))
print("Augmented EMPTY:", len(augmented_empty))
print("Augmented OBJECT_PRESENT:", len(augmented_object))


balanced_count = min(
    MAX_AUGMENTED_PER_CLASS,
    len(augmented_empty),
    len(augmented_object)
)


if balanced_count <= VAL_PER_CLASS:
    raise RuntimeError(
        "Not enough augmented images."
    )


rng = random.Random(SEED)


selected_empty = rng.sample(
    augmented_empty,
    balanced_count
)

selected_object = rng.sample(
    augmented_object,
    balanced_count
)


rng.shuffle(selected_empty)
rng.shuffle(selected_object)


val_empty = selected_empty[:VAL_PER_CLASS]
val_object = selected_object[:VAL_PER_CLASS]

train_aug_empty = selected_empty[VAL_PER_CLASS:]
train_aug_object = selected_object[VAL_PER_CLASS:]


reset_folder(DATASET_DIR)


for folder in [
    TRAIN_DIR / "EMPTY",
    TRAIN_DIR / "OBJECT_PRESENT",
    VAL_DIR / "EMPTY",
    VAL_DIR / "OBJECT_PRESENT",
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


for index, image_path in enumerate(original_empty):

    copy_image(
        image_path,
        TRAIN_DIR / "EMPTY",
        "original_empty",
        index
    )


for index, image_path in enumerate(original_object):

    copy_image(
        image_path,
        TRAIN_DIR / "OBJECT_PRESENT",
        "original_object",
        index
    )


for index, image_path in enumerate(train_aug_empty):

    copy_image(
        image_path,
        TRAIN_DIR / "EMPTY",
        "aug_empty",
        index
    )


for index, image_path in enumerate(train_aug_object):

    copy_image(
        image_path,
        TRAIN_DIR / "OBJECT_PRESENT",
        "aug_object",
        index
    )


for index, image_path in enumerate(val_empty):

    copy_image(
        image_path,
        VAL_DIR / "EMPTY",
        "val_empty",
        index
    )


for index, image_path in enumerate(val_object):

    copy_image(
        image_path,
        VAL_DIR / "OBJECT_PRESENT",
        "val_object",
        index
    )


train_empty_count = (
    len(original_empty)
    +
    len(train_aug_empty)
)

train_object_count = (
    len(original_object)
    +
    len(train_aug_object)
)


print()
print("TRAIN EMPTY:", train_empty_count)
print("TRAIN OBJECT_PRESENT:", train_object_count)
print("VAL EMPTY:", len(val_empty))
print("VAL OBJECT_PRESENT:", len(val_object))


if RUNS_DIR.exists():
    shutil.rmtree(RUNS_DIR)


model = YOLO(
    BASE_MODEL
)


model.train(
    data=str(DATASET_DIR),
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=0,
    workers=0,
    optimizer="AdamW",
    lr0=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    pretrained=True,
    seed=SEED,
    deterministic=True,
    project=str(RUNS_DIR),
    name="improved_empty_vs_object",
    exist_ok=True,
    verbose=True,
    hsv_h=0.01,
    hsv_s=0.10,
    hsv_v=0.10,
    degrees=3.0,
    translate=0.03,
    scale=0.10,
    fliplr=0.0,
    flipud=0.0,
)


BEST_SOURCE = (
    RUNS_DIR
    / "improved_empty_vs_object"
    / "weights"
    / "best.pt"
)


if not BEST_SOURCE.exists():
    raise FileNotFoundError(
        f"best.pt not found:\n{BEST_SOURCE}"
    )


shutil.copy2(
    BEST_SOURCE,
    FINAL_MODEL
)


best_model = YOLO(
    str(FINAL_MODEL)
)


validation_results = best_model.val(
    data=str(DATASET_DIR),
    split="val",
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=0,
    workers=0,
    verbose=True,
)


top1 = None


if hasattr(validation_results, "top1"):

    try:
        top1 = float(
            validation_results.top1
        )

    except Exception:
        pass


config = {

    "task": "EMPTY vs OBJECT_PRESENT",

    "image_requirement": {
        "EMPTY":
        "whole pallet visible with no object",

        "OBJECT_PRESENT":
        "whole pallet visible with object on pallet",
    },

    "base_model": BASE_MODEL,

    "final_model": str(FINAL_MODEL),

    "image_size": IMAGE_SIZE,

    "balanced_augmented_per_class":
    balanced_count,

    "train_counts": {
        "EMPTY": train_empty_count,
        "OBJECT_PRESENT": train_object_count,
    },

    "validation_counts": {
        "EMPTY": len(val_empty),
        "OBJECT_PRESENT": len(val_object),
    },

    "validation_top1": top1,
}


CONFIG_PATH.write_text(
    json.dumps(
        config,
        indent=2
    ),
    encoding="utf-8"
)


README_TEXT = f"""
CREST - EMPTY VS OBJECT_PRESENT CLASSIFIER
==========================================

PURPOSE

Classify a detected pallet as:

EMPTY

or

OBJECT_PRESENT


CORRECT TRAINING IMAGE FORMAT

EMPTY:
whole pallet visible with no object

OBJECT_PRESENT:
whole pallet visible with object sitting on pallet


SOURCE FOLDERS

Original EMPTY:
{ORIGINAL_EMPTY}

Original OBJECT_PRESENT:
{ORIGINAL_OBJECT}

Augmented EMPTY:
{AUGMENTED_EMPTY}

Augmented OBJECT_PRESENT:
{AUGMENTED_OBJECT}


COUNTS

Original EMPTY:
{len(original_empty)}

Original OBJECT_PRESENT:
{len(original_object)}

Augmented EMPTY:
{len(augmented_empty)}

Augmented OBJECT_PRESENT:
{len(augmented_object)}


TRAINING

EMPTY:
{train_empty_count}

OBJECT_PRESENT:
{train_object_count}


VALIDATION

EMPTY:
{len(val_empty)}

OBJECT_PRESENT:
{len(val_object)}


MODEL

YOLO11n Classification


IMAGE SIZE

{IMAGE_SIZE}


VALIDATION TOP-1

{top1}


MODEL FILE

{FINAL_MODEL}


THIS MODEL DOES NOT DETECT X.

Current flow:

YOLO pallet detector
        |
        v
pallet crop
        |
        v
EMPTY / OBJECT_PRESENT classifier

Later:

OBJECT_PRESENT
        |
        v
X detector
        |
    +---+---+
    |       |
    X      NO X
    |       |
REJECTED ACCEPTED
"""


README_PATH.write_text(
    README_TEXT.strip() + "\n",
    encoding="utf-8"
)


print()
print("=" * 72)
print("TRAINING COMPLETE")
print("=" * 72)

print()
print("Model:")
print(FINAL_MODEL)

print()
print("Validation Top-1:")
print(top1)

print()
print("README updated:")
print(README_PATH)
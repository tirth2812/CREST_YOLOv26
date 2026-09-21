from pathlib import Path
import json
import shutil

from ultralytics import YOLO


# ============================================================
# CREST - EMPTY VS OBJECT_PRESENT CLASSIFIER TRAINING
# ============================================================
#
# PURPOSE:
#
# Fine-tune a lightweight YOLO11n classification model.
#
# INPUT:
# Close-up pallet image
#
# OUTPUT:
# EMPTY
# or
# OBJECT_PRESENT
#
# IMPORTANT:
#
# - REJECTED images currently mean OBJECT_PRESENT.
# - X detection is NOT part of this model.
# - TEST images are NOT used here.
# - Only TRAIN and VALIDATION folders are used.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Do_Not_Delete_PLC\original images"
)

PROJECT_ROOT = (
    ROOT
    / "06_Empty_vs_Object_Classifier"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "01_Safe_Dataset_Split"
)

DATASET_DIR = (
    SPLIT_DIR
    / "dataset"
)

TRAINING_DIR = (
    PROJECT_ROOT
    / "02_Classifier_Training"
)

RUNS_DIR = (
    TRAINING_DIR
    / "training_output"
)

FINAL_MODEL_PATH = (
    TRAINING_DIR
    / "empty_vs_object_yolo11n_cls_best.pt"
)

CONFIG_PATH = (
    TRAINING_DIR
    / "classifier_training_config.json"
)

README_PATH = (
    TRAINING_DIR
    / "README_CLASSIFIER_TRAINING.txt"
)


# ============================================================
# CHECK DATASET
# ============================================================

required_folders = [

    DATASET_DIR
    / "train"
    / "EMPTY",

    DATASET_DIR
    / "train"
    / "OBJECT_PRESENT",

    DATASET_DIR
    / "val"
    / "EMPTY",

    DATASET_DIR
    / "val"
    / "OBJECT_PRESENT",

    DATASET_DIR
    / "test"
    / "EMPTY",

    DATASET_DIR
    / "test"
    / "OBJECT_PRESENT",
]


print("=" * 72)
print("CREST - EMPTY VS OBJECT_PRESENT CLASSIFIER TRAINING")
print("=" * 72)


for folder in required_folders:

    if not folder.exists():

        raise FileNotFoundError(
            f"Missing dataset folder:\n{folder}"
        )

    print(
        "FOUND:",
        folder
    )


# ============================================================
# COUNT IMAGES
# ============================================================

SUPPORTED = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def count_images(folder: Path):

    return sum(

        1

        for path in folder.iterdir()

        if (
            path.is_file()
            and
            path.suffix.lower()
            in SUPPORTED
        )
    )


train_empty = count_images(
    DATASET_DIR
    / "train"
    / "EMPTY"
)

train_object = count_images(
    DATASET_DIR
    / "train"
    / "OBJECT_PRESENT"
)

val_empty = count_images(
    DATASET_DIR
    / "val"
    / "EMPTY"
)

val_object = count_images(
    DATASET_DIR
    / "val"
    / "OBJECT_PRESENT"
)

test_empty = count_images(
    DATASET_DIR
    / "test"
    / "EMPTY"
)

test_object = count_images(
    DATASET_DIR
    / "test"
    / "OBJECT_PRESENT"
)


print()
print("DATASET")
print("-" * 40)

print(
    f"TRAIN EMPTY:          {train_empty}"
)

print(
    f"TRAIN OBJECT_PRESENT: {train_object}"
)

print(
    f"VAL EMPTY:            {val_empty}"
)

print(
    f"VAL OBJECT_PRESENT:   {val_object}"
)

print(
    f"LOCKED TEST EMPTY:    {test_empty}"
)

print(
    f"LOCKED TEST OBJECT:   {test_object}"
)


print()
print(
    "IMPORTANT: TEST DATA WILL NOT BE USED DURING TRAINING."
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BASE_MODEL = "yolo11n-cls.pt"

IMAGE_SIZE = 224

EPOCHS = 60

PATIENCE = 12

BATCH_SIZE = 4

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0005

SEED = 42


# ============================================================
# PRINT SETTINGS
# ============================================================

print()
print("=" * 72)
print("TRAINING CONFIGURATION")
print("=" * 72)

print(
    "Base model:",
    BASE_MODEL
)

print(
    "Task: EMPTY vs OBJECT_PRESENT"
)

print(
    "Image size:",
    IMAGE_SIZE
)

print(
    "Epoch limit:",
    EPOCHS
)

print(
    "Early stopping patience:",
    PATIENCE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Optimizer: AdamW"
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Workers: 0"
)

print(
    "Device: GPU 0"
)

print()
print(
    "TRAINING-TIME AUGMENTATION WILL BE USED."
)

print(
    "EXISTING PRE-AUGMENTED DATA IS NOT USED."
)

print(
    "TEST SET REMAINS LOCKED."
)


# ============================================================
# LOAD PRETRAINED CLASSIFIER
# ============================================================

print()
print("=" * 72)
print("LOADING PRETRAINED YOLO11n CLASSIFIER")
print("=" * 72)

model = YOLO(
    BASE_MODEL
)

print(
    "Model loaded."
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 72)
print("STARTING TRAINING")
print("=" * 72)


results = model.train(

    data=str(
        DATASET_DIR
    ),

    epochs=EPOCHS,

    patience=PATIENCE,

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    device=0,

    workers=0,

    optimizer="AdamW",

    lr0=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    seed=SEED,

    deterministic=True,

    pretrained=True,

    project=str(
        RUNS_DIR
    ),

    name="empty_vs_object",

    exist_ok=True,

    verbose=True,
)


# ============================================================
# FIND BEST MODEL
# ============================================================

BEST_MODEL_SOURCE = (
    RUNS_DIR
    / "empty_vs_object"
    / "weights"
    / "best.pt"
)


if not BEST_MODEL_SOURCE.exists():

    raise FileNotFoundError(
        f"Training finished but best.pt was not found:\n"
        f"{BEST_MODEL_SOURCE}"
    )


shutil.copy2(
    BEST_MODEL_SOURCE,
    FINAL_MODEL_PATH
)


print()
print("=" * 72)
print("BEST MODEL SAVED")
print("=" * 72)

print(
    FINAL_MODEL_PATH
)


# ============================================================
# VALIDATE BEST MODEL
# ============================================================

print()
print("=" * 72)
print("VALIDATING BEST MODEL")
print("=" * 72)


best_model = YOLO(
    str(
        FINAL_MODEL_PATH
    )
)


validation_results = best_model.val(

    data=str(
        DATASET_DIR
    ),

    split="val",

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    device=0,

    workers=0,

    verbose=True,
)


# ============================================================
# READ VALIDATION METRICS
# ============================================================

top1 = None
top5 = None


if hasattr(
    validation_results,
    "top1"
):

    try:

        top1 = float(
            validation_results.top1
        )

    except Exception:

        pass


if hasattr(
    validation_results,
    "top5"
):

    try:

        top5 = float(
            validation_results.top5
        )

    except Exception:

        pass


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "task": (
        "EMPTY vs OBJECT_PRESENT"
    ),

    "model_type": (
        "YOLO11n Classification"
    ),

    "base_model": BASE_MODEL,

    "final_model": str(
        FINAL_MODEL_PATH
    ),

    "dataset": str(
        DATASET_DIR
    ),

    "classes": [
        "EMPTY",
        "OBJECT_PRESENT",
    ],

    "training_counts": {

        "EMPTY":
        train_empty,

        "OBJECT_PRESENT":
        train_object,
    },

    "validation_counts": {

        "EMPTY":
        val_empty,

        "OBJECT_PRESENT":
        val_object,
    },

    "locked_test_counts": {

        "EMPTY":
        test_empty,

        "OBJECT_PRESENT":
        test_object,
    },

    "training_settings": {

        "imgsz":
        IMAGE_SIZE,

        "epochs":
        EPOCHS,

        "patience":
        PATIENCE,

        "batch":
        BATCH_SIZE,

        "optimizer":
        "AdamW",

        "lr0":
        LEARNING_RATE,

        "weight_decay":
        WEIGHT_DECAY,

        "device":
        0,

        "workers":
        0,

        "seed":
        SEED,
    },

    "augmentation_policy": (
        "Ultralytics training-time augmentation only. "
        "Existing pre-augmented dataset intentionally "
        "excluded from this clean experiment."
    ),

    "test_policy": (
        "Test images were not used during training "
        "or validation."
    ),

    "validation_top1": top1,

    "validation_top5": top5,
}


CONFIG_PATH.write_text(

    json.dumps(
        config,
        indent=2
    ),

    encoding="utf-8"
)


# ============================================================
# CREATE README
# ============================================================

README_TEXT = f"""
CREST PROJECT
EMPTY VS OBJECT_PRESENT CLASSIFIER
TRAINING STAGE
===============================================


PURPOSE
-----------------------------------------------

This folder contains the first trained classifier used
after pallet localization.

The model answers one question:

IS THERE AN OBJECT ON THE PALLET?


OUTPUT CLASSES
-----------------------------------------------

EMPTY

OBJECT_PRESENT


IMPORTANT
-----------------------------------------------

The OBJECT_PRESENT training examples currently come from
the original close-up REJECTED images.

For this stage, the X mark does not matter.

A rejected image simply means:

AN OBJECT IS PRESENT.


X detection will be developed separately later.


MODEL
-----------------------------------------------

YOLO11n Classification

Pretrained starting model:

{BASE_MODEL}


Final trained model:

empty_vs_object_yolo11n_cls_best.pt


WHY PRETRAINED WEIGHTS ARE USED
-----------------------------------------------

The current original dataset is small.

Using a pretrained classification model allows the system
to start from useful visual features instead of learning
everything from zero.


TRAIN DATA
-----------------------------------------------

EMPTY:

{train_empty}


OBJECT_PRESENT:

{train_object}


TOTAL TRAIN ORIGINALS:

{train_empty + train_object}


VALIDATION DATA
-----------------------------------------------

EMPTY:

{val_empty}


OBJECT_PRESENT:

{val_object}


TOTAL VALIDATION ORIGINALS:

{val_empty + val_object}


LOCKED TEST DATA
-----------------------------------------------

EMPTY:

{test_empty}


OBJECT_PRESENT:

{test_object}


TOTAL LOCKED TEST IMAGES:

{test_empty + test_object}


IMPORTANT:

The test images were NOT used during this training stage.


TRAINING SETTINGS
-----------------------------------------------

Image size:

{IMAGE_SIZE}


Epoch limit:

{EPOCHS}


Early stopping patience:

{PATIENCE}


Batch size:

{BATCH_SIZE}


Optimizer:

AdamW


Initial learning rate:

{LEARNING_RATE}


Weight decay:

{WEIGHT_DECAY}


GPU:

device 0


Windows DataLoader workers:

0


Random seed:

{SEED}


AUGMENTATION
-----------------------------------------------

Training-time image augmentation is used while fine-tuning.

The previously generated:

500 augmented EMPTY images

and

1199 augmented OBJECT_PRESENT images

are NOT used in this clean experiment.

They were intentionally excluded because their exact
relationship to the original validation/test images is
not guaranteed.

This prevents possible data leakage.


MODEL OUTPUT
-----------------------------------------------

For each pallet crop, the classifier will return
probabilities such as:

EMPTY:
0.97

OBJECT_PRESENT:
0.03


or:

EMPTY:
0.02

OBJECT_PRESENT:
0.98


The class with the highest confidence becomes the predicted
result.


CURRENT PIPELINE
-----------------------------------------------

LIVE CAMERA
        |
        v
YOLO PALLET LOCALIZER
        |
        v
6 PALLET BOXES
        |
        v
CROP EACH PALLET
        |
        v
THIS CLASSIFIER
        |
     +--+--+
     |     |
     v     v
   EMPTY  OBJECT_PRESENT


FUTURE STAGE
-----------------------------------------------

If the result is:

EMPTY

then final state is:

EMPTY


If the result is:

OBJECT_PRESENT

then another stage checks:

IS AN X PRESENT?


X PRESENT:

REJECTED


NO X:

ACCEPTED


FULL FUTURE FLOW
-----------------------------------------------

CAMERA
   |
   v
YOLO PALLET DETECTION
   |
   v
PALLET CROP
   |
   v
EMPTY VS OBJECT_PRESENT
   |
   +----------------+
   |                |
 EMPTY        OBJECT_PRESENT
                     |
                     v
                 X DETECTOR
                     |
                +----+----+
                |         |
               X        NO X
                |         |
                v         v
            REJECTED   ACCEPTED


NEXT STEP
-----------------------------------------------

After training completes, the next step is a completely
separate evaluation on the four locked original TEST images.

The test set must not be used to change model settings.
"""


README_PATH.write_text(

    README_TEXT.strip()
    + "\n",

    encoding="utf-8"
)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 72)
print("CLASSIFIER TRAINING COMPLETE")
print("=" * 72)

print()
print("Best model:")
print(
    FINAL_MODEL_PATH
)

print()
print("Configuration:")
print(
    CONFIG_PATH
)

print()
print("README:")
print(
    README_PATH
)

print()
print("Validation Top-1:")
print(
    top1
)

print()
print(
    "IMPORTANT: LOCKED TEST SET HAS NOT BEEN USED."
)

print()
print(
    "Next step: evaluate the final model on the "
    "4 locked original test images."
)
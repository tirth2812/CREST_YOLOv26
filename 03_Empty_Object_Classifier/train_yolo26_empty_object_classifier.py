from pathlib import Path
import csv
import json
import shutil

import torch
from ultralytics import YOLO


# ============================================================
# CREST YOLO26
# FAIR EMPTY vs OBJECT_PRESENT COMPARISON
#
# IMPORTANT:
#
# This script intentionally uses EXACTLY the final real
# YOLO11 train/validation dataset.
#
# It does NOT create a new random split.
# It does NOT use the old balanced augmented dataset.
# It does NOT remove holder-empty examples.
# It does NOT remove accepted/tick object examples.
#
# Only the model family changes:
#
# YOLO11n-cls -> YOLO26n-cls
# ============================================================


# ============================================================
# PATHS
# ============================================================

STAGE_DIR = Path(__file__).resolve().parent

SOURCE_DATASET = Path(
    r"C:\Do_Not_Delete_PLC\original images"
    r"\06_Empty_vs_Object_Classifier"
    r"\04_Improved_Classifier_Training"
    r"\final_real_live_classifier_dataset"
)

TRAIN_EMPTY = (
    SOURCE_DATASET
    / "train"
    / "EMPTY"
)

TRAIN_OBJECT = (
    SOURCE_DATASET
    / "train"
    / "OBJECT_PRESENT"
)

VAL_EMPTY = (
    SOURCE_DATASET
    / "val"
    / "EMPTY"
)

VAL_OBJECT = (
    SOURCE_DATASET
    / "val"
    / "OBJECT_PRESENT"
)


MODEL_DIR = (
    STAGE_DIR
    / "MODEL"
)

RESULTS_DIR = (
    STAGE_DIR
    / "RESULTS"
)

TRAINING_OUTPUT = (
    RESULTS_DIR
    / "training_output"
)

RUN_NAME = (
    "yolo26n_empty_object_fair"
)

RUN_DIR = (
    TRAINING_OUTPUT
    / RUN_NAME
)

FINAL_MODEL = (
    MODEL_DIR
    / "empty_object_yolo26n_cls_best.pt"
)

BACKUP_MODEL = (
    MODEL_DIR
    / "empty_object_yolo26n_cls_best_BEFORE_FAIR_RETRAIN.pt"
)

CONFIG_PATH = (
    RESULTS_DIR
    / "yolo26_empty_object_fair_config.json"
)

METRICS_PATH = (
    RESULTS_DIR
    / "yolo26_empty_object_fair_metrics.json"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "yolo26_empty_object_val_predictions.csv"
)

SPLIT_MANIFEST_PATH = (
    RESULTS_DIR
    / "yolo26_empty_object_exact_split.json"
)


# ============================================================
# MODEL / TRAINING SETTINGS
#
# Keep the CREST settings aligned with the established
# EMPTY / OBJECT_PRESENT experiment.
# ============================================================

BASE_MODEL = "yolo26n-cls.pt"

IMAGE_SIZE = 320

EPOCHS = 100

PATIENCE = 20

BATCH_SIZE = 8

DEVICE = 0

WORKERS = 0

OPTIMIZER = "AdamW"

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0005

SEED = 42


# ============================================================
# AUGMENTATION
# ============================================================

DEGREES = 5.0

TRANSLATE = 0.02

SCALE = 0.05

FLIP_LR = 0.0

FLIP_UD = 0.0

ERASING = 0.0


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# HELPERS
# ============================================================

def collect_images(folder):

    if not folder.exists():

        raise FileNotFoundError(
            f"Required dataset folder not found:\n"
            f"{folder}"
        )

    return sorted(
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


def normalize_class(name):

    text = str(
        name
    ).strip().upper()

    text = text.replace(
        "-",
        "_"
    )

    text = text.replace(
        " ",
        "_"
    )

    if text == "EMPTY":

        return "EMPTY"

    if text == "OBJECT_PRESENT":

        return "OBJECT_PRESENT"

    return text


# ============================================================
# VERIFY EXACT FINAL DATASET
# ============================================================

def verify_dataset():

    train_empty = collect_images(
        TRAIN_EMPTY
    )

    train_object = collect_images(
        TRAIN_OBJECT
    )

    val_empty = collect_images(
        VAL_EMPTY
    )

    val_object = collect_images(
        VAL_OBJECT
    )


    print()
    print("=" * 70)
    print("FINAL REAL YOLO11 DATASET BEING REUSED")
    print("=" * 70)

    print(
        f"TRAIN EMPTY:          "
        f"{len(train_empty)}"
    )

    print(
        f"TRAIN OBJECT_PRESENT: "
        f"{len(train_object)}"
    )

    print(
        f"VAL EMPTY:            "
        f"{len(val_empty)}"
    )

    print(
        f"VAL OBJECT_PRESENT:   "
        f"{len(val_object)}"
    )


    expected = {

        "train_empty":
            95,

        "train_object":
            95,

        "val_empty":
            25,

        "val_object":
            25,
    }


    actual = {

        "train_empty":
            len(train_empty),

        "train_object":
            len(train_object),

        "val_empty":
            len(val_empty),

        "val_object":
            len(val_object),
    }


    if actual != expected:

        raise RuntimeError(
            "\nDataset count does not match the "
            "final YOLO11 real dataset.\n\n"
            f"Expected:\n{expected}\n\n"
            f"Found:\n{actual}\n"
        )


    manifest = {

        "policy":
            (
                "Exact final YOLO11 real train/validation "
                "dataset reused for YOLO26. No resplitting."
            ),

        "source_dataset":
            str(SOURCE_DATASET),

        "counts": {

            "train": {

                "EMPTY":
                    len(train_empty),

                "OBJECT_PRESENT":
                    len(train_object),
            },

            "val": {

                "EMPTY":
                    len(val_empty),

                "OBJECT_PRESENT":
                    len(val_object),
            },
        },

        "train": {

            "EMPTY": [
                path.name
                for path
                in train_empty
            ],

            "OBJECT_PRESENT": [
                path.name
                for path
                in train_object
            ],
        },

        "val": {

            "EMPTY": [
                path.name
                for path
                in val_empty
            ],

            "OBJECT_PRESENT": [
                path.name
                for path
                in val_object
            ],
        },
    }


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    SPLIT_MANIFEST_PATH.write_text(

        json.dumps(
            manifest,
            indent=2
        ),

        encoding="utf-8"
    )


    return manifest


# ============================================================
# VALIDATION EVALUATION
# ============================================================

def evaluate_validation(model):

    rows = []

    total = 0

    correct = 0


    per_class = {

        "EMPTY": {
            "correct": 0,
            "total": 0
        },

        "OBJECT_PRESENT": {
            "correct": 0,
            "total": 0
        },
    }


    class_folders = {

        "EMPTY":
            VAL_EMPTY,

        "OBJECT_PRESENT":
            VAL_OBJECT,
    }


    for actual_class, folder in class_folders.items():

        for image_path in collect_images(
            folder
        ):

            result = model.predict(

                source=str(
                    image_path
                ),

                imgsz=IMAGE_SIZE,

                device=DEVICE,

                verbose=False,

            )[0]


            if result.probs is None:

                predicted_class = "UNKNOWN"

                confidence = 0.0

            else:

                class_id = int(
                    result.probs.top1
                )

                predicted_class = normalize_class(
                    result.names[
                        class_id
                    ]
                )

                confidence = float(
                    result.probs.top1conf.item()
                )


            is_correct = (
                predicted_class
                ==
                actual_class
            )


            total += 1

            correct += int(
                is_correct
            )


            per_class[
                actual_class
            ][
                "total"
            ] += 1


            per_class[
                actual_class
            ][
                "correct"
            ] += int(
                is_correct
            )


            rows.append({

                "image":
                    image_path.name,

                "actual":
                    actual_class,

                "predicted":
                    predicted_class,

                "confidence":
                    f"{confidence:.6f}",

                "result":
                    (
                        "CORRECT"
                        if is_correct
                        else "WRONG"
                    ),
            })


    for class_name in per_class:

        class_total = (
            per_class[
                class_name
            ][
                "total"
            ]
        )

        class_correct = (
            per_class[
                class_name
            ][
                "correct"
            ]
        )

        per_class[
            class_name
        ][
            "accuracy"
        ] = (
            class_correct
            /
            class_total
            if class_total
            else 0.0
        )


    metrics = {

        "total":
            total,

        "correct":
            correct,

        "wrong":
            total - correct,

        "accuracy":
            (
                correct
                /
                total
                if total
                else 0.0
            ),

        "per_class":
            per_class,
    }


    with PREDICTIONS_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=[
                "image",
                "actual",
                "predicted",
                "confidence",
                "result",
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is not available."
        )


    manifest = verify_dataset()


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAINING_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # BACK UP CURRENT YOLO26 MODEL
    # ========================================================

    if (
        FINAL_MODEL.exists()
        and
        not BACKUP_MODEL.exists()
    ):

        shutil.copy2(
            FINAL_MODEL,
            BACKUP_MODEL
        )

        print()
        print(
            "Previous YOLO26 model backed up:"
        )

        print(
            BACKUP_MODEL
        )


    # ========================================================
    # REMOVE ONLY OLD FAIR-COMPARISON RUN
    # ========================================================

    if RUN_DIR.exists():

        shutil.rmtree(
            RUN_DIR
        )


    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config = {

        "architecture":
            "YOLO26n Classification",

        "base_model":
            BASE_MODEL,

        "task":
            "EMPTY vs OBJECT_PRESENT",

        "source_dataset":
            str(SOURCE_DATASET),

        "fair_comparison":
            True,

        "same_yolo11_split":
            True,

        "classes": [
            "EMPTY",
            "OBJECT_PRESENT"
        ],

        "training_counts":
            manifest["counts"],

        "settings": {

            "imgsz":
                IMAGE_SIZE,

            "epochs":
                EPOCHS,

            "patience":
                PATIENCE,

            "batch":
                BATCH_SIZE,

            "optimizer":
                OPTIMIZER,

            "lr0":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "degrees":
                DEGREES,

            "translate":
                TRANSLATE,

            "scale":
                SCALE,

            "fliplr":
                FLIP_LR,

            "flipud":
                FLIP_UD,

            "erasing":
                ERASING,

            "seed":
                SEED,

            "deterministic":
                True,
        },
    }


    CONFIG_PATH.write_text(

        json.dumps(
            config,
            indent=2
        ),

        encoding="utf-8"
    )


    # ========================================================
    # TRAIN YOLO26
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING FAIR YOLO26 EMPTY / OBJECT TRAINING")
    print("=" * 70)

    print()
    print(
        "Base model:",
        BASE_MODEL
    )

    print(
        "Dataset:",
        SOURCE_DATASET
    )

    print()


    model = YOLO(
        BASE_MODEL
    )


    model.train(

        data=str(
            SOURCE_DATASET
        ),

        imgsz=IMAGE_SIZE,

        epochs=EPOCHS,

        patience=PATIENCE,

        batch=BATCH_SIZE,

        device=DEVICE,

        workers=WORKERS,

        optimizer=OPTIMIZER,

        lr0=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        degrees=DEGREES,

        translate=TRANSLATE,

        scale=SCALE,

        fliplr=FLIP_LR,

        flipud=FLIP_UD,

        erasing=ERASING,

        seed=SEED,

        deterministic=True,

        pretrained=True,

        project=str(
            TRAINING_OUTPUT
        ),

        name=RUN_NAME,

        exist_ok=True,

        plots=True,

        verbose=True,
    )


    # ========================================================
    # COPY BEST MODEL
    # ========================================================

    best_source = (
        RUN_DIR
        / "weights"
        / "best.pt"
    )


    if not best_source.exists():

        raise FileNotFoundError(
            f"Training finished but best.pt "
            f"was not found:\n"
            f"{best_source}"
        )


    shutil.copy2(
        best_source,
        FINAL_MODEL
    )


    print()
    print(
        "Best YOLO26 model copied to:"
    )

    print(
        FINAL_MODEL
    )


    # ========================================================
    # VERIFY CLASS MAPPING
    # ========================================================

    best_model = YOLO(
        str(
            FINAL_MODEL
        )
    )


    print()
    print(
        "MODEL CLASS MAPPING:"
    )

    print(
        best_model.names
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    metrics = evaluate_validation(
        best_model
    )


    METRICS_PATH.write_text(

        json.dumps(
            metrics,
            indent=2
        ),

        encoding="utf-8"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("YOLO26 EMPTY / OBJECT TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Validation: "
        f"{metrics['correct']}/"
        f"{metrics['total']} correct"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )


    for class_name in (
        "EMPTY",
        "OBJECT_PRESENT"
    ):

        class_metrics = (
            metrics[
                "per_class"
            ][
                class_name
            ]
        )

        print(
            f"{class_name}: "
            f"{class_metrics['correct']}/"
            f"{class_metrics['total']} "
            f"("
            f"{class_metrics['accuracy']:.4f}"
            f")"
        )


    print()
    print(
        "Final model:"
    )

    print(
        FINAL_MODEL
    )

    print()
    print(
        "Metrics:"
    )

    print(
        METRICS_PATH
    )

    print()
    print(
        "Predictions:"
    )

    print(
        PREDICTIONS_PATH
    )


if __name__ == "__main__":

    main()
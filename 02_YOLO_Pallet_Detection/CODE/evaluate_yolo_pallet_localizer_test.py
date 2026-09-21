from pathlib import Path
import importlib.util
import json
import sys

import cv2
from ultralytics import YOLO


# ============================================================
# CREST - YOLO PALLET LOCALIZER TEST EVALUATION
#
# PURPOSE:
# Evaluate the already-trained YOLO11n pallet localizer on
# the 5 source images that were NOT used for YOLO training
# or YOLO confidence-threshold selection.
#
# IMPORTANT:
# - NO TRAINING
# - NO THRESHOLD TUNING
# - Confidence stays locked at 0.05
# - Image size stays locked at 960
# - FULL SOURCE IMAGE
# - NO TOP_CROP
# ============================================================


ROOT = Path(
    r"C:\Do_Not_Delete_PLC\original images"
)

PROJECT_DIR = (
    ROOT
    / "01_YOLO_Pallet_Localizer_Training"
)

CREST_PIPELINE = (
    ROOT
    / "crest_pipeline.py"
)

ANNOTATIONS = (
    ROOT
    / "pallet_annotations.json"
)

MODEL_PATH = (
    PROJECT_DIR
    / "pallet_yolo11n_best.pt"
)

SPLIT_PATH = (
    PROJECT_DIR
    / "split_used_for_yolo.json"
)

CONFIG_PATH = (
    PROJECT_DIR
    / "yolo_localizer_config.json"
)

RESULT_PATH = (
    PROJECT_DIR
    / "yolo_test_metrics.json"
)

DEBUG_DIR = (
    PROJECT_DIR
    / "test_debug_images"
)

README_PATH = (
    PROJECT_DIR
    / "README_YOLO_PALLET_LOCALIZER.txt"
)

RUNTIME_ARTIFACTS = (
    PROJECT_DIR
    / "crest_runtime_artifacts"
)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 72)
print("YOLO11n PALLET LOCALIZER - TEST EVALUATION")
print("=" * 72)

required_files = [
    CREST_PIPELINE,
    ANNOTATIONS,
    MODEL_PATH,
    SPLIT_PATH,
    CONFIG_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

    print("FOUND:", path)


# ============================================================
# IMPORT CREST PIPELINE
# ============================================================

spec = importlib.util.spec_from_file_location(
    "crest_pipeline",
    CREST_PIPELINE,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Could not load crest_pipeline.py"
    )

crest = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = crest

spec.loader.exec_module(crest)


# ============================================================
# LOAD CONFIG + ANNOTATIONS
# ============================================================

cfg = crest.Config(
    data_root=ROOT,
    artifacts=RUNTIME_ARTIFACTS,
    annotations=ANNOTATIONS,
)

records, _ = crest.parse_annotations(cfg)

split = json.loads(
    SPLIT_PATH.read_text(
        encoding="utf-8"
    )
)

yolo_config = json.loads(
    CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)

test_names = list(
    split["test"]
)

CONFIDENCE = float(
    yolo_config["confidence_threshold"]
)

IMAGE_SIZE = int(
    yolo_config["imgsz"]
)

NMS_IOU = float(
    yolo_config["nms_iou"]
)


print("\n" + "=" * 72)
print("LOCKED TEST CONFIGURATION")
print("=" * 72)

print("Model:")
print(MODEL_PATH)

print("\nTest images:", len(test_names))
print("Confidence:", CONFIDENCE)
print("Image size:", IMAGE_SIZE)
print("NMS IoU:", NMS_IOU)
print("FULL SOURCE IMAGE")
print("NO TOP_CROP")
print("NO PARAMETER TUNING")


# ============================================================
# GROUND TRUTH
# ============================================================

grouped = crest.gt_by_image(
    records
)

grouped_test_gt = crest.gt_by_image(
    records,
    set(test_names),
)

total_gt = sum(
    len(items)
    for items in grouped_test_gt.values()
)

print(
    "\nGround-truth test pallets:",
    total_gt,
)

if len(test_names) != 5:
    raise RuntimeError(
        f"Expected 5 test images, found {len(test_names)}."
    )

if total_gt != 30:
    raise RuntimeError(
        f"Expected 30 test pallets, found {total_gt}."
    )


# ============================================================
# LOAD TRAINED YOLO
# ============================================================

print("\nLoading YOLO11n model...")

detector = YOLO(
    str(MODEL_PATH)
)

print("Model loaded.")


# ============================================================
# RUN TEST INFERENCE
# ============================================================

predictions = {}

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\n" + "=" * 72)
print("RUNNING TEST INFERENCE")
print("=" * 72)


for index, image_name in enumerate(
    test_names,
    start=1,
):

    gt_items = grouped.get(
        image_name,
        [],
    )

    if not gt_items:
        raise RuntimeError(
            f"No GT annotations for:\n{image_name}"
        )

    image_path = Path(
        gt_items[0].image_path
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    result_list = detector.predict(
        source=str(image_path),

        imgsz=IMAGE_SIZE,

        conf=CONFIDENCE,

        iou=NMS_IOU,

        max_det=20,

        device=0,

        verbose=False,
    )

    detections = []

    if result_list:

        result = result_list[0]

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            scores = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, score in zip(
                boxes,
                scores,
            ):

                detections.append(
                    {
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ],

                        "localization_score": float(
                            score
                        ),
                    }
                )

    predictions[
        image_name
    ] = detections

    print(
        f"{index}/{len(test_names)} "
        f"{image_name}: "
        f"{len(detections)} detections"
    )


    # ========================================================
    # SAVE DEBUG IMAGE
    #
    # BLUE  = ground truth
    # GREEN = YOLO prediction
    # ========================================================

    debug = image.copy()

    for gt_index, item in enumerate(
        gt_items,
        start=1,
    ):

        x1, y1, x2, y2 = map(
            int,
            item.bbox,
        )

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            debug,
            f"GT_{gt_index}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )


    for pred_index, detection in enumerate(
        detections,
        start=1,
    ):

        x1, y1, x2, y2 = map(
            int,
            detection["bbox"],
        )

        score = detection[
            "localization_score"
        ]

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            debug,
            f"YOLO_{pred_index} {score:.2f}",
            (x1, min(
                debug.shape[0] - 10,
                y2 + 18,
            )),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


    safe_name = (
        image_name
        .replace("/", "__")
        .replace("\\", "__")
    )

    cv2.imwrite(
        str(
            DEBUG_DIR
            / safe_name
        ),
        debug,
    )


# ============================================================
# EXACT CREST METRICS
# ============================================================

metrics = crest.localization_metrics(
    predictions,
    grouped_test_gt,
)


# ============================================================
# SAVE RESULT
# ============================================================

output = {
    "detector": "YOLO11n",

    "model": str(
        MODEL_PATH
    ),

    "confidence_threshold": CONFIDENCE,

    "imgsz": IMAGE_SIZE,

    "nms_iou": NMS_IOU,

    "test_images": len(
        test_names
    ),

    "test_pallet_boxes": total_gt,

    "metrics": metrics,

    "parameters_tuned_on_test": False,
}


RESULT_PATH.write_text(
    json.dumps(
        output,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# APPEND TEST RESULT TO README
# ============================================================

with README_PATH.open(
    "a",
    encoding="utf-8",
) as file:

    file.write(
        "\n\n"
        "YOLO TEST RESULT\n"
        "================\n"
    )

    file.write(
        "Test parameters were locked before evaluation.\n"
    )

    file.write(
        f"Confidence: {CONFIDENCE}\n"
    )

    file.write(
        f"Precision: {metrics['precision']}\n"
    )

    file.write(
        f"Recall: {metrics['recall']}\n"
    )

    file.write(
        f"F1: {metrics['f1']}\n"
    )

    file.write(
        f"Mean IoU: {metrics['mean_iou']}\n"
    )

    file.write(
        f"True positives: {metrics['true_positives']}\n"
    )

    file.write(
        f"False positives: {metrics['false_positives']}\n"
    )

    file.write(
        f"False negatives: {metrics['false_negatives']}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 72)
print("YOLO11n HELD-OUT TEST RESULT")
print("=" * 72)

for key, value in metrics.items():
    print(
        f"{key}: {value}"
    )

print(
    "\nDetected:",
    metrics["true_positives"],
    "/",
    (
        metrics["true_positives"]
        + metrics["false_negatives"]
    ),
)

print(
    "False positives:",
    metrics["false_positives"],
)

print(
    "Missed pallets:",
    metrics["false_negatives"],
)


print("\n" + "=" * 72)
print("DINO TEST BASELINE")
print("=" * 72)

print("Detected: 25 / 30")
print("Precision: 0.9615")
print("Recall:    0.8333")
print("F1:        0.8929")
print("Mean IoU:  0.7023")
print("FP: 1")
print("FN: 5")


print("\nSaved test metrics:")
print(RESULT_PATH)

print("\nSaved debug images:")
print(DEBUG_DIR)

print(
    "\nIMPORTANT: "
    "NO TEST PARAMETERS WERE TUNED."
)
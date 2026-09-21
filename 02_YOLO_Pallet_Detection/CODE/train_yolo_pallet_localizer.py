from pathlib import Path
import json
import shutil

import cv2
import torch
from ultralytics import YOLO


STAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = STAGE_DIR.parent

ANNOTATIONS_PATH = (
    PROJECT_DIR
    / "01_Pallet_Annotation"
    / "DATA"
    / "pallet_annotations.json"
)
SOURCE_IMAGES_DIR = (
    PROJECT_DIR
    / "01_Pallet_Annotation"
    / "DATA"
    / "images"
)
SPLIT_PATH = STAGE_DIR / "CONFIG_AND_RESULTS" / "split_used_for_yolo.json"

DATASET_DIR = STAGE_DIR / "DATASET" / "yolo_dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
TRAINING_DIR = STAGE_DIR / "CONFIG_AND_RESULTS" / "training_output"
MODEL_DIR = STAGE_DIR / "MODEL"
FINAL_MODEL = MODEL_DIR / "pallet_yolo26n_best.pt"
METRICS_PATH = STAGE_DIR / "CONFIG_AND_RESULTS" / "yolo26_test_metrics.json"
CONFIG_PATH = STAGE_DIR / "CONFIG_AND_RESULTS" / "yolo26_localizer_config.json"
VISUAL_DIR = STAGE_DIR / "VISUAL_TEST_RESULTS"

BASE_MODEL = "yolo26n.pt"
IMAGE_SIZE = 640
EPOCHS = 150
PATIENCE = 30
BATCH_SIZE = 8
SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def recreate_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_source_image(relative_name):
    filename = Path(relative_name).name
    direct_path = SOURCE_IMAGES_DIR / filename
    if direct_path.exists():
        return direct_path

    matches = list(SOURCE_IMAGES_DIR.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source image for {relative_name}; found {len(matches)}"
        )
    return matches[0]


def get_annotation_entry(annotation_data, relative_name):
    images = annotation_data.get("images", {})
    if relative_name in images:
        return images[relative_name]

    filename = Path(relative_name).name
    matches = [entry for key, entry in images.items() if Path(key).name == filename]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one annotation entry for {relative_name}; found {len(matches)}"
        )
    return matches[0]


def yolo_label_line(bbox, image_width, image_height):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0

    values = (
        0,
        center_x / image_width,
        center_y / image_height,
        box_width / image_width,
        box_height / image_height,
    )
    return f"{values[0]} {values[1]:.8f} {values[2]:.8f} {values[3]:.8f} {values[4]:.8f}"


def create_subset(subset_name, relative_names, annotation_data):
    image_destination = DATASET_DIR / "images" / subset_name
    label_destination = DATASET_DIR / "labels" / subset_name
    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)

    image_count = 0
    box_count = 0

    for relative_name in relative_names:
        source_image = find_source_image(relative_name)
        entry = get_annotation_entry(annotation_data, relative_name)

        image = cv2.imread(str(source_image))
        if image is None:
            raise RuntimeError(f"Could not read image:\n{source_image}")

        actual_height, actual_width = image.shape[:2]
        recorded_width = int(entry.get("width", actual_width))
        recorded_height = int(entry.get("height", actual_height))

        if (actual_width, actual_height) != (recorded_width, recorded_height):
            raise ValueError(
                f"Image-size mismatch for {source_image.name}: "
                f"actual={actual_width}x{actual_height}, "
                f"annotation={recorded_width}x{recorded_height}"
            )

        output_image = image_destination / source_image.name
        shutil.copy2(source_image, output_image)

        label_lines = []
        for pallet in entry.get("pallets", []):
            bbox = pallet.get("bbox")
            if not bbox or len(bbox) != 4:
                raise ValueError(f"Invalid pallet bbox in {relative_name}: {bbox}")
            label_lines.append(
                yolo_label_line(bbox, actual_width, actual_height)
            )

        if not label_lines:
            raise ValueError(f"No pallet boxes found for {relative_name}")

        label_path = label_destination / f"{source_image.stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        image_count += 1
        box_count += len(label_lines)

    return image_count, box_count


def metric_value(metric_object, attribute):
    value = getattr(metric_object, attribute, None)
    return None if value is None else float(value)


def evaluate_model(model, split_name):
    result = model.val(
        data=str(DATA_YAML),
        split=split_name,
        imgsz=IMAGE_SIZE,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    return {
        "precision": metric_value(result.box, "mp"),
        "recall": metric_value(result.box, "mr"),
        "mAP50": metric_value(result.box, "map50"),
        "mAP50_95": metric_value(result.box, "map"),
    }


def save_test_visuals(model, test_names):
    recreate_directory(VISUAL_DIR)
    for relative_name in test_names:
        image_path = DATASET_DIR / "images" / "test" / Path(relative_name).name
        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=0.25,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]
        output_path = VISUAL_DIR / image_path.name
        if not cv2.imwrite(str(output_path), result.plot()):
            raise RuntimeError(f"Could not save visual result:\n{output_path}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. In Kaggle, enable a GPU accelerator before training."
        )

    annotation_data = load_json(ANNOTATIONS_PATH)
    split_data = load_json(SPLIT_PATH)

    expected_counts = {"train": 10, "val": 5, "test": 5}
    for split_name, expected_count in expected_counts.items():
        actual_count = len(split_data.get(split_name, []))
        if actual_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} {split_name} images; found {actual_count}"
            )

    all_names = [name for split_name in expected_counts for name in split_data[split_name]]
    if len(all_names) != len(set(all_names)):
        raise ValueError("Train, validation, and test splits contain overlapping images.")

    recreate_directory(DATASET_DIR)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split_name in ("train", "val", "test"):
        image_count, box_count = create_subset(
            split_name,
            split_data[split_name],
            annotation_data,
        )
        counts[split_name] = {"images": image_count, "pallet_boxes": box_count}

    DATA_YAML.write_text(
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: pallet\n",
        encoding="utf-8",
    )

    configuration = {
        "architecture": "YOLO26n",
        "base_model": BASE_MODEL,
        "task": "pallet detection",
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "device": "CUDA GPU",
        "split_counts": counts,
    }
    CONFIG_PATH.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    print("CREST YOLO26 PALLET DETECTOR TRAINING")
    print("=" * 50)
    for split_name, split_counts in counts.items():
        print(
            f"{split_name.upper()}: {split_counts['images']} images, "
            f"{split_counts['pallet_boxes']} pallet boxes"
        )

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        pretrained=True,
        device=0,
        workers=2,
        project=str(TRAINING_DIR),
        name="yolo26n_pallet",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    best_source = TRAINING_DIR / "yolo26n_pallet" / "weights" / "best.pt"
    if not best_source.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found:\n{best_source}")

    shutil.copy2(best_source, FINAL_MODEL)
    best_model = YOLO(str(FINAL_MODEL))

    metrics = {
        "model": str(FINAL_MODEL),
        "validation": evaluate_model(best_model, "val"),
        "heldout_test": evaluate_model(best_model, "test"),
        "split_counts": counts,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_test_visuals(best_model, split_data["test"])

    print("\nTRAINING COMPLETE")
    print(f"Model: {FINAL_MODEL}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Test visuals: {VISUAL_DIR}")
    print(json.dumps(metrics["heldout_test"], indent=2))


if __name__ == "__main__":
    main()

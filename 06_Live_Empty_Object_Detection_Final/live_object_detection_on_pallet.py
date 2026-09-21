from pathlib import Path
import importlib.util
import time

import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Do_Not_Delete_PLC\original images\09_Live_Object_Detection_On_Pallet"
)

PALLET_STAGE_FILE = Path(
    r"C:\Do_Not_Delete_PLC\original images"
    r"\08_Dynamic_Live_Pallet_Detection_No_Fixed_Count"
    r"\dynamic_live_pallet_detection.py"
)

PALLET_MODEL_PATH = Path(
    r"C:\Do_Not_Delete_PLC\original images"
    r"\02_YOLO_Pallet_Localizer_Final"
    r"\MODEL"
    r"\pallet_yolo11n_best.pt"
)

CLASSIFIER_MODEL_PATH = Path(
    r"C:\Do_Not_Delete_PLC\original images"
    r"\06_Empty_vs_Object_Classifier"
    r"\04_Improved_Classifier_Training"
    r"\improved_empty_vs_object_yolo11n_cls_best.pt"
)

BOUNDARY_FILE = BASE_DIR / "conveyor_boundary.json"

README_FILE = BASE_DIR / "README_LIVE_OBJECT_DETECTION.txt"

SAVE_DIR = BASE_DIR / "saved_live_object_detection_frames"


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 1

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080


# ============================================================
# CLASSIFIER
# ============================================================

CLASSIFIER_IMAGE_SIZE = 320

CLASSIFIER_MIN_CONFIDENCE = 0.50

PALLET_CROP_PADDING = 0.03


SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD STAGE 08 FUNCTIONS
# ============================================================

def load_pallet_stage():

    spec = importlib.util.spec_from_file_location(
        "dynamic_pallet_stage",
        str(PALLET_STAGE_FILE),
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    # Use the boundary copy inside stage 09.
    module.BOUNDARY_FILE = BOUNDARY_FILE

    # Final settings that worked in Stage 08.
    module.YOLO_CONFIDENCE = 0.15
    module.CONFIRM_FRAMES = 5

    return module


# ============================================================
# README
# ============================================================

def write_readme():

    text = f"""
CREST LIVE OBJECT DETECTION ON PALLET
===============================================

WORKING FOLDER

{BASE_DIR}


PURPOSE

This stage takes the final dynamic pallet detections
and checks each detected pallet for:

EMPTY

or

OBJECT_PRESENT


PIPELINE

LIVE CAMERA
    |
    v
Fine-tuned YOLO pallet detector
    |
    v
Saved conveyor boundary filter
    |
    v
Dynamic pallet tracking
    |
    v
Crop each confirmed pallet
    |
    v
EMPTY vs OBJECT_PRESENT classifier
    |
    +-------------------+
    |                   |
    v                   v
EMPTY             OBJECT_PRESENT


PALLET MODEL

{PALLET_MODEL_PATH}


OBJECT CLASSIFIER

{CLASSIFIER_MODEL_PATH}


CONVEYOR BOUNDARY

{BOUNDARY_FILE}


CAMERA

Index:
{CAMERA_INDEX}

Resolution:
{CAMERA_WIDTH} x {CAMERA_HEIGHT}


FINAL PALLET DETECTION SETTINGS

YOLO confidence:
0.15

Temporal confirmation:
5 frames

NO fixed pallet count.


CLASSIFIER SETTINGS

Image size:
{CLASSIFIER_IMAGE_SIZE}

Minimum displayed confidence:
{CLASSIFIER_MIN_CONFIDENCE}


LIVE CONTROLS

S
Save current detection screenshot.

Q
Quit.


IMPORTANT

This stage currently uses the improved
EMPTY vs OBJECT_PRESENT classifier.

If live classification is not reliable,
the next step is to retrain the classifier
using the real live pallet crops that were
collected from the conveyor.
""".strip()

    README_FILE.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# CROP PALLET
# ============================================================

def crop_pallet(frame, box):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(
        box_width * PALLET_CROP_PADDING
    )

    pad_y = int(
        box_height * PALLET_CROP_PADDING
    )

    x1 = max(
        0,
        x1 - pad_x,
    )

    y1 = max(
        0,
        y1 - pad_y,
    )

    x2 = min(
        width,
        x2 + pad_x,
    )

    y2 = min(
        height,
        y2 + pad_y,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[
        y1:y2,
        x1:x2
    ].copy()

    if crop.size == 0:
        return None

    return crop


# ============================================================
# NORMALIZE CLASS NAME
# ============================================================

def normalize_class_name(name):

    text = str(name).strip().upper()

    text = text.replace(
        "-",
        "_",
    )

    text = text.replace(
        " ",
        "_",
    )

    if "EMPTY" in text:
        return "EMPTY"

    if (
        "OBJECT" in text
        or "REJECT" in text
        or "PRESENT" in text
    ):
        return "OBJECT_PRESENT"

    return text


# ============================================================
# CLASSIFY PALLET CROP
# ============================================================

def classify_pallet(
    classifier,
    crop,
):

    result = classifier.predict(
        source=crop,
        imgsz=CLASSIFIER_IMAGE_SIZE,
        device=0,
        verbose=False,
    )[0]

    if result.probs is None:
        return (
            "UNKNOWN",
            0.0,
        )

    class_id = int(
        result.probs.top1
    )

    confidence = float(
        result.probs.top1conf.item()
    )

    class_names = result.names

    raw_name = class_names[
        class_id
    ]

    class_name = normalize_class_name(
        raw_name
    )

    return (
        class_name,
        confidence,
    )


# ============================================================
# DRAW RESULT
# ============================================================

def draw_result(
    frame,
    outer_polygon,
    inner_polygon,
    tracks,
    classifications,
    raw_count,
    valid_count,
):

    output = frame.copy()

    cv2.polylines(
        output,
        [outer_polygon],
        True,
        (0, 255, 0),
        3,
    )

    cv2.polylines(
        output,
        [inner_polygon],
        True,
        (0, 255, 255),
        3,
    )

    empty_count = 0
    object_count = 0

    for track in tracks:

        track_id = track[
            "track_id"
        ]

        x1, y1, x2, y2 = track[
            "box"
        ]

        class_name, class_conf = classifications.get(
            track_id,
            (
                "WAITING",
                0.0,
            ),
        )

        if class_name == "EMPTY":

            empty_count += 1

            box_color = (
                0,
                255,
                0,
            )

        elif class_name == "OBJECT_PRESENT":

            object_count += 1

            box_color = (
                0,
                165,
                255,
            )

        else:

            box_color = (
                255,
                255,
                0,
            )

        cv2.rectangle(
            output,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            box_color,
            4,
        )

        label = (
            f"P{track_id} "
            f"{class_name} "
            f"{class_conf:.2f}"
        )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    30,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            box_color,
            2,
        )

    cv2.rectangle(
        output,
        (
            10,
            10,
        ),
        (
            660,
            190,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    cv2.putText(
        output,
        (
            "PHYSICAL PALLETS: "
            f"{len(tracks)}"
        ),
        (
            25,
            42,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (
            255,
            255,
            255,
        ),
        2,
    )

    cv2.putText(
        output,
        (
            "EMPTY: "
            f"{empty_count}"
        ),
        (
            25,
            78,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (
            0,
            255,
            0,
        ),
        2,
    )

    cv2.putText(
        output,
        (
            "OBJECT PRESENT: "
            f"{object_count}"
        ),
        (
            25,
            113,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (
            0,
            165,
            255,
        ),
        2,
    )

    cv2.putText(
        output,
        (
            "YOLO RAW: "
            f"{raw_count}"
            "   "
            "BOUNDARY VALID: "
            f"{valid_count}"
        ),
        (
            25,
            148,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (
            255,
            255,
            255,
        ),
        2,
    )

    cv2.putText(
        output,
        (
            "S: screenshot   "
            "Q: quit"
        ),
        (
            25,
            180,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (
            255,
            255,
            255,
        ),
        2,
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    write_readme()

    if not PALLET_STAGE_FILE.exists():

        raise FileNotFoundError(
            f"Stage 08 code not found:\n"
            f"{PALLET_STAGE_FILE}"
        )

    if not PALLET_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Pallet model not found:\n"
            f"{PALLET_MODEL_PATH}"
        )

    if not CLASSIFIER_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Classifier model not found:\n"
            f"{CLASSIFIER_MODEL_PATH}"
        )

    if not BOUNDARY_FILE.exists():

        raise FileNotFoundError(
            f"Boundary file not found:\n"
            f"{BOUNDARY_FILE}"
        )

    pallet_stage = load_pallet_stage()

    print(
        "Loading pallet detector..."
    )

    pallet_model = YOLO(
        str(PALLET_MODEL_PATH)
    )

    print(
        "Loading EMPTY / OBJECT_PRESENT classifier..."
    )

    classifier = YOLO(
        str(CLASSIFIER_MODEL_PATH)
    )

    print(
        f"Opening camera index "
        f"{CAMERA_INDEX}"
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():

        cap.release()

        cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Camera resolution: "
        f"{actual_width} x "
        f"{actual_height}"
    )

    boundary = pallet_stage.load_boundary()

    if boundary is None:

        cap.release()

        raise RuntimeError(
            "Could not load conveyor boundary."
        )

    tracker = pallet_stage.DynamicTracker()

    window_name = (
        "CREST - Live Object Detection On Pallet"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    for _ in range(15):

        cap.read()

    while True:

        success, frame = cap.read()

        if not success:

            print(
                "Camera read failed."
            )

            break

        (
            ring_mask,
            outer_polygon,
            inner_polygon,
        ) = pallet_stage.create_ring_mask(
            frame.shape,
            boundary,
        )

        raw_detections = (
            pallet_stage.get_yolo_detections(
                pallet_model,
                frame,
            )
        )

        (
            valid_detections,
            rejected_detections,
        ) = pallet_stage.filter_by_ring(
            raw_detections,
            ring_mask,
        )

        confirmed_tracks = tracker.update(
            valid_detections
        )

        classifications = {}

        for track in confirmed_tracks:

            track_id = track[
                "track_id"
            ]

            crop = crop_pallet(
                frame,
                track["box"],
            )

            if crop is None:

                classifications[
                    track_id
                ] = (
                    "UNKNOWN",
                    0.0,
                )

                continue

            (
                class_name,
                class_confidence,
            ) = classify_pallet(
                classifier,
                crop,
            )

            classifications[
                track_id
            ] = (
                class_name,
                class_confidence,
            )

        output = draw_result(
            frame=frame,
            outer_polygon=outer_polygon,
            inner_polygon=inner_polygon,
            tracks=confirmed_tracks,
            classifications=classifications,
            raw_count=len(
                raw_detections
            ),
            valid_count=len(
                valid_detections
            ),
        )

        cv2.imshow(
            window_name,
            output,
        )

        key = cv2.waitKey(
            1
        ) & 0xFF

        if key == ord(
            "q"
        ):

            break

        elif key == ord(
            "s"
        ):

            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )

            save_path = (
                SAVE_DIR
                / (
                    "live_object_detection_"
                    f"{timestamp}.jpg"
                )
            )

            cv2.imwrite(
                str(save_path),
                output,
            )

            print(
                f"Saved: "
                f"{save_path}"
            )

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
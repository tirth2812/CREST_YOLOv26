from pathlib import Path
from datetime import datetime

import cv2
from ultralytics import YOLO


ROOT = Path(
    r"C:\Do_Not_Delete_PLC\original images"
)

PALLET_MODEL_PATH = (
    ROOT
    / "02_YOLO_Pallet_Localizer_Final"
    / "MODEL"
    / "pallet_yolo11n_best.pt"
)

PROJECT_DIR = (
    ROOT
    / "07_Live_Empty_Object_Classification"
)

SCREENSHOT_DIR = (
    PROJECT_DIR
    / "saved_live_results"
)

EMPTY_SAVE_DIR = Path(
    r"C:\Do_Not_Delete_PLC\PLC\PLC\close_original\empty"
)

OBJECT_SAVE_DIR = Path(
    r"C:\Do_Not_Delete_PLC\PLC\PLC\close_original\rejcted"
)


SCREENSHOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EMPTY_SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OBJECT_SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 1

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30


# ============================================================
# YOLO
# ============================================================

PALLET_CONFIDENCE = 0.05
PALLET_IMAGE_SIZE = 960
PALLET_NMS_IOU = 0.60

MAX_PALLETS = 6


# ============================================================
# CONVEYOR AREA
# ============================================================

ROI_ZONES = [

    (0.27, 0.46, 0.93, 0.68),

    (0.12, 0.72, 0.91, 0.98),

    (0.10, 0.52, 0.36, 0.91),

    (0.77, 0.47, 0.96, 0.94),
]


# ============================================================
# LABELS
# ============================================================

manual_labels = [
    "EMPTY",
    "EMPTY",
    "EMPTY",
    "EMPTY",
    "EMPTY",
    "EMPTY",
]


# ============================================================
# FREEZE STORAGE
# ============================================================

frozen = False

frozen_frame = None

frozen_pallets = []

frozen_crops = []


# ============================================================
# HELPERS
# ============================================================

def center_inside_conveyor(
    center_x,
    center_y,
    frame_width,
    frame_height
):

    nx = center_x / frame_width
    ny = center_y / frame_height

    for left, top, right, bottom in ROI_ZONES:

        if (
            left <= nx <= right
            and
            top <= ny <= bottom
        ):
            return True

    return False


def toggle_label(index):

    if manual_labels[index] == "EMPTY":

        manual_labels[index] = (
            "OBJECT_PRESENT"
        )

    else:

        manual_labels[index] = (
            "EMPTY"
        )


def reset_labels():

    for i in range(6):

        manual_labels[i] = "EMPTY"


# ============================================================
# MODEL
# ============================================================

if not PALLET_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Missing model:\n{PALLET_MODEL_PATH}"
    )


print("=" * 75)
print("CREST - SAFE MIXED PALLET DATA COLLECTION")
print("=" * 75)

print()
print("F = FREEZE current 6 pallets")
print("1-6 = toggle EMPTY / OBJECT_PRESENT")
print("K = SAVE frozen labeled crops")
print("U = UNFREEZE and continue live video")
print("R = reset all labels to EMPTY")
print("S = screenshot")
print("Q = quit")
print()

print("IMPORTANT:")
print("Freeze with F BEFORE labeling.")
print()


pallet_model = YOLO(
    str(PALLET_MODEL_PATH)
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    cap.release()

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )


if not cap.isOpened():

    raise RuntimeError(
        "Could not open camera index 1."
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)


WINDOW_NAME = (
    "CREST - SAFE MIXED DATA COLLECTION"
)

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    720
)


# ============================================================
# LIVE LOOP
# ============================================================

current_frame = None
current_pallets = []
current_crops = []


while True:

    # ========================================================
    # LIVE MODE
    # ========================================================

    if not frozen:

        ret, frame = cap.read()

        if not ret:

            print(
                "Could not read camera."
            )

            break


        frame_height, frame_width = (
            frame.shape[:2]
        )


        results = pallet_model.predict(

            source=frame,

            imgsz=PALLET_IMAGE_SIZE,

            conf=PALLET_CONFIDENCE,

            iou=PALLET_NMS_IOU,

            max_det=20,

            device=0,

            verbose=False,
        )


        possible_pallets = []


        if results:

            result = results[0]

            if (
                result.boxes is not None
                and
                len(result.boxes) > 0
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
                    scores
                ):

                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    center_x = (
                        x1 + x2
                    ) / 2.0

                    center_y = (
                        y1 + y2
                    ) / 2.0


                    if center_inside_conveyor(
                        center_x,
                        center_y,
                        frame_width,
                        frame_height
                    ):

                        possible_pallets.append(
                            {
                                "bbox": (
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ),

                                "center_x":
                                center_x,

                                "score":
                                float(score)
                            }
                        )


        # Keep strongest 6
        possible_pallets.sort(
            key=lambda item:
            item["score"],
            reverse=True
        )


        current_pallets = (
            possible_pallets[
                :MAX_PALLETS
            ]
        )


        # Stable display order for current frozen frame
        current_pallets.sort(
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0]
            )
        )


        current_crops = []


        for detection in current_pallets:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )


            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )


            pad_x = int(
                box_width * 0.08
            )

            pad_y = int(
                box_height * 0.12
            )


            crop_x1 = max(
                0,
                x1 - pad_x
            )

            crop_y1 = max(
                0,
                y1 - pad_y
            )

            crop_x2 = min(
                frame_width,
                x2 + pad_x
            )

            crop_y2 = min(
                frame_height,
                y2 + pad_y
            )


            crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]


            current_crops.append(
                crop.copy()
            )


        current_frame = (
            frame.copy()
        )


    # ========================================================
    # USE FROZEN OR LIVE DATA
    # ========================================================

    if frozen:

        display_frame = (
            frozen_frame.copy()
        )

        display_pallets = (
            frozen_pallets
        )

        display_crops = (
            frozen_crops
        )

    else:

        display_frame = (
            current_frame.copy()
        )

        display_pallets = (
            current_pallets
        )

        display_crops = (
            current_crops
        )


    # ========================================================
    # DRAW PALLETS
    # ========================================================

    empty_count = 0
    object_count = 0


    for index, detection in enumerate(
        display_pallets
    ):

        x1, y1, x2, y2 = (
            detection["bbox"]
        )


        label = (
            manual_labels[index]
        )


        if label == "EMPTY":

            color = (
                0,
                255,
                0
            )

            empty_count += 1

        else:

            color = (
                0,
                0,
                255
            )

            object_count += 1


        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )


        cv2.putText(
            display_frame,
            f"P{index + 1} {label}",
            (
                x1,
                max(
                    30,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # INFO PANEL
    # ========================================================

    cv2.rectangle(
        display_frame,
        (10, 10),
        (790, 190),
        (0, 0, 0),
        -1
    )


    mode_text = (
        "FROZEN - LABEL NOW"
        if frozen
        else
        "LIVE - PRESS F TO FREEZE"
    )


    cv2.putText(
        display_frame,
        mode_text,
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (
            0,
            255,
            255
        ),
        2
    )


    cv2.putText(
        display_frame,
        f"PALLETS: {len(display_crops)} / 6",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display_frame,
        f"EMPTY: {empty_count}",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        display_frame,
        f"OBJECT: {object_count}",
        (220, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )


    cv2.putText(
        display_frame,
        "F=Freeze  1-6=Toggle  K=Save  U=Unfreeze",
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display_frame,
        "R=Reset  S=Screenshot  Q=Quit",
        (25, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # ========================================================
    # QUIT
    # ========================================================

    if (
        key == ord("q")
        or
        key == 27
    ):

        break


    # ========================================================
    # FREEZE
    # ========================================================

    elif key == ord("f"):

        if frozen:

            print(
                "Already frozen."
            )

        elif len(current_crops) != 6:

            print(
                "Cannot freeze: need exactly 6 pallets."
            )

        else:

            frozen = True

            frozen_frame = (
                current_frame.copy()
            )

            frozen_pallets = [
                item.copy()
                for item in current_pallets
            ]

            frozen_crops = [
                crop.copy()
                for crop in current_crops
            ]

            reset_labels()

            print()
            print(
                "FRAME FROZEN."
            )

            print(
                "Now set labels using keys 1-6."
            )


    # ========================================================
    # UNFREEZE
    # ========================================================

    elif key == ord("u"):

        frozen = False

        frozen_frame = None
        frozen_pallets = []
        frozen_crops = []

        reset_labels()

        print(
            "Returned to LIVE mode."
        )


    # ========================================================
    # TOGGLE LABELS ONLY WHEN FROZEN
    # ========================================================

    elif key in [
        ord("1"),
        ord("2"),
        ord("3"),
        ord("4"),
        ord("5"),
        ord("6"),
    ]:

        if not frozen:

            print(
                "Press F first to freeze."
            )

        else:

            index = (
                key - ord("1")
            )

            toggle_label(
                index
            )

            print(
                f"P{index + 1} -> "
                f"{manual_labels[index]}"
            )


    # ========================================================
    # RESET LABELS
    # ========================================================

    elif key == ord("r"):

        reset_labels()

        print(
            "All labels reset to EMPTY."
        )


    # ========================================================
    # SAVE LABELED FROZEN CROPS
    # ========================================================

    elif key == ord("k"):

        if not frozen:

            print(
                "NOT SAVED: press F first."
            )

            continue


        if len(frozen_crops) != 6:

            print(
                "NOT SAVED: need exactly 6 frozen crops."
            )

            continue


        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )


        saved_empty = 0
        saved_object = 0


        for index, (
            crop,
            label
        ) in enumerate(

            zip(
                frozen_crops,
                manual_labels
            ),

            start=1
        ):

            if label == "EMPTY":

                save_path = (
                    EMPTY_SAVE_DIR
                    / (
                        f"live_empty_"
                        f"{timestamp}_"
                        f"p{index}.jpg"
                    )
                )

                saved_empty += 1

            else:

                save_path = (
                    OBJECT_SAVE_DIR
                    / (
                        f"live_object_"
                        f"{timestamp}_"
                        f"p{index}.jpg"
                    )
                )

                saved_object += 1


            cv2.imwrite(
                str(save_path),
                crop
            )


        print()
        print("=" * 45)
        print("SAVED CORRECT FROZEN SET")
        print("=" * 45)

        print(
            "EMPTY:",
            saved_empty
        )

        print(
            "OBJECT_PRESENT:",
            saved_object
        )

        print()
        print(
            "Press U for next live sample."
        )


    # ========================================================
    # SCREENSHOT
    # ========================================================

    elif key == ord("s"):

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )


        save_path = (
            SCREENSHOT_DIR
            / (
                f"collection_"
                f"{timestamp}.jpg"
            )
        )


        cv2.imwrite(
            str(save_path),
            display_frame
        )

        print(
            "Screenshot:",
            save_path
        )


cap.release()

cv2.destroyAllWindows()

print()
print("Camera closed.")
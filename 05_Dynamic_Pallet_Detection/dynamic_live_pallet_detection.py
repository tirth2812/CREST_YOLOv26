from pathlib import Path
import json
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(
    r"C:\Do_Not_Delete_PLC\original images\08_Dynamic_Live_Pallet_Detection_No_Fixed_Count"
)

MODEL_PATH = Path(
    r"C:\Do_Not_Delete_PLC\original images"
    r"\02_YOLO_Pallet_Localizer_Final"
    r"\MODEL"
    r"\pallet_yolo11n_best.pt"
)

BOUNDARY_FILE = BASE_DIR / "conveyor_boundary.json"
README_FILE = BASE_DIR / "README_DYNAMIC_PALLET_DETECTION.txt"
SAVE_DIR = BASE_DIR / "saved_dynamic_detection_frames"

CAMERA_INDEX = 1
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

YOLO_CONFIDENCE = 0.15
YOLO_IMAGE_SIZE = 960
YOLO_NMS_IOU = 0.60
YOLO_MAX_DETECTIONS = 100

RING_OVERLAP_THRESHOLD = 0.25

CONFIRM_FRAMES = 5
TRACK_IOU_THRESHOLD = 0.25
TRACK_MAX_MISSES = 5

PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720

FREEHAND_MIN_DISTANCE = 3

WINDOW_LIVE = "CREST Dynamic Pallet Detection"
WINDOW_CALIBRATION = "CREST Freehand Conveyor Boundary Calibration"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def write_readme():
    text = f"""
CREST DYNAMIC LIVE PALLET DETECTION
===============================================

WORKING FOLDER

{BASE_DIR}


PURPOSE

Detect the actual number of pallets physically present
on the conveyor.

NO fixed pallet count is used.


CAMERA

Camera index:
{CAMERA_INDEX}

Requested resolution:
{CAMERA_WIDTH} x {CAMERA_HEIGHT}

Full camera image is used.

NO TOP CROP.


PALLET MODEL

{MODEL_PATH}


CURRENT DETECTION FLOW

LIVE CAMERA
    |
    v
Fine-tuned YOLO11n pallet detector
    |
    v
Freehand conveyor boundary filter
    |
    v
Temporal confirmation
    |
    v
Dynamic physical pallet count


FREEHAND CONVEYOR CALIBRATION

The conveyor is defined using two freehand closed paths.

OUTER boundary:
Draw around the outside edge of the complete conveyor.

INNER boundary:
Draw around the inside edge of the conveyor.

Allowed detection region:

OUTER boundary
minus
INNER boundary


BOUNDARY FILE

{BOUNDARY_FILE}


CALIBRATION CONTROLS

OUTER:

Hold LEFT MOUSE BUTTON
and draw continuously around the complete OUTER edge.

Release the mouse when the outer loop is finished.

Press N.


INNER:

Hold LEFT MOUSE BUTTON
and draw continuously around the complete INNER edge.

Release the mouse when the inner loop is finished.


R
Clear both boundaries and start again.

U
Clear the boundary currently being drawn.

S
Save calibration.

ENTER
Start pallet detection.

Q
Quit.


EXISTING SAVED CALIBRATION

If conveyor_boundary.json exists:

ENTER
Use saved calibration.

C or R
Clear it and redraw.


LIVE CONTROLS

C
Recalibrate boundary.

S
Save screenshot.

Q
Quit.


YOLO SETTINGS

Confidence:
{YOLO_CONFIDENCE}

Image size:
{YOLO_IMAGE_SIZE}

NMS IoU:
{YOLO_NMS_IOU}

Raw maximum detections:
{YOLO_MAX_DETECTIONS}

This is only a YOLO processing limit.

It is NOT a fixed pallet count.


BOUNDARY FILTER

A pallet candidate is accepted only when:

1. Bounding-box center is inside the conveyor ring.

AND

2. At least {RING_OVERLAP_THRESHOLD * 100:.0f}% of the bounding box overlaps
   the conveyor ring.


TEMPORAL CONFIRMATION

Required consecutive matches:
{CONFIRM_FRAMES}

Track IoU:
{TRACK_IOU_THRESHOLD}

Maximum missed frames:
{TRACK_MAX_MISSES}


IMPORTANT

The physical pallet count is dynamic.

The system does NOT assume 6 pallets.
""".strip()

    README_FILE.write_text(text, encoding="utf-8")


def point_distance(point_a, point_b):
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )


def normalize_points(points):
    return [
        [
            float(x) / PREVIEW_WIDTH,
            float(y) / PREVIEW_HEIGHT,
        ]
        for x, y in points
    ]


def denormalize_points(points, width, height):
    result = []

    for x, y in points:
        result.append(
            [
                int(round(float(x) * width)),
                int(round(float(y) * height)),
            ]
        )

    return np.array(result, dtype=np.int32)


def clean_polygon(points):
    if len(points) < 3:
        return points

    contour = np.array(
        points,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    epsilon = max(
        1.0,
        0.0015 * perimeter,
    )

    simplified = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    return [
        (
            int(point[0][0]),
            int(point[0][1]),
        )
        for point in simplified
    ]


def save_boundary(outer_points, inner_points):
    outer_clean = clean_polygon(outer_points)
    inner_clean = clean_polygon(inner_points)

    data = {
        "version": 2,
        "type": "freehand_conveyor_ring",
        "coordinate_type": "normalized",
        "outer_polygon": normalize_points(outer_clean),
        "inner_polygon": normalize_points(inner_clean),
    }

    with BOUNDARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    return data


def load_boundary():
    if not BOUNDARY_FILE.exists():
        return None

    try:
        with BOUNDARY_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        outer = data.get(
            "outer_polygon",
            [],
        )

        inner = data.get(
            "inner_polygon",
            [],
        )

        if len(outer) < 3:
            return None

        if len(inner) < 3:
            return None

        return {
            "outer_polygon": outer,
            "inner_polygon": inner,
        }

    except Exception as error:
        print(
            f"Boundary load error: {error}"
        )
        return None


def create_ring_mask(frame_shape, boundary):
    height, width = frame_shape[:2]

    outer = denormalize_points(
        boundary["outer_polygon"],
        width,
        height,
    )

    inner = denormalize_points(
        boundary["inner_polygon"],
        width,
        height,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [outer],
        255,
    )

    cv2.fillPoly(
        mask,
        [inner],
        0,
    )

    return mask, outer, inner


def calibration_screen(
    frame,
    existing_boundary=None,
):
    preview = cv2.resize(
        frame,
        (
            PREVIEW_WIDTH,
            PREVIEW_HEIGHT,
        ),
    )

    state = {
        "outer": [],
        "inner": [],
        "mode": "outer",
        "drawing": False,
    }

    if existing_boundary is not None:
        state["outer"] = [
            (
                int(x * PREVIEW_WIDTH),
                int(y * PREVIEW_HEIGHT),
            )
            for x, y
            in existing_boundary["outer_polygon"]
        ]

        state["inner"] = [
            (
                int(x * PREVIEW_WIDTH),
                int(y * PREVIEW_HEIGHT),
            )
            for x, y
            in existing_boundary["inner_polygon"]
        ]

        state["mode"] = "saved"

    def add_point(x, y):
        if state["mode"] == "outer":
            points = state["outer"]

        elif state["mode"] == "inner":
            points = state["inner"]

        else:
            return

        new_point = (x, y)

        if not points:
            points.append(new_point)
            return

        if point_distance(
            points[-1],
            new_point,
        ) >= FREEHAND_MIN_DISTANCE:
            points.append(new_point)

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param,
    ):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["mode"] in (
                "outer",
                "inner",
            ):
                state["drawing"] = True
                add_point(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if state["drawing"]:
                add_point(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if state["drawing"]:
                add_point(x, y)
                state["drawing"] = False

    cv2.namedWindow(
        WINDOW_CALIBRATION,
        cv2.WINDOW_AUTOSIZE,
    )

    cv2.setMouseCallback(
        WINDOW_CALIBRATION,
        mouse_callback,
    )

    while True:
        canvas = preview.copy()

        outer = state["outer"]
        inner = state["inner"]

        if len(outer) >= 2:
            outer_array = np.array(
                outer,
                dtype=np.int32,
            )

            cv2.polylines(
                canvas,
                [outer_array],
                state["mode"] != "outer",
                (0, 255, 0),
                3,
            )

        if len(inner) >= 2:
            inner_array = np.array(
                inner,
                dtype=np.int32,
            )

            cv2.polylines(
                canvas,
                [inner_array],
                state["mode"] != "inner",
                (0, 255, 255),
                3,
            )

        if (
            len(outer) >= 3
            and len(inner) >= 3
            and state["mode"] == "saved"
        ):
            overlay = canvas.copy()

            cv2.fillPoly(
                overlay,
                [
                    np.array(
                        outer,
                        dtype=np.int32,
                    )
                ],
                (0, 120, 0),
            )

            cv2.fillPoly(
                overlay,
                [
                    np.array(
                        inner,
                        dtype=np.int32,
                    )
                ],
                (0, 0, 0),
            )

            canvas = cv2.addWeighted(
                overlay,
                0.18,
                canvas,
                0.82,
                0,
            )

            cv2.polylines(
                canvas,
                [
                    np.array(
                        outer,
                        dtype=np.int32,
                    )
                ],
                True,
                (0, 255, 0),
                3,
            )

            cv2.polylines(
                canvas,
                [
                    np.array(
                        inner,
                        dtype=np.int32,
                    )
                ],
                True,
                (0, 255, 255),
                3,
            )

        cv2.rectangle(
            canvas,
            (0, 0),
            (
                PREVIEW_WIDTH,
                115,
            ),
            (0, 0, 0),
            -1,
        )

        if state["mode"] == "outer":
            title = (
                "DRAW OUTER BOUNDARY - "
                "HOLD LEFT MOUSE AND DRAG"
            )

        elif state["mode"] == "inner":
            title = (
                "DRAW INNER BOUNDARY - "
                "HOLD LEFT MOUSE AND DRAG"
            )

        else:
            title = "BOUNDARY READY"

        cv2.putText(
            canvas,
            title,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            canvas,
            "N: OUTER -> INNER   R/C: redraw   U: clear current boundary",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            canvas,
            "S: save   ENTER: start detection   Q: quit",
            (20, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            WINDOW_CALIBRATION,
            canvas,
        )

        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            cv2.destroyWindow(
                WINDOW_CALIBRATION
            )
            return None

        if key in (
            ord("r"),
            ord("c"),
        ):
            state["outer"] = []
            state["inner"] = []
            state["mode"] = "outer"
            state["drawing"] = False

        elif key == ord("u"):
            if state["mode"] == "outer":
                state["outer"] = []

            elif state["mode"] == "inner":
                state["inner"] = []

        elif key == ord("n"):
            if (
                state["mode"] == "outer"
                and len(state["outer"]) >= 3
            ):
                state["outer"] = clean_polygon(
                    state["outer"]
                )

                state["mode"] = "inner"
                state["drawing"] = False

        elif key == ord("s"):
            if (
                len(state["outer"]) >= 3
                and len(state["inner"]) >= 3
            ):
                data = save_boundary(
                    state["outer"],
                    state["inner"],
                )

                state["outer"] = [
                    (
                        int(x * PREVIEW_WIDTH),
                        int(y * PREVIEW_HEIGHT),
                    )
                    for x, y
                    in data["outer_polygon"]
                ]

                state["inner"] = [
                    (
                        int(x * PREVIEW_WIDTH),
                        int(y * PREVIEW_HEIGHT),
                    )
                    for x, y
                    in data["inner_polygon"]
                ]

                state["mode"] = "saved"

                print(
                    f"Boundary saved: "
                    f"{BOUNDARY_FILE}"
                )

        elif key in (
            13,
            10,
        ):
            if (
                len(state["outer"]) >= 3
                and len(state["inner"]) >= 3
            ):
                data = save_boundary(
                    state["outer"],
                    state["inner"],
                )

                boundary = {
                    "outer_polygon":
                        data["outer_polygon"],
                    "inner_polygon":
                        data["inner_polygon"],
                }

                cv2.destroyWindow(
                    WINDOW_CALIBRATION
                )

                return boundary


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1,
    )

    ih = max(
        0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = max(
        0,
        ax2 - ax1,
    ) * max(
        0,
        ay2 - ay1,
    )

    area_b = max(
        0,
        bx2 - bx1,
    ) * max(
        0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def ring_overlap_ratio(
    box,
    ring_mask,
):
    height, width = ring_mask.shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(
            width - 1,
            int(x1),
        ),
    )

    y1 = max(
        0,
        min(
            height - 1,
            int(y1),
        ),
    )

    x2 = max(
        0,
        min(
            width,
            int(x2),
        ),
    )

    y2 = max(
        0,
        min(
            height,
            int(y2),
        ),
    )

    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = ring_mask[
        y1:y2,
        x1:x2,
    ]

    if region.size == 0:
        return 0.0

    ring_pixels = cv2.countNonZero(
        region
    )

    return (
        ring_pixels
        / float(region.size)
    )


def box_center_inside_ring(
    box,
    ring_mask,
):
    height, width = ring_mask.shape[:2]

    x1, y1, x2, y2 = box

    center_x = int(
        (x1 + x2) / 2
    )

    center_y = int(
        (y1 + y2) / 2
    )

    center_x = max(
        0,
        min(
            width - 1,
            center_x,
        ),
    )

    center_y = max(
        0,
        min(
            height - 1,
            center_y,
        ),
    )

    return (
        ring_mask[
            center_y,
            center_x,
        ] > 0
    )


class DynamicTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        for track in self.tracks.values():
            track["matched"] = False

        used_tracks = set()
        used_detections = set()

        matches = []

        for detection_index, detection in enumerate(
            detections
        ):
            best_track_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue

                score = box_iou(
                    detection["box"],
                    track["box"],
                )

                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if (
                best_track_id is not None
                and best_iou >= TRACK_IOU_THRESHOLD
            ):
                matches.append(
                    (
                        detection_index,
                        best_track_id,
                    )
                )

                used_detections.add(
                    detection_index
                )

                used_tracks.add(
                    best_track_id
                )

        for detection_index, track_id in matches:
            detection = detections[
                detection_index
            ]

            track = self.tracks[
                track_id
            ]

            track["box"] = detection["box"]
            track["confidence"] = detection[
                "confidence"
            ]
            track["overlap"] = detection[
                "overlap"
            ]

            track["hits"] += 1
            track["misses"] = 0
            track["matched"] = True

        for detection_index, detection in enumerate(
            detections
        ):
            if detection_index in used_detections:
                continue

            self.tracks[
                self.next_id
            ] = {
                "box":
                    detection["box"],
                "confidence":
                    detection["confidence"],
                "overlap":
                    detection["overlap"],
                "hits": 1,
                "misses": 0,
                "matched": True,
            }

            self.next_id += 1

        remove_ids = []

        for track_id, track in self.tracks.items():
            if not track["matched"]:
                track["misses"] += 1

            if (
                track["misses"]
                > TRACK_MAX_MISSES
            ):
                remove_ids.append(
                    track_id
                )

        for track_id in remove_ids:
            del self.tracks[
                track_id
            ]

        confirmed = []

        for track_id, track in self.tracks.items():
            if (
                track["matched"]
                and track["hits"]
                >= CONFIRM_FRAMES
            ):
                confirmed.append(
                    {
                        "track_id":
                            track_id,
                        **track,
                    }
                )

        return confirmed


def get_yolo_detections(
    model,
    frame,
):
    result = model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_NMS_IOU,
        imgsz=YOLO_IMAGE_SIZE,
        device=0,
        max_det=YOLO_MAX_DETECTIONS,
        verbose=False,
    )[0]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(
            box.cls[0].item()
        )

        if class_id != 0:
            continue

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        detections.append(
            {
                "box": (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ),
                "confidence":
                    confidence,
            }
        )

    return detections


def filter_by_ring(
    detections,
    ring_mask,
):
    accepted = []
    rejected = []

    for detection in detections:
        box = detection["box"]

        center_inside = (
            box_center_inside_ring(
                box,
                ring_mask,
            )
        )

        overlap = ring_overlap_ratio(
            box,
            ring_mask,
        )

        detection["overlap"] = overlap

        if (
            center_inside
            and overlap
            >= RING_OVERLAP_THRESHOLD
        ):
            accepted.append(
                detection
            )

        else:
            rejected.append(
                detection
            )

    return accepted, rejected


def draw_live_frame(
    frame,
    outer_polygon,
    inner_polygon,
    raw_detections,
    valid_detections,
    rejected_detections,
    confirmed_tracks,
):
    output = frame.copy()

    overlay = output.copy()

    cv2.fillPoly(
        overlay,
        [outer_polygon],
        (40, 100, 40),
    )

    cv2.fillPoly(
        overlay,
        [inner_polygon],
        (0, 0, 0),
    )

    output = cv2.addWeighted(
        overlay,
        0.10,
        output,
        0.90,
        0,
    )

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

    for detection in rejected_detections:
        x1, y1, x2, y2 = detection[
            "box"
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

    for detection in valid_detections:
        x1, y1, x2, y2 = detection[
            "box"
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 150, 0),
            2,
        )

    for track in confirmed_tracks:
        x1, y1, x2, y2 = track[
            "box"
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4,
        )

        cv2.putText(
            output,
            f"PALLET {track['confidence']:.2f}",
            (
                x1,
                max(
                    30,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

    cv2.rectangle(
        output,
        (10, 10),
        (630, 160),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        (
            "PHYSICAL PALLETS DETECTED: "
            f"{len(confirmed_tracks)}"
        ),
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        output,
        (
            "YOLO RAW: "
            f"{len(raw_detections)}"
        ),
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        output,
        (
            "BOUNDARY VALID: "
            f"{len(valid_detections)}"
        ),
        (25, 113),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        output,
        (
            "FILTERED FALSE: "
            f"{len(rejected_detections)}"
        ),
        (25, 146),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        output,
        (
            "C: recalibrate   "
            "S: screenshot   "
            "Q: quit"
        ),
        (
            20,
            output.shape[0] - 25,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return output


def main():
    write_readme()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        f"Loading YOLO model: "
        f"{MODEL_PATH}"
    )

    model = YOLO(
        str(MODEL_PATH)
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
            f"Could not open camera "
            f"index {CAMERA_INDEX}"
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

    for _ in range(15):
        cap.read()

    success, frame = cap.read()

    if not success:
        cap.release()

        raise RuntimeError(
            "Could not read initial "
            "camera frame."
        )

    saved_boundary = load_boundary()

    boundary = calibration_screen(
        frame,
        existing_boundary=saved_boundary,
    )

    if boundary is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    tracker = DynamicTracker()

    cv2.namedWindow(
        WINDOW_LIVE,
        cv2.WINDOW_NORMAL,
    )

    while True:
        success, frame = cap.read()

        if not success:
            print(
                "Camera frame read failed."
            )
            break

        (
            ring_mask,
            outer_polygon,
            inner_polygon,
        ) = create_ring_mask(
            frame.shape,
            boundary,
        )

        raw_detections = (
            get_yolo_detections(
                model,
                frame,
            )
        )

        (
            valid_detections,
            rejected_detections,
        ) = filter_by_ring(
            raw_detections,
            ring_mask,
        )

        confirmed_tracks = (
            tracker.update(
                valid_detections
            )
        )

        output = draw_live_frame(
            frame,
            outer_polygon,
            inner_polygon,
            raw_detections,
            valid_detections,
            rejected_detections,
            confirmed_tracks,
        )

        cv2.imshow(
            WINDOW_LIVE,
            output,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )

            save_path = (
                SAVE_DIR
                / (
                    "dynamic_pallet_detection_"
                    f"{timestamp}.jpg"
                )
            )

            cv2.imwrite(
                str(save_path),
                output,
            )

            print(
                f"Saved: {save_path}"
            )

        elif key == ord("c"):
            tracker = DynamicTracker()

            boundary = calibration_screen(
                frame,
                existing_boundary=boundary,
            )

            if boundary is None:
                break

            cv2.namedWindow(
                WINDOW_LIVE,
                cv2.WINDOW_NORMAL,
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CREST YOLO26 - PHASE 27 FAST NEW-OBJECT + OCCLUSION-HOLD LIVE
# SELF-CONTAINED VERSION
#
# IMPORTANT FIX
# -------------
# This version does NOT import helper functions from
# live_final_empty_accepted_rejected.py.
#
# The previous candidate failed because the runtime copy of that
# module did not expose merge_pallet_detections as an attribute.
#
# This file now contains the required live helper logic directly
# and only loads dynamic_live_pallet_detection.py as the tracker stage.
#
# EXISTING live_final_empty_accepted_rejected.py IS NOT MODIFIED.
# ============================================================


# ============================================================
# PATHS
# ============================================================

CODE_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = CODE_DIR.parent

STAGE08_FILE = (
    CODE_DIR
    / "dynamic_live_pallet_detection.py"
)

PALLET_MODEL_PATH = (
    SYSTEM_ROOT
    / "MODELS"
    / "pallet_yolo26n_best.pt"
)

OBJECT_MODEL_PATH = (
    SYSTEM_ROOT
    / "MODELS"
    / "empty_object_yolo26n_cls_best.pt"
)

SYMBOL_MODEL_PATH = (
    SYSTEM_ROOT
    / "MODELS"
    / "x_no_x_yolo26n_medium_focus_final_hard_best.pt"
)

BOUNDARY_FILE = (
    SYSTEM_ROOT
    / "CONFIG"
    / "conveyor_boundary.json"
)

SAVE_DIR = (
    SYSTEM_ROOT
    / "RESULTS"
    / "27_Live_Fast_New_Object_Final"
)

DEBUG_DIR = (
    SAVE_DIR
    / "Latest_Medium_Focus_ROIs"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CAMERA
# ============================================================

# Try the normal CREST index first, then alternatives.
CAMERA_INDEX_CANDIDATES = [
    1,
    0,
    2,
    3,
    4,
]

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CAMERA_EXPOSURE = -7.0


# ============================================================
# PALLET DETECTION / TRACKING
# ============================================================

PALLET_CONFIDENCE = 0.10
PALLET_CONFIRM_FRAMES = 5
PALLET_CROP_PADDING = 0.03

# Draw a missing track for only a very short time so we do not create ghosts.
TRACK_DISPLAY_GRACE_FRAMES = 2

# Keep its INTERNAL state longer than we draw it.  This allows the same pallet
# to pass behind the black pole/wire for a short time without losing its
# EMPTY/OBJECT and X/NO_X decision history.  Hidden stale tracks are NOT counted
# and are NOT drawn.
TRACK_STATE_MEMORY_FRAMES = 12

# Only bridge a NEW tracker ID across a genuinely brief disappearance.
# This is deliberately shorter than the internal state memory to reduce
# the chance of copying one pallet's state to a neighboring pallet.
STATE_BRIDGE_MAX_MISSED_FRAMES = 10

# Strict one-to-one state bridge used only when the tracker assigns a new ID
# after a short occlusion.
STATE_BRIDGE_MIN_SIZE_RATIO = 0.80
STATE_BRIDGE_MIN_SMALLER_OVERLAP = 0.20
STATE_BRIDGE_MIN_IOU = 0.08
STATE_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO = 0.45
STATE_BRIDGE_MAX_VERTICAL_CENTER_RATIO = 0.35
STATE_BRIDGE_MIN_SCORE = 1.10
STATE_BRIDGE_UNIQUE_MARGIN = 0.25

DUPLICATE_MIN_SIZE_RATIO = 0.60
DUPLICATE_MIN_VERTICAL_OVERLAP = 0.70
DUPLICATE_MIN_SMALLER_OVERLAP = 0.18

DUPLICATE_MAX_HORIZONTAL_CENTER_RATIO = 0.82
DUPLICATE_MAX_VERTICAL_CENTER_RATIO = 0.35

STALE_TRANSFER_OVERLAP = 0.10
STALE_TRANSFER_IOU = 0.02
STALE_TRANSFER_CENTER_RATIO = 1.25


# ============================================================
# PALLET SOFTWARE BRIGHTENING
# ============================================================

PALLET_GAMMA = 0.60

pallet_clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)


# ============================================================
# EMPTY / OBJECT
# ============================================================

OBJECT_IMAGE_SIZE = 320
OBJECT_MIN_CONFIDENCE = 0.50
OBJECT_HISTORY_FRAMES = 7

# Asymmetric evidence: seeing a real object is positive evidence, while EMPTY
# is allowed to take a little longer because the pole/wire can temporarily hide
# the object and make the crop look empty.
OBJECT_PRESENT_INITIAL_VOTES = 2
EMPTY_INITIAL_VOTES = 5

# If a new/reacquired track was temporarily committed as EMPTY while the pole
# covered the object, allow strong positive object evidence to upgrade it.
# This transition is one-way: OBJECT_PRESENT never downgrades back to EMPTY.
OBJECT_RECOVERY_VOTES = 2
OBJECT_RECOVERY_MIN_CONFIDENCE = 0.75

# FAST NEW-OBJECT DETECTION
# -------------------------
# An EMPTY pallet must not wait for the lower clean-view gate before we notice
# that a real object has been placed on it.  Three consecutive strong
# OBJECT_PRESENT observations are enough to start a fast symbol check anywhere
# the tracked pallet is visible.  This is intentionally stronger than one or two
# frames so a passing hand/wire does not turn an EMPTY pallet into an object.
OBJECT_FAST_ADD_CONSECUTIVE_VOTES = 3
OBJECT_FAST_ADD_MIN_CONFIDENCE = 0.85

# Automatic real-change detection while the conveyor is moving.
# A brief pole/wire occlusion may create one or two EMPTY observations,
# but a physically removed object creates sustained high-confidence EMPTY
# observations.  Require a consecutive streak before changing an already
# established OBJECT_PRESENT pallet to EMPTY.
# Contradictory EMPTY/OBJECT evidence is sampled slowly enough that a
# split-second hand/wire/occlusion cannot erase a known pallet state.
OBJECT_OBSERVATION_MIN_SECONDS = 0.18

# An established OBJECT_PRESENT pallet changes to EMPTY only after sustained
# clean-view evidence.  Five spaced observations require roughly 0.7-0.9 s.
OBJECT_REMOVAL_CONSECUTIVE_VOTES = 5
OBJECT_REMOVAL_MIN_CONFIDENCE = 0.85


# ============================================================
# SYMBOL MODEL / FROZEN REPRESENTATION
# ============================================================

SYMBOL_IMAGE_SIZE = 640
SYMBOL_MIN_CONFIDENCE = 0.65

FOCUS_X1 = 0.15
FOCUS_Y1 = 0.00
FOCUS_X2 = 0.85
FOCUS_Y2 = 0.67


# ============================================================
# LIVE 3-VIEW DECISION
# ============================================================

SYMBOL_REQUIRED_OBSERVATIONS = 3
SYMBOL_REQUIRED_VOTES = 2

# New observation must come from a meaningfully different pallet location.
SYMBOL_SAMPLE_MIN_CENTER_DISTANCE_RATIO = 0.22

# Prevent adjacent frames from becoming separate "views".
SYMBOL_SAMPLE_MIN_SECONDS = 0.20

# No fixed purple rectangle is used in Phase 25.
#
# The conveyor moves CLOCKWISE.  Therefore the clean section immediately after
# the right-side wire/turn is identified dynamically by TRACK MOTION:
#
#   top straight    -> pallet moves mainly RIGHT  -> no symbol voting
#   right turn/wire -> direction/box unstable     -> no symbol voting
#   lower straight  -> pallet moves mainly LEFT   -> symbol voting allowed
#   left turn       -> direction changes          -> no symbol voting
#
# This removes the temporary hard-coded spatial offset.  A camera shift does not
# move the decision logic to the wrong physical place.
SYMBOL_CLEAN_MIN_LEFTWARD_VX = 0.60
SYMBOL_CLEAN_HORIZONTAL_DOMINANCE = 1.25
SYMBOL_CLEAN_MIN_DETECT_CONFIDENCE = 0.20
SYMBOL_CLEAN_MIN_RING_OVERLAP = 0.30

# After the wire or any box instability, require several consecutive stable
# tracked frames before X/NO_X is allowed to vote.
SYMBOL_CLEAN_REQUIRED_STABLE_FRAMES = 5
SYMBOL_CLEAN_RECOVERY_SECONDS = 0.25
SYMBOL_CLEAN_MIN_BOX_IOU = 0.50
SYMBOL_CLEAN_MIN_SIZE_RATIO = 0.82
SYMBOL_CLEAN_MAX_HORIZONTAL_CENTER_RATIO = 0.22
SYMBOL_CLEAN_MAX_VERTICAL_CENTER_RATIO = 0.20

# Initial classification still uses the externally validated 2-of-3 rule.
# Once ACCEPTED/REJECTED exists, a genuine physical X <-> NO_X change requires
# 3/3 opposite CLEAN observations.  Bad wire/turn views cannot rewrite it.
SYMBOL_CHANGE_REQUIRED_OPPOSITE_VOTES = 3

# FAST SYMBOL CHECK AFTER A NEW OBJECT APPEARS
# --------------------------------------------
# The old clean-view-only logic could make a newly placed object wait one or two
# full conveyor laps before collecting three symbol samples.  After a confirmed
# EMPTY -> OBJECT_PRESENT transition, open a short fast window instead.
#
# We wait briefly for the placing hand to move away, then take three
# time-separated / slightly-spaced samples.  A fast decision requires 3/3
# identical votes.  If they disagree, the fast window expires and the normal
# clean-view logic takes over.
FAST_SYMBOL_START_DELAY_SECONDS = 0.35
FAST_SYMBOL_WINDOW_SECONDS = 2.50
FAST_SYMBOL_SAMPLE_MIN_SECONDS = 0.16
FAST_SYMBOL_MIN_CENTER_DISTANCE_RATIO = 0.05
FAST_SYMBOL_REQUIRED_IDENTICAL_VOTES = 3
FAST_SYMBOL_MIN_DETECT_CONFIDENCE = 0.20
FAST_SYMBOL_MIN_RING_OVERLAP = 0.25


# ============================================================
# DISPLAY COLORS
# ============================================================

COLOR_EMPTY = (0, 255, 0)
COLOR_ACCEPTED = (255, 255, 0)
COLOR_REJECTED = (0, 0, 255)
COLOR_CHECKING = (0, 255, 255)
COLOR_UNKNOWN = (255, 255, 255)


# ============================================================
# LOAD DYNAMIC PALLET STAGE
# ============================================================

def load_stage08():
    spec = importlib.util.spec_from_file_location(
        "dynamic_pallet_stage",
        str(STAGE08_FILE),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import tracker stage:\n{STAGE08_FILE}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    module.BOUNDARY_FILE = BOUNDARY_FILE
    module.YOLO_CONFIDENCE = PALLET_CONFIDENCE
    module.YOLO_IMAGE_SIZE = 640
    module.YOLO_NMS_IOU = 0.50
    module.CONFIRM_FRAMES = PALLET_CONFIRM_FRAMES

    return module


# ============================================================
# CAMERA OPEN
# ============================================================

def open_camera():
    for index in CAMERA_INDEX_CANDIDATES:
        print(
            f"Trying camera index {index}..."
        )

        cap = cv2.VideoCapture(
            index,
            cv2.CAP_DSHOW,
        )

        if not cap.isOpened():
            cap.release()

            cap = cv2.VideoCapture(
                index
            )

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            ),
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )

        cap.set(
            cv2.CAP_PROP_FPS,
            CAMERA_FPS,
        )

        cap.set(
            cv2.CAP_PROP_AUTO_EXPOSURE,
            0.25,
        )

        cap.set(
            cv2.CAP_PROP_EXPOSURE,
            CAMERA_EXPOSURE,
        )

        readable = False

        for _ in range(5):
            success, frame = cap.read()

            if success and frame is not None:
                readable = True
                break

        if readable:
            print(
                f"Using camera index {index}."
            )

            return cap, index

        cap.release()

    raise RuntimeError(
        "Could not open any camera index in "
        f"{CAMERA_INDEX_CANDIDATES}."
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def make_pallet_frame(frame):
    table = np.array(
        [
            (
                (
                    value
                    / 255.0
                )
                ** PALLET_GAMMA
            )
            * 255
            for value in range(
                256
            )
        ],
        dtype=np.uint8,
    )

    bright = cv2.LUT(
        frame,
        table,
    )

    lab = cv2.cvtColor(
        bright,
        cv2.COLOR_BGR2LAB,
    )

    (
        l_channel,
        a_channel,
        b_channel,
    ) = cv2.split(
        lab
    )

    l_channel = pallet_clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        (
            l_channel,
            a_channel,
            b_channel,
        )
    )

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )


def make_symbol_frame(frame):
    gamma = 1.15

    table = np.array(
        [
            (
                (
                    value
                    / 255.0
                )
                ** gamma
            )
            * 255
            for value in range(
                256
            )
        ],
        dtype=np.uint8,
    )

    processed = cv2.LUT(
        frame,
        table,
    )

    (
        blue,
        green,
        red,
    ) = cv2.split(
        processed
    )

    blue = np.clip(
        blue.astype(
            np.float32
        )
        * 0.80,
        0,
        255,
    ).astype(
        np.uint8
    )

    green = np.clip(
        green.astype(
            np.float32
        )
        * 1.00,
        0,
        255,
    ).astype(
        np.uint8
    )

    red = np.clip(
        red.astype(
            np.float32
        )
        * 0.90,
        0,
        255,
    ).astype(
        np.uint8
    )

    processed = cv2.merge(
        (
            blue,
            green,
            red,
        )
    )

    blurred = cv2.GaussianBlur(
        processed,
        (0, 0),
        0.7,
    )

    return cv2.addWeighted(
        processed,
        1.30,
        blurred,
        -0.30,
        0,
    )


# ============================================================
# PALLET CROP
# ============================================================

def crop_pallet(
    frame,
    box,
):
    height, width = (
        frame.shape[:2]
    )

    (
        x1,
        y1,
        x2,
        y2,
    ) = box

    box_width = (
        x2 - x1
    )

    box_height = (
        y2 - y1
    )

    pad_x = int(
        box_width
        * PALLET_CROP_PADDING
    )

    pad_y = int(
        box_height
        * PALLET_CROP_PADDING
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

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    crop = frame[
        y1:y2,
        x1:x2,
    ].copy()

    if crop.size == 0:
        return None

    return crop


# ============================================================
# BOX GEOMETRY
# ============================================================

def box_iou(
    box_a,
    box_b,
):
    (
        ax1,
        ay1,
        ax2,
        ay2,
    ) = box_a

    (
        bx1,
        by1,
        bx2,
        by2,
    ) = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    iw = max(
        0,
        ix2 - ix1,
    )

    ih = max(
        0,
        iy2 - iy1,
    )

    intersection = (
        iw * ih
    )

    area_a = max(
        1,
        (
            ax2 - ax1
        )
        * (
            ay2 - ay1
        ),
    )

    area_b = max(
        1,
        (
            bx2 - bx1
        )
        * (
            by2 - by1
        ),
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


def smaller_box_overlap_ratio(
    box_a,
    box_b,
):
    (
        ax1,
        ay1,
        ax2,
        ay2,
    ) = box_a

    (
        bx1,
        by1,
        bx2,
        by2,
    ) = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    iw = max(
        0,
        ix2 - ix1,
    )

    ih = max(
        0,
        iy2 - iy1,
    )

    intersection = (
        iw * ih
    )

    area_a = max(
        1,
        (
            ax2 - ax1
        )
        * (
            ay2 - ay1
        ),
    )

    area_b = max(
        1,
        (
            bx2 - bx1
        )
        * (
            by2 - by1
        ),
    )

    smaller_area = min(
        area_a,
        area_b,
    )

    return (
        intersection
        / smaller_area
    )


def vertical_overlap_ratio(
    box_a,
    box_b,
):
    (
        _,
        ay1,
        _,
        ay2,
    ) = box_a

    (
        _,
        by1,
        _,
        by2,
    ) = box_b

    overlap = max(
        0,
        min(
            ay2,
            by2,
        )
        - max(
            ay1,
            by1,
        ),
    )

    height_a = max(
        1,
        ay2 - ay1,
    )

    height_b = max(
        1,
        by2 - by1,
    )

    return (
        overlap
        / min(
            height_a,
            height_b,
        )
    )


def box_size_ratio(
    box_a,
    box_b,
):
    (
        ax1,
        ay1,
        ax2,
        ay2,
    ) = box_a

    (
        bx1,
        by1,
        bx2,
        by2,
    ) = box_b

    area_a = max(
        1,
        (
            ax2 - ax1
        )
        * (
            ay2 - ay1
        ),
    )

    area_b = max(
        1,
        (
            bx2 - bx1
        )
        * (
            by2 - by1
        ),
    )

    return (
        min(
            area_a,
            area_b,
        )
        / max(
            area_a,
            area_b,
        )
    )


def center_ratios(
    box_a,
    box_b,
):
    (
        ax1,
        ay1,
        ax2,
        ay2,
    ) = box_a

    (
        bx1,
        by1,
        bx2,
        by2,
    ) = box_b

    acx = (
        ax1 + ax2
    ) / 2.0

    acy = (
        ay1 + ay2
    ) / 2.0

    bcx = (
        bx1 + bx2
    ) / 2.0

    bcy = (
        by1 + by2
    ) / 2.0

    avg_width = max(
        1.0,
        (
            (
                ax2 - ax1
            )
            + (
                bx2 - bx1
            )
        )
        / 2.0,
    )

    avg_height = max(
        1.0,
        (
            (
                ay2 - ay1
            )
            + (
                by2 - by1
            )
        )
        / 2.0,
    )

    horizontal_ratio = (
        abs(
            acx - bcx
        )
        / avg_width
    )

    vertical_ratio = (
        abs(
            acy - bcy
        )
        / avg_height
    )

    return (
        horizontal_ratio,
        vertical_ratio,
    )


# ============================================================
# MERGE RAW + BRIGHT PALLET DETECTIONS
# ============================================================

def merge_pallet_detections(
    primary,
    secondary,
):
    merged = [
        dict(
            detection
        )
        for detection
        in primary
    ]

    for candidate in secondary:
        matching_index = None
        matching_iou = 0.0

        for index, existing in enumerate(
            merged
        ):
            overlap = box_iou(
                candidate[
                    "box"
                ],
                existing[
                    "box"
                ],
            )

            if (
                overlap
                > matching_iou
            ):
                matching_iou = (
                    overlap
                )

                matching_index = (
                    index
                )

        if (
            matching_index
            is not None
            and
            matching_iou
            >= 0.45
        ):
            if (
                candidate[
                    "confidence"
                ]
                >
                merged[
                    matching_index
                ][
                    "confidence"
                ]
            ):
                merged[
                    matching_index
                ] = dict(
                    candidate
                )

        else:
            merged.append(
                dict(
                    candidate
                )
            )

    return merged


# ============================================================
# DUPLICATE TRACK FILTER
# ============================================================

def current_boxes_are_duplicate(
    box_a,
    box_b,
):
    size_ratio = (
        box_size_ratio(
            box_a,
            box_b,
        )
    )

    vertical_ratio = (
        vertical_overlap_ratio(
            box_a,
            box_b,
        )
    )

    smaller_overlap = (
        smaller_box_overlap_ratio(
            box_a,
            box_b,
        )
    )

    (
        horizontal_center_ratio,
        vertical_center_ratio,
    ) = center_ratios(
        box_a,
        box_b,
    )

    return (
        size_ratio
        >= DUPLICATE_MIN_SIZE_RATIO

        and

        vertical_ratio
        >= DUPLICATE_MIN_VERTICAL_OVERLAP

        and

        smaller_overlap
        >= DUPLICATE_MIN_SMALLER_OVERLAP

        and

        horizontal_center_ratio
        <= DUPLICATE_MAX_HORIZONTAL_CENTER_RATIO

        and

        vertical_center_ratio
        <= DUPLICATE_MAX_VERTICAL_CENTER_RATIO
    )


def filter_current_duplicate_tracks(
    confirmed_tracks,
    symbol_history,
    track_cache,
):
    ordered_tracks = sorted(
        confirmed_tracks,
        key=lambda track: (
            0
            if (
                track[
                    "track_id"
                ]
                in symbol_history
            )
            else 1,

            0
            if (
                track[
                    "track_id"
                ]
                in track_cache
            )
            else 1,

            track[
                "track_id"
            ],
        ),
    )

    kept_tracks = []
    suppressed_ids = []

    for candidate in ordered_tracks:
        candidate_id = (
            candidate[
                "track_id"
            ]
        )

        candidate_box = (
            candidate[
                "box"
            ]
        )

        duplicate = False

        for kept in kept_tracks:
            if current_boxes_are_duplicate(
                candidate_box,
                kept[
                    "box"
                ],
            ):
                duplicate = True
                break

        if duplicate:
            suppressed_ids.append(
                candidate_id
            )

        else:
            kept_tracks.append(
                candidate
            )

    return (
        kept_tracks,
        suppressed_ids,
    )


# ============================================================
# OLD / NEW TRACK MATCH
# ============================================================

def old_new_tracks_match(
    old_box,
    new_box,
):
    overlap = (
        smaller_box_overlap_ratio(
            old_box,
            new_box,
        )
    )

    iou = box_iou(
        old_box,
        new_box,
    )

    (
        horizontal_ratio,
        vertical_ratio,
    ) = center_ratios(
        old_box,
        new_box,
    )

    combined_center_ratio = (
        (
            horizontal_ratio
            ** 2
        )
        + (
            vertical_ratio
            ** 2
        )
    ) ** 0.5

    return (
        overlap
        >= STALE_TRANSFER_OVERLAP

        and

        iou
        >= STALE_TRANSFER_IOU

        and

        combined_center_ratio
        <= STALE_TRANSFER_CENTER_RATIO
    )


# ============================================================
# SHORT-OCCLUSION STATE BRIDGE
# ============================================================

def state_bridge_score(
    old_box,
    new_box,
):
    """Return a strict geometric continuity score, or None if unsafe."""

    size_ratio = box_size_ratio(
        old_box,
        new_box,
    )

    if (
        size_ratio
        < STATE_BRIDGE_MIN_SIZE_RATIO
    ):
        return None

    smaller_overlap = (
        smaller_box_overlap_ratio(
            old_box,
            new_box,
        )
    )

    iou = box_iou(
        old_box,
        new_box,
    )

    (
        horizontal_ratio,
        vertical_ratio,
    ) = center_ratios(
        old_box,
        new_box,
    )

    if (
        horizontal_ratio
        > STATE_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO
        or
        vertical_ratio
        > STATE_BRIDGE_MAX_VERTICAL_CENTER_RATIO
    ):
        return None

    # Require some real spatial overlap.  This blocks state jumping to the next
    # pallet even if two boxes have similar size.
    if (
        smaller_overlap
        < STATE_BRIDGE_MIN_SMALLER_OVERLAP
        and
        iou
        < STATE_BRIDGE_MIN_IOU
    ):
        return None

    score = (
        1.8 * iou
        + 1.4 * smaller_overlap
        + 0.55 * size_ratio
        - 0.35 * horizontal_ratio
        - 0.55 * vertical_ratio
    )

    return float(score)


def has_meaningful_track_state(
    track_id,
    object_history,
    object_decision,
    symbol_history,
    symbol_decision,
):
    return (
        track_id in object_decision
        or track_id in symbol_decision
        or (
            track_id in object_history
            and len(object_history[track_id]) > 0
        )
        or (
            track_id in symbol_history
            and len(symbol_history[track_id]) > 0
        )
    )


def move_track_state(
    old_track_id,
    new_track_id,
    new_box,
    track_cache,
    object_history,
    object_decision,
    symbol_history,
    symbol_decision,
    symbol_sample_state,
    fast_symbol_start_after,
    fast_symbol_until,
):
    """Move state from a lost tracker ID to a strict reacquired ID."""

    if old_track_id in object_history:
        object_history[new_track_id] = object_history.pop(
            old_track_id
        )

    if old_track_id in object_decision:
        object_decision[new_track_id] = object_decision.pop(
            old_track_id
        )

    if old_track_id in symbol_history:
        symbol_history[new_track_id] = symbol_history.pop(
            old_track_id
        )

    if old_track_id in symbol_decision:
        symbol_decision[new_track_id] = symbol_decision.pop(
            old_track_id
        )

    if old_track_id in symbol_sample_state:
        symbol_sample_state[new_track_id] = symbol_sample_state.pop(
            old_track_id
        )

    if old_track_id in fast_symbol_start_after:
        fast_symbol_start_after[new_track_id] = fast_symbol_start_after.pop(
            old_track_id
        )

    if old_track_id in fast_symbol_until:
        fast_symbol_until[new_track_id] = fast_symbol_until.pop(
            old_track_id
        )

    old_cache = track_cache.pop(
        old_track_id,
        None,
    )

    if old_cache is not None:
        old_cache["box"] = new_box
        old_cache["missed"] = 0
        track_cache[new_track_id] = old_cache


def bridge_reacquired_tracks(
    confirmed_tracks,
    current_track_ids,
    track_cache,
    object_history,
    object_decision,
    symbol_history,
    symbol_decision,
    symbol_sample_state,
    fast_symbol_start_after,
    fast_symbol_until,
):
    """
    Strict one-to-one state transfer after a short tracker-ID break.

    A transfer is allowed only when:
      * old track is currently missing but still inside internal memory,
      * old track has meaningful state,
      * geometry is a strong match,
      * best match is unique enough to avoid neighbor contamination.
    """

    stale_ids = [
        track_id
        for track_id, data
        in track_cache.items()
        if (
            track_id not in current_track_ids
            and data.get("missed", 0) <= STATE_BRIDGE_MAX_MISSED_FRAMES
            and has_meaningful_track_state(
                track_id,
                object_history,
                object_decision,
                symbol_history,
                symbol_decision,
            )
        )
    ]

    if not stale_ids:
        return

    used_old_ids = set()

    for track in confirmed_tracks:
        new_track_id = track["track_id"]
        new_box = track["box"]

        # Existing IDs already own their state; do not remap them.
        if (
            new_track_id in object_history
            or new_track_id in object_decision
            or new_track_id in symbol_history
            or new_track_id in symbol_decision
        ):
            continue

        candidates = []

        for old_track_id in stale_ids:
            if old_track_id in used_old_ids:
                continue

            old_cache = track_cache.get(
                old_track_id
            )

            if old_cache is None:
                continue

            score = state_bridge_score(
                old_cache["box"],
                new_box,
            )

            if score is None:
                continue

            candidates.append(
                (
                    score,
                    old_track_id,
                )
            )

        if not candidates:
            continue

        candidates.sort(
            reverse=True
        )

        best_score, best_old_id = candidates[0]

        if (
            best_score
            < STATE_BRIDGE_MIN_SCORE
        ):
            continue

        if len(candidates) > 1:
            second_score = candidates[1][0]

            if (
                best_score - second_score
                < STATE_BRIDGE_UNIQUE_MARGIN
            ):
                # Ambiguous: safer to re-check than to copy another pallet's state.
                continue

        old_missed = track_cache[
            best_old_id
        ].get(
            "missed",
            0,
        )

        old_object_state = object_decision.get(
            best_old_id
        )

        old_symbol_state = symbol_decision.get(
            best_old_id
        )

        move_track_state(
            best_old_id,
            new_track_id,
            new_box,
            track_cache,
            object_history,
            object_decision,
            symbol_history,
            symbol_decision,
            symbol_sample_state,
            fast_symbol_start_after,
            fast_symbol_until,
        )

        used_old_ids.add(
            best_old_id
        )

        print(
            f"STATE BRIDGE P{best_old_id} -> P{new_track_id} | "
            f"missed={old_missed} | score={best_score:.3f} | "
            f"object={old_object_state} | symbol={old_symbol_state}"
        )


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def normalize_object_class(name):
    text = (
        str(
            name
        )
        .strip()
        .upper()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    if "EMPTY" in text:
        return "EMPTY"

    if (
        "OBJECT" in text
        or "PRESENT" in text
        or "REJECT" in text
    ):
        return "OBJECT_PRESENT"

    return text


def normalize_symbol_class(name):
    text = (
        str(
            name
        )
        .strip()
        .upper()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    if text == "X":
        return "X"

    if (
        text == "NO_X"
        or "NO_X" in text
    ):
        return "NO_X"

    return text


# ============================================================
# EMPTY / OBJECT CLASSIFIER
# ============================================================

def classify_empty_object(
    classifier,
    crop,
):
    result = classifier.predict(
        source=crop,
        imgsz=OBJECT_IMAGE_SIZE,
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

    class_name = (
        normalize_object_class(
            result.names[
                class_id
            ]
        )
    )

    return (
        class_name,
        confidence,
    )


def update_object_decision(
    track_id,
    observed_state,
    observed_confidence,
    object_history,
    object_decision,
    object_sample_state,
    allow_state_change,
):
    """
    Stable EMPTY / OBJECT_PRESENT state machine.

    Key behavior:
      * New pallets may establish their initial state normally.
      * Once a pallet is OBJECT_PRESENT, an unsafe EMPTY observation caused by
        a hand, pole, wire, partial occlusion, or unstable view is IGNORED.
      * Once a pallet is EMPTY, an unsafe OBJECT_PRESENT observation is IGNORED.
      * Real state changes are allowed only in a clean/stable view and require
        spaced, repeated evidence.

    This prevents a temporary obstruction from clearing ACCEPTED/REJECTED and
    forcing the symbol stage to start CHECKING again.
    """

    if track_id not in object_history:
        object_history[track_id] = deque(
            maxlen=OBJECT_HISTORY_FRAMES
        )

    history = object_history[track_id]
    current_decision = object_decision.get(track_id)

    valid_observation = (
        observed_state in ("EMPTY", "OBJECT_PRESENT")
        and observed_confidence >= OBJECT_MIN_CONFIDENCE
    )

    now = time.time()
    previous_sample_time = float(
        object_sample_state.get(track_id, 0.0)
    )

    time_ready = (
        now - previous_sample_time
        >= OBJECT_OBSERVATION_MIN_SECONDS
    )

    should_append = (
        valid_observation
        and time_ready
    )

    # Once a stable state exists, contradictory evidence from an unsafe view
    # is treated as occlusion/noise and is NOT allowed into history.
    if should_append and current_decision == "OBJECT_PRESENT":
        if (
            observed_state == "EMPTY"
            and not allow_state_change
        ):
            should_append = False

    elif should_append and current_decision == "EMPTY":
        if (
            observed_state == "OBJECT_PRESENT"
            and not allow_state_change
            and observed_confidence < OBJECT_FAST_ADD_MIN_CONFIDENCE
        ):
            # Weak contradictory evidence from an unsafe view is ignored.
            # Strong repeated OBJECT_PRESENT evidence is still allowed so a
            # newly placed object can be discovered without waiting a full lap.
            should_append = False

    if should_append:
        history.append(
            (
                observed_state,
                observed_confidence,
            )
        )
        object_sample_state[track_id] = now

    empty_votes = sum(
        1
        for state, _
        in history
        if state == "EMPTY"
    )

    object_votes = sum(
        1
        for state, _
        in history
        if state == "OBJECT_PRESENT"
    )

    current_decision = object_decision.get(track_id)

    if current_decision is None:
        # Initial classification is still asymmetric: positive object evidence
        # can establish OBJECT_PRESENT quickly, while EMPTY needs more support.
        if object_votes >= OBJECT_PRESENT_INITIAL_VOTES:
            object_decision[track_id] = "OBJECT_PRESENT"

        elif empty_votes >= EMPTY_INITIAL_VOTES:
            object_decision[track_id] = "EMPTY"

    elif current_decision == "EMPTY":
        # A real object addition must be detected quickly even if the pallet is
        # not yet in the normal lower-straight clean-view gate.
        #
        # Clean view: keep the existing quick 2-vote recovery rule.
        # Other visible views: require 3 stronger consecutive observations.
        if allow_state_change:
            required_votes = OBJECT_RECOVERY_VOTES
            minimum_confidence = OBJECT_RECOVERY_MIN_CONFIDENCE
            transition_name = "clean"
        else:
            required_votes = OBJECT_FAST_ADD_CONSECUTIVE_VOTES
            minimum_confidence = OBJECT_FAST_ADD_MIN_CONFIDENCE
            transition_name = "fast"

        strong_object_streak = 0

        for state, confidence in reversed(history):
            if (
                state == "OBJECT_PRESENT"
                and confidence >= minimum_confidence
            ):
                strong_object_streak += 1
            else:
                break

        if strong_object_streak >= required_votes:
            object_decision[track_id] = "OBJECT_PRESENT"

            print(
                f"P{track_id} OBJECT CHANGE: "
                f"EMPTY -> OBJECT_PRESENT "
                f"({strong_object_streak} {transition_name} strong object observations)"
            )

    elif current_decision == "OBJECT_PRESENT":
        # Never let a wire/hand/partial occlusion erase a known object state.
        # Only CLEAN, sustained EMPTY evidence may confirm real object removal.
        if allow_state_change:
            strong_empty_streak = 0

            for state, confidence in reversed(history):
                if (
                    state == "EMPTY"
                    and confidence >= OBJECT_REMOVAL_MIN_CONFIDENCE
                ):
                    strong_empty_streak += 1
                else:
                    break

            if (
                strong_empty_streak
                >= OBJECT_REMOVAL_CONSECUTIVE_VOTES
            ):
                object_decision[track_id] = "EMPTY"

                print(
                    f"P{track_id} OBJECT CHANGE: "
                    f"OBJECT_PRESENT -> EMPTY "
                    f"({strong_empty_streak} clean strong EMPTY observations)"
                )

    stable_state = object_decision.get(track_id)

    matching_confidences = [
        confidence
        for state, confidence
        in history
        if state == stable_state
    ]

    stable_confidence = (
        sum(matching_confidences)
        / len(matching_confidences)
        if matching_confidences
        else 0.0
    )

    return (
        stable_state,
        stable_confidence,
        empty_votes,
        object_votes,
    )


# ============================================================
# MEDIUM-FOCUS REPRESENTATION
# ============================================================

def prepare_medium_focus(
    symbol_crop,
):
    if (
        symbol_crop is None
        or symbol_crop.size == 0
    ):
        return None

    height, width = (
        symbol_crop.shape[:2]
    )

    x1 = int(
        round(
            FOCUS_X1
            * width
        )
    )

    y1 = int(
        round(
            FOCUS_Y1
            * height
        )
    )

    x2 = int(
        round(
            FOCUS_X2
            * width
        )
    )

    y2 = int(
        round(
            FOCUS_Y2
            * height
        )
    )

    x1 = max(
        0,
        min(
            width - 1,
            x1,
        ),
    )

    y1 = max(
        0,
        min(
            height - 1,
            y1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            width,
            x2,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            height,
            y2,
        ),
    )

    focused = symbol_crop[
        y1:y2,
        x1:x2,
    ].copy()

    if focused.size == 0:
        return None

    (
        focus_height,
        focus_width,
    ) = focused.shape[:2]

    scale = min(
        SYMBOL_IMAGE_SIZE
        / focus_width,
        SYMBOL_IMAGE_SIZE
        / focus_height,
    )

    new_width = max(
        1,
        int(
            round(
                focus_width
                * scale
            )
        ),
    )

    new_height = max(
        1,
        int(
            round(
                focus_height
                * scale
            )
        ),
    )

    resized = cv2.resize(
        focused,
        (
            new_width,
            new_height,
        ),
        interpolation=(
            cv2.INTER_LINEAR
        ),
    )

    mean_bgr = np.mean(
        focused.reshape(
            -1,
            3,
        ),
        axis=0,
    )

    fill = tuple(
        int(
            round(
                value
            )
        )
        for value in mean_bgr
    )

    canvas = np.empty(
        (
            SYMBOL_IMAGE_SIZE,
            SYMBOL_IMAGE_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[:, :] = fill

    left = (
        SYMBOL_IMAGE_SIZE
        - new_width
    ) // 2

    top = (
        SYMBOL_IMAGE_SIZE
        - new_height
    ) // 2

    canvas[
        top:
        top + new_height,
        left:
        left + new_width,
    ] = resized

    return canvas


# ============================================================
# MEDIUM-FOCUS SYMBOL CLASSIFIER
# ============================================================

def classify_medium_focus_symbol(
    verifier,
    prepared_crop,
):
    result = verifier.predict(
        source=prepared_crop,
        imgsz=SYMBOL_IMAGE_SIZE,
        device=0,
        verbose=False,
    )[0]

    if result.probs is None:
        return (
            "UNKNOWN",
            0.0,
            0.0,
            0.0,
        )

    probabilities = (
        result.probs
        .data
        .detach()
        .cpu()
        .tolist()
    )

    probability_map = {
        normalize_symbol_class(
            result.names[
                index
            ]
        ): float(
            probability
        )
        for index, probability
        in enumerate(
            probabilities
        )
    }

    p_x = (
        probability_map.get(
            "X",
            0.0,
        )
    )

    p_no_x = (
        probability_map.get(
            "NO_X",
            0.0,
        )
    )

    if p_x >= p_no_x:
        return (
            "X",
            p_x,
            p_x,
            p_no_x,
        )

    return (
        "NO_X",
        p_no_x,
        p_x,
        p_no_x,
    )


# ============================================================
# SPATIALLY SEPARATED OBSERVATIONS
# ============================================================

def box_center_and_diagonal(
    box,
):
    (
        x1,
        y1,
        x2,
        y2,
    ) = box

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    width = max(
        1.0,
        x2 - x1,
    )

    height = max(
        1.0,
        y2 - y1,
    )

    diagonal = float(
        np.hypot(
            width,
            height,
        )
    )

    return (
        center_x,
        center_y,
        diagonal,
    )


def is_left_turn_hold_zone(
    box,
    frame_width,
):
    """Return True when a pallet is in the known unstable left-hand turn."""

    x1, _, x2, _ = box

    center_x = (
        x1 + x2
    ) / 2.0

    if frame_width <= 0:
        return False

    center_ratio = (
        center_x
        / float(frame_width)
    )

    return (
        center_ratio
        <= LEFT_TURN_HOLD_X_RATIO
    )


def update_dynamic_symbol_gate(
    track_id,
    track,
    symbol_quality_state,
    manual_recheck=False,
):
    """
    Decide whether the CURRENT tracked view is safe for X/NO_X voting.

    No fixed image rectangle is used.  During normal clockwise motion a trusted
    symbol view must be:
      * moving left (the lower straight after the right-side wire),
      * moving mostly horizontally (not a turn),
      * detected with reasonable confidence/ring overlap,
      * geometrically stable for several consecutive frames.

    Any unstable/wire/turn view resets the stability counter and starts a short
    recovery timer.
    """

    now = time.time()
    box = track["box"]

    if manual_recheck:
        symbol_quality_state[track_id] = {
            "box": box,
            "stable_frames": SYMBOL_CLEAN_REQUIRED_STABLE_FRAMES,
            "recover_until": 0.0,
        }
        return True, "MANUAL"

    velocity_x = float(track.get("velocity_x", 0.0))
    velocity_y = float(track.get("velocity_y", 0.0))
    confidence = float(track.get("confidence", 0.0))
    ring_overlap = float(track.get("overlap", 0.0))

    moving_left = (
        velocity_x
        <= -SYMBOL_CLEAN_MIN_LEFTWARD_VX
    )

    mostly_horizontal = (
        abs(velocity_x)
        >= (
            SYMBOL_CLEAN_HORIZONTAL_DOMINANCE
            * max(abs(velocity_y), 0.25)
        )
    )

    detector_good = (
        confidence
        >= SYMBOL_CLEAN_MIN_DETECT_CONFIDENCE
        and ring_overlap
        >= SYMBOL_CLEAN_MIN_RING_OVERLAP
    )

    previous = symbol_quality_state.get(track_id)

    if previous is None:
        symbol_quality_state[track_id] = {
            "box": box,
            "stable_frames": 1 if (moving_left and mostly_horizontal and detector_good) else 0,
            "recover_until": now + SYMBOL_CLEAN_RECOVERY_SECONDS,
        }
        return False, "WARMUP"

    previous_box = previous.get("box", box)

    current_iou = box_iou(
        previous_box,
        box,
    )

    current_size_ratio = box_size_ratio(
        previous_box,
        box,
    )

    (
        horizontal_center_ratio,
        vertical_center_ratio,
    ) = center_ratios(
        previous_box,
        box,
    )

    geometry_stable = (
        current_iou
        >= SYMBOL_CLEAN_MIN_BOX_IOU
        and current_size_ratio
        >= SYMBOL_CLEAN_MIN_SIZE_RATIO
        and horizontal_center_ratio
        <= SYMBOL_CLEAN_MAX_HORIZONTAL_CENTER_RATIO
        and vertical_center_ratio
        <= SYMBOL_CLEAN_MAX_VERTICAL_CENTER_RATIO
    )

    clean_now = (
        moving_left
        and mostly_horizontal
        and detector_good
        and geometry_stable
    )

    if clean_now:
        stable_frames = int(
            previous.get(
                "stable_frames",
                0,
            )
        ) + 1
        recover_until = float(
            previous.get(
                "recover_until",
                0.0,
            )
        )
    else:
        stable_frames = 0
        recover_until = (
            now
            + SYMBOL_CLEAN_RECOVERY_SECONDS
        )

    symbol_quality_state[track_id] = {
        "box": box,
        "stable_frames": stable_frames,
        "recover_until": recover_until,
    }

    if not moving_left:
        return False, "HOLD-DIRECTION"

    if not mostly_horizontal:
        return False, "HOLD-TURN"

    if not detector_good:
        return False, "HOLD-DETECT"

    if not geometry_stable:
        return False, "HOLD-RECOVER"

    if now < recover_until:
        return False, "HOLD-RECOVER"

    if (
        stable_frames
        < SYMBOL_CLEAN_REQUIRED_STABLE_FRAMES
    ):
        return False, (
            f"STABILIZE-{stable_frames}/"
            f"{SYMBOL_CLEAN_REQUIRED_STABLE_FRAMES}"
        )

    return True, "CLEAN"


def should_take_symbol_sample(
    track_id,
    box,
    symbol_sample_state,
    manual_recheck=False,
    fast_mode=False,
):
    now = time.time()

    (
        center_x,
        center_y,
        diagonal,
    ) = box_center_and_diagonal(
        box
    )

    previous = (
        symbol_sample_state.get(
            track_id
        )
    )

    if previous is None:
        return (
            True,
            now,
            center_x,
            center_y,
            diagonal,
        )

    elapsed = (
        now
        - previous[
            "time"
        ]
    )

    center_distance = float(
        np.hypot(
            center_x
            - previous[
                "center_x"
            ],
            center_y
            - previous[
                "center_y"
            ],
        )
    )

    reference_diagonal = max(
        1.0,
        (
            diagonal
            + previous[
                "diagonal"
            ]
        )
        / 2.0,
    )

    distance_ratio = (
        center_distance
        / reference_diagonal
    )

    if manual_recheck:
        # During an intentional stopped-belt manual test, allow three
        # time-separated observations even though the pallet is stationary.
        allowed = (
            elapsed
            >= SYMBOL_SAMPLE_MIN_SECONDS
        )

    elif fast_mode:
        # Newly added object: do not wait for the large 0.22-diagonal spacing
        # used by the normal lap-based verifier.  A small amount of conveyor
        # motion plus time separation is enough to obtain three quick fresh
        # views in well under one lap.
        allowed = (
            elapsed
            >= FAST_SYMBOL_SAMPLE_MIN_SECONDS

            and

            distance_ratio
            >= FAST_SYMBOL_MIN_CENTER_DISTANCE_RATIO
        )

    else:
        allowed = (
            elapsed
            >= SYMBOL_SAMPLE_MIN_SECONDS

            and

            distance_ratio
            >= SYMBOL_SAMPLE_MIN_CENTER_DISTANCE_RATIO
        )

    return (
        allowed,
        now,
        center_x,
        center_y,
        diagonal,
    )


def commit_sample_position(
    track_id,
    sample_time,
    center_x,
    center_y,
    diagonal,
    symbol_sample_state,
):
    symbol_sample_state[
        track_id
    ] = {
        "time": sample_time,
        "center_x": (
            center_x
        ),
        "center_y": (
            center_y
        ),
        "diagonal": (
            diagonal
        ),
    }


# ============================================================
# SYMBOL DECISION STATE
# ============================================================

def clear_symbol_state(
    track_id,
    symbol_history,
    symbol_decision,
    symbol_sample_state,
):
    symbol_history.pop(
        track_id,
        None,
    )

    symbol_decision.pop(
        track_id,
        None,
    )

    symbol_sample_state.pop(
        track_id,
        None,
    )


def update_three_view_decision(
    track_id,
    symbol_history,
    symbol_decision,
    fast_mode=False,
):
    history = (
        symbol_history.get(
            track_id,
            deque(),
        )
    )

    observations = len(
        history
    )

    x_votes = sum(
        1
        for state, _
        in history
        if state == "X"
    )

    no_x_votes = sum(
        1
        for state, _
        in history
        if state == "NO_X"
    )

    current_decision = (
        symbol_decision.get(
            track_id
        )
    )

    # --------------------------------------------------------
    # INITIAL DECISION
    # --------------------------------------------------------
    # Keep the externally validated 2-of-3 rule for a pallet that has not yet
    # reached a terminal decision.
    if (
        current_decision is None
        and observations
        >= SYMBOL_REQUIRED_OBSERVATIONS
    ):
        if fast_mode:
            # Fast mode trades spatial separation for speed, so demand complete
            # agreement before showing a terminal result.
            if (
                x_votes
                >= FAST_SYMBOL_REQUIRED_IDENTICAL_VOTES
            ):
                symbol_decision[
                    track_id
                ] = "REJECTED"

            elif (
                no_x_votes
                >= FAST_SYMBOL_REQUIRED_IDENTICAL_VOTES
            ):
                symbol_decision[
                    track_id
                ] = "ACCEPTED"

        else:
            if (
                x_votes
                >= SYMBOL_REQUIRED_VOTES
            ):
                symbol_decision[
                    track_id
                ] = "REJECTED"

            elif (
                no_x_votes
                >= SYMBOL_REQUIRED_VOTES
            ):
                symbol_decision[
                    track_id
                ] = "ACCEPTED"

    # --------------------------------------------------------
    # TERMINAL-STATE HYSTERESIS
    # --------------------------------------------------------
    # Once ACCEPTED/REJECTED exists, do NOT switch on a normal 2-of-3 vote.
    # A real physical symbol change must produce three consecutive reliable
    # opposite observations. This prevents left-turn or partial-occlusion
    # flicker while still allowing a genuine X <-> NO_X replacement to update.
    elif (
        current_decision == "ACCEPTED"
        and observations
        >= SYMBOL_REQUIRED_OBSERVATIONS
        and x_votes
        >= SYMBOL_CHANGE_REQUIRED_OPPOSITE_VOTES
    ):
        symbol_decision[
            track_id
        ] = "REJECTED"

        print(
            f"P{track_id} SYMBOL CHANGE CONFIRMED: "
            f"ACCEPTED -> REJECTED | "
            f"reliable votes X={x_votes} NO_X={no_x_votes}"
        )

    elif (
        current_decision == "REJECTED"
        and observations
        >= SYMBOL_REQUIRED_OBSERVATIONS
        and no_x_votes
        >= SYMBOL_CHANGE_REQUIRED_OPPOSITE_VOTES
    ):
        symbol_decision[
            track_id
        ] = "ACCEPTED"

        print(
            f"P{track_id} SYMBOL CHANGE CONFIRMED: "
            f"REJECTED -> ACCEPTED | "
            f"reliable votes X={x_votes} NO_X={no_x_votes}"
        )

    return (
        symbol_decision.get(
            track_id
        ),
        x_votes,
        no_x_votes,
        observations,
    )


# ============================================================
# DRAW PALLET
# ============================================================

def draw_pallet(
    output,
    box,
    label,
    color,
):
    (
        x1,
        y1,
        x2,
        y2,
    ) = box

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
        color,
        4,
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
        0.65,
        color,
        2,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    required_files = [
        STAGE08_FILE,
        PALLET_MODEL_PATH,
        OBJECT_MODEL_PATH,
        SYMBOL_MODEL_PATH,
        BOUNDARY_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    print()
    print(
        "=" * 76
    )
    print(
        "CREST LIVE FINAL-HARD YOLO26 CANDIDATE"
    )
    print(
        "=" * 76
    )
    print()
    print(
        "Frozen external validation:"
    )
    print(
        "  image level: 58/60 = 96.67%"
    )
    print(
        "  physical-pass majority: 20/20 = 100.00%"
    )
    print()
    print(
        "Live symbol rule:"
    )
    print(
        "  3 spatially separated observations"
    )
    print(
        "  2/3 majority"
    )
    print(
        "  occlusion-hold clean-view voting; no fixed decision rectangle"
    )
    print(
        "  obstruction/turn = HOLD previous state; only stable clean views can change it"
    )
    print()

    pallet_stage = (
        load_stage08()
    )

    print(
        "Loading pallet detector..."
    )

    pallet_model = YOLO(
        str(
            PALLET_MODEL_PATH
        )
    )

    print(
        "Loading EMPTY / OBJECT_PRESENT classifier..."
    )

    object_classifier = YOLO(
        str(
            OBJECT_MODEL_PATH
        )
    )

    print(
        "Loading frozen medium-focus X / NO_X classifier..."
    )

    symbol_verifier = YOLO(
        str(
            SYMBOL_MODEL_PATH
        )
    )

    print(
        "All models loaded."
    )

    boundary = (
        pallet_stage.load_boundary()
    )

    if boundary is None:
        raise RuntimeError(
            "Could not load conveyor boundary."
        )

    tracker = (
        pallet_stage.DynamicTracker()
    )

    object_history = {}
    object_decision = {}
    object_sample_state = {}

    symbol_history = {}
    symbol_decision = {}
    symbol_sample_state = {}

    # Dynamic clean-view state.  No hard-coded decision rectangle is used.
    # Each track must recover from wire/turn instability and move left on the
    # clean lower straight before X/NO_X votes are accepted.
    symbol_quality_state = {}

    # Short event-driven symbol verification window used when an EMPTY pallet
    # receives a newly placed object.  This avoids waiting one/two complete laps.
    fast_symbol_start_after = {}
    fast_symbol_until = {}

    track_cache = {}

    # Optional stopped-belt test helper: R forces a semantic recheck when you
    # physically change a stationary pallet.  Moving pallets now adapt
    # automatically, so R is not needed during normal conveyor operation.
    manual_recheck_until = 0.0

    cap, camera_index = (
        open_camera()
    )

    print(
        f"Live camera index: {camera_index}"
    )

    for _ in range(
        15
    ):
        cap.read()

    window_name = (
        "CREST - FAST NEW-OBJECT + OCCLUSION-HOLD LIVE"
    )

    while True:
        success, frame = (
            cap.read()
        )

        if not success:
            print(
                "Camera frame failed."
            )
            break

        pallet_frame = (
            make_pallet_frame(
                frame
            )
        )

        symbol_frame = (
            make_symbol_frame(
                frame
            )
        )

        (
            ring_mask,
            outer_polygon,
            inner_polygon,
        ) = (
            pallet_stage.create_ring_mask(
                pallet_frame.shape,
                boundary,
            )
        )

        raw_view_detections = (
            pallet_stage.get_yolo_detections(
                pallet_model,
                frame,
            )
        )

        bright_view_detections = (
            pallet_stage.get_yolo_detections(
                pallet_model,
                pallet_frame,
            )
        )

        raw_detections = (
            merge_pallet_detections(
                raw_view_detections,
                bright_view_detections,
            )
        )

        (
            valid_detections,
            rejected_detections,
        ) = (
            pallet_stage.filter_by_ring(
                raw_detections,
                ring_mask,
            )
        )

        confirmed_tracks = (
            tracker.update(
                valid_detections
            )
        )

        (
            confirmed_tracks,
            suppressed_track_ids,
        ) = (
            filter_current_duplicate_tracks(
                confirmed_tracks,
                symbol_history,
                track_cache,
            )
        )

        for suppressed_id in (
            suppressed_track_ids
        ):
            track_cache.pop(
                suppressed_id,
                None,
            )

            clear_symbol_state(
                suppressed_id,
                symbol_history,
                symbol_decision,
                symbol_sample_state,
            )

            symbol_quality_state.pop(
                suppressed_id,
                None,
            )

            fast_symbol_start_after.pop(
                suppressed_id,
                None,
            )

            fast_symbol_until.pop(
                suppressed_id,
                None,
            )

            object_history.pop(
                suppressed_id,
                None,
            )

            object_decision.pop(
                suppressed_id,
                None,
            )

            object_sample_state.pop(
                suppressed_id,
                None,
            )

        current_track_ids = {
            track[
                "track_id"
            ]
            for track
            in confirmed_tracks
        }

        current_track_boxes = [
            track[
                "box"
            ]
            for track
            in confirmed_tracks
        ]

        for track_id in list(
            track_cache.keys()
        ):
            track_cache[
                track_id
            ][
                "missed"
            ] += 1

        # ----------------------------------------------------
        # SHORT OCCLUSION / TRACK-ID CONTINUITY
        # ----------------------------------------------------
        # Keep the old state internally while it is hidden, then move it only
        # to a strong, unique geometric reacquisition.  We never draw or count
        # the hidden stale track, so this does not recreate the old ghost-pallet
        # problem.
        bridge_reacquired_tracks(
            confirmed_tracks,
            current_track_ids,
            track_cache,
            object_history,
            object_decision,
            symbol_history,
            symbol_decision,
            symbol_sample_state,
            fast_symbol_start_after,
            fast_symbol_until,
        )

        output = (
            symbol_frame.copy()
        )

        cv2.polylines(
            output,
            [
                outer_polygon
            ],
            True,
            (
                0,
                255,
                0,
            ),
            3,
        )

        cv2.polylines(
            output,
            [
                inner_polygon
            ],
            True,
            (
                0,
                255,
                255,
            ),
            3,
        )

        # No purple trusted-zone rectangle is drawn in Phase 25.
        # Symbol voting is enabled dynamically from track motion + box stability.

        # ====================================================
        # PROCESS CURRENT TRACKS
        # ====================================================

        for track in confirmed_tracks:
            track_id = (
                track[
                    "track_id"
                ]
            )

            box = (
                track[
                    "box"
                ]
            )

            object_crop = (
                crop_pallet(
                    pallet_frame,
                    box,
                )
            )

            if object_crop is None:
                continue

            (
                observed_object_state,
                observed_object_confidence,
            ) = (
                classify_empty_object(
                    object_classifier,
                    object_crop,
                )
            )

            # ---------------------------------------------------------
            # CLEAN-VIEW GATE IS ALSO USED FOR OBJECT-STATE CHANGES.
            #
            # Temporary hand/wire/pole/turn interference is allowed to affect
            # the raw classifier output, but it is NOT allowed to erase an
            # already-established EMPTY/OBJECT_PRESENT state.
            # ---------------------------------------------------------
            manual_recheck = (
                time.time()
                < manual_recheck_until
            )

            (
                clean_view_ready,
                gate_reason,
            ) = update_dynamic_symbol_gate(
                track_id,
                track,
                symbol_quality_state,
                manual_recheck=manual_recheck,
            )

            previous_object_state = object_decision.get(
                track_id
            )

            (
                object_state,
                object_confidence,
                empty_votes,
                object_votes,
            ) = (
                update_object_decision(
                    track_id,
                    observed_object_state,
                    observed_object_confidence,
                    object_history,
                    object_decision,
                    object_sample_state,
                    allow_state_change=(
                        clean_view_ready
                        or manual_recheck
                    ),
                )
            )

            # ---------------------------------------------------------
            # FAST NEW-OBJECT EVENT
            # ---------------------------------------------------------
            # Start a short verification burst whenever this track changes from
            # EMPTY/unknown to OBJECT_PRESENT.  The burst is event-driven and
            # does not wait for a complete conveyor lap or the lower clean zone.
            if (
                object_state == "OBJECT_PRESENT"
                and previous_object_state != "OBJECT_PRESENT"
            ):
                clear_symbol_state(
                    track_id,
                    symbol_history,
                    symbol_decision,
                    symbol_sample_state,
                )

                symbol_quality_state.pop(
                    track_id,
                    None,
                )

                event_time = time.time()

                fast_symbol_start_after[
                    track_id
                ] = (
                    event_time
                    + FAST_SYMBOL_START_DELAY_SECONDS
                )

                fast_symbol_until[
                    track_id
                ] = (
                    event_time
                    + FAST_SYMBOL_WINDOW_SECONDS
                )

                print(
                    f"P{track_id} FAST NEW-OBJECT VERIFY STARTED | "
                    f"wait={FAST_SYMBOL_START_DELAY_SECONDS:.2f}s | "
                    f"window={FAST_SYMBOL_WINDOW_SECONDS:.2f}s"
                )

            # =================================================
            # EMPTY
            # =================================================

            if (
                object_state
                == "EMPTY"
            ):
                clear_symbol_state(
                    track_id,
                    symbol_history,
                    symbol_decision,
                    symbol_sample_state,
                )

                symbol_quality_state.pop(
                    track_id,
                    None,
                )

                fast_symbol_start_after.pop(
                    track_id,
                    None,
                )

                fast_symbol_until.pop(
                    track_id,
                    None,
                )

                color = (
                    COLOR_EMPTY
                )

                label = (
                    f"P{track_id} "
                    f"EMPTY "
                    f"{object_confidence:.2f}"
                )

            # =================================================
            # OBJECT PRESENT
            # =================================================

            elif (
                object_state
                == "OBJECT_PRESENT"
            ):
                if (
                    track_id
                    not in symbol_history
                ):
                    symbol_history[
                        track_id
                    ] = deque(
                        maxlen=(
                            SYMBOL_REQUIRED_OBSERVATIONS
                        )
                    )

                current_decision = (
                    symbol_decision.get(
                        track_id
                    )
                )

                # -----------------------------------------------------
                # FAST NEW-OBJECT VERIFY OR NORMAL CLEAN-VIEW VERIFY
                # -----------------------------------------------------
                now_for_symbol = time.time()

                fast_start = fast_symbol_start_after.get(
                    track_id,
                    0.0,
                )

                fast_until = fast_symbol_until.get(
                    track_id,
                    0.0,
                )

                fast_pending = (
                    fast_until > 0.0
                    and now_for_symbol < fast_start
                )

                fast_mode = (
                    fast_until > 0.0
                    and fast_start <= now_for_symbol < fast_until
                )

                fast_detector_good = (
                    float(track.get("confidence", 0.0))
                    >= FAST_SYMBOL_MIN_DETECT_CONFIDENCE
                    and float(track.get("overlap", 0.0))
                    >= FAST_SYMBOL_MIN_RING_OVERLAP
                )

                if (
                    fast_mode
                    and fast_detector_good
                ):
                    (
                        allowed,
                        sample_time,
                        center_x,
                        center_y,
                        diagonal,
                    ) = should_take_symbol_sample(
                        track_id,
                        box,
                        symbol_sample_state,
                        manual_recheck=False,
                        fast_mode=True,
                    )

                elif clean_view_ready:
                    (
                        allowed,
                        sample_time,
                        center_x,
                        center_y,
                        diagonal,
                    ) = should_take_symbol_sample(
                        track_id,
                        box,
                        symbol_sample_state,
                        manual_recheck=manual_recheck,
                        fast_mode=False,
                    )

                else:
                    allowed = False
                    sample_time = time.time()
                    (
                        center_x,
                        center_y,
                        diagonal,
                    ) = box_center_and_diagonal(
                        box
                    )

                if allowed:
                    full_symbol_crop = (
                        crop_pallet(
                            symbol_frame,
                            box,
                        )
                    )

                    if (
                        full_symbol_crop
                        is not None
                    ):
                        prepared_symbol_crop = (
                            prepare_medium_focus(
                                full_symbol_crop
                            )
                        )

                        if (
                            prepared_symbol_crop
                            is not None
                        ):
                            (
                                symbol_state,
                                symbol_confidence,
                                p_x,
                                p_no_x,
                            ) = (
                                classify_medium_focus_symbol(
                                    symbol_verifier,
                                    prepared_symbol_crop,
                                )
                            )

                            observation_number = (
                                len(
                                    symbol_history[
                                        track_id
                                    ]
                                )
                                + 1
                            )

                            cv2.imwrite(
                                str(
                                    DEBUG_DIR
                                    / (
                                        f"P{track_id}"
                                        f"_obs{observation_number}"
                                        f"_latest.jpg"
                                    )
                                ),
                                prepared_symbol_crop,
                            )

                            if (
                                symbol_state
                                in (
                                    "X",
                                    "NO_X",
                                )
                                and
                                symbol_confidence
                                >= SYMBOL_MIN_CONFIDENCE
                            ):
                                symbol_history[
                                    track_id
                                ].append(
                                    (
                                        symbol_state,
                                        symbol_confidence,
                                    )
                                )

                                commit_sample_position(
                                    track_id,
                                    sample_time,
                                    center_x,
                                    center_y,
                                    diagonal,
                                    symbol_sample_state,
                                )

                                print(
                                    f"P{track_id} "
                                    f"symbol sample "
                                    f"{len(symbol_history[track_id])}/3: "
                                    f"{symbol_state} "
                                    f"conf={symbol_confidence:.3f} "
                                    f"P(X)={p_x:.3f} "
                                    f"P(NO_X)={p_no_x:.3f}"
                                )

                            else:
                                print(
                                    f"P{track_id} "
                                    f"symbol sample ignored: "
                                    f"{symbol_state} "
                                    f"conf={symbol_confidence:.3f}"
                                )

                (
                    current_decision,
                    x_votes,
                    no_x_votes,
                    observations,
                ) = (
                    update_three_view_decision(
                        track_id,
                        symbol_history,
                        symbol_decision,
                        fast_mode=fast_mode,
                    )
                )

                # Once the fast burst has produced a terminal answer, stop the
                # burst immediately.  Future changes use the safer normal
                # clean-view hysteresis.
                if (
                    current_decision
                    in ("ACCEPTED", "REJECTED")
                    and fast_until > 0.0
                ):
                    fast_symbol_start_after.pop(
                        track_id,
                        None,
                    )
                    fast_symbol_until.pop(
                        track_id,
                        None,
                    )
                    fast_mode = False
                    fast_pending = False

                # ---------------------------------------------------------
                # DISPLAY
                # ---------------------------------------------------------
                # Keep an established terminal result stable whenever the view
                # is unsafe.  A terminal result can change only after 3/3
                # opposite CLEAN observations.  A new/unclassified pallet shows
                # CHECKING until enough clean evidence is collected.
                if (
                    current_decision
                    == "REJECTED"
                ):
                    color = (
                        COLOR_REJECTED
                    )

                    label = (
                        f"P{track_id} "
                        f"REJECTED "
                        f"X:{x_votes}/"
                        f"{observations}"
                        + (
                            " HOLD"
                            if not clean_view_ready
                            else ""
                        )
                    )

                elif (
                    current_decision
                    == "ACCEPTED"
                ):
                    color = (
                        COLOR_ACCEPTED
                    )

                    label = (
                        f"P{track_id} "
                        f"ACCEPTED "
                        f"NOX:{no_x_votes}/"
                        f"{observations}"
                        + (
                            " HOLD"
                            if not clean_view_ready
                            else ""
                        )
                    )

                else:
                    color = (
                        COLOR_CHECKING
                    )

                    if fast_pending:
                        mode_text = "FAST-WAIT"
                    elif fast_mode:
                        mode_text = "FAST-CHECK"
                    else:
                        mode_text = gate_reason

                    label = (
                        f"P{track_id} "
                        f"CHECKING "
                        f"X:{x_votes} "
                        f"NOX:{no_x_votes} "
                        f"{observations}/3 "
                        f"{mode_text}"
                    )

            # =================================================
            # UNKNOWN
            # =================================================

            else:
                color = (
                    COLOR_UNKNOWN
                )

                label = (
                    f"P{track_id} "
                    f"UNKNOWN "
                    f"{object_confidence:.2f}"
                )

            track_cache[
                track_id
            ] = {
                "box": box,
                "missed": 0,
                "label": label,
                "color": color,
            }

        # ====================================================
        # OCCLUSION MEMORY CLEANUP
        # ====================================================
        # Do not immediately delete a missing track simply because another ID
        # appears nearby.  The strict state bridge above handles safe one-to-one
        # reacquisition.  Missing tracks remain INTERNAL only until the memory
        # timeout below.

        # ====================================================
        # INTERNAL STATE MEMORY VS DISPLAY GRACE
        # ====================================================
        # A track can remain internally remembered for a short occlusion, but
        # after TRACK_DISPLAY_GRACE_FRAMES it is NOT drawn and NOT counted.
        # This preserves continuity without ghost pallets.

        for track_id in list(
            track_cache.keys()
        ):
            if (
                track_cache[
                    track_id
                ][
                    "missed"
                ]
                >
                TRACK_STATE_MEMORY_FRAMES
            ):
                del track_cache[
                    track_id
                ]

                clear_symbol_state(
                    track_id,
                    symbol_history,
                    symbol_decision,
                    symbol_sample_state,
                )

                symbol_quality_state.pop(
                    track_id,
                    None,
                )

                fast_symbol_start_after.pop(
                    track_id,
                    None,
                )

                fast_symbol_until.pop(
                    track_id,
                    None,
                )

                object_history.pop(
                    track_id,
                    None,
                )

                object_decision.pop(
                    track_id,
                    None,
                )

                object_sample_state.pop(
                    track_id,
                    None,
                )

        visible_track_cache = {
            track_id: data
            for track_id, data
            in track_cache.items()
            if (
                data.get(
                    "missed",
                    0,
                )
                <= TRACK_DISPLAY_GRACE_FRAMES
            )
        }

        # ====================================================
        # COUNTS
        # ====================================================

        empty_count = 0
        accepted_count = 0
        rejected_count = 0
        checking_count = 0
        unknown_count = 0

        for data in (
            visible_track_cache.values()
        ):
            label = (
                data[
                    "label"
                ]
            )

            if "EMPTY" in label:
                empty_count += 1

            elif "ACCEPTED" in label:
                accepted_count += 1

            elif "REJECTED" in label:
                rejected_count += 1

            elif "CHECKING" in label:
                checking_count += 1

            else:
                unknown_count += 1

        # ====================================================
        # DRAW
        # ====================================================

        for (
            track_id,
            data,
        ) in visible_track_cache.items():
            draw_pallet(
                output,
                data[
                    "box"
                ],
                data[
                    "label"
                ],
                data[
                    "color"
                ],
            )

        cv2.rectangle(
            output,
            (
                10,
                10,
            ),
            (
                830,
                278,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )

        status_lines = [
            (
                f"PALLETS: "
                f"{len(visible_track_cache)}",
                COLOR_UNKNOWN,
            ),
            (
                f"EMPTY: "
                f"{empty_count}",
                COLOR_EMPTY,
            ),
            (
                f"ACCEPTED: "
                f"{accepted_count}",
                COLOR_ACCEPTED,
            ),
            (
                f"REJECTED: "
                f"{rejected_count}",
                COLOR_REJECTED,
            ),
            (
                f"CHECKING: "
                f"{checking_count}",
                COLOR_CHECKING,
            ),
            (
                f"UNKNOWN: "
                f"{unknown_count}",
                COLOR_UNKNOWN,
            ),
            (
                "FAST NEW-OBJECT VERIFY | OCCLUSION HOLD | R=MANUAL RECHECK",
                COLOR_UNKNOWN,
            ),
        ]

        y = 40

        for (
            text,
            color,
        ) in status_lines:
            cv2.putText(
                output,
                text,
                (
                    25,
                    y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
            )

            y += 34

        cv2.imshow(
            window_name,
            output,
        )

        key = (
            cv2.waitKey(
                1
            )
            & 0xFF
        )

        if (
            key
            == ord(
                "q"
            )
            or key == 27
        ):
            break

        if key == ord(
            "r"
        ):
            # ------------------------------------------------
            # MANUAL TEST RESET
            # ------------------------------------------------
            # Use this ONLY when the belt is stopped and you intentionally
            # remove/change the object or swap X <-> NO_X on a pallet.
            # The production latch is intentionally preserved unless R is
            # pressed, because a real pallet should not physically change
            # after it has already been classified.
            object_history.clear()
            object_decision.clear()
            object_sample_state.clear()
            symbol_history.clear()
            symbol_decision.clear()
            symbol_sample_state.clear()
            symbol_quality_state.clear()
            fast_symbol_start_after.clear()
            fast_symbol_until.clear()

            manual_recheck_until = (
                time.time()
                + 2.0
            )

            for reset_track_id, reset_data in track_cache.items():
                if reset_data.get(
                    "missed",
                    0,
                ) <= TRACK_DISPLAY_GRACE_FRAMES:
                    reset_data[
                        "label"
                    ] = (
                        f"P{reset_track_id} RECHECKING"
                    )
                    reset_data[
                        "color"
                    ] = COLOR_CHECKING

            print()
            print(
                "MANUAL RECHECK: cleared current semantic states. "
                "Stationary pallets will be re-evaluated for 2 seconds."
            )
            print()

        if key == ord(
            "s"
        ):
            timestamp = (
                time.strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            save_path = (
                SAVE_DIR
                / (
                    "medium_focus_live_"
                    f"{timestamp}.jpg"
                )
            )

            cv2.imwrite(
                str(
                    save_path
                ),
                output,
            )

            print(
                f"Saved: {save_path}"
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


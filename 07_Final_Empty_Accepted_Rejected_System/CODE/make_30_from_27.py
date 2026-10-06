"""
Builds 30_live_full_conveyor_consistency_test.py from YOUR local
27_live_fast_new_object_final.py.

* 27 is only READ, never modified.
* If any expected piece of 27 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P30]".

Run:  python make_30_from_27.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "27_live_fast_new_object_final.py"
TARGET = CODE_DIR / "30_live_full_conveyor_consistency_test.py"

if not SOURCE.exists():
    sys.exit(f"Missing source file:\n{SOURCE}")

text = SOURCE.read_text(encoding="utf-8")
problems = []


def once(old, new, label):
    global text

    count = text.count(old)

    if count != 1:
        problems.append(f"[{label}] expected exactly 1 match in 27, found {count}")
        return

    text = text.replace(old, new)


# ---------------------------------------------------------------- header
once('''# ============================================================
# CREST YOLO26 - PHASE 27 FAST NEW-OBJECT + OCCLUSION-HOLD LIVE
# SELF-CONTAINED VERSION
''', '''# ============================================================
# CREST YOLO26 - PHASE 30: FULL-CONVEYOR CONSISTENCY TEST
#
# COPY of 27_live_fast_new_object_final.py (Phase 27 is NOT modified).
# Only two things were changed, every edit is marked "# [P30]":
#
#   A. BOX PERSISTENCE  - a pallet that the tracker stops returning keeps its
#      box on screen (moving with its last velocity) for a short, time-based
#      hold, and its state is kept/bridgeable long enough to survive the
#      tracker's 5-miss deletion + 5-hit re-confirmation.
#   B. CLASSIFICATION EVERYWHERE - the clean-view gate no longer requires
#      LEFTWARD motion, so TOP (rightward) pallets are classified too.
#
# Nothing else changed: models, thresholds of the object/symbol voting,
# asymmetric object logic, 2-of-3 / 3-of-3 symbol rules, fast new-object path,
# duplicate filter, bridge geometry, camera, preprocessing, boundary.
# ============================================================

# ORIGINAL PHASE 27 HEADER FOLLOWS
# ------------------------------------------------------------
# CREST YOLO26 - PHASE 27 FAST NEW-OBJECT + OCCLUSION-HOLD LIVE
# SELF-CONTAINED VERSION
''', "header")

once('''    / "27_Live_Fast_New_Object_Final"''',
     '''    / "30_Live_Full_Conveyor_Consistency"  # [P30] own results folder''', "save dir")

# ------------------------------------------------------------- constants
once('''# ============================================================
# PALLET SOFTWARE BRIGHTENING
# ============================================================
''', '''# ============================================================
# [P30] BOX PERSISTENCE + FULL-CONVEYOR CONSISTENCY SETTINGS
# ============================================================
# The three frame-count constants above (TRACK_DISPLAY_GRACE_FRAMES,
# TRACK_STATE_MEMORY_FRAMES, STATE_BRIDGE_MAX_MISSED_FRAMES) are SUPERSEDED by
# the time-based values below (frame-rate independent).  Why: the tracker deletes
# a track after 5 missed frames and a replacement id needs 5 confirming hits, so
# after a blind gap of 6+ frames the new id is first usable ~11 frames after the
# last sighting - already past the old bridge limit (10) and state memory (12).

# How long a pallet that the tracker no longer returns keeps being DRAWN.
# Must cover the blind gap PLUS the ~5 frames a replacement tracker id needs to
# be confirmed (about 0.17 s at 30 FPS), so 1.0 s hides dropouts up to ~0.8 s.
TRACK_DISPLAY_HOLD_SECONDS = 1.00

# How long its EMPTY/OBJECT + X/NO_X state is kept internally (never drawn
# beyond the hold above).
TRACK_STATE_MEMORY_SECONDS = 2.50

# A NEW tracker id may take over a lost id's state only if the lost id was last
# seen no longer ago than this (strict geometry/uniqueness checks unchanged).
STATE_BRIDGE_MAX_MISSING_SECONDS = 2.00

# While a pallet is not detected, its remembered box keeps moving with its last
# tracker velocity (px/frame) so it follows the pallet instead of freezing, and
# so the bridge compares against where the pallet should be.  1.0 = constant belt
# speed (correct on a straight; a decaying box would stall and could be caught up
# by the NEXT pallet).  Lower it only if boxes visibly run off at turns.
HELD_BOX_VELOCITY_DECAY = 1.0
HELD_BOX_MAX_SHIFT_PER_FRAME = 15.0

# A held box is hidden when a live track already covers it (prevents a ghost
# next to the re-acquired pallet).
HELD_BOX_HIDE_OVERLAP = 0.50

# Classification everywhere.  False = original Phase 27 gate (leftward only).
SYMBOL_CLEAN_ALLOW_BOTH_DIRECTIONS = True
# True  = keep rejecting vertical-dominant motion (turns); top AND bottom pass.
# False = also allow vertical motion (left/right sides); geometry-stability
#         checks still apply.
SYMBOL_CLEAN_REQUIRE_HORIZONTAL = True
SYMBOL_CLEAN_MIN_SPEED = 0.60  # px/frame, any allowed direction

# Compact diagnostics (event lines + one table every N seconds).
CONSISTENCY_DEBUG = True
CONSISTENCY_DEBUG_PRINT_SECONDS = 10.0


# ============================================================
# PALLET SOFTWARE BRIGHTENING
# ============================================================
''', "constants")

# ------------------------------------------------------- bridge window
once('''            and data.get("missed", 0) <= STATE_BRIDGE_MAX_MISSED_FRAMES''',
     '''            and (
                time.monotonic() - data.get("t_seen", time.monotonic())
                <= STATE_BRIDGE_MAX_MISSING_SECONDS  # [P30] was a 10-frame limit
            )''', "bridge window")

once('''        old_cache["box"] = new_box
        old_cache["missed"] = 0
        track_cache[new_track_id] = old_cache''',
     '''        old_cache["box"] = new_box
        old_cache["missed"] = 0
        old_cache["t_seen"] = time.monotonic()  # [P30]
        track_cache[new_track_id] = old_cache''', "move_track_state")

# ------------------------------------------------------------------ gate
once('''    moving_left = (
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
''', '''    if SYMBOL_CLEAN_ALLOW_BOTH_DIRECTIONS:
        # [P30] direction-agnostic: any sufficiently fast motion.  Phase 27
        # required velocity_x <= -0.60, i.e. LEFTWARD only, so a pallet on the
        # rightward-moving straight stayed in HOLD-DIRECTION forever.
        # (variable name kept: it now means "moving in an allowed direction")
        if SYMBOL_CLEAN_REQUIRE_HORIZONTAL:
            moving_left = abs(velocity_x) >= SYMBOL_CLEAN_MIN_SPEED
        else:
            moving_left = (
                float(np.hypot(velocity_x, velocity_y))
                >= SYMBOL_CLEAN_MIN_SPEED
            )
    else:
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
        or not SYMBOL_CLEAN_REQUIRE_HORIZONTAL  # [P30]
    )
''', "gate")

# ------------------------------------------------ draw_pallet thickness
once('''def draw_pallet(
    output,
    box,
    label,
    color,
):''', '''def draw_pallet(
    output,
    box,
    label,
    color,
    thickness=4,  # [P30] held (undetected) boxes are drawn thinner
):''', "draw_pallet signature")

once('''        color,
        4,
    )

    cv2.putText(
        output,
        label,''', '''        color,
        thickness,
    )

    cv2.putText(
        output,
        label,''', "draw_pallet thickness")

# --------------------------------------------------------------- helpers
once('''# ============================================================
# MAIN
# ============================================================

def main():''', '''# ============================================================
# [P30] HELD-BOX PERSISTENCE HELPERS
# ============================================================

def advance_held_box(
    data,
    frame_width,
    frame_height,
):
    """Move a not-detected pallet's remembered box by its last velocity."""

    vx = float(data.get("vx", 0.0))
    vy = float(data.get("vy", 0.0))

    shift_x = max(
        -HELD_BOX_MAX_SHIFT_PER_FRAME,
        min(HELD_BOX_MAX_SHIFT_PER_FRAME, vx),
    )

    shift_y = max(
        -HELD_BOX_MAX_SHIFT_PER_FRAME,
        min(HELD_BOX_MAX_SHIFT_PER_FRAME, vy),
    )

    x1, y1, x2, y2 = data["box"]

    box_width = x2 - x1
    box_height = y2 - y1

    x1 = max(0, min(frame_width - box_width, x1 + shift_x))
    y1 = max(0, min(frame_height - box_height, y1 + shift_y))

    data["box"] = (
        int(round(x1)),
        int(round(y1)),
        int(round(x1 + box_width)),
        int(round(y1 + box_height)),
    )

    data["vx"] = vx * HELD_BOX_VELOCITY_DECAY
    data["vy"] = vy * HELD_BOX_VELOCITY_DECAY


def build_visible_track_cache(
    track_cache,
    current_track_ids,
    now,
):
    """
    Boxes to draw/count: every track seen within the hold window, except a
    HELD (not currently detected) box that is already covered by a live track.
    """

    live_boxes = [
        data["box"]
        for track_id, data in track_cache.items()
        if (
            track_id in current_track_ids
            and data.get("missed", 0) == 0
        )
    ]

    visible = {}

    for track_id, data in track_cache.items():
        if (
            now - data.get("t_seen", now)
            > TRACK_DISPLAY_HOLD_SECONDS
        ):
            continue

        held = (
            track_id not in current_track_ids
            or data.get("missed", 0) > 0
        )

        if held:
            covered = any(
                smaller_box_overlap_ratio(data["box"], live_box)
                >= HELD_BOX_HIDE_OVERLAP
                or current_boxes_are_duplicate(data["box"], live_box)
                for live_box in live_boxes
            )

            if covered:
                continue

        visible[track_id] = data

    return visible


def conveyor_side(
    box,
    side_geometry,
):
    """TOP / RIGHT / BOTTOM / LEFT of the conveyor ring (diagnostics only)."""

    center_x, center_y, half_w, half_h = side_geometry

    box_cx = (box[0] + box[2]) / 2.0
    box_cy = (box[1] + box[3]) / 2.0

    nx = (box_cx - center_x) / half_w
    ny = (box_cy - center_y) / half_h

    if abs(nx) > abs(ny):
        return "RIGHT" if nx > 0 else "LEFT"

    return "BOTTOM" if ny > 0 else "TOP"


# ============================================================
# MAIN
# ============================================================

def main():''', "helpers")

# ------------------------------------------------------------ main state
once('''    track_cache = {}

    # Optional stopped-belt test helper''', '''    track_cache = {}

    # [P30] diagnostics
    side_geometry = None
    side_stats = {}
    gate_bucket_memory = {}
    last_stats_print = time.monotonic()

    # Optional stopped-belt test helper''', "main state")

# ------------------------------------------------------------- miss loop
once('''        for track_id in list(
            track_cache.keys()
        ):
            track_cache[
                track_id
            ][
                "missed"
            ] += 1
''', '''        frame_now = time.monotonic()  # [P30]

        if side_geometry is None:  # [P30] ring geometry is constant
            inner_points = inner_polygon.reshape(-1, 2)
            side_geometry = (
                float(inner_points[:, 0].mean()),
                float(inner_points[:, 1].mean()),
                max(1.0, float(np.ptp(inner_points[:, 0])) / 2.0),
                max(1.0, float(np.ptp(inner_points[:, 1])) / 2.0),
            )

        for track_id in list(
            track_cache.keys()
        ):
            track_cache[
                track_id
            ][
                "missed"
            ] += 1

            # [P30] not returned by the tracker this frame: keep following it
            if track_id not in current_track_ids:
                held_data = track_cache[track_id]

                advance_held_box(
                    held_data,
                    frame.shape[1],
                    frame.shape[0],
                )

                if (
                    CONSISTENCY_DEBUG
                    and held_data["missed"] == 1
                ):
                    print(
                        f"P{track_id} "
                        f"{conveyor_side(held_data['box'], side_geometry)} "
                        f"NOT DETECTED -> holding bbox + state "
                        f"({held_data['label']})"
                    )
''', "miss loop")

# ----------------------------------------------------- per-track diagnostics
once('''            if object_crop is None:
                continue

            (
                observed_object_state,
                observed_object_confidence,
            ) = (
                classify_empty_object(''', '''            if object_crop is None:
                continue

            # [P30] diagnostics: which side, and did we just recover from a gap
            side_name = conveyor_side(box, side_geometry)
            previous_entry = track_cache.get(track_id)

            if (
                CONSISTENCY_DEBUG
                and previous_entry is not None
                and previous_entry.get("missed", 1) >= 3
            ):
                print(
                    f"P{track_id} {side_name} detection returned after "
                    f"{previous_entry['missed'] - 1} frames "
                    f"(state kept: {previous_entry['label']})"
                )

            side_counters = side_stats.setdefault(
                side_name,
                {
                    "frames": 0,
                    "object_calls": 0,
                    "clean": 0,
                    "symbol_samples": 0,
                    "symbol_ignored": 0,
                    "hold": {},
                },
            )

            (
                observed_object_state,
                observed_object_confidence,
            ) = (
                classify_empty_object(''', "diagnostics 1")

once('''            previous_object_state = object_decision.get(
                track_id
            )
''', '''            # [P30] diagnostics (counts what the existing code already did)
            side_counters["frames"] += 1
            side_counters["object_calls"] += 1

            if clean_view_ready:
                side_counters["clean"] += 1
            else:
                gate_key = (
                    "STABILIZE"
                    if gate_reason.startswith("STABILIZE")
                    else gate_reason
                )
                side_counters["hold"][gate_key] = (
                    side_counters["hold"].get(gate_key, 0) + 1
                )

            if CONSISTENCY_DEBUG:
                bucket = (
                    "CLEAN"
                    if clean_view_ready
                    else (
                        "STABILIZE"
                        if gate_reason.startswith("STABILIZE")
                        else gate_reason
                    )
                )

                if gate_bucket_memory.get(track_id) != bucket:
                    print(
                        f"P{track_id} {side_name} view gate -> {bucket} "
                        f"(vx={track.get('velocity_x', 0.0):+.2f} "
                        f"vy={track.get('velocity_y', 0.0):+.2f})"
                    )
                    gate_bucket_memory[track_id] = bucket

            previous_object_state = object_decision.get(
                track_id
            )
''', "diagnostics 2")

once('''                                print(
                                    f"P{track_id} "
                                    f"symbol sample "
                                    f"{len(symbol_history[track_id])}/3: "''',
     '''                                side_counters["symbol_samples"] += 1  # [P30]

                                print(
                                    f"P{track_id} {side_name} "
                                    f"symbol sample "
                                    f"{len(symbol_history[track_id])}/3: "''', "sample accepted")

once('''                                print(
                                    f"P{track_id} "
                                    f"symbol sample ignored: "''',
     '''                                side_counters["symbol_ignored"] += 1  # [P30]

                                print(
                                    f"P{track_id} {side_name} "
                                    f"symbol sample ignored: "''', "sample ignored")

# ---------------------------------------------------------- cache record
once('''            track_cache[
                track_id
            ] = {
                "box": box,
                "missed": 0,
                "label": label,
                "color": color,
            }''', '''            track_cache[
                track_id
            ] = {
                "box": box,
                "missed": 0,
                "label": label,
                "color": color,
                "vx": float(track.get("velocity_x", 0.0)),  # [P30]
                "vy": float(track.get("velocity_y", 0.0)),  # [P30]
                "t_seen": frame_now,  # [P30]
            }''', "cache record")

# -------------------------------------------------------- memory cleanup
once('''            if (
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
''', '''            if (
                frame_now
                - track_cache[
                    track_id
                ].get(
                    "t_seen",
                    frame_now,
                )
                > TRACK_STATE_MEMORY_SECONDS  # [P30] was 12 frames
            ):
                if CONSISTENCY_DEBUG:
                    print(
                        f"P{track_id} lost for "
                        f"{TRACK_STATE_MEMORY_SECONDS:.1f}s -> "
                        f"box + state removed (no ghost)"
                    )

                del track_cache[
                    track_id
                ]
''', "memory cleanup")

# --------------------------------------------------------- visible cache
once('''        visible_track_cache = {
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
        }''', '''        # [P30] time-based hold (was: missed <= 2 frames), minus held boxes
        # that a live track already covers.
        visible_track_cache = build_visible_track_cache(
            track_cache,
            current_track_ids,
            frame_now,
        )''', "visible cache")

once('''                data[
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
            ),''', '''                data[
                    "label"
                ],
                data[
                    "color"
                ],
                thickness=(
                    2
                    if data.get("missed", 0) > 0
                    else 4
                ),  # [P30] held box = thinner
            )

        cv2.rectangle(
            output,
            (
                10,
                10,
            ),''', "draw held thinner")

once('''                if reset_data.get(
                    "missed",
                    0,
                ) <= TRACK_DISPLAY_GRACE_FRAMES:''', '''                if (
                    time.monotonic()
                    - reset_data.get("t_seen", 0.0)
                    <= TRACK_DISPLAY_HOLD_SECONDS
                ):  # [P30]''', "manual recheck")

# ---------------------------------------------------- texts + stats table
once('''"FAST NEW-OBJECT VERIFY | OCCLUSION HOLD | R=MANUAL RECHECK",''',
     '''"PHASE 30: BOX HOLD | ALL-SIDES CLASSIFY | R=MANUAL RECHECK",''', "status text")

once('''"CREST - FAST NEW-OBJECT + OCCLUSION-HOLD LIVE"''',
     '''"CREST - PHASE 30 FULL-CONVEYOR CONSISTENCY TEST"''', "window title")

once('''        cv2.imshow(
            window_name,
            output,
        )
''', '''        # [P30] one compact per-side table every N seconds
        if (
            CONSISTENCY_DEBUG
            and frame_now - last_stats_print
            >= CONSISTENCY_DEBUG_PRINT_SECONDS
        ):
            last_stats_print = frame_now

            print()
            print("---- PER-SIDE CLASSIFICATION ACTIVITY (last window) ----")
            print("side   | track-frames | obj-clf calls | clean-view | symbol samples (ignored) | gate holds")

            for side_name_key in ("TOP", "RIGHT", "BOTTOM", "LEFT"):
                counters = side_stats.get(side_name_key)

                if counters is None:
                    continue

                print(
                    f"{side_name_key:6s} | {counters['frames']:12d} | "
                    f"{counters['object_calls']:13d} | {counters['clean']:10d} | "
                    f"{counters['symbol_samples']:6d} ({counters['symbol_ignored']:d})"
                    f"{'':14s} | {counters['hold']}"
                )

            print("--------------------------------------------------------")
            print()

            side_stats.clear()

        cv2.imshow(
            window_name,
            output,
        )
''', "stats table")

# ------------------------------------------------------------------ write
if problems:
    print("NOTHING WAS WRITTEN. Your 27 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("27_live_fast_new_object_final.py was NOT modified.")

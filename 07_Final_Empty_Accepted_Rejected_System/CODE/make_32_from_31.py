"""
Builds 32_live_full_conveyor_consistency_test.py from YOUR local
31_live_full_conveyor_consistency_test.py.

* 31 is only READ, never modified.
* If any expected piece of 31 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P32]".

Phase 32 fixes ONE thing: a box that is held while its pallet is not detected
must not slide by itself.  It now follows two physical facts:
  1. If the belt is not carrying pallets (no detected pallet is moving), a
     pallet that is briefly undetected has not moved either -> its box stays.
  2. A pallet on the conveyor cannot leave the conveyor -> a held box is never
     moved to a position outside the conveyor ring.

Run:  python make_32_from_31.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "31_live_full_conveyor_consistency_test.py"
TARGET = CODE_DIR / "32_live_full_conveyor_consistency_test.py"

if not SOURCE.exists():
    sys.exit(f"Missing source file:\n{SOURCE}")

text = SOURCE.read_text(encoding="utf-8")
problems = []


def once(old, new, label):
    global text

    count = text.count(old)

    if count != 1:
        problems.append(f"[{label}] expected exactly 1 match in 31, found {count}")
        return

    text = text.replace(old, new)


def cut(start, end, replacement, label):
    """Replace everything from `start` up to (not including) `end`."""
    global text

    if text.count(start) != 1:
        problems.append(f"[{label}] start marker found {text.count(start)} times (need 1)")
        return

    start_index = text.index(start)
    end_index = text.find(end, start_index + len(start))

    if end_index < 0:
        problems.append(f"[{label}] end marker not found after start marker")
        return

    text = text[:start_index] + replacement + text[end_index:]


# ------------------------------------------------------------ header / names
once('''# ------------------------------------------------------------
# PHASE 31 (this file) = Phase 30 plus:''', '''# ------------------------------------------------------------
# PHASE 32 (this file) = Phase 31 plus ONE fix: a held (undetected) pallet's
# box no longer slides by itself.  It only moves while the belt is actually
# carrying pallets, and never outside the conveyor ring.
# ------------------------------------------------------------

# ------------------------------------------------------------
# PHASE 31 = Phase 30 plus:''', "header")

once('''    / "31_Live_Full_Conveyor_Consistency"''',
     '''    / "32_Live_Full_Conveyor_Consistency"''', "save dir")

once('''"PHASE 31: BOX HOLD | ALL-SIDES | KEEP DECISION | R=RECHECK",''',
     '''"PHASE 32: BOX HOLD | ALL-SIDES | KEEP DECISION | R=RECHECK",''', "status text")

once('''"CREST - PHASE 31 FULL-CONVEYOR CONSISTENCY TEST"''',
     '''"CREST - PHASE 32 FULL-CONVEYOR CONSISTENCY TEST"''', "window title")

# --------------------------------------------------------- held-box motion
cut('''def advance_held_box(''', '''def build_visible_track_cache(''', '''def advance_held_box(
    data,
    frame_width,
    frame_height,
    belt_moving=True,  # [P32]
    ring_mask=None,  # [P32]
):
    """
    Move a not-detected pallet's remembered box by its last velocity - but only
    when that is physically plausible:

      * belt_moving False  -> nothing on the belt is moving, so a pallet that is
        merely undetected has not moved either: the box stays where it is.
      * ring_mask given    -> a pallet on the conveyor cannot leave the
        conveyor: a move that would put the box centre outside the ring is not
        made (the box stays at its last position inside it).
    """

    if not belt_moving:  # [P32]
        return

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

    if ring_mask is not None:  # [P32]
        center_x = int(round(x1 + box_width / 2.0))
        center_y = int(round(y1 + box_height / 2.0))

        mask_height, mask_width = ring_mask.shape[:2]

        if not (
            0 <= center_x < mask_width
            and 0 <= center_y < mask_height
            and ring_mask[center_y, center_x] > 0
        ):
            return

    data["box"] = (
        int(round(x1)),
        int(round(y1)),
        int(round(x1 + box_width)),
        int(round(y1 + box_height)),
    )

    data["vx"] = vx * HELD_BOX_VELOCITY_DECAY
    data["vy"] = vy * HELD_BOX_VELOCITY_DECAY


''', "advance_held_box")

once('''                advance_held_box(
                    held_data,
                    frame.shape[1],
                    frame.shape[0],
                )''', '''                advance_held_box(
                    held_data,
                    frame.shape[1],
                    frame.shape[0],
                    belt_moving=belt_moving,  # [P32]
                    ring_mask=ring_mask,  # [P32]
                )''', "call site")

once('''        frame_now = time.monotonic()  # [P30]
''', '''        frame_now = time.monotonic()  # [P30]

        # [P32] Is the belt carrying pallets right now?  Same definition of
        # "moving" as the clean-view gate.  If no pallet is visible this frame
        # the previous answer is kept.
        if confirmed_tracks:
            belt_moving = any(
                float(
                    np.hypot(
                        track.get("velocity_x", 0.0),
                        track.get("velocity_y", 0.0),
                    )
                )
                >= SYMBOL_CLEAN_MIN_SPEED
                for track in confirmed_tracks
            )
''', "belt moving")

once('''    track_cache = {}

    # Optional stopped-belt test helper''', '''    track_cache = {}

    belt_moving = True  # [P32] updated every frame from the visible pallets

    # Optional stopped-belt test helper''', "belt init")

# ------------------------------------------------------------------ write
if problems:
    print("NOTHING WAS WRITTEN. Your 31 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("31_live_full_conveyor_consistency_test.py was NOT modified.")

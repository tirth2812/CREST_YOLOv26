"""
Builds 37_live_static_ghost_filter_test.py from YOUR local
32_live_full_conveyor_consistency_test.py.

* 32 is only READ, never modified.
* If any expected piece of 32 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P37]".

Phase 37 = Phase 32 plus ONE thing: a box that does not move at all for 6 s
while the belt is carrying other pallets (the wrapped carton, the arrow sign)
is not a pallet and is no longer drawn or counted.  A real pallet always moves
with the belt; if that box starts moving it is a pallet again at once.
Nothing is hidden while the belt is stopped.  Classification is untouched.

Run:  python make_37_from_32.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "32_live_full_conveyor_consistency_test.py"
TARGET = CODE_DIR / "37_live_static_ghost_filter_test.py"

if not SOURCE.exists():
    sys.exit(f"Missing source file:\n{SOURCE}")

text = SOURCE.read_text(encoding="utf-8")
problems = []


def once(old, new, label):
    global text

    count = text.count(old)

    if count != 1:
        problems.append(f"[{label}] expected exactly 1 match in 32, found {count}")
        return

    text = text.replace(old, new)


once('''# PHASE 32 (this file) = Phase 31 plus ONE fix:''',
     '''# PHASE 37 (this file) = Phase 32 plus ONE thing: a box that never moves while
# the belt carries other pallets (carton, arrow sign) is not a pallet - hidden.
# ------------------------------------------------------------
#
# ------------------------------------------------------------
# PHASE 32 = Phase 31 plus ONE fix:''', "header")

once('''    / "32_Live_Full_Conveyor_Consistency"''',
     '''    / "37_Live_Static_Ghost_Filter"''', "save dir")

once('''"PHASE 32: BOX HOLD | ALL-SIDES | KEEP DECISION | R=RECHECK",''',
     '''"PHASE 37: BOX HOLD | NO STATIC GHOSTS | KEEP DECISION | R=RECHECK",''', "status text")

once('''"CREST - PHASE 32 FULL-CONVEYOR CONSISTENCY TEST"''',
     '''"CREST - PHASE 37 STATIC GHOST FILTER TEST"''', "window title")

once('''def draw_pallet(
''', '''# [P37] STATIC GHOST FILTER ---------------------------------------------
STATIC_GHOST_MAX_MOVE_PX = 40.0   # [P37] still = centre stayed within this
STATIC_GHOST_SECONDS = 6.0        # [P37] still this long while the belt moves
STATIC_GHOST_BELT_WINDOW = 1.0    # [P37] belt "moving" = some pallet travelled this recently

static_ghost_state = {}  # [P37]


def drop_static_ghosts(tracks, now):  # [P37]
    """Returns (tracks_to_keep, ids_dropped).

    A pallet on this conveyor always travels with the belt.  A box whose centre
    stays put for STATIC_GHOST_SECONDS while another box has recently travelled
    is a fixed object (carton, sign): dropped until it moves.  While the belt is
    stopped nothing is dropped and the timers do not run.
    """

    state = static_ghost_state
    live = {track["track_id"] for track in tracks}

    for track_id in list(state):
        if track_id not in live:
            del state[track_id]

    for track in tracks:
        x1, y1, x2, y2 = track["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        entry = state.get(track["track_id"])

        if entry is None:
            state[track["track_id"]] = {
                "cx": cx, "cy": cy, "still": 0.0,
                "last": now, "moved_at": -1e9, "ghost": False,
            }
            continue

        if float(np.hypot(cx - entry["cx"], cy - entry["cy"])) > STATIC_GHOST_MAX_MOVE_PX:
            entry.update(cx=cx, cy=cy, still=0.0, moved_at=now, ghost=False)

    belt_moving_now = any(
        now - entry["moved_at"] <= STATIC_GHOST_BELT_WINDOW
        for entry in state.values()
    )

    kept = []
    dropped = []

    for track in tracks:
        entry = state[track["track_id"]]
        dt = min(0.5, max(0.0, now - entry["last"]))
        entry["last"] = now

        if belt_moving_now and now - entry["moved_at"] > STATIC_GHOST_BELT_WINDOW:
            entry["still"] += dt

        if entry["still"] >= STATIC_GHOST_SECONDS:
            entry["ghost"] = True

        if entry["ghost"]:
            dropped.append(track["track_id"])
        else:
            kept.append(track)

    return kept, dropped


def draw_pallet(
''', "ghost filter function")

once('''        for suppressed_id in (
            suppressed_track_ids
        ):
''', '''        # [P37] a box that never moves while the belt runs is not a pallet
        confirmed_tracks, static_ghost_ids = drop_static_ghosts(
            confirmed_tracks,
            time.monotonic(),
        )
        suppressed_track_ids = list(suppressed_track_ids) + static_ghost_ids

        for suppressed_id in (
            suppressed_track_ids
        ):
''', "apply filter")

if problems:
    print("NOTHING WAS WRITTEN. Your 32 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("32_live_full_conveyor_consistency_test.py was NOT modified.")

"""
Builds 31_live_full_conveyor_consistency_test.py from YOUR local
30_live_full_conveyor_consistency_test.py.

* 30 is only READ, never modified.
* If any expected piece of 30 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P31]".

Phase 31 =
  1. A pallet keeps its decision when the tracker gives it a new id in the OPEN
     (no wire): the state bridge can now retry while the new id is still young,
     and a second-chance "only one lost pallet <-> only one new id" match exists.
  2. Nothing is printed while running.  Decision events go to a log FILE.

Run:  python make_31_from_30.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "30_live_full_conveyor_consistency_test.py"
TARGET = CODE_DIR / "31_live_full_conveyor_consistency_test.py"

if not SOURCE.exists():
    sys.exit(f"Missing source file:\n{SOURCE}")

text = SOURCE.read_text(encoding="utf-8")
problems = []


def once(old, new, label):
    global text

    count = text.count(old)

    if count != 1:
        problems.append(f"[{label}] expected exactly 1 match in 30, found {count}")
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
once('''# ORIGINAL PHASE 27 HEADER FOLLOWS''', '''# ------------------------------------------------------------
# PHASE 31 (this file) = Phase 30 plus:
#   1. A pallet in the OPEN keeps its decision when the tracker gives it a new
#      id: the state bridge may retry while the new id is still young (it used
#      to get ONE attempt), and a second-chance match transfers state when
#      exactly one lost pallet and exactly one new id sit in the same place.
#   2. Nothing is printed while running.  Decision events are appended to
#      decision_events.log in the results folder.
# ------------------------------------------------------------

# ORIGINAL PHASE 27 HEADER FOLLOWS''', "header")

once('''    / "30_Live_Full_Conveyor_Consistency"''',
     '''    / "31_Live_Full_Conveyor_Consistency"''', "save dir")

once('''"PHASE 30: BOX HOLD | ALL-SIDES CLASSIFY | R=MANUAL RECHECK",''',
     '''"PHASE 31: BOX HOLD | ALL-SIDES | KEEP DECISION | R=RECHECK",''', "status text")

once('''"CREST - PHASE 30 FULL-CONVEYOR CONSISTENCY TEST"''',
     '''"CREST - PHASE 31 FULL-CONVEYOR CONSISTENCY TEST"''', "window title")

# ------------------------------------------- constants + silent event logger
once('''# Compact diagnostics (event lines + one table every N seconds).
CONSISTENCY_DEBUG = True
CONSISTENCY_DEBUG_PRINT_SECONDS = 10.0
''', '''# [P31] A NEW tracker id may still receive a lost pallet's state while it is
# YOUNG: no X/NO_X decision yet and at most this many EMPTY/OBJECT samples
# (samples are >= 0.18 s apart, so this is roughly its first 0.5-0.7 s).
BRIDGE_YOUNG_TRACK_MAX_OBJECT_SAMPLES = 3

# [P31] Second-chance match, used only when the strict bridge found nothing:
# exactly ONE lost pallet and exactly ONE brand-new id (first confirmed frame)
# lie within this neighbourhood of each other (and nothing else does), so they
# are the same pallet.  Looser than the strict geometry because after a longer
# gap the remembered box can be off at a turn or the new box can be partly cut
# off.  Kept below the pallet spacing (~1.2 box widths) so a neighbouring pallet
# is never in range.
RELAXED_BRIDGE_MIN_SIZE_RATIO = 0.50
RELAXED_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO = 1.00
RELAXED_BRIDGE_MAX_VERTICAL_CENTER_RATIO = 0.50

# [P31] SILENT decision-event log (a FILE, never the terminal).  Events are
# rare (a state bridge, a state change, a new untracked id), so this costs
# nothing per frame.  Set False to disable completely.
EVENT_LOG_ENABLED = True
EVENT_LOG_FILE = SAVE_DIR / "decision_events.log"


def log_event(message):  # [P31]
    if not EVENT_LOG_ENABLED:
        return

    try:
        with open(
            EVENT_LOG_FILE,
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                time.strftime("%H:%M:%S ")
                + message
                + "\\n"
            )
    except Exception:
        pass  # logging must never disturb the live system
''', "constants")

# ------------------------------------------ state bridge: retry + second chance
once('''def bridge_reacquired_tracks(
    confirmed_tracks,''', '''def track_can_receive_state(  # [P31]
    track_id,
    object_history,
    symbol_decision,
):
    """
    A new id may take over a lost pallet's state only while it is still young.
    (Phase 27 refused it as soon as it had ANY history, i.e. after its very
    first processed frame, so one imperfect first frame made the loss permanent.)
    """

    return (
        track_id not in symbol_decision
        and len(object_history.get(track_id, ()))
        <= BRIDGE_YOUNG_TRACK_MAX_OBJECT_SAMPLES
    )


def transfer_bridged_state(  # [P31]
    old_track_id,
    new_track_id,
    new_box,
    how,
    track_cache,
    object_history,
    object_decision,
    symbol_history,
    symbol_decision,
    symbol_sample_state,
    fast_symbol_start_after,
    fast_symbol_until,
):
    log_event(
        f"STATE BRIDGE ({how}) P{old_track_id} -> P{new_track_id} | "
        f"object={object_decision.get(old_track_id)} | "
        f"symbol={symbol_decision.get(old_track_id)}"
    )

    # Whatever the young id worked out on its own is replaced by the
    # established state (including a fast-verify window it may have opened).
    for store in (
        object_history,
        object_decision,
        symbol_history,
        symbol_decision,
        symbol_sample_state,
        fast_symbol_start_after,
        fast_symbol_until,
    ):
        store.pop(new_track_id, None)

    move_track_state(
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
    )


def bridge_reacquired_tracks(
    confirmed_tracks,''', "bridge helpers")

once('''    used_old_ids = set()

    for track in confirmed_tracks:
        new_track_id = track["track_id"]
        new_box = track["box"]
''', '''    used_old_ids = set()
    bridged_new_ids = set()  # [P31]

    for track in confirmed_tracks:
        new_track_id = track["track_id"]
        new_box = track["box"]
''', "bridge init")

once('''        # Existing IDs already own their state; do not remap them.
        if (
            new_track_id in object_history
            or new_track_id in object_decision
            or new_track_id in symbol_history
            or new_track_id in symbol_decision
        ):
            continue
''', '''        # [P31] Only a YOUNG id may still receive a lost pallet's state
        # (Phase 27: any history at all blocked it, so it had ONE attempt).
        if not track_can_receive_state(
            new_track_id,
            object_history,
            symbol_decision,
        ):
            continue
''', "bridge eligibility")

cut('''        old_missed = track_cache[
            best_old_id
        ].get(
            "missed",
            0,
        )
''', '''# ============================================================
# CLASS NORMALIZATION''', '''        transfer_bridged_state(  # [P31]
            best_old_id,
            new_track_id,
            new_box,
            "strict",
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

        bridged_new_ids.add(
            new_track_id
        )

    # [P31] SECOND CHANCE: for every BRAND-NEW id (first confirmed frame, not in
    # the cache yet) the strict test could not match, transfer the state only
    # when it is a MUTUALLY UNIQUE pair - exactly one lost pallet is near it,
    # and it is the only new id near that lost pallet.  Ambiguous = left alone.
    pending_new = [
        candidate
        for candidate in confirmed_tracks
        if (
            candidate["track_id"] not in bridged_new_ids
            and candidate["track_id"] not in track_cache
            and track_can_receive_state(
                candidate["track_id"],
                object_history,
                symbol_decision,
            )
        )
    ]

    pending_old = [
        old_id
        for old_id in stale_ids
        if (
            old_id not in used_old_ids
            and old_id in track_cache
        )
    ]

    def relaxed_near(old_id, candidate):
        old_box = track_cache[old_id]["box"]

        if (
            box_size_ratio(old_box, candidate["box"])
            < RELAXED_BRIDGE_MIN_SIZE_RATIO
        ):
            return False

        horizontal, vertical = center_ratios(
            old_box,
            candidate["box"],
        )

        return (
            horizontal <= RELAXED_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO
            and vertical <= RELAXED_BRIDGE_MAX_VERTICAL_CENTER_RATIO
        )

    for candidate in pending_new:
        near_old = [
            old_id
            for old_id in pending_old
            if relaxed_near(old_id, candidate)
        ]

        if len(near_old) != 1:
            continue

        old_id = near_old[0]

        near_new = [
            other
            for other in pending_new
            if relaxed_near(old_id, other)
        ]

        if len(near_new) != 1:
            continue

        transfer_bridged_state(
            old_id,
            candidate["track_id"],
            candidate["box"],
            "second-chance",
            track_cache,
            object_history,
            object_decision,
            symbol_history,
            symbol_decision,
            symbol_sample_state,
            fast_symbol_start_after,
            fast_symbol_until,
        )

        pending_old.remove(old_id)
        used_old_ids.add(old_id)
        bridged_new_ids.add(candidate["track_id"])


''', "bridge transfer + second chance")

# ------------------------------------------------ prints -> silent event log
once('''            print(
                f"P{track_id} OBJECT CHANGE: "
                f"EMPTY -> OBJECT_PRESENT "''', '''            log_event(  # [P31]
                f"P{track_id} OBJECT CHANGE: "
                f"EMPTY -> OBJECT_PRESENT "''', "object change 1")

once('''                print(
                    f"P{track_id} OBJECT CHANGE: "
                    f"OBJECT_PRESENT -> EMPTY "''', '''                log_event(  # [P31]
                    f"P{track_id} OBJECT CHANGE: "
                    f"OBJECT_PRESENT -> EMPTY "''', "object change 2")

once('''        print(
            f"P{track_id} SYMBOL CHANGE CONFIRMED: "
            f"ACCEPTED -> REJECTED | "''', '''        log_event(  # [P31]
            f"P{track_id} SYMBOL CHANGE CONFIRMED: "
            f"ACCEPTED -> REJECTED | "''', "symbol change 1")

once('''        print(
            f"P{track_id} SYMBOL CHANGE CONFIRMED: "
            f"REJECTED -> ACCEPTED | "''', '''        log_event(  # [P31]
            f"P{track_id} SYMBOL CHANGE CONFIRMED: "
            f"REJECTED -> ACCEPTED | "''', "symbol change 2")

once('''                print(
                    f"P{track_id} FAST NEW-OBJECT VERIFY STARTED | "''', '''                log_event(  # [P31]
                    f"P{track_id} FAST NEW-OBJECT VERIFY STARTED | "''', "fast verify")

once('''                if CONSISTENCY_DEBUG:
                    print(
                        f"P{track_id} lost for "
                        f"{TRACK_STATE_MEMORY_SECONDS:.1f}s -> "
                        f"box + state removed (no ghost)"
                    )
''', '''                log_event(  # [P31]
                    f"P{track_id} STATE EXPIRED after "
                    f"{TRACK_STATE_MEMORY_SECONDS:.1f}s "
                    f"(last: {track_cache[track_id]['label']})"
                )
''', "expiry")

once('''            track_cache[
                track_id
            ] = {
                "box": box,''', '''            if track_id not in track_cache:  # [P31] id with NO state carried over
                log_event(
                    f"P{track_id} NEW TRACK, no state carried | "
                    f"object={object_decision.get(track_id)} "
                    f"symbol={symbol_decision.get(track_id)} box={box}"
                )

            track_cache[
                track_id
            ] = {
                "box": box,''', "new track log")

# ------------------------------------- remove the Phase 30 terminal diagnostics
cut('''                if (
                    CONSISTENCY_DEBUG
                    and held_data["missed"] == 1
                ):''', '''        # ----------------------------------------------------
        # SHORT OCCLUSION / TRACK-ID CONTINUITY''', '''
''', "held print")

cut('''        if side_geometry is None:  # [P30] ring geometry is constant''', '''        for track_id in list(
            track_cache.keys()
        ):
            track_cache[
                track_id
            ][
                "missed"
            ] += 1
''', '', "side geometry")

cut('''            # [P30] diagnostics: which side, and did we just recover from a gap''', '''            (
                observed_object_state,
                observed_object_confidence,
            ) = (
                classify_empty_object(''', '', "diag 1")

cut('''            # [P30] diagnostics (counts what the existing code already did)''', '''            previous_object_state = object_decision.get(''', '', "diag 2")

cut('''
                                side_counters["symbol_samples"] += 1  # [P30]''', '''                (
                    current_decision,
                    x_votes,
                    no_x_votes,
                    observations,''', '''

''', "symbol sample prints")

cut('''        # [P30] one compact per-side table every N seconds''', '''        cv2.imshow(
            window_name,
            output,
        )
''', '', "stats table")

cut('''            print()
            print(
                "MANUAL RECHECK: ''', '''        if key == ord(
            "s"
        ):''', '''
''', "manual recheck print")

cut('''    print()
    print(
        "=" * 76
    )
    print(
        "CREST LIVE FINAL-HARD YOLO26 CANDIDATE"''', '''    pallet_stage = (
        load_stage08()
    )''', '', "startup banner")

once('''    # [P30] diagnostics
    side_geometry = None
    side_stats = {}
    gate_bucket_memory = {}
    last_stats_print = time.monotonic()

''', '', "diag state")

cut('''def conveyor_side(''', '''# ============================================================
# MAIN
# ============================================================''', '', "conveyor_side")

# ------------------------------------------------------------------ write
if problems:
    print("NOTHING WAS WRITTEN. Your 30 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("30_live_full_conveyor_consistency_test.py was NOT modified.")

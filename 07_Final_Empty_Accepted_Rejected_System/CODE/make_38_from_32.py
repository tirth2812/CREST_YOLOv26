"""
Builds 38_live_asymmetric_fast_symbol_test.py from YOUR local
32_live_full_conveyor_consistency_test.py.

* 32 is only READ, never modified.
* If any expected piece of 32 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P38]".

Phase 38 = Phase 32 plus TWO things, both about evidence quality (no positions,
no offsets, no new thresholds):

  1. ASYMMETRIC FAST SYMBOL EVIDENCE
     Seeing an X is positive evidence.  NOT seeing it from an unreliable view
     (turn, wire, unstable box - i.e. the existing dynamic clean-view gate says
     "not clean") proves nothing: the X may simply be hidden.  So a NO_X sample
     taken during the fast new-object window while the view is NOT clean is not
     stored as a vote.  The pallet stays CHECKING until clean-view samples decide.
       - X samples: unchanged (stored from any view, 3/3 still decides REJECTED).
       - NO_X samples taken in a clean view: unchanged.
       - fast EMPTY -> OBJECT_PRESENT detection: unchanged.

  2. BRIDGE DIAGNOSTICS (log file only, changes no behaviour)
     Whenever a brand-new tracker id gets no state although lost pallets with
     state exist, bridge_diagnostics.log records every candidate's metrics and
     the exact bridge condition that failed.

Run:  python make_38_from_32.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "32_live_full_conveyor_consistency_test.py"
TARGET = CODE_DIR / "38_live_asymmetric_fast_symbol_test.py"

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
     '''# PHASE 38 (this file) = Phase 32 plus (1) ASYMMETRIC FAST SYMBOL EVIDENCE: a NO_X
# sample taken in the fast window from a NOT-clean view is not stored (the X may
# just be hidden), so the pallet stays CHECKING until clean-view samples decide;
# X evidence and the fast EMPTY->OBJECT detection are unchanged.  (2) bridge
# DIAGNOSTICS to bridge_diagnostics.log (no behaviour change).
# ------------------------------------------------------------
#
# ------------------------------------------------------------
# PHASE 32 = Phase 31 plus ONE fix:''', "header")

once('''    / "32_Live_Full_Conveyor_Consistency"''',
     '''    / "38_Live_Asymmetric_Fast_Symbol"''', "save dir")

once('''"PHASE 32: BOX HOLD | ALL-SIDES | KEEP DECISION | R=RECHECK",''',
     '''"PHASE 38: ASYMMETRIC FAST SYMBOL | KEEP DECISION | R=RECHECK",''', "status text")

once('''"CREST - PHASE 32 FULL-CONVEYOR CONSISTENCY TEST"''',
     '''"CREST - PHASE 38 ASYMMETRIC FAST SYMBOL TEST"''', "window title")

once('''FAST_SYMBOL_MIN_RING_OVERLAP = 0.25
''', '''FAST_SYMBOL_MIN_RING_OVERLAP = 0.25

# [P38] bridge diagnostics (file only, never the terminal) and the per-track
# count of NO_X samples that were NOT stored because the view was not clean.
BRIDGE_DIAG_FILE = SAVE_DIR / "bridge_diagnostics.log"  # [P38]
fast_no_x_deferred = {}  # [P38]
''', "constants")

once('''def clear_symbol_state(
    track_id,
    symbol_history,
    symbol_decision,
    symbol_sample_state,
):
''', '''def clear_symbol_state(
    track_id,
    symbol_history,
    symbol_decision,
    symbol_sample_state,
):
    fast_no_x_deferred.pop(track_id, None)  # [P38]

''', "clear deferred count")

once('''def update_three_view_decision(
''', '''def should_store_symbol_sample(symbol_state, view_is_clean):  # [P38]
    """
    Evidence quality rule for X / NO_X samples.

    X seen            -> positive evidence, stored from any view.
    NO_X, clean view  -> reliable, stored.
    NO_X, NOT clean   -> the X may be hidden by the angle / turn / wire / an
                         unstable box, so it proves nothing: NOT stored.
    (A larger number of such samples would not help: they all come from the
    same bad view.  Only a clean view can show that no X is present.)
    """

    if symbol_state == "NO_X":
        return bool(view_is_clean)

    return True


def note_deferred_no_x(track_id):  # [P38]
    count = fast_no_x_deferred.get(track_id, 0) + 1
    fast_no_x_deferred[track_id] = count

    if count == 3:
        log_event(
            f"P{track_id} FAST NO_X DEFERRED | 3 NO_X samples from a NOT-clean view "
            f"were not counted (Phase 32 would have committed ACCEPTED here); "
            f"staying CHECKING until a clean view decides"
        )


def log_bridge_diag(message):  # [P38]
    try:
        with open(BRIDGE_DIAG_FILE, "a", encoding="utf-8") as handle:
            handle.write(time.strftime("%H:%M:%S ") + message + "\\n")
    except Exception:
        pass  # logging must never disturb the live system


def diagnose_unbridged_tracks(  # [P38]
    confirmed_tracks,
    current_track_ids,
    track_cache,
    object_history,
    object_decision,
    symbol_history,
    symbol_decision,
):
    """
    DIAGNOSTIC ONLY - reads state, changes nothing.

    For every brand-new tracker id that got no state although a lost pallet
    with state is still remembered, write every candidate's metrics and the
    first bridge condition that failed (same order as state_bridge_score).
    """

    now = time.monotonic()

    lost = [
        old_id
        for old_id, data in track_cache.items()
        if old_id not in current_track_ids
        and has_meaningful_track_state(
            old_id,
            object_history,
            object_decision,
            symbol_history,
            symbol_decision,
        )
    ]

    if not lost:
        return

    for track in confirmed_tracks:
        new_id = track["track_id"]

        if new_id in track_cache:
            continue

        new_box = track["box"]
        young = track_can_receive_state(
            new_id,
            object_history,
            symbol_decision,
        )

        rows = []
        scored = []

        for old_id in lost:
            old = track_cache[old_id]
            old_box = old["box"]
            missing = now - old.get("t_seen", now)
            size_ratio = box_size_ratio(old_box, new_box)
            iou = box_iou(old_box, new_box)
            overlap = smaller_box_overlap_ratio(old_box, new_box)
            horizontal, vertical = center_ratios(old_box, new_box)
            score = state_bridge_score(old_box, new_box)

            if missing > STATE_BRIDGE_MAX_MISSING_SECONDS:
                reason = (
                    f"missing {missing:.2f}s > {STATE_BRIDGE_MAX_MISSING_SECONDS:.2f}s"
                )
            elif size_ratio < STATE_BRIDGE_MIN_SIZE_RATIO:
                reason = (
                    f"size_ratio {size_ratio:.2f} < {STATE_BRIDGE_MIN_SIZE_RATIO:.2f}"
                )
            elif horizontal > STATE_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO:
                reason = (
                    f"horizontal {horizontal:.2f} > "
                    f"{STATE_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO:.2f}"
                )
            elif vertical > STATE_BRIDGE_MAX_VERTICAL_CENTER_RATIO:
                reason = (
                    f"vertical {vertical:.2f} > "
                    f"{STATE_BRIDGE_MAX_VERTICAL_CENTER_RATIO:.2f}"
                )
            elif (
                overlap < STATE_BRIDGE_MIN_SMALLER_OVERLAP
                and iou < STATE_BRIDGE_MIN_IOU
            ):
                reason = (
                    f"overlap {overlap:.2f} < {STATE_BRIDGE_MIN_SMALLER_OVERLAP:.2f} "
                    f"and iou {iou:.2f} < {STATE_BRIDGE_MIN_IOU:.2f}"
                )
            elif score is not None and score < STATE_BRIDGE_MIN_SCORE:
                reason = f"score {score:.2f} < {STATE_BRIDGE_MIN_SCORE:.2f}"
            else:
                reason = "geometry passes"
                scored.append((score, old_id))

            rows.append(
                f"    old=P{old_id} symbol={symbol_decision.get(old_id)} "
                f"object={object_decision.get(old_id)} missing={missing:.2f}s "
                f"old_box={tuple(int(v) for v in old_box)} "
                f"size_ratio={size_ratio:.2f} iou={iou:.2f} overlap={overlap:.2f} "
                f"h={horizontal:.2f} v={vertical:.2f} "
                f"score={'-' if score is None else format(score, '.2f')} "
                f"-> {reason}"
            )

        scored.sort(reverse=True)

        if not young:
            verdict = "new id is not young (already has a symbol decision or > 3 object samples)"
        elif len(scored) > 1 and scored[0][0] - scored[1][0] < STATE_BRIDGE_UNIQUE_MARGIN:
            verdict = (
                f"AMBIGUOUS: best {scored[0][0]:.2f} vs second {scored[1][0]:.2f} "
                f"(margin < {STATE_BRIDGE_UNIQUE_MARGIN:.2f})"
            )
        elif scored and scored[0][0] < STATE_BRIDGE_MIN_SCORE:
            verdict = "best score below minimum"
        elif scored:
            verdict = f"strict geometry passes for P{scored[0][1]} (taken by another id?)"
        else:
            verdict = "no candidate passes the strict geometry"

        log_bridge_diag(
            f"BRIDGE REJECT new=P{new_id} new_box={tuple(int(v) for v in new_box)} "
            f"young={young} | {verdict}"
        )

        for row in rows:
            log_bridge_diag(row)


def update_three_view_decision(
''', "helpers")

once('''                                symbol_history[
                                    track_id
                                ].append(
                                    (
                                        symbol_state,
                                        symbol_confidence,
                                    )
                                )

                                commit_sample_position(''', '''                                # [P38] a NO_X from a NOT-clean view proves nothing
                                if should_store_symbol_sample(
                                    symbol_state,
                                    clean_view_ready or manual_recheck,
                                ):
                                    symbol_history[
                                        track_id
                                    ].append(
                                        (
                                            symbol_state,
                                            symbol_confidence,
                                        )
                                    )
                                else:
                                    note_deferred_no_x(track_id)

                                commit_sample_position(''', "store rule")

once('''                (
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
                )''', '''                decision_before = symbol_decision.get(track_id)  # [P38]

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

                # [P38] log the first terminal commit of this id
                if decision_before is None and current_decision in (
                    "ACCEPTED",
                    "REJECTED",
                ):
                    if current_decision == "REJECTED" and fast_mode:
                        log_event(
                            f"P{track_id} FAST X COMMIT | X={x_votes} NO_X={no_x_votes}"
                        )
                    else:
                        log_event(
                            f"P{track_id} CLEAN {current_decision} | "
                            f"X={x_votes} NO_X={no_x_votes}"
                        )''', "commit log")

once('''        output = (
            symbol_frame.copy()
        )
''', '''        # [P38] diagnostics only: why did a brand-new id receive no state?
        diagnose_unbridged_tracks(
            confirmed_tracks,
            current_track_ids,
            track_cache,
            object_history,
            object_decision,
            symbol_history,
            symbol_decision,
        )

        output = (
            symbol_frame.copy()
        )
''', "diagnostics call")

if problems:
    print("NOTHING WAS WRITTEN. Your 32 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("32_live_full_conveyor_consistency_test.py was NOT modified.")

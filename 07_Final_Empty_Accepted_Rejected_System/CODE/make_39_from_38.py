"""
Builds 39_live_decision_audit_test.py from YOUR local
38_live_asymmetric_fast_symbol_test.py.

* 38 is only READ, never modified.
* If any expected piece of 38 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P39]".

Phase 39 = Phase 38 + a DECISION AUDIT.  It changes NO decision logic, NO
threshold, NO sampling: classification is exactly Phase 38.  It only records
evidence, so that an occasional wrong result can be traced instead of guessed:

  * every X/NO_X sample taken (stored or deferred): prediction, confidence,
    p(X), p(NO_X), FAST/CLEAN mode, clean-view flag + gate reason, velocity,
    detector confidence, ring overlap, stable-frame count, box stability
    (IoU / size ratio / centre motion), position, time, and the exact crops
  * every terminal decision (first ACCEPTED/REJECTED and every change): the
    samples that produced it, movement between the samples, the tracker-id
    lineage (bridge chain), object state, and a RULE-VIOLATION check that
    proves the asymmetric fast rule is active
  * files: Decision_Samples\\<time>_P<id>_<DECISION>_<FAST|CLEAN>_obs<n>.jpg,
    Decision_Audit\\<time>_P<id>_<DECISION>_<MODE>\\ (frame, pallet crop,
    decision.json) and decision_audit.log

Run:  python make_39_from_38.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "38_live_asymmetric_fast_symbol_test.py"
TARGET = CODE_DIR / "39_live_decision_audit_test.py"

if not SOURCE.exists():
    sys.exit(f"Missing source file:\n{SOURCE}")

text = SOURCE.read_text(encoding="utf-8")
problems = []


def once(old, new, label):
    global text

    count = text.count(old)

    if count != 1:
        problems.append(f"[{label}] expected exactly 1 match in 38, found {count}")
        return

    text = text.replace(old, new)


once('''# PHASE 38 (this file) = Phase 32 plus (1)''',
     '''# PHASE 39 (this file) = Phase 38 + a DECISION AUDIT (evidence capture only; no
# decision logic, threshold or sampling is changed).  See make_39_from_38.py.
# ------------------------------------------------------------
#
# ------------------------------------------------------------
# PHASE 38 (this file) = Phase 32 plus (1)''', "header")

once('''    / "38_Live_Asymmetric_Fast_Symbol"''',
     '''    / "39_Live_Decision_Audit"''', "save dir")

once('''"PHASE 38: ASYMMETRIC FAST SYMBOL | KEEP DECISION | R=RECHECK",''',
     '''"PHASE 39: DECISION AUDIT | KEEP DECISION | R=RECHECK",''', "status text")

once('''"CREST - PHASE 38 ASYMMETRIC FAST SYMBOL TEST"''',
     '''"CREST - PHASE 39 DECISION AUDIT TEST"''', "window title")

once('''fast_no_x_deferred = {}  # [P38]
''', '''fast_no_x_deferred = {}  # [P38]

# [P39] decision audit state (evidence capture only)
AUDIT_DIR = SAVE_DIR / "Decision_Audit"  # [P39]
SAMPLES_DIR = SAVE_DIR / "Decision_Samples"  # [P39]
AUDIT_LOG_FILE = SAVE_DIR / "decision_audit.log"  # [P39]
AUDIT_MAX_SAMPLES_PER_TRACK = 8  # [P39]
sample_audit = {}  # [P39] track id -> deque of sample records (with crops)
audit_prev_box = {}  # [P39] track id -> box of the previous frame
audit_frame_info = {}  # [P39] track id -> gate/geometry info of the latest frame
track_born = {}  # [P39] track id -> time it first appeared
track_lineage = {}  # [P39] track id -> ids its state came from (bridge chain)
track_bridged_at = {}  # [P39] track id -> time of the last bridge into it
''', "audit state")

once('''    fast_no_x_deferred.pop(track_id, None)  # [P38]
''', '''    fast_no_x_deferred.pop(track_id, None)  # [P38]
    sample_audit.pop(track_id, None)  # [P39]
''', "clear audit with symbol state")

once('''    old_cache = track_cache.pop(
        old_track_id,
        None,
    )
''', '''    # [P39] the audit evidence and the id lineage follow the state
    if old_track_id in sample_audit:
        sample_audit[new_track_id] = sample_audit.pop(old_track_id)

    track_lineage[new_track_id] = (
        track_lineage.pop(old_track_id, [old_track_id]) + [new_track_id]
    )
    track_bridged_at[new_track_id] = time.time()

    old_cache = track_cache.pop(
        old_track_id,
        None,
    )
''', "move audit with state")

once('''def should_store_symbol_sample(symbol_state, view_is_clean):  # [P38]''',
     '''def _audit_float(value):  # [P39]
    try:
        return round(float(value), 4)
    except Exception:
        return None


def audit_note_tracks(confirmed_tracks):  # [P39]
    try:
        now = time.time()

        for track in confirmed_tracks:
            track_born.setdefault(track["track_id"], now)
    except Exception:
        pass  # the audit must never disturb the live system


def audit_note_frame(  # [P39]
    track_id,
    track,
    clean_view_ready,
    gate_reason,
    symbol_quality_state,
):
    """Remember this frame's gate result + box-stability numbers (read only)."""

    try:
        box = track["box"]
        previous = audit_prev_box.get(track_id)

        if previous is None:
            iou = size_ratio = h = v = None
        else:
            iou = box_iou(previous, box)
            size_ratio = box_size_ratio(previous, box)
            h, v = center_ratios(previous, box)

        audit_prev_box[track_id] = box

        audit_frame_info[track_id] = {
            "clean_view": bool(clean_view_ready),
            "gate_reason": str(gate_reason),
            "vx": _audit_float(track.get("velocity_x", 0.0)),
            "vy": _audit_float(track.get("velocity_y", 0.0)),
            "detector_confidence": _audit_float(track.get("confidence", 0.0)),
            "ring_overlap": _audit_float(track.get("overlap", 0.0)),
            "stable_frames": symbol_quality_state.get(track_id, {}).get("stable_frames"),
            "box_iou_vs_prev_frame": _audit_float(iou) if iou is not None else None,
            "size_ratio_vs_prev_frame": _audit_float(size_ratio) if size_ratio is not None else None,
            "center_h_ratio": _audit_float(h) if h is not None else None,
            "center_v_ratio": _audit_float(v) if v is not None else None,
        }
    except Exception:
        pass  # the audit must never disturb the live system


def audit_record_sample(  # [P39]
    track_id,
    symbol_state,
    symbol_confidence,
    p_x,
    p_no_x,
    sample_time,
    center_x,
    center_y,
    diagonal,
    box,
    mode,
    stored,
    prepared_crop,
    full_crop,
):
    """One X/NO_X sample that passed the minimum confidence (stored or deferred)."""

    try:
        if track_id not in sample_audit:
            sample_audit[track_id] = deque(maxlen=AUDIT_MAX_SAMPLES_PER_TRACK)

        info = dict(audit_frame_info.get(track_id, {}))
        info.update(
            {
                "time": float(sample_time),
                "time_text": time.strftime("%H:%M:%S", time.localtime(sample_time)),
                "center": (round(float(center_x), 1), round(float(center_y), 1)),
                "diagonal": round(float(diagonal), 1),
                "box": tuple(int(v) for v in box),
                "state": str(symbol_state),
                "confidence": _audit_float(symbol_confidence),
                "p_x": _audit_float(p_x),
                "p_no_x": _audit_float(p_no_x),
                "mode": mode,
                "stored_as_vote": bool(stored),
                "crop": None if prepared_crop is None else prepared_crop.copy(),
                "full_crop": None if full_crop is None else full_crop.copy(),
            }
        )
        sample_audit[track_id].append(info)
    except Exception:
        pass  # the audit must never disturb the live system


def audit_commit(  # [P39]
    track_id,
    decision_before,
    decision_after,
    fast_mode,
    clean_view_ready,
    gate_reason,
    x_votes,
    no_x_votes,
    observations,
    object_state,
    object_confidence,
    track,
    box,
    frame,
):
    """Everything needed to reconstruct why a terminal decision was made."""

    try:
        import json

        now = time.time()
        stamp = time.strftime("%H%M%S", time.localtime(now)) + f"_{int((now % 1) * 1000):03d}"
        mode = "FAST" if fast_mode else "CLEAN"
        kind = "FIRST" if decision_before is None else f"CHANGE {decision_before}->{decision_after}"
        tag = f"{stamp}_P{track_id}_{decision_after}_{mode}"

        records = list(sample_audit.get(track_id, ()))
        stored = [r for r in records if r["stored_as_vote"]][-3:]
        deferred = [r for r in records if not r["stored_as_vote"]]

        # --- are the votes really different views? -------------------------------
        movement = []
        for first, second in zip(stored, stored[1:]):
            dx = second["center"][0] - first["center"][0]
            dy = second["center"][1] - first["center"][1]
            distance = (dx * dx + dy * dy) ** 0.5
            diagonal = max(1.0, (first["diagonal"] + second["diagonal"]) / 2.0)
            movement.append(
                {
                    "px": round(distance, 1),
                    "ratio_of_diagonal": round(distance / diagonal, 3),
                    "seconds": round(second["time"] - first["time"], 3),
                }
            )

        total_px = None
        if len(stored) >= 2:
            dx = stored[-1]["center"][0] - stored[0]["center"][0]
            dy = stored[-1]["center"][1] - stored[0]["center"][1]
            total_px = round((dx * dx + dy * dy) ** 0.5, 1)

        # --- proof that the asymmetric fast rule is active ----------------------
        violation = [
            r["time_text"]
            for r in stored
            if r["state"] == "NO_X" and not r["clean_view"] and r["mode"] != "MANUAL"
        ]

        born = track_born.get(track_id)
        bridged = track_bridged_at.get(track_id)
        lineage = track_lineage.get(track_id, [track_id])

        files = []
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        for number, record in enumerate(stored, start=1):
            if record["crop"] is not None:
                name = f"{tag}_obs{number}.jpg"
                cv2.imwrite(str(SAMPLES_DIR / name), record["crop"])
                files.append(name)

        audit_folder = AUDIT_DIR / tag
        audit_folder.mkdir(parents=True, exist_ok=True)

        annotated = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            annotated,
            f"P{track_id} {decision_after} {mode}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imwrite(str(audit_folder / "frame.jpg"), annotated)

        if stored and stored[-1]["full_crop"] is not None:
            cv2.imwrite(str(audit_folder / "pallet_crop.jpg"), stored[-1]["full_crop"])

        def public(record):
            return {k: v for k, v in record.items() if k not in ("crop", "full_crop")}

        audit = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "track": f"P{track_id}",
            "kind": kind,
            "decision": decision_after,
            "mode": mode,
            "clean_view_ready_at_commit": bool(clean_view_ready),
            "gate_reason_at_commit": str(gate_reason),
            "votes": {"X": x_votes, "NO_X": no_x_votes, "observations": observations},
            "object_state": object_state,
            "object_confidence": _audit_float(object_confidence),
            "box": tuple(int(v) for v in box),
            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            "detector_confidence": _audit_float(track.get("confidence", 0.0)),
            "ring_overlap": _audit_float(track.get("overlap", 0.0)),
            "vx": _audit_float(track.get("velocity_x", 0.0)),
            "vy": _audit_float(track.get("velocity_y", 0.0)),
            "tracker_lineage": " -> ".join(f"P{i}" for i in lineage),
            "seconds_since_id_first_seen": None if born is None else round(now - born, 2),
            "seconds_since_last_bridge_into_id": None if bridged is None else round(now - bridged, 2),
            "samples_used": [public(r) for r in stored],
            "sample_crop_files": files,
            "movement_between_samples": movement,
            "total_movement_px_first_to_last": total_px,
            "samples_NOT_stored_unsafe_view_NO_X": [public(r) for r in deferred][-6:],
            "RULE_VIOLATION_unsafe_NO_X_counted": violation,
        }

        with open(audit_folder / "decision.json", "w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, default=str)

        lines = [
            "",
            f"DECISION COMMIT  {audit['time']}  track=P{track_id}  {kind}",
            f"  decision={decision_after}  mode={mode}  clean_view={bool(clean_view_ready)}  gate={gate_reason}",
            f"  votes X={x_votes} NO_X={no_x_votes} obs={observations}  object={object_state} {object_confidence:.2f}",
            f"  box={audit['box']} center=({audit['center'][0]:.0f},{audit['center'][1]:.0f}) "
            f"det_conf={audit['detector_confidence']} ring={audit['ring_overlap']} vx={audit['vx']} vy={audit['vy']}",
            f"  lineage={audit['tracker_lineage']}  id_age={audit['seconds_since_id_first_seen']}s  "
            f"since_bridge={audit['seconds_since_last_bridge_into_id']}s",
        ]
        for number, record in enumerate(stored, start=1):
            lines.append(
                f"  sample {number}: {record['state']} conf={record['confidence']} "
                f"p_x={record['p_x']} p_no_x={record['p_no_x']} mode={record['mode']} "
                f"clean={record.get('clean_view')} gate={record.get('gate_reason')} "
                f"center={record['center']} t={record['time_text']} stable={record.get('stable_frames')} "
                f"iou={record.get('box_iou_vs_prev_frame')} size={record.get('size_ratio_vs_prev_frame')}"
            )
        lines.append(f"  movement={movement}  total={total_px}px")
        lines.append(
            f"  deferred unsafe NO_X samples (not counted): {len(deferred)}  "
            f"RULE_VIOLATION={violation if violation else 'none'}"
        )
        lines.append(f"  files: {files}  folder: {audit_folder.name}")

        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write("\\n".join(lines) + "\\n")

        if violation:
            log_event(f"P{track_id} AUDIT RULE VIOLATION: unsafe NO_X counted at {violation}")
    except Exception as error:
        try:
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as handle:
                handle.write(f"AUDIT ERROR P{track_id}: {error!r}\\n")
        except Exception:
            pass  # the audit must never disturb the live system


def should_store_symbol_sample(symbol_state, view_is_clean):  # [P38]''', "audit helpers")

once('''            ) = update_dynamic_symbol_gate(
                track_id,
                track,
                symbol_quality_state,
                manual_recheck=manual_recheck,
            )
''', '''            ) = update_dynamic_symbol_gate(
                track_id,
                track,
                symbol_quality_state,
                manual_recheck=manual_recheck,
            )

            audit_note_frame(  # [P39] read-only
                track_id,
                track,
                clean_view_ready,
                gate_reason,
                symbol_quality_state,
            )
''', "note frame")

once('''                                else:
                                    note_deferred_no_x(track_id)
''', '''                                else:
                                    note_deferred_no_x(track_id)

                                audit_record_sample(  # [P39] read-only
                                    track_id,
                                    symbol_state,
                                    symbol_confidence,
                                    p_x,
                                    p_no_x,
                                    sample_time,
                                    center_x,
                                    center_y,
                                    diagonal,
                                    box,
                                    (
                                        "MANUAL"
                                        if manual_recheck
                                        else (
                                            "FAST"
                                            if (fast_mode and fast_detector_good)
                                            else "CLEAN"
                                        )
                                    ),
                                    should_store_symbol_sample(
                                        symbol_state,
                                        clean_view_ready or manual_recheck,
                                    ),
                                    prepared_symbol_crop,
                                    full_symbol_crop,
                                )
''', "record sample")

once('''                            f"P{track_id} CLEAN {current_decision} | "
                            f"X={x_votes} NO_X={no_x_votes}"
                        )
''', '''                            f"P{track_id} CLEAN {current_decision} | "
                            f"X={x_votes} NO_X={no_x_votes}"
                        )

                # [P39] full audit of every terminal decision and every change
                if (
                    current_decision in ("ACCEPTED", "REJECTED")
                    and decision_before != current_decision
                ):
                    audit_commit(
                        track_id,
                        decision_before,
                        current_decision,
                        fast_mode,
                        clean_view_ready,
                        gate_reason,
                        x_votes,
                        no_x_votes,
                        observations,
                        object_state,
                        object_confidence,
                        track,
                        box,
                        frame,
                    )
''', "commit audit")

once('''        # [P38] diagnostics only: why did a brand-new id receive no state?
''', '''        audit_note_tracks(confirmed_tracks)  # [P39] read-only

        # [P38] diagnostics only: why did a brand-new id receive no state?
''', "track born")

if problems:
    print("NOTHING WAS WRITTEN. Your 38 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("38_live_asymmetric_fast_symbol_test.py was NOT modified.")

"""
Builds 35_live_rfid_identity_mapping_test.py from YOUR local
34_live_rfid_identity_mapping_test.py.

* 34 is only READ, never modified.
* If any expected piece of 34 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P35]".

Phase 35 = Phase 34 with a SIMPLE display:
  * each RFID station is ONE plain single-colour box (no lines, no text)
    -> needs the updated crest_rfid_fusion.py
  * --rfid already gives operator labels: "<number> <STATE>"  e.g. "1 ACCEPTED",
    "2 EMPTY"  ("? <STATE>" until the pallet has passed an RFID reader)
  * rfid_stations.json is read in its real format (box: {x1,y1,x2,y2})
  * one-time warning if the camera resolution differs from the one the boxes
    were calibrated for

Run:  python make_35_from_34.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "34_live_rfid_identity_mapping_test.py"
TARGET = CODE_DIR / "35_live_rfid_identity_mapping_test.py"

if not SOURCE.exists():
    sys.exit(f"Missing source file:\n{SOURCE}")

text = SOURCE.read_text(encoding="utf-8")
problems = []


def once(old, new, label):
    global text

    count = text.count(old)

    if count != 1:
        problems.append(f"[{label}] expected exactly 1 match in 34, found {count}")
        return

    text = text.replace(old, new)


once('''# PHASE 34 (this file) = Phase 32 plus the RFID IDENTITY layer.''',
     '''# PHASE 35 (this file) = Phase 34 with a SIMPLE display: one plain box per RFID
# station, and operator labels "<number> <STATE>" (e.g. "1 ACCEPTED") with --rfid.
# ------------------------------------------------------------
#
# PHASE 34 (this file) = Phase 32 plus the RFID IDENTITY layer.''', "header")

once('''    / "34_Live_RFID_Identity_Mapping"''',
     '''    / "35_Live_RFID_Identity_Mapping"''', "save dir")

once('''"PHASE 34: RFID IDENTITY | KEEP DECISION | R=RECHECK",''',
     '''"PHASE 35: RFID IDENTITY | KEEP DECISION | R=RECHECK",''', "status text")

once('''"CREST - PHASE 34 RFID IDENTITY MAPPING TEST"''',
     '''"CREST - PHASE 35 RFID IDENTITY MAPPING TEST"''', "window title")

once('''    return arguments.rfid, arguments.final
''', '''    # [P35] --rfid already means operator labels ("1 ACCEPTED"); --final alone
    # shows the labels without reading the PLC.
    return arguments.rfid, (arguments.final or arguments.rfid)
''', "cli")

once('''                if (
                    isinstance(box, (list, tuple))
                    and len(box) == 4
                    and all(isinstance(v, (int, float)) for v in box)
                ):''', '''                if isinstance(box, dict) and all(  # [P35] {"x1":..,"y1":..,"x2":..,"y2":..}
                    k in box for k in ("x1", "y1", "x2", "y2")
                ):
                    box = [box["x1"], box["y1"], box["x2"], box["y2"]]

                if (
                    isinstance(box, (list, tuple))
                    and len(box) == 4
                    and all(isinstance(v, (int, float)) for v in box)
                ):''', "json box format")

once('''def display_label(label, track_id):  # [P34]''', '''def read_rfid_frame_size():  # [P35]
    """(width, height) the station boxes were calibrated for, or None."""

    try:
        data = json.loads(RFID_STATIONS_FILE.read_text(encoding="utf-8-sig"))
        return int(data["image_width"]), int(data["image_height"])
    except Exception:
        return None


def display_label(label, track_id):  # [P34]''', "frame size helper")

once('''    check_rfid_station_file(fusion_module)
''', '''    check_rfid_station_file(fusion_module)
    rfid_frame_size = read_rfid_frame_size()  # [P35]
''', "frame size at startup")

once('''            break

        pallet_frame = (''', '''            break

        # [P35] one-time check: the RFID boxes are pixel coordinates
        if rfid_frame_size is not None:
            if (frame.shape[1], frame.shape[0]) != rfid_frame_size:
                print(
                    f"WARNING: camera frame is {frame.shape[1]}x{frame.shape[0]} but the RFID "
                    f"boxes were calibrated for {rfid_frame_size[0]}x{rfid_frame_size[1]} - "
                    f"the station boxes will NOT line up with the conveyor."
                )

            rfid_frame_size = None

        pallet_frame = (''', "resolution check")

once('''        # [P34] station boxes + entry(green)/exit(red) lines (text only in debug)
        FUSION.draw_debug(
            output,
            frame_timestamp,
            labels=not FINAL_LABELS,
        )
''', '''        # [P35] each RFID station = ONE plain single-colour box
        FUSION.draw_debug(
            output,
            frame_timestamp,
        )
''', "simple overlay")

if problems:
    print("NOTHING WAS WRITTEN. Your 34 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("34_live_rfid_identity_mapping_test.py was NOT modified.")

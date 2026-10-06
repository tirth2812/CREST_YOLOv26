"""
Builds 34_live_rfid_identity_mapping_test.py from YOUR local
32_live_full_conveyor_consistency_test.py.

* 32 is only READ, never modified.
* If any expected piece of 32 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P34]".
* Detection, tracking, EMPTY/OBJECT, X/NO_X, hold/bridge logic are NOT touched.

Needs crest_rfid_fusion.py in the same CODE folder (the identity layer).

Run:  python make_34_from_32.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "32_live_full_conveyor_consistency_test.py"
TARGET = CODE_DIR / "34_live_rfid_identity_mapping_test.py"
FUSION = CODE_DIR / "crest_rfid_fusion.py"

if not SOURCE.exists():
    sys.exit(f"Missing source file:\n{SOURCE}")

if not FUSION.exists():
    sys.exit(f"Missing identity layer (copy it first):\n{FUSION}")

text = SOURCE.read_text(encoding="utf-8")
problems = []


def once(old, new, label):
    global text

    count = text.count(old)

    if count != 1:
        problems.append(f"[{label}] expected exactly 1 match in 32, found {count}")
        return

    text = text.replace(old, new)


# ------------------------------------------------------------ header / names
once('''# PHASE 32 (this file) = Phase 31 plus ONE fix: a held (undetected) pallet's''',
     '''# PHASE 34 (this file) = Phase 32 plus the RFID IDENTITY layer.
#   python 34_live_rfid_identity_mapping_test.py           geometry check only
#   python 34_live_rfid_identity_mapping_test.py --rfid    + read PLC, map numbers
#   python 34_live_rfid_identity_mapping_test.py --final   labels: "<number> <STATE>"
# Vision behaviour is unchanged; all added lines are marked "# [P34]".
# ------------------------------------------------------------
#
# PHASE 32 = Phase 31 plus ONE fix: a held (undetected) pallet's''', "header")

once('''    / "32_Live_Full_Conveyor_Consistency"''',
     '''    / "34_Live_RFID_Identity_Mapping"''', "save dir")

once('''"PHASE 32: BOX HOLD | ALL-SIDES | KEEP DECISION | R=RECHECK",''',
     '''"PHASE 34: RFID IDENTITY | KEEP DECISION | R=RECHECK",''', "status text")

once('''"CREST - PHASE 32 FULL-CONVEYOR CONSISTENCY TEST"''',
     '''"CREST - PHASE 34 RFID IDENTITY MAPPING TEST"''', "window title")

once('''import importlib.util
from pathlib import Path
import time
''', '''import argparse  # [P34]
import importlib.util
import json  # [P34]
from pathlib import Path
import sys  # [P34]
import time
''', "imports")

# ------------------------------------------- settings + helpers (module level)
once('''COLOR_UNKNOWN = (255, 255, 255)
''', '''COLOR_UNKNOWN = (255, 255, 255)


# ============================================================
# [P34] RFID IDENTITY LAYER
# ============================================================
# Run modes come from the command line (nothing to edit):
#   (none)   geometry check: station boxes + passage logs, NO PLC reading,
#            normal labels  -> verify which track crosses each station
#   --rfid   also read the PLC (background thread) and map pallet numbers
#   --final  operator labels only: "<number> <STATE>" (e.g. "3 ACCEPTED")

RFID_READER_ENABLED = False
FINAL_LABELS = False
FUSION = None

FUSION_FILE = CODE_DIR / "crest_rfid_fusion.py"
RFID_STATIONS_FILE = SYSTEM_ROOT / "CONFIG" / "rfid_stations.json"
RFID_LOG_FILE = (
    SAVE_DIR
    / f"rfid_identity_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
)

# Wide provisional RFID timing windows, from the reader check on the real
# hardware: a tag stays in the RFID1 field ~1.34 s and in the RFID2 field ~0.42 s
# and the event time is the moment it APPEARS, i.e. up to ~0.7 s before the
# pallet is level with the station box.  Neighbouring pallets are ~3.3 s apart,
# so a wide window cannot confuse them.  Tighten later from the log
# (analyze_rfid_fusion_log.py).
RFID_TIMING = dict(
    rfid_pre_margin_sec=1.5,
    rfid_post_margin_sec=1.0,
    pending_hold_sec=2.5,
    ambiguity_hold_sec=2.5,
    event_max_age_sec=5.0,
)

# Console shows only these (everything goes to the log file).
RFID_CONSOLE_EVENTS = (
    "PASSAGE_OPEN",
    "PASSAGE_EXIT",
    "PASSAGE_EXPIRED",
    "PASSAGE_TRACK_SWITCH",
    "STATION_OVERLAP",
    "RFID_EVENT",
    "RFID_DEBOUNCE_DROP",
    "RFID_PASSAGE_MATCH",
    "RFID_PASSAGE_DUPLICATE",
    "RFID_PASSAGE_AMBIGUOUS",
    "RFID_UNMATCHED",
    "RFID_STALE_DROP",
    "RFID_OFFSET_SAMPLE",
    "RFID_UNKNOWN_TAG",
    "RFID_READER_ERROR",
    "RFID_READER_DISABLED",
    "IDENTITY_ASSIGN",
    "IDENTITY_VERIFY",
    "IDENTITY_REMAP",
    "IDENTITY_CLEAR",
    "IDENTITY_UNBOUND",
    "IDENTITY_REACQUIRE",
)


def parse_rfid_arguments(argv=None):  # [P34]
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfid", action="store_true")
    parser.add_argument("--final", action="store_true")
    arguments, _ = parser.parse_known_args(argv)
    return arguments.rfid, arguments.final


def load_fusion_module():  # [P34]
    spec = importlib.util.spec_from_file_location(
        "crest_rfid_fusion",
        str(FUSION_FILE),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import identity layer:\\n{FUSION_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["crest_rfid_fusion"] = module
    spec.loader.exec_module(module)
    return module


def _find_station_boxes(node, found):  # [P34]
    if isinstance(node, dict):
        for key, value in node.items():
            name = str(key).upper().replace(" ", "")

            if name.startswith("RFID"):
                box = None

                if isinstance(value, dict):
                    for field in ("box", "bbox", "rect", "rectangle"):
                        if field in value:
                            box = value[field]
                            break

                    if box is None and all(k in value for k in ("x1", "y1", "x2", "y2")):
                        box = [value["x1"], value["y1"], value["x2"], value["y2"]]

                elif isinstance(value, (list, tuple)):
                    box = value

                if (
                    isinstance(box, (list, tuple))
                    and len(box) == 4
                    and all(isinstance(v, (int, float)) for v in box)
                ):
                    found[name] = tuple(int(v) for v in box)
                    continue

            _find_station_boxes(value, found)

    elif isinstance(node, list):
        for item in node:
            _find_station_boxes(item, found)


def check_rfid_station_file(fusion_module):  # [P34]
    """Compare CONFIG\\rfid_stations.json with the built-in boxes (warn only)."""

    built_in = {
        name: tuple(config["box"])
        for name, config in fusion_module.DEFAULT_STATIONS.items()
    }

    if not RFID_STATIONS_FILE.exists():
        print(f"RFID stations: {RFID_STATIONS_FILE.name} not found, using built-in {built_in}")
        return

    try:
        data = json.loads(RFID_STATIONS_FILE.read_text(encoding="utf-8-sig"))
    except Exception as error:
        print(f"RFID stations: could not read {RFID_STATIONS_FILE.name}: {error}")
        return

    found = {}
    _find_station_boxes(data, found)

    for name, box in built_in.items():
        if name not in found:
            print(f"RFID stations: {name} box not found in file (using built-in {box})")
        elif found[name] == box:
            print(f"RFID stations: {name} {box} matches {RFID_STATIONS_FILE.name}")
        else:
            print(
                f"RFID stations: WARNING {name} file={found[name]} "
                f"built-in={box} (built-in is used)"
            )


def display_label(label, track_id):  # [P34]
    """Draw-time label only.  The raw label (used for the counts) is untouched."""

    if FUSION is None:
        return label

    physical_id = FUSION.get_physical_id(track_id)

    if FINAL_LABELS:
        state_word = ""

        for word in ("EMPTY", "ACCEPTED", "REJECTED", "CHECKING", "UNKNOWN"):
            if word in label:
                state_word = word
                break

        number = str(physical_id) if physical_id is not None else "?"
        return f"{number} {state_word}".strip()

    prefix = f"P{track_id} "

    if physical_id is not None and label.startswith(prefix):
        return f"{physical_id} [P{track_id}] " + label[len(prefix):]

    return label
''', "settings + helpers")

# ------------------------------------------------ Phase 27 bridge -> identity
once('''def track_can_receive_state(  # [P31]''', '''# [P34] Tell the identity layer whenever the PROVEN Phase 27 bridge moves a state
# from an old tracker id to a new one (strict and second-chance both go through
# move_track_state).  The original function is called unchanged first.
_phase27_move_track_state = move_track_state


def move_track_state(old_track_id, new_track_id, *args, **kwargs):  # [P34]
    _phase27_move_track_state(old_track_id, new_track_id, *args, **kwargs)

    if FUSION is not None:
        FUSION.notify_bridge(old_track_id, new_track_id)


def track_can_receive_state(  # [P31]''', "bridge hook")

# ----------------------------------------------------------- main(): startup
once('''    tracker = (
        pallet_stage.DynamicTracker()
    )

    object_history = {}''', '''    tracker = (
        pallet_stage.DynamicTracker()
    )

    # [P34] identity layer (+ optional background RFID reader thread)
    global FUSION, RFID_READER_ENABLED, FINAL_LABELS

    RFID_READER_ENABLED, FINAL_LABELS = parse_rfid_arguments()

    fusion_module = load_fusion_module()
    check_rfid_station_file(fusion_module)

    FUSION = fusion_module.FusionIdentityManager(
        config=fusion_module.FusionConfig(
            log_path=str(RFID_LOG_FILE),
            log_echo=not FINAL_LABELS,
            log_echo_events=RFID_CONSOLE_EVENTS,
            **RFID_TIMING,
        ),
    )

    rfid_reader = None

    if RFID_READER_ENABLED:
        rfid_reader = fusion_module.RFIDReaderWorker(
            out_queue=FUSION.event_queue,
            log=FUSION.log,
        )
        rfid_reader.start()

    print(
        f"RFID identity layer: reader={'ON' if RFID_READER_ENABLED else 'OFF (geometry check)'} "
        f"| labels={'FINAL' if FINAL_LABELS else 'debug'} | log: {RFID_LOG_FILE}"
    )

    object_history = {}''', "main startup")

once('''        success, frame = (
            cap.read()
        )

        if not success:''', '''        success, frame = (
            cap.read()
        )

        # [P34] timestamp at frame acquisition (read() just returned), same
        # monotonic clock as the RFID thread; NOT after inference.
        frame_timestamp = time.monotonic()

        if not success:''', "frame timestamp")

once('''            fast_symbol_start_after,
            fast_symbol_until,
        )

        output = (
            symbol_frame.copy()
        )
''', '''            fast_symbol_start_after,
            fast_symbol_until,
        )

        # [P34] identity layer sees the FINAL post-duplicate-filter tracks, after
        # the bridge.  It never changes any vision state.
        FUSION.update_tracks(
            frame_timestamp,
            confirmed_tracks,
        )

        FUSION.process_rfid_events(
            time.monotonic(),
        )

        output = (
            symbol_frame.copy()
        )
''', "fusion update")

once('''                data[
                    "label"
                ],
                data[
                    "color"
                ],
                thickness=(''', '''                display_label(  # [P34] draw-time only; counts use the raw label
                    data[
                        "label"
                    ],
                    track_id,
                ),
                data[
                    "color"
                ],
                thickness=(''', "draw label")

once('''        cv2.imshow(
            window_name,
            output,
        )
''', '''        # [P34] station boxes + entry(green)/exit(red) lines (text only in debug)
        FUSION.draw_debug(
            output,
            frame_timestamp,
            labels=not FINAL_LABELS,
        )

        cv2.imshow(
            window_name,
            output,
        )
''', "overlay")

once('''    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":''', '''    # [P34]
    if rfid_reader is not None:
        rfid_reader.stop()

    FUSION.log.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":''', "shutdown")

# ------------------------------------------------------------------ write
if problems:
    print("NOTHING WAS WRITTEN. Your 32 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("32_live_full_conveyor_consistency_test.py was NOT modified.")

"""
Builds 36_live_rfid_identity_mapping_test.py from YOUR local
35_live_rfid_identity_mapping_test.py.

* 35 is only READ, never modified.
* If any expected piece of 35 is not found exactly once, NOTHING is written.
* Every inserted/changed line in the new file is marked "# [P36]".
* Needs the updated crest_rfid_fusion.py in the same CODE folder.

Phase 36 adds a CALIBRATION mode.  The RFID number is given to the pallet that
the CAMERA sees at the reader at the right moment.  The camera sees things a
little (or a lot) later than the RFID reports them, and the antenna may not be
exactly where the box is drawn.  Instead of guessing, run ONE pallet around the
conveyor:

    python 36_live_rfid_identity_mapping_test.py --calibrate

It measures the delay of each reader, saves CONFIG\\rfid_timing.json and prints a
summary.  Afterwards the normal run uses that file automatically:

    python 36_live_rfid_identity_mapping_test.py --rfid

Run:  python make_36_from_35.py
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
SOURCE = CODE_DIR / "35_live_rfid_identity_mapping_test.py"
TARGET = CODE_DIR / "36_live_rfid_identity_mapping_test.py"

if not SOURCE.exists():
    sys.exit(f"Missing source file:\n{SOURCE}")

text = SOURCE.read_text(encoding="utf-8")
problems = []


def once(old, new, label):
    global text

    count = text.count(old)

    if count != 1:
        problems.append(f"[{label}] expected exactly 1 match in 35, found {count}")
        return

    text = text.replace(old, new)


once('''# PHASE 35 (this file) = Phase 34 with a SIMPLE display:''',
     '''# PHASE 36 (this file) = Phase 35 plus a CALIBRATION mode (--calibrate, ONE pallet)
# that measures how much later the camera sees a pallet at each RFID reader than the
# reader reports it; the measured delay (CONFIG\\rfid_timing.json) is then used.
# ------------------------------------------------------------
#
# PHASE 35 (this file) = Phase 34 with a SIMPLE display:''', "header")

once('''    / "35_Live_RFID_Identity_Mapping"''',
     '''    / "36_Live_RFID_Identity_Mapping"''', "save dir")

once('''"PHASE 35: RFID IDENTITY | KEEP DECISION | R=RECHECK",''',
     '''"PHASE 36: RFID IDENTITY | KEEP DECISION | R=RECHECK",''', "status text")

once('''"CREST - PHASE 35 RFID IDENTITY MAPPING TEST"''',
     '''"CREST - PHASE 36 RFID IDENTITY MAPPING TEST"''', "window title")

once('''FUSION_FILE = CODE_DIR / "crest_rfid_fusion.py"
''', '''FUSION_FILE = CODE_DIR / "crest_rfid_fusion.py"
RFID_TIMING_FILE = SYSTEM_ROOT / "CONFIG" / "rfid_timing.json"  # [P36]
''', "timing file")

once('''def parse_rfid_arguments(argv=None):  # [P34]''', '''def parse_calibrate_flag(argv=None):  # [P36]
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    arguments, _ = parser.parse_known_args(argv)
    return arguments.calibrate


def parse_rfid_arguments(argv=None):  # [P34]''', "calibrate flag")

once('''    RFID_READER_ENABLED, FINAL_LABELS = parse_rfid_arguments()
''', '''    RFID_READER_ENABLED, FINAL_LABELS = parse_rfid_arguments()

    CALIBRATE = parse_calibrate_flag()  # [P36]

    if CALIBRATE:
        RFID_READER_ENABLED = True
''', "main flags")

once('''            log_echo=True,  # [P35] keep the RFID / identity event lines in the console
''', '''            log_echo=True,  # [P35] keep the RFID / identity event lines in the console
            calibrate=CALIBRATE,  # [P36] measure the camera delay, assign nothing
            vision_delay_sec=fusion_module.load_vision_delay(RFID_TIMING_FILE),  # [P36]
''', "config")

once('''        f"| labels={'FINAL' if FINAL_LABELS else 'debug'} | log: {RFID_LOG_FILE}"
    )
''', '''        f"| labels={'FINAL' if FINAL_LABELS else 'debug'} | log: {RFID_LOG_FILE}"
    )

    # [P36]
    if CALIBRATE:
        print(
            "CALIBRATION MODE: put exactly ONE pallet on the conveyor and let it run "
            "3+ laps. Stop with q / Esc. Numbers are NOT assigned in this mode."
        )
    else:
        print(
            "RFID camera delay per reader (s): "
            f"{FUSION.cfg.vision_delay_sec or 'none yet - run once with --calibrate and ONE pallet'}"
        )
''', "startup message")

once('''    FUSION.log.close()
''', '''    # [P36] save what the calibration measured
    if CALIBRATE:
        calibration = FUSION.write_calibration(RFID_TIMING_FILE)

        if calibration:
            print("CALIBRATION RESULT (saved to " + str(RFID_TIMING_FILE) + "):")
            for reader, values in calibration.items():
                print(f"   {reader}: {values}")
        else:
            print("CALIBRATION: no usable samples - see the RFID_CALIBRATION_SKIP lines above.")

    FUSION.log.close()
''', "shutdown")

if problems:
    print("NOTHING WAS WRITTEN. Your 35 file differs from the one this script expects:")
    for problem in problems:
        print("  - " + problem)
    sys.exit(1)

TARGET.write_text(text, encoding="utf-8")
print(f"Created: {TARGET}")
print("35_live_rfid_identity_mapping_test.py was NOT modified.")

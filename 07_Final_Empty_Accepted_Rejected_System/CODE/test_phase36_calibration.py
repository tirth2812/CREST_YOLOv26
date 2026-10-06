"""Offline checks for 36_live_rfid_identity_mapping_test.py:  python test_phase36_calibration.py"""
import importlib.util, json, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
for n in ("cv2", "ultralytics"): sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["cv2"].createCLAHE = lambda **k: object(); sys.modules["ultralytics"].YOLO = object
import numpy  # noqa
def load(label, filename):
    s = importlib.util.spec_from_file_location(label, HERE / filename)
    m = importlib.util.module_from_spec(s); sys.modules[label] = m; s.loader.exec_module(m); return m
load("dyn_tracker", "dynamic_live_pallet_detection.py")
P36 = load("p36", "36_live_rfid_identity_mapping_test.py")
F = P36.load_fusion_module()

def test_flags():
    assert P36.parse_calibrate_flag([]) is False and P36.parse_calibrate_flag(["--calibrate"]) is True
    assert P36.parse_rfid_arguments(["--calibrate"]) == (False, False)      # --calibrate alone does not change labels
    assert P36.parse_rfid_arguments(["--rfid"]) == (True, True)

def test_config_built_exactly_like_main_does():
    path = Path(tempfile.gettempdir()) / "rfid_timing_p36.json"
    path.write_text(json.dumps({"RFID1": {"vision_delay_sec": 2.2}, "RFID2": {"vision_delay_sec": 2.1, "samples": 4}}))
    delays = F.load_vision_delay(path)
    cfg = F.FusionConfig(log_path=None, log_echo=False, log_echo_events=P36.RFID_TIMING and P36.RFID_CONSOLE_EVENTS,
                         calibrate=False, vision_delay_sec=delays, **P36.RFID_TIMING)
    fusion = F.FusionIdentityManager(config=cfg)
    assert fusion.cfg.vision_delay_sec == {"RFID1": 2.2, "RFID2": 2.1} and fusion.cfg.match_mode == "closest"
    assert P36.RFID_TIMING_FILE.name == "rfid_timing.json"
    assert F.load_vision_delay(Path(tempfile.gettempdir()) / "nope.json") == {}

def test_calibration_file_roundtrip_and_reuse():
    fusion = F.FusionIdentityManager(F.FusionConfig(log_echo=False, calibrate=True))
    fusion._calibration["RFID1"] = [{"delay_sec": 2.0, "dwell_sec": 1.3}, {"delay_sec": 2.2, "dwell_sec": 1.3}, {"delay_sec": 2.1, "dwell_sec": 1.4}]
    path = Path(tempfile.gettempdir()) / "rfid_timing_rt.json"
    path.unlink(missing_ok=True)
    out = fusion.write_calibration(path)
    assert out["RFID1"]["vision_delay_sec"] == 2.1 and "RFID2" not in out
    assert F.load_vision_delay(path) == {"RFID1": 2.1}
    empty = F.FusionIdentityManager(F.FusionConfig(log_echo=False))
    path2 = Path(tempfile.gettempdir()) / "rfid_timing_empty.json"; path2.unlink(missing_ok=True)
    assert empty.write_calibration(path2) == {} and not path2.exists(), "no samples -> no file"

def test_vision_untouched_vs_35():
    a = [l for l in (HERE / "35_live_rfid_identity_mapping_test.py").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    b = set((HERE / "36_live_rfid_identity_mapping_test.py").read_text(encoding="utf-8-sig").splitlines())
    removed = [l.strip() for l in a if l not in b]
    print("   lines of 35 not in 36:", removed)
    assert len(removed) <= 6

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for n, f in tests:
        try: f()
        except Exception as e:
            bad += 1; import traceback; print("FAIL", n, repr(e)); traceback.print_exc(limit=4)
    print(f"{len(tests) - bad}/{len(tests)} passed"); sys.exit(1 if bad else 0)

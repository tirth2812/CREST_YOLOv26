"""Offline checks for 35_live_rfid_identity_mapping_test.py:  python test_phase35_display.py"""
import contextlib, importlib.util, io, json, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
for n in ("cv2", "ultralytics"): sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["cv2"].createCLAHE = lambda **k: object(); sys.modules["ultralytics"].YOLO = object
import numpy  # noqa

def load(label, filename):
    s = importlib.util.spec_from_file_location(label, HERE / filename)
    m = importlib.util.module_from_spec(s); sys.modules[label] = m; s.loader.exec_module(m); return m

load("dyn_tracker", "dynamic_live_pallet_detection.py")
P35 = load("p35", "35_live_rfid_identity_mapping_test.py")
F = P35.load_fusion_module()

YOUR_JSON = {"image_width": 1920, "image_height": 1080,
  "rfid1": {"name": "RFID1", "location": "TOP", "direction": "RIGHT_TO_LEFT", "box": {"x1": 1123, "y1": 578, "x2": 1183, "y2": 675}},
  "rfid2": {"name": "RFID2", "location": "BOTTOM", "direction": "LEFT_TO_RIGHT", "box": {"x1": 823, "y1": 734, "x2": 884, "y2": 815}}}

def with_file(content):
    path = Path(tempfile.gettempdir()) / "rfid_stations_p35.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    P35.RFID_STATIONS_FILE = path

def test_your_rfid_stations_json_is_read_and_matches():
    with_file(YOUR_JSON)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): P35.check_rfid_station_file(F)
    out = buf.getvalue()
    print("\n1)", out.strip().splitlines())
    assert out.count("matches") == 2 and "WARNING" not in out
    assert P35.read_rfid_frame_size() == (1920, 1080)
    bad = json.loads(json.dumps(YOUR_JSON)); bad["rfid1"]["box"]["x1"] = 1100
    with_file(bad)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): P35.check_rfid_station_file(F)
    assert "WARNING RFID1" in buf.getvalue()

def test_cli_rfid_implies_operator_labels():
    assert P35.parse_rfid_arguments([]) == (False, False)
    assert P35.parse_rfid_arguments(["--rfid"]) == (True, True)
    assert P35.parse_rfid_arguments(["--final"]) == (False, True)

def test_station_is_one_plain_box_no_lines_no_text():
    calls = []
    cv = types.ModuleType("cv2")
    cv.FONT_HERSHEY_SIMPLEX = 0
    for name in ("rectangle", "line", "putText", "circle", "polylines"):
        setattr(cv, name, lambda *a, _n=name, **k: calls.append((_n, a, k)))
    sys.modules["cv2"] = cv
    P35.FUSION = F.FusionIdentityManager(F.FusionConfig(log_echo=False))
    P35.FUSION.draw_debug(object(), 0.0)                      # exactly how file 35 calls it
    kinds = [c[0] for c in calls]
    colours = {c[1][3] for c in calls}
    boxes = [c[1][1:3] for c in calls]
    print("\n2) drawing calls per frame:", kinds, "| colours:", colours, "| boxes:", boxes)
    assert kinds == ["rectangle", "rectangle"] and len(colours) == 1
    assert boxes == [((1123, 578), (1183, 675)), ((823, 734), (884, 815))]
    sys.modules["cv2"] = types.ModuleType("cv2"); sys.modules["cv2"].createCLAHE = lambda **k: object()

def test_pallet_labels_are_number_and_state_only():
    P35.FUSION = F.FusionIdentityManager(F.FusionConfig(log_echo=False))
    f = P35.FUSION
    P35.FINAL_LABELS = True
    t1 = F.Trajectory(1, 41, (0, 0, 130, 120), 0.0, f.stations); f.trajectories[1] = t1; f.track_to_traj[41] = t1; f._bind(t1, 1, "t")
    t2 = F.Trajectory(2, 52, (200, 0, 330, 120), 0.0, f.stations); f.trajectories[2] = t2; f.track_to_traj[52] = t2; f._bind(t2, 2, "t")
    t3 = F.Trajectory(3, 63, (400, 0, 530, 120), 0.0, f.stations); f.trajectories[3] = t3; f.track_to_traj[63] = t3
    got = [P35.display_label("P41 ACCEPTED NOX:3/3 HOLD", 41), P35.display_label("P52 EMPTY 0.98", 52), P35.display_label("P63 REJECTED X:3/3", 63)]
    print("\n3) labels:", got)
    assert got == ["1 ACCEPTED", "2 EMPTY", "? REJECTED"]
    P35.FINAL_LABELS = False; P35.FUSION = None

def test_resolution_check_present_and_vision_untouched():
    src = (HERE / "35_live_rfid_identity_mapping_test.py").read_text(encoding="utf-8-sig")
    assert "rfid_frame_size = None" in src and "will NOT line up" in src
    assert "log_echo=True,  # [P35]" in src and "log_echo=not FINAL_LABELS" not in src
    a = (HERE / "34_live_rfid_identity_mapping_test.py").read_text(encoding="utf-8-sig").splitlines()
    b = set((HERE / "35_live_rfid_identity_mapping_test.py").read_text(encoding="utf-8-sig").splitlines())
    removed = [l.strip() for l in a if l.strip() and l not in b]
    print("\n4) lines of 34 not in 35:", removed)
    assert len(removed) <= 9

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for n, f in tests:
        try: f()
        except Exception as e:
            bad += 1; import traceback; print("FAIL", n, repr(e)); traceback.print_exc(limit=4)
    print(f"\n{len(tests) - bad}/{len(tests)} passed"); sys.exit(1 if bad else 0)

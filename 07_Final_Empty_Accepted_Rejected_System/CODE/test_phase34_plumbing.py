"""
Offline checks for 34_live_rfid_identity_mapping_test.py (no camera / GPU / PLC):
    python test_phase34_plumbing.py
"""
import contextlib, importlib.util, io, json, sys, tempfile, types
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
for n in ("cv2", "ultralytics"):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["cv2"].createCLAHE = lambda **k: object()
sys.modules["ultralytics"].YOLO = object
import numpy  # noqa: F401

def load(label, filename):
    s = importlib.util.spec_from_file_location(label, HERE / filename)
    m = importlib.util.module_from_spec(s); sys.modules[label] = m; s.loader.exec_module(m); return m

trk = load("dyn_tracker", "dynamic_live_pallet_detection.py")
P32 = load("p32", "32_live_full_conveyor_consistency_test.py")
P34 = load("p34", "34_live_rfid_identity_mapping_test.py")
fusion_mod = P34.load_fusion_module()
DT = 1 / 30.0

class Clock:
    t = 7000.0
    def monotonic(self): return self.t
    def time(self): return self.t
    def strftime(self, *a): return "00:00:00 "

def box_at(cx, cy=626.0, w=130, h=120):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

def new_fusion():
    return fusion_mod.FusionIdentityManager(fusion_mod.FusionConfig(log_echo=False, **P34.RFID_TIMING))

def test_cli_modes_need_no_editing():
    assert P34.parse_rfid_arguments([]) == (False, False)
    assert P34.parse_rfid_arguments(["--rfid"]) == (True, False)
    assert P34.parse_rfid_arguments(["--rfid", "--final"]) == (True, True)
    assert P34.parse_rfid_arguments(["--final", "--other"]) == (False, True)

def test_labels_final_and_debug():
    P34.FUSION = new_fusion()
    f = P34.FUSION
    traj = fusion_mod.Trajectory(1, 57, box_at(1000), 0.0, f.stations); f.trajectories[1] = traj; f.track_to_traj[57] = traj
    raw = "P57 CHECKING X:1 NOX:0 1/3 STABILIZE-2/5"
    # unknown number
    P34.FINAL_LABELS = False
    assert P34.display_label(raw, 57) == raw
    P34.FINAL_LABELS = True
    assert P34.display_label("P57 ACCEPTED NOX:3/3 HOLD", 57) == "? ACCEPTED"
    f._bind(traj, 3, "test")
    cases = {"P57 ACCEPTED NOX:3/3 HOLD": "3 ACCEPTED", "P57 REJECTED X:3/3": "3 REJECTED",
             "P57 EMPTY 0.97": "3 EMPTY", raw: "3 CHECKING", "P57 RECHECKING": "3 CHECKING", "P57 UNKNOWN 0.00": "3 UNKNOWN"}
    for label, expected in cases.items():
        assert P34.display_label(label, 57) == expected, (label, P34.display_label(label, 57))
    final = P34.display_label("P57 ACCEPTED NOX:3/3 HOLD", 57)
    for banned in ("P57", "NOX", "HOLD", "conf", "/", "RFID"):
        assert banned not in final
    P34.FINAL_LABELS = False
    assert P34.display_label("P57 ACCEPTED NOX:3/3", 57) == "3 [P57] ACCEPTED NOX:3/3"
    assert P34.display_label("P99 EMPTY", 99) == "P99 EMPTY"
    P34.FUSION = None
    assert P34.display_label("P57 ACCEPTED", 57) == "P57 ACCEPTED"
    print("\n1) final label examples:", list(cases.values())[:3], "| raw label (used for counts) is never modified")

def test_station_file_check_variants():
    out = {}
    def run(content):
        path = Path(tempfile.gettempdir()) / "rfid_stations_test.json"
        if content is None: path.unlink(missing_ok=True)
        else: path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8-sig")
        P34.RFID_STATIONS_FILE = path
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): P34.check_rfid_station_file(fusion_mod)
        return buf.getvalue()
    good = {"RFID1": {"box": [1123, 578, 1183, 675]}, "RFID2": {"box": [823, 734, 884, 815]}}
    o = run(good); assert o.count("matches") == 2, o
    o = run({"image": {"w": 1920, "h": 1080}, "stations": {"RFID1": {"bbox": [1123, 578, 1183, 675], "direction": "RIGHT_TO_LEFT"},
                                                        "RFID2": {"x1": 823, "y1": 734, "x2": 884, "y2": 815}}}); assert o.count("matches") == 2, o
    o = run({"RFID1": {"box": [1100, 578, 1183, 675]}, "RFID2": {"box": [823, 734, 884, 815]}}); assert "WARNING RFID1" in o and "RFID2" in o and "matches" in o
    o = run({"RFID1": {"box": [1123, 578, 1183, 675]}}); assert "RFID2 box not found" in o
    o = run("{not json"); assert "could not read" in o
    o = run(None); assert "not found" in o
    print("\n2) station file check: match / nested / mismatch / missing station / bad json / no file -> all handled, warn only")

def test_bridge_hook_moves_identity_and_leaves_vision_state_alone():
    P34.time = clock = Clock()
    P34.FUSION = f = new_fusion()
    f.update_tracks(clock.t, [{"track_id": 41, "box": box_at(1000)}])
    f._bind(f.track_to_traj[41], 3, "test")
    od, oh, sh, sd, ss, fa, fu = {41: "OBJECT_PRESENT"}, {41: deque([1])}, {41: deque([("NO_X", .9)])}, {41: "ACCEPTED"}, {41: {}}, {}, {}
    cache = {41: {"box": box_at(1000), "missed": 3, "label": "P41 ACCEPTED", "color": 0, "t_seen": clock.t - 0.2}}
    f.update_tracks(clock.t + 0.2, [])                                   # P41 lost
    clock.t += 0.2
    tracks = [{"track_id": 57, "box": box_at(1000)}]
    P34.bridge_reacquired_tracks(tracks, {57}, cache, oh, od, sh, sd, ss, fa, fu)   # Phase 27 bridge (strict)
    assert sd.get(57) == "ACCEPTED" and 41 not in sd, "vision state transfer unchanged"
    f.update_tracks(clock.t, tracks)
    assert f.get_physical_id(57) == 3 and f.registry[3].trajectory is f.track_to_traj[57]
    print("\n3) Phase 27 bridge P41 -> P57: vision state moved AND physical id 3 followed")
    P34.FUSION = None

def test_end_to_end_dropout_with_real_tracker_and_bridge():
    print("\n4) real tracker + real Phase 27/31 bridge + identity layer: number 3 through a dropout in the open")
    print("   dropout | physical number after return")
    for gap in (3, 8, 18, 30, 45):
        P34.time = clock = Clock()
        P34.FUSION = f = new_fusion()
        tracker = trk.DynamicTracker()
        cache, oh, od, sh, sd, ss, fa, fu = {}, {}, {}, {}, {}, {}, {}, {}
        x, first, assigned = 1500.0, None, False
        result = None
        for frame in range(40 + gap + 70):
            clock.t += DT
            blind = 40 <= frame < 40 + gap
            w = 90 if (frame >= 40 + gap and frame < 40 + gap + 6) else 130            # box partly cut off on return
            dets = [] if blind else [{"box": box_at(x, w=w, h=int(120 * w / 130)), "confidence": .9, "overlap": 1.0}]
            x -= 4.0
            tracks = tracker.update(dets); current = {t["track_id"] for t in tracks}
            for tid in list(cache):
                cache[tid]["missed"] += 1
                if tid not in current: P34.advance_held_box(cache[tid], 1920, 1080)
            P34.bridge_reacquired_tracks(tracks, current, cache, oh, od, sh, sd, ss, fa, fu)
            f.update_tracks(clock.t, tracks)
            for t in tracks:
                tid = t["track_id"]
                if first is None:
                    first = tid; od[tid] = "OBJECT_PRESENT"; sd[tid] = "ACCEPTED"
                    oh[tid] = deque([("OBJECT_PRESENT", .9)] * 5, maxlen=7); sh[tid] = deque([("NO_X", .9)] * 3, maxlen=3)
                oh.setdefault(tid, deque(maxlen=7))
                cache[tid] = {"box": t["box"], "missed": 0, "label": "x", "color": 0, "vx": t["velocity_x"], "vy": t["velocity_y"], "t_seen": clock.t}
            if frame == 20 and tracks:                                       # RFID said: this pallet is number 3
                f._bind(f.track_to_traj[tracks[0]["track_id"]], 3, "rfid")
            for tid in list(cache):
                if clock.t - cache[tid]["t_seen"] > P34.TRACK_STATE_MEMORY_SECONDS:
                    del cache[tid]
                    for d in (oh, od, sh, sd): d.pop(tid, None)
        live = tracks[0]["track_id"] if tracks else None
        result = (f.get_physical_id(live) if live else None, sd.get(live))
        print(f"   {gap:5d}f  | number={result[0]}  state={result[1]}")
        assert result == (3, "ACCEPTED"), f"gap {gap}: {result}"
    P34.FUSION = None

def test_vision_code_paths_unchanged():
    # the only differences between 32 and 34 are the [P34] lines + renamed labels
    a = [l for l in (HERE / "32_live_full_conveyor_consistency_test.py").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    b = (HERE / "34_live_rfid_identity_mapping_test.py").read_text(encoding="utf-8-sig").splitlines()
    removed = [l for l in a if l not in set(b)]
    print("\n5) lines of Phase 32 that are not present in 34 (expected: names / label call only):")
    for l in removed: print("     ", l.strip()[:90])
    assert len(removed) <= 6

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for n, f in tests:
        try: f()
        except Exception as e:
            bad += 1; import traceback; print("FAIL", n, repr(e)); traceback.print_exc(limit=4)
    print(f"\n{len(tests) - bad}/{len(tests)} passed"); sys.exit(1 if bad else 0)

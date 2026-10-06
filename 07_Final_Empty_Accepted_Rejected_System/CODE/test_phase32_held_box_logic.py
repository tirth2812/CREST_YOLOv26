"""
Offline checks for the Phase 32 held-box logic (no camera/GPU/models).
Real DynamicTracker; Phase 31 vs Phase 32 side by side:   python test_phase32_held_box_logic.py
"""
import importlib.util, sys, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for n in ("cv2", "ultralytics"):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["cv2"].createCLAHE = lambda **k: object()
sys.modules["ultralytics"].YOLO = object

def load(label, filename):
    s = importlib.util.spec_from_file_location(label, HERE / filename)
    m = importlib.util.module_from_spec(s); sys.modules[label] = m; s.loader.exec_module(m); return m

trk = load("dyn_tracker", "dynamic_live_pallet_detection.py")
P31 = load("p31", "31_live_full_conveyor_consistency_test.py")
P32 = load("p32", "32_live_full_conveyor_consistency_test.py")
DT = 1 / 30.0

class Clock:
    t = 9000.0
    def monotonic(self): return self.t
    def time(self): return self.t
    def strftime(self, *a): return ""

def box_at(cx, cy, w=130, h=120):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

def det(cx, cy):
    return {"box": box_at(cx, cy), "confidence": .9, "overlap": 1.0}

def ring_mask(w=1920, h=1080):
    m = np.zeros((h, w), np.uint8); m[500:900, 200:1700] = 255; return m   # simple ring band

def belt_state(tracks, previous):
    if tracks:
        return any(float(np.hypot(t.get("velocity_x", 0), t.get("velocity_y", 0))) >= 0.60 for t in tracks)
    return previous

def stop_scenario(module, version):
    """Two pallets run, the belt is stopped by hand at frame 40 and pallet A is not detected from then on."""
    module.time = clock = Clock()
    tracker = trk.DynamicTracker(); cache = {}; moving = True; mask = ring_mask()
    xa, xb = 1000.0, 1400.0
    start = None
    for frame in range(100):
        clock.t += DT
        stopped = frame >= 40
        if not stopped:
            xa -= 4.0; xb -= 4.0
        dets = [det(xb, 700)] + ([] if stopped else [det(xa, 700)])      # A vanishes at the stop
        tracks = tracker.update(dets); current = {t["track_id"] for t in tracks}
        moving = belt_state(tracks, moving)
        for tid in list(cache):
            cache[tid]["missed"] += 1
            if tid not in current:
                if version == 31: module.advance_held_box(cache[tid], 1920, 1080)
                else: module.advance_held_box(cache[tid], 1920, 1080, belt_moving=moving, ring_mask=mask)
        for t in tracks:
            cache[t["track_id"]] = {"box": t["box"], "missed": 0, "label": "x", "color": 0,
                                    "vx": t["velocity_x"], "vy": t["velocity_y"], "t_seen": clock.t}
        held = [c for c in cache.values() if c["missed"] > 0]
        if frame == 40 and held == [] : pass
        if held and start is None:
            start = held[0]["box"][0]
    final = [c for c in cache.values() if c["missed"] > 0]
    return (start, final[0]["box"][0]) if final else (start, None)

def test_held_box_does_not_slide_when_belt_is_stopped():
    s31, e31 = stop_scenario(P31, 31)
    s32, e32 = stop_scenario(P32, 32)
    print(f"\n1) belt stopped, pallet A undetected for ~2 s: held box x first={s31} -> last")
    print(f"   Phase 31 slid {abs(e31 - s31):4d} px   |   Phase 32 slid {abs(e32 - s32):4d} px")
    assert abs(e31 - s31) >= 60, "scenario should reproduce the slide in Phase 31"
    assert abs(e32 - s32) <= 12, "Phase 32 box must stay (only the few frames before the stop is detected)"

def test_box_still_follows_a_moving_belt():
    P32.time = Clock()
    entry = {"box": box_at(1000, 700), "missed": 1, "vx": -4.0, "vy": 0.0}
    for _ in range(10):
        P32.advance_held_box(entry, 1920, 1080, belt_moving=True, ring_mask=ring_mask())
    cx = (entry["box"][0] + entry["box"][2]) / 2
    print(f"\n2) belt running: held box follows the pallet: x={cx:.0f} (true {1000 - 40})")
    assert abs(cx - 960) < 6

def test_held_box_never_leaves_the_conveyor():
    mask = ring_mask()
    entry = {"box": box_at(260, 700), "missed": 1, "vx": -4.0, "vy": 0.0}     # heading out of the ring on the left
    for _ in range(60):
        P32.advance_held_box(entry, 1920, 1080, belt_moving=True, ring_mask=mask)
    cx = (entry["box"][0] + entry["box"][2]) // 2; cy = (entry["box"][1] + entry["box"][3]) // 2
    print(f"\n3) box heading off the conveyor stops at its edge: centre=({cx},{cy}), inside ring: {mask[cy, cx] > 0}")
    assert mask[cy, cx] > 0
    entry2 = {"box": box_at(1000, 520), "missed": 1, "vx": 0.0, "vy": -4.0}   # heading out of the top
    for _ in range(60):
        P32.advance_held_box(entry2, 1920, 1080, belt_moving=True, ring_mask=mask)
    cy2 = (entry2["box"][1] + entry2["box"][3]) // 2
    assert mask[cy2, 1000] > 0

def test_belt_state_logic():
    mk = lambda v: {"velocity_x": v, "velocity_y": 0.0}
    assert belt_state([mk(-4.0), mk(0.0)], False) is True       # one moving pallet = belt running
    assert belt_state([mk(0.1), mk(-0.2)], True) is False       # all (nearly) still = stopped
    assert belt_state([], True) is True and belt_state([], False) is False   # nothing visible: keep last answer
    print("\n4) belt state: any pallet moving -> running; all still -> stopped; none visible -> unchanged")

def test_old_call_signature_still_works():
    entry = {"box": box_at(1000, 700), "missed": 1, "vx": -4.0, "vy": 0.0}
    P32.advance_held_box(entry, 1920, 1080)          # defaults = Phase 31 behaviour (used by older tests)
    assert entry["box"][0] < box_at(1000, 700)[0]

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for n, f in tests:
        try: f()
        except Exception as e:
            bad += 1; import traceback; print("FAIL", n, repr(e)); traceback.print_exc(limit=3)
    print(f"\n{len(tests) - bad}/{len(tests)} passed"); sys.exit(1 if bad else 0)

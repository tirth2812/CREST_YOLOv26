"""
Offline checks for 30_live_full_conveyor_consistency_test.py (no camera/GPU/models).
Runs the REAL Phase 27 and Phase 30 functions side by side:
  python test_phase30_consistency.py
"""
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

for name in ("cv2", "ultralytics"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["cv2"].createCLAHE = lambda **k: object()
sys.modules["ultralytics"].YOLO = object
import numpy  # noqa: E402,F401


def load(label, filename):
    spec = importlib.util.spec_from_file_location(label, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    spec.loader.exec_module(module)
    return module


tracker_mod = load("dyn_tracker", "dynamic_live_pallet_detection.py")
P27 = load("p27", "27_live_fast_new_object_final.py")
P30 = load("p30", "30_live_full_conveyor_consistency_test.py")

FPS = 30.0
DT = 1.0 / FPS


class Clock:
    def __init__(self):
        self.t = 5000.0

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def strftime(self, *a):
        return "x"


def box_at(cx, cy=616.0, w=130, h=120):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def detection(cx, cy=616.0):
    return {"box": box_at(cx, cy), "confidence": 0.9, "overlap": 1.0}


# ---------------------------------------------------------------------
# 1. BOX FLICKER: real tracker + dropout, Phase 27 rule vs Phase 30 helpers
# ---------------------------------------------------------------------
def run_display(module, version, gap_frames, direction=-1, total=120):
    clock = Clock()
    module.time = clock
    tracker = tracker_mod.DynamicTracker()
    cache = {}
    x = 1400.0 if direction < 0 else 300.0
    blank_frames = 0
    ids_shown = set()
    first_drop = 30
    shown_history = []

    for frame in range(total):
        clock.t += DT
        dets = [] if first_drop <= frame < first_drop + gap_frames else [detection(x)]
        x += direction * 4.0
        tracks = tracker.update(dets)
        current = {t["track_id"] for t in tracks}

        for tid in list(cache):
            cache[tid]["missed"] += 1
            if version == 30 and tid not in current:
                module.advance_held_box(cache[tid], 1920, 1080)

        for t in tracks:
            cache[t["track_id"]] = {
                "box": t["box"], "missed": 0, "label": "P ACCEPTED", "color": (0, 0, 0),
                "vx": t["velocity_x"], "vy": t["velocity_y"], "t_seen": clock.t,
            }

        if version == 27:
            visible = {k: v for k, v in cache.items() if v["missed"] <= module.TRACK_DISPLAY_GRACE_FRAMES}
        else:
            visible = module.build_visible_track_cache(cache, current, clock.t)
            for tid in list(cache):
                if clock.t - cache[tid]["t_seen"] > module.TRACK_STATE_MEMORY_SECONDS:
                    del cache[tid]

        if frame >= 12:
            if not visible:
                blank_frames += 1
            shown_history.append(len(visible))
        ids_shown |= set(visible)
    return blank_frames, max(shown_history), len(cache)


def test_box_does_not_flicker_through_dropouts():
    print("\n1) frames with NO box drawn although the pallet is on the conveyor")
    print("   dropout | Phase 27 | Phase 30 | max boxes at once (ghost check)")
    for gap in (2, 3, 5, 8, 12, 18, 24):
        b27, m27, _ = run_display(P27, 27, gap)
        b30, m30, _ = run_display(P30, 30, gap)
        print(f"   {gap:5d}f  | {b27:8d} | {b30:8d} | {m30}")
        assert b30 == 0, f"Phase 30 still blanks for a {gap}-frame dropout"
        assert m30 <= 1, "ghost/duplicate box"
    print("   -> PASS: no blank frames up to 24 frames (0.8 s), never two boxes for one pallet")


def test_box_eventually_disappears_when_pallet_really_gone():
    clock = Clock()
    P30.time = clock
    tracker = tracker_mod.DynamicTracker()
    cache = {}
    x = 1400.0
    gone_at = None
    for frame in range(200):
        clock.t += DT
        dets = [detection(x)] if frame < 20 else []     # pallet physically removed at frame 20
        x -= 4.0
        tracks = tracker.update(dets)
        current = {t["track_id"] for t in tracks}
        for tid in list(cache):
            cache[tid]["missed"] += 1
            if tid not in current:
                P30.advance_held_box(cache[tid], 1920, 1080)
        for t in tracks:
            cache[t["track_id"]] = {"box": t["box"], "missed": 0, "label": "x", "color": 0,
                                    "vx": t["velocity_x"], "vy": t["velocity_y"], "t_seen": clock.t}
        visible = P30.build_visible_track_cache(cache, current, clock.t)
        for tid in list(cache):
            if clock.t - cache[tid]["t_seen"] > P30.TRACK_STATE_MEMORY_SECONDS:
                del cache[tid]
        if frame > 20 and not visible and gone_at is None:
            gone_at = (frame - 19) * DT
    print(f"\n2) removed pallet: box disappears {gone_at:.2f}s after last detection "
          f"(hold = {P30.TRACK_DISPLAY_HOLD_SECONDS}s), state purged after {P30.TRACK_STATE_MEMORY_SECONDS}s, cache empty={not cache}")
    assert gone_at is not None and gone_at <= P30.TRACK_DISPLAY_HOLD_SECONDS + 3 * DT
    assert not cache


def test_held_box_follows_pallet_instead_of_freezing():
    clock = Clock()
    P30.time = clock
    entry = {"box": box_at(1000), "missed": 0, "vx": -4.0, "vy": 0.0, "t_seen": clock.t}
    for _ in range(10):
        P30.advance_held_box(entry, 1920, 1080)
    cx = (entry["box"][0] + entry["box"][2]) / 2
    true_cx = 1000 - 4.0 * 10
    print(f"\n3) held box after 10 missed frames: x={cx:.0f} vs true {true_cx:.0f} (frozen would be 1000)")
    assert abs(cx - true_cx) < 8


# ---------------------------------------------------------------------
# 2. STATE LOSS: real bridge across a long blind gap
# ---------------------------------------------------------------------
def run_bridge(module, version, gap_frames):
    clock = Clock()
    module.time = clock
    tracker = tracker_mod.DynamicTracker()
    cache = {}
    object_history, object_decision = {}, {}
    symbol_history, symbol_decision = {}, {}
    symbol_sample_state, fast_after, fast_until = {}, {}, {}
    x = 1400.0
    owner_state_after_gap = None
    old_id = None

    for frame in range(30 + gap_frames + 40):
        clock.t += DT
        in_gap = 30 <= frame < 30 + gap_frames
        dets = [] if in_gap else [detection(x)]
        x -= 4.0
        tracks = tracker.update(dets)
        current = {t["track_id"] for t in tracks}

        for tid in list(cache):
            cache[tid]["missed"] += 1
            if version == 30 and tid not in current:
                module.advance_held_box(cache[tid], 1920, 1080)

        module.bridge_reacquired_tracks(
            tracks, current, cache, object_history, object_decision,
            symbol_history, symbol_decision, symbol_sample_state, fast_after, fast_until,
        )

        for t in tracks:
            tid = t["track_id"]
            if old_id is None:
                old_id = tid
                object_decision[tid] = "OBJECT_PRESENT"
                symbol_decision[tid] = "ACCEPTED"
                object_history[tid] = [1]
                symbol_history[tid] = [("NO_X", 0.9)]
            cache[tid] = {"box": t["box"], "missed": 0, "label": "x", "color": 0,
                          "vx": t["velocity_x"], "vy": t["velocity_y"], "t_seen": clock.t}

        # inline memory cleanup of main() (same condition as the file)
        for tid in list(cache):
            if version == 27:
                expired = cache[tid]["missed"] > module.TRACK_STATE_MEMORY_FRAMES
            else:
                expired = clock.t - cache[tid]["t_seen"] > module.TRACK_STATE_MEMORY_SECONDS
            if expired:
                del cache[tid]
                for d in (object_history, object_decision, symbol_history, symbol_decision):
                    d.pop(tid, None)

        if frame == 30 + gap_frames + 39 and tracks:
            tid = tracks[0]["track_id"]
            owner_state_after_gap = (tid, object_decision.get(tid), symbol_decision.get(tid))
    return old_id, owner_state_after_gap


def test_state_survives_long_occlusions():
    print("\n4) ACCEPTED state after a blind gap (real tracker + real bridge)")
    print("   dropout | Phase 27                      | Phase 30")
    for gap in (3, 5, 6, 8, 12, 18, 30, 45):
        o27, s27 = run_bridge(P27, 27, gap)
        o30, s30 = run_bridge(P30, 30, gap)
        ok27 = s27 is not None and s27[2] == "ACCEPTED"
        ok30 = s30 is not None and s30[2] == "ACCEPTED"
        print(f"   {gap:5d}f  | {'kept ACCEPTED' if ok27 else 'STATE LOST (re-evaluated)':29s} | {'kept ACCEPTED' if ok30 else 'STATE LOST'}")
        if gap <= 45:      # 45 frames = 1.5 s < 2.0 s bridge window
            assert ok30, f"Phase 30 lost state for a {gap}-frame dropout"
    print("   -> PASS up to 1.5 s; (Phase 27 loses it from 6-8 frames on)")


# ---------------------------------------------------------------------
# 3. NEIGHBOUR SAFETY: state must not jump to the next pallet
# ---------------------------------------------------------------------
def _neighbour_case(give_follower_state):
    clock = Clock()
    P30.time = clock
    tracker = tracker_mod.DynamicTracker()
    cache, oh, od, sh, sd, ss, fa, fu = {}, {}, {}, {}, {}, {}, {}, {}
    xa, xb = 1400.0, 1560.0           # leader A and follower B, 160 px apart
    id_a = None
    for frame in range(120):
        clock.t += DT
        dets = [detection(xb)]        # A is removed (blind forever); B keeps moving
        if frame < 30:
            dets.append(detection(xa))
        xa -= 4.0
        xb -= 4.0
        tracks = tracker.update(dets)
        current = {t["track_id"] for t in tracks}
        for tid in list(cache):
            cache[tid]["missed"] += 1
            if tid not in current:
                P30.advance_held_box(cache[tid], 1920, 1080)
        P30.bridge_reacquired_tracks(tracks, current, cache, oh, od, sh, sd, ss, fa, fu)
        for t in tracks:
            tid = t["track_id"]
            if frame == 29 and abs((t["box"][0] + t["box"][2]) / 2 - (xa + 4)) < 20:
                id_a = tid
                od[tid], sd[tid] = "OBJECT_PRESENT", "REJECTED"
            elif give_follower_state and frame == 29:
                od[tid] = "EMPTY"       # follower already classified, like in main()
            cache.setdefault(tid, {})
            cache[tid] = {"box": t["box"], "missed": 0, "label": "x", "color": 0,
                          "vx": t["velocity_x"], "vy": t["velocity_y"], "t_seen": clock.t}
    return [tid for tid in sd if tid != id_a]


def test_state_is_not_given_to_neighbour_pallet():
    for with_state in (True, False):
        others = _neighbour_case(with_state)
        print(f"\n5) removed leader A (REJECTED), follower B {'already has its own state' if with_state else 'has NO state yet (worst case)'}: "
              f"symbol states on other ids = {others}")
        assert not others, "state leaked to a neighbouring pallet"


# ---------------------------------------------------------------------
# 4. CLASSIFICATION EVERYWHERE: real gate, rightward (top) vs leftward (bottom)
# ---------------------------------------------------------------------
def gate_trace(module, vx, vy=0.0, frames=30):
    clock = Clock()
    module.time = clock
    state = {}
    reasons = []
    cx, cy = 900.0, 400.0
    for _ in range(frames):
        clock.t += DT
        cx += vx
        cy += vy
        track = {"box": box_at(cx, cy), "velocity_x": vx, "velocity_y": vy, "confidence": 0.9, "overlap": 1.0}
        ready, reason = module.update_dynamic_symbol_gate(7, track, state)
        reasons.append(reason)
    return reasons[-1], any(r == "CLEAN" for r in reasons)


def test_gate_allows_both_horizontal_directions():
    print("\n6) clean-view gate (is this view allowed to vote?)")
    rows = [("TOP    (moving right, vx=+4)", 4.0, 0.0), ("BOTTOM (moving left,  vx=-4)", -4.0, 0.0),
            ("turn   (diagonal vx=+2,vy=+4)", 2.0, 4.0), ("vertical (vx=0,vy=+4)       ", 0.0, 4.0),
            ("stopped (belt halted)        ", 0.0, 0.0)]
    for label, vx, vy in rows:
        l27, c27 = gate_trace(P27, vx, vy)
        l30, c30 = gate_trace(P30, vx, vy)
        print(f"   {label}: Phase 27 -> {l27:15s} | Phase 30 -> {l30}")
    assert gate_trace(P27, 4.0)[0] == "HOLD-DIRECTION", "expected Phase 27 to block rightward motion"
    assert gate_trace(P30, 4.0)[0] == "CLEAN"
    assert gate_trace(P30, -4.0)[0] == "CLEAN"
    assert gate_trace(P27, -4.0)[0] == "CLEAN"
    assert gate_trace(P30, 2.0, 4.0)[0] == "HOLD-TURN", "turns must still hold"
    assert gate_trace(P30, 0.0, 4.0)[0] != "CLEAN", "vertical motion must not vote in the default mode"
    assert gate_trace(P30, 0.0)[0] != "CLEAN", "stopped belt must not vote"


def test_gate_keeps_rejecting_unstable_boxes():
    clock = Clock()
    P30.time = clock
    state = {}
    reasons = set()
    cx = 900.0
    for i in range(30):                        # box width pumping 130 <-> 60 (wire in front)
        clock.t += DT
        cx += 4.0
        w = 130 if i % 2 == 0 else 60
        track = {"box": box_at(cx, 400.0, w=w), "velocity_x": 4.0, "velocity_y": 0.0, "confidence": 0.9, "overlap": 1.0}
        ready, reason = P30.update_dynamic_symbol_gate(9, track, state)
        reasons.add(reason)
        assert not ready
    print(f"\n7) unstable (wire-occluded) boxes never vote in Phase 30: {sorted(reasons)}")


def test_original_gate_still_available_by_flag():
    P30.SYMBOL_CLEAN_ALLOW_BOTH_DIRECTIONS = False
    try:
        assert gate_trace(P30, 4.0)[0] == "HOLD-DIRECTION"
    finally:
        P30.SYMBOL_CLEAN_ALLOW_BOTH_DIRECTIONS = True


def test_held_box_hidden_when_live_track_covers_it():
    clock = Clock()
    cache = {1: {"box": box_at(1000), "missed": 4, "t_seen": clock.t - 0.1},
             2: {"box": box_at(1010), "missed": 0, "t_seen": clock.t}}
    visible = P30.build_visible_track_cache(cache, {2}, clock.t)
    print(f"\n8) held box overlapped by live track is hidden: visible={sorted(visible)}")
    assert sorted(visible) == [2]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as error:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL {name}: {type(error).__name__}: {error}")
            traceback.print_exc(limit=4)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

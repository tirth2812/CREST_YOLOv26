"""
Offline checks for 31_live_full_conveyor_consistency_test.py (no camera/GPU/models).
Real tracker + real bridge, Phase 30 vs Phase 31 side by side.
    python test_phase31_hold_decision.py
"""
import ast
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
from collections import deque
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
P30 = load("p30", "30_live_full_conveyor_consistency_test.py")
P31 = load("p31", "31_live_full_conveyor_consistency_test.py")
P31.EVENT_LOG_FILE = Path(tempfile.gettempdir()) / "p31_test_events.log"

DT = 1.0 / 30.0


class Clock:
    def __init__(self):
        self.t = 9000.0

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def strftime(self, *a):
        return "00:00:00 "


def box_at(cx, cy=616.0, w=130, h=120):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def fresh_state():
    return dict(object_history={}, object_decision={}, symbol_history={}, symbol_decision={},
                symbol_sample_state={}, fast_after={}, fast_until={})


def bridge(module, tracks, cache, st):
    current = {t["track_id"] for t in tracks}
    with contextlib.redirect_stdout(io.StringIO()):
        module.bridge_reacquired_tracks(
            tracks, current, cache, st["object_history"], st["object_decision"],
            st["symbol_history"], st["symbol_decision"], st["symbol_sample_state"],
            st["fast_after"], st["fast_until"])


def lost_pallet(clock, st, pid, box, label="P1 ACCEPTED"):
    st["object_decision"][pid] = "OBJECT_PRESENT"
    st["symbol_decision"][pid] = "ACCEPTED"
    st["object_history"][pid] = deque([("OBJECT_PRESENT", .9)] * 4, maxlen=7)
    st["symbol_history"][pid] = deque([("NO_X", .9)] * 3, maxlen=3)
    return {"box": box, "missed": 10, "label": label, "color": 0, "vx": -4.0, "vy": 0.0,
            "t_seen": clock.t - 0.4}


def as_processed(st, pid):
    """what update_object_decision does to a track on its first processed frame"""
    st["object_history"].setdefault(pid, deque(maxlen=7))


# ---------------------------------------------------------------------
def test_retry_first_frame_imperfect_then_good():
    print("\n1) new id appears with a shrunken box (wire edge), full box a few frames later")
    results = {}
    for name, module in (("Phase 30", P30), ("Phase 31", P31)):
        module.time = clock = Clock()
        if hasattr(module, "RELAXED_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO"):
            module.RELAXED_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO = 0.0     # isolate the RETRY rule
        st = fresh_state()
        cache = {1: lost_pallet(clock, st, 1, box_at(900))}
        kept_at = None
        for frame in range(1, 8):
            clock.t += DT
            w = 90 if frame <= 2 else 130            # strict size gate (0.8) fails for 90/130
            tracks = [{"track_id": 2, "box": box_at(900 - 4 * frame, w=w, h=int(120 * w / 130)),
                       "velocity_x": -4.0, "velocity_y": 0.0}]
            bridge(module, tracks, cache, st)
            as_processed(st, 2)
            if st["symbol_decision"].get(2) == "ACCEPTED" and kept_at is None:
                kept_at = frame
            cache.setdefault(2, {"box": tracks[0]["box"], "missed": 0, "label": "x", "color": 0,
                                 "vx": -4.0, "vy": 0.0, "t_seen": clock.t})
        results[name] = kept_at
        print(f"   {name}: state {'KEPT at frame %d' % kept_at if kept_at else 'LOST -> pallet shows UNKNOWN'}")
    P31.RELAXED_BRIDGE_MAX_HORIZONTAL_CENTER_RATIO = 1.00
    assert results["Phase 30"] is None
    assert results["Phase 31"] is not None


def test_second_chance_when_prediction_is_off():
    print("\n2) new id 0.9 box-widths from where the pallet was expected (turn / long gap)")
    out = {}
    for name, module in (("Phase 30", P30), ("Phase 31", P31)):
        module.time = clock = Clock()
        st = fresh_state()
        cache = {1: lost_pallet(clock, st, 1, box_at(900))}
        clock.t += DT
        tracks = [{"track_id": 2, "box": box_at(900 - 117, cy=616 + 10), "velocity_x": -4.0, "velocity_y": 0.0}]
        bridge(module, tracks, cache, st)
        out[name] = st["symbol_decision"].get(2)
        print(f"   {name}: new id state = {out[name]}")
    assert out["Phase 30"] is None and out["Phase 31"] == "ACCEPTED"


def test_second_chance_never_guesses():
    print("\n3) neighbour safety (Phase 31 must transfer NOTHING in these cases)")
    cases = {}

    # a) leader lost; brand-new id is one pallet spacing behind it (1.23 widths) -> out of range
    P31.time = clock = Clock(); st = fresh_state()
    cache = {1: lost_pallet(clock, st, 1, box_at(900))}
    bridge(P31, [{"track_id": 2, "box": box_at(900 + 160)}], cache, st)
    cases["follower one spacing behind"] = st["symbol_decision"].get(2)

    # b) two lost pallets both near the new id -> ambiguous
    P31.time = clock = Clock(); st = fresh_state()
    cache = {1: lost_pallet(clock, st, 1, box_at(880)), 3: lost_pallet(clock, st, 3, box_at(930))}
    bridge(P31, [{"track_id": 2, "box": box_at(905)}], cache, st)
    cases["two lost pallets near one new id"] = st["symbol_decision"].get(2)

    # c) one lost pallet, two brand-new ids near it -> ambiguous
    P31.time = clock = Clock(); st = fresh_state()
    cache = {1: lost_pallet(clock, st, 1, box_at(900))}
    # both beyond the strict range (0.45 widths) but inside the relaxed one -> ambiguous
    bridge(P31, [{"track_id": 2, "box": box_at(900 - 91)}, {"track_id": 4, "box": box_at(900 + 91)}], cache, st)
    cases["two new ids near one lost pallet"] = (st["symbol_decision"].get(2), st["symbol_decision"].get(4))

    # d) an already-tracked young follower (in cache) is not a second-chance target
    P31.time = clock = Clock(); st = fresh_state()
    cache = {1: lost_pallet(clock, st, 1, box_at(900)),
             2: {"box": box_at(820), "missed": 1, "label": "x", "color": 0, "vx": -4, "vy": 0, "t_seen": clock.t}}
    bridge(P31, [{"track_id": 2, "box": box_at(820)}], cache, st)
    cases["existing tracked follower"] = st["symbol_decision"].get(2)

    # e) lost pallet is too old
    P31.time = clock = Clock(); st = fresh_state()
    cache = {1: lost_pallet(clock, st, 1, box_at(900))}
    clock.t += P31.STATE_BRIDGE_MAX_MISSING_SECONDS + 0.5
    bridge(P31, [{"track_id": 2, "box": box_at(900)}], cache, st)
    cases["lost pallet older than bridge window"] = st["symbol_decision"].get(2)

    for label, result in cases.items():
        print(f"   {label:40s} -> {result}")
    assert all(v in (None, (None, None)) for v in cases.values())


def test_established_pallet_is_never_overwritten():
    P31.time = clock = Clock(); st = fresh_state()
    cache = {1: lost_pallet(clock, st, 1, box_at(900))}
    st["symbol_decision"][2] = "REJECTED"           # an already classified pallet
    st["object_decision"][2] = "OBJECT_PRESENT"
    st["object_history"][2] = deque([("OBJECT_PRESENT", .9)] * 6, maxlen=7)
    bridge(P31, [{"track_id": 2, "box": box_at(900)}], cache, st)
    print("\n4) classified pallet next to a lost one keeps its own decision:", st["symbol_decision"][2])
    assert st["symbol_decision"][2] == "REJECTED"
    # a young id's own half-formed state is REPLACED, not merged, when it is bridged
    P31.time = clock = Clock(); st = fresh_state()
    cache = {1: lost_pallet(clock, st, 1, box_at(900))}
    st["object_decision"][2] = "OBJECT_PRESENT"
    st["object_history"][2] = deque([("OBJECT_PRESENT", .8)] * 2, maxlen=7)
    st["fast_until"][2] = clock.t + 2.0              # it had opened its own fast-verify window
    bridge(P31, [{"track_id": 2, "box": box_at(900)}], cache, st)
    assert st["symbol_decision"].get(2) == "ACCEPTED" and 2 not in st["fast_until"]


def test_real_tracker_open_area_dropout_keeps_decision():
    print("\n5) real tracker, pallet in the open, dropout then partly cut-off first box (ACCEPTED must survive)")
    print("   dropout | Phase 30            | Phase 31")

    def run(module, gap):
        module.time = clock = Clock()
        tracker = tracker_mod.DynamicTracker()
        st = fresh_state()
        cache = {}
        x, old_id, reappear = 1400.0, None, None
        for frame in range(30 + gap + 60):
            clock.t += DT
            blind = 30 <= frame < 30 + gap
            if reappear is None and frame >= 30 + gap:
                reappear = frame
            w = 88 if (reappear is not None and frame - reappear < 9) else 130   # box partly cut off on return
            dets = [] if blind else [{"box": box_at(x, w=w, h=int(120 * w / 130)), "confidence": .9, "overlap": 1.0}]
            x -= 4.0
            tracks = tracker.update(dets)
            current = {t["track_id"] for t in tracks}
            for tid in list(cache):
                cache[tid]["missed"] += 1
                if tid not in current:
                    module.advance_held_box(cache[tid], 1920, 1080)
            bridge(module, tracks, cache, st)
            for t in tracks:
                tid = t["track_id"]
                if old_id is None:
                    old_id = tid
                    st["object_decision"][tid] = "OBJECT_PRESENT"
                    st["symbol_decision"][tid] = "ACCEPTED"
                    st["object_history"][tid] = deque([("OBJECT_PRESENT", .9)] * 5, maxlen=7)
                    st["symbol_history"][tid] = deque([("NO_X", .9)] * 3, maxlen=3)
                else:
                    as_processed(st, tid)
                    if tid not in st["symbol_decision"] and frame % 6 == 0:     # a young id forming its own opinion
                        st["object_history"][tid].append(("OBJECT_PRESENT", .9))
                        if len(st["object_history"][tid]) >= 2:
                            st["object_decision"][tid] = "OBJECT_PRESENT"
                cache[tid] = {"box": t["box"], "missed": 0, "label": "x", "color": 0,
                              "vx": t["velocity_x"], "vy": t["velocity_y"], "t_seen": clock.t}
            for tid in list(cache):
                if clock.t - cache[tid]["t_seen"] > module.TRACK_STATE_MEMORY_SECONDS:
                    del cache[tid]
                    for d in (st["object_history"], st["object_decision"], st["symbol_history"], st["symbol_decision"]):
                        d.pop(tid, None)
        live = [t for t in tracks]
        return bool(live) and st["symbol_decision"].get(live[0]["track_id"]) == "ACCEPTED"

    lost30 = lost31 = 0
    for gap in (6, 8, 12, 18, 30, 40):
        a, b = run(P30, gap), run(P31, gap)
        lost30 += (not a)
        print(f"   {gap:5d}f  | {'kept' if a else 'LOST -> UNKNOWN':19s} | {'kept' if b else 'LOST'}")
        assert b, f"Phase 31 lost the decision for a {gap}-frame dropout"
    assert lost30 > 0, "scenario should expose the Phase 30 weakness"


def test_nothing_is_printed_in_the_live_loop():
    tree = ast.parse((HERE / "31_live_full_conveyor_consistency_test.py").read_text(encoding="utf-8-sig"))
    allowed = {"Camera frame failed.", "Loading pallet detector...", "Loading EMPTY / OBJECT_PRESENT classifier...",
               "Loading frozen medium-focus X / NO_X classifier...", "All models loaded."}
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loops = [n for n in ast.walk(main) if isinstance(n, ast.While)]
    prints = [c for loop in loops for c in ast.walk(loop)
              if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "print"]
    texts = [ast.unparse(c.args[0]) if c.args else "" for c in prints]
    print("\n6) print() calls inside the live while-loop:", texts)
    # allowed inside the loop: the camera-failure exit and the user-pressed 'S' screenshot message
    for t in texts:
        assert "Camera frame failed" in t or "Saved:" in t, t
    # helper functions used per frame never print either
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in (
                "update_object_decision", "update_three_view_decision", "bridge_reacquired_tracks",
                "transfer_bridged_state", "move_track_state", "update_dynamic_symbol_gate", "build_visible_track_cache"):
            assert not [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "print"], fn.name


def test_event_log_is_file_only():
    P31.time = Clock()
    path = Path(tempfile.gettempdir()) / "p31_events_check.log"
    path.unlink(missing_ok=True)
    P31.EVENT_LOG_FILE = path
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P31.log_event("P7 OBJECT CHANGE: test")
    assert buf.getvalue() == "" and "OBJECT CHANGE" in path.read_text()
    P31.EVENT_LOG_FILE = Path("Z:/definitely/not/writable/x.log")      # must never raise
    P31.log_event("ignored")
    P31.EVENT_LOG_ENABLED = False
    P31.log_event("ignored")
    P31.EVENT_LOG_ENABLED = True
    print("\n7) log_event writes only to the file, never prints, never raises")


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

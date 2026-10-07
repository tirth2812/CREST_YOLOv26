"""Offline checks for the Phase 37 static-ghost filter (real tracker, no camera):  python test_phase37_static_ghost.py"""
import importlib.util, sys, types, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
for n in ("cv2", "ultralytics"):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["cv2"].createCLAHE = lambda **k: object()
sys.modules["ultralytics"].YOLO = object


def load(label, filename):
    s = importlib.util.spec_from_file_location(label, HERE / filename)
    m = importlib.util.module_from_spec(s); sys.modules[label] = m; s.loader.exec_module(m); return m


trk = load("dyn_tracker", "dynamic_live_pallet_detection.py")
P37 = load("p37", "37_live_static_ghost_filter_test.py")
DT = 1 / 30.0


def box_at(cx, cy, w=130, h=120):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def det(cx, cy):
    return {"box": box_at(cx, cy), "confidence": .9, "overlap": 1.0}


def run(frames, pallets_move, ghost_moves_from=None, belt_stops_at=None, with_ghost=True, seed=1):
    rnd = random.Random(seed)
    P37.static_ghost_state.clear()
    tracker = trk.DynamicTracker()
    xs = [600.0, 900.0, 1200.0, 1500.0]
    gx, gy = 250.0, 300.0
    log = []
    for f in range(frames):
        now = 5000.0 + f * DT
        stopped = belt_stops_at is not None and f >= belt_stops_at
        if pallets_move and not stopped:
            xs = [x - 3.0 for x in xs]
        dets = [det(x, 800) for x in xs]
        if with_ghost:
            if ghost_moves_from is not None and f >= ghost_moves_from and not stopped:
                gx += 3.0
            dets.append(det(gx + rnd.uniform(-4, 4), gy + rnd.uniform(-4, 4)))
        tracks = tracker.update(dets)
        kept, dropped = P37.drop_static_ghosts(tracks, now)
        log.append((len(tracks), len(kept), len(dropped), {t["track_id"] for t in kept}))
    return log


ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (" " + extra if extra else ""))


# 1 belt runs, carton static: carton hidden after ~6 s, 4 pallets always shown
log = run(30 * 12, True)
check("1 ghost still shown at 3 s", log[90][1] == 5)
check("1 ghost hidden after 8 s", log[8 * 30][1] == 4 and log[-1][1] == 4, f"kept={log[-1][1]}")
check("1 real pallets never hidden (after the 5-hit confirmation)", all(k >= 4 for _, k, _, _ in log[10:]))

# 2 belt stopped the whole time: nothing hidden even after a long time
log = run(30 * 20, False)
check("2 stopped belt: nothing dropped", all(d == 0 for _, _, d, _ in log) and log[-1][1] == 5)

# 3 belt runs, then stops for 30 s: real pallets must not be dropped by the stop
log = run(30 * 40, True, belt_stops_at=30 * 10)
check("3 pallets survive a long belt stop", log[-1][1] >= 4, f"kept={log[-1][1]}")

# 4 fixed object later starts moving: shown again
log = run(30 * 20, True, ghost_moves_from=30 * 10)
check("4 hidden then visible again when it moves", log[9 * 30][1] == 4 and log[-1][1] == 5, f"{log[9*30][1]} -> {log[-1][1]}")

# 5 only moving pallets, no ghost: nothing hidden ever
log = run(30 * 15, True, with_ghost=False)
check("5 no ghost: nothing dropped", all(d == 0 for _, _, d, _ in log))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)

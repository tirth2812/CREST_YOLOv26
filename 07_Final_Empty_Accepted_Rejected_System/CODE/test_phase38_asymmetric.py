"""
Offline checks for Phase 38 (no camera / GPU / models):  python test_phase38_asymmetric.py
Uses the REAL decision functions of the Phase 38 file.  "Phase 32 rule" = store every sample.
"""
import importlib.util, sys, tempfile, time, types
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
for n in ("cv2", "ultralytics"):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["cv2"].createCLAHE = lambda **k: object()
sys.modules["ultralytics"].YOLO = object


def load(label, filename):
    s = importlib.util.spec_from_file_location(label, HERE / filename)
    m = importlib.util.module_from_spec(s); sys.modules[label] = m; s.loader.exec_module(m); return m


P38 = load("p38", "38_live_asymmetric_fast_symbol_test.py")
P38.log_event = lambda *a, **k: None
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (" " + extra if extra else ""))


def run(samples, phase38, initial=None):
    """samples: list of (state, clean_view, fast_mode).  Returns the decision after each sample."""
    h = {1: deque(maxlen=3)}
    d = {1: initial} if initial else {}
    out = []
    for state, clean, fast in samples:
        store = P38.should_store_symbol_sample(state, clean) if phase38 else True
        if store:
            h[1].append((state, 0.9))
        out.append(P38.update_three_view_decision(1, h, d, fast_mode=fast)[0])
    return out


# --- the rule itself -------------------------------------------------------
S = P38.should_store_symbol_sample
check("rule: X from a bad view is stored", S("X", False))
check("rule: X from a clean view is stored", S("X", True))
check("rule: NO_X from a clean view is stored", S("NO_X", True))
check("rule: NO_X from a bad view is NOT stored", not S("NO_X", False))

# --- Problem 1: X pallet starting in a bad view ---------------------------
bad_start = [("NO_X", False, True)] * 3 + [("X", True, False)] * 3
old = run(bad_start, False)
new = run(bad_start, True)
check("Phase 32 rule reproduces the bug (ACCEPTED then REJECTED)", "ACCEPTED" in old and old[-1] == "REJECTED", str(old))
check("Phase 38: never ACCEPTED, CHECKING then REJECTED", "ACCEPTED" not in new and new[2] is None and new[-1] == "REJECTED", str(new))

# --- X from a good / strong view: fast REJECTED kept -----------------------
new = run([("X", False, True)] * 3, True)
check("fast X x3 still commits REJECTED quickly", new[-1] == "REJECTED", str(new))
new = run([("X", True, True)] * 3, True)
check("fast X x3 from a clean view commits REJECTED", new[-1] == "REJECTED", str(new))

# --- NO_X pallet --------------------------------------------------------------
new = run([("NO_X", False, True)] * 3 + [("NO_X", True, False)] * 3, True)
check("NO_X from a bad view: CHECKING, then ACCEPTED only from clean samples",
      new[:5] == [None] * 5 and new[-1] == "ACCEPTED", str(new))
new = run([("NO_X", True, True)] * 3, True)
check("NO_X from a clean view in the fast window: ACCEPTED without delay", new[-1] == "ACCEPTED", str(new))

# --- mixed fast evidence ------------------------------------------------------
new = run([("NO_X", False, True), ("X", False, True), ("X", False, True)], True)
check("mixed unsafe fast samples: no premature decision", new == [None, None, None], str(new))
new = run([("NO_X", False, True), ("X", False, True), ("X", False, True), ("X", True, False)], True)
check("...and a clean X then decides REJECTED", new[-1] == "REJECTED", str(new))

# --- real physical changes still work (terminal-state hysteresis untouched) -
new = run([("X", True, False)] * 3, True, initial="ACCEPTED")
check("ACCEPTED + 3 clean X -> REJECTED", new[-1] == "REJECTED", str(new))
new = run([("NO_X", True, False)] * 3, True, initial="REJECTED")
check("REJECTED + 3 clean NO_X -> ACCEPTED", new[-1] == "ACCEPTED", str(new))
new = run([("X", True, False), ("X", True, False), ("NO_X", True, False)], True, initial="ACCEPTED")
check("ACCEPTED + X X NO_X stays ACCEPTED", new[-1] == "ACCEPTED", str(new))

# --- deferred counter -----------------------------------------------------------
logged = []
P38.log_event = lambda m: logged.append(m)
P38.fast_no_x_deferred.clear()
for _ in range(3):
    P38.note_deferred_no_x(7)
check("FAST NO_X DEFERRED logged once, at the 3rd deferred sample", len(logged) == 1 and "P7 FAST NO_X DEFERRED" in logged[0])
P38.clear_symbol_state(7, {}, {}, {})
check("deferred count cleared with the symbol state", 7 not in P38.fast_no_x_deferred)
P38.log_event = lambda *a, **k: None

# --- bridge diagnostics ------------------------------------------------------------
tmp = Path(tempfile.mkdtemp()) / "bridge_diagnostics.log"
P38.BRIDGE_DIAG_FILE = tmp
real_time = P38.time
now = 5000.0
P38.time = types.SimpleNamespace(monotonic=lambda: now, time=lambda: now, strftime=lambda *a: "00:00:00 ")


def box(cx, cy, w=130, h=120):
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def scenario(old_boxes, new_box, missing=0.7, new_has_symbol=False):
    tmp.write_text("")
    cache, od, sd, oh, sh = {}, {}, {}, {}, {}
    for oid, ob in old_boxes.items():
        cache[oid] = {"box": ob, "missed": 5, "label": "x", "color": (0, 0, 0), "vx": 0, "vy": 0, "t_seen": now - missing}
        od[oid] = "OBJECT_PRESENT"
        sd[oid] = "REJECTED"
    if new_has_symbol:
        sd[99] = "ACCEPTED"
    P38.diagnose_unbridged_tracks([{"track_id": 99, "box": new_box}], set(), cache, oh, od, sh, sd)
    return tmp.read_text()


log = scenario({111: box(1000, 800)}, box(1700, 800))
check("diag: far horizontal jump names the failed criterion", "BRIDGE REJECT new=P99" in log and "horizontal" in log and "> 0.45" in log, log.splitlines()[1] if log else "")
log = scenario({111: box(1000, 800)}, box(1000, 800), missing=2.4)
check("diag: lost too long names the failed criterion", "missing 2.40s > 2.00s" in log)
log = scenario({111: box(1000, 800, 130, 120)}, box(1000, 800, 60, 55))
check("diag: size mismatch named", "size_ratio" in log and "< 0.80" in log)
log = scenario({111: box(1000, 800)}, box(1010, 805), new_has_symbol=True)
check("diag: new id that already decided is reported as not young", "young=False" in log and "not young" in log)
log = scenario({111: box(1000, 800), 112: box(1030, 800)}, box(1015, 800))
check("diag: two near candidates reported as AMBIGUOUS", "AMBIGUOUS" in log)
log = scenario({111: box(1000, 800)}, box(1010, 805))
check("diag: a clean match that was not transferred says so", "strict geometry passes for P111" in log)
P38.time = real_time

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)

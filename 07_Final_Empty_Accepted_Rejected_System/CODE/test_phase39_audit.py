"""
Offline checks for the Phase 39 decision audit (no camera / GPU / models):  python test_phase39_audit.py
"""
import importlib.util, inspect, json, sys, tempfile, time, types
from collections import deque
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


P38 = load("p38", "38_live_asymmetric_fast_symbol_test.py")
P39 = load("p39", "39_live_decision_audit_test.py")
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (" " + extra if extra else ""))


# --- 0. the decision logic is byte-identical to Phase 38 -------------------------
for fn in ("update_object_decision", "update_three_view_decision", "should_store_symbol_sample",
           "update_dynamic_symbol_gate", "should_take_symbol_sample", "commit_sample_position",
           "state_bridge_score", "bridge_reacquired_tracks", "track_can_receive_state"):
    check(f"decision logic unchanged vs Phase 38: {fn}", inspect.getsource(getattr(P38, fn)) == inspect.getsource(getattr(P39, fn)))

# --- fake cv2 + temp folders --------------------------------------------------------
class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0
    @staticmethod
    def imwrite(path, image):
        Path(path).write_bytes(b"jpg" + bytes(str(getattr(image, "shape", "")), "ascii")); return True
    @staticmethod
    def rectangle(*a, **k): return None
    @staticmethod
    def putText(*a, **k): return None


root = Path(tempfile.mkdtemp())
P39.cv2 = FakeCV2
P39.SAMPLES_DIR = root / "Decision_Samples"
P39.AUDIT_DIR = root / "Decision_Audit"
P39.AUDIT_LOG_FILE = root / "decision_audit.log"
P39.log_event = lambda *a, **k: None
now0 = 7000.0


def sample(track_id, state, clean, mode, t, cx, cy, conf=0.9, stored=None):
    crop = np.zeros((40, 40, 3), np.uint8)
    P39.audit_frame_info[track_id] = {"clean_view": clean, "gate_reason": "CLEAN" if clean else "HOLD-TURN", "vx": 5.0, "vy": 0.5,
                                      "detector_confidence": 0.8, "ring_overlap": 0.9, "stable_frames": 6 if clean else 0}
    P39.audit_record_sample(track_id, state, conf, 1 - conf if state == "NO_X" else conf, conf if state == "NO_X" else 1 - conf,
                            t, cx, cy, 200.0, (int(cx) - 100, int(cy) - 70, int(cx) + 100, int(cy) + 70), mode,
                            P39.should_store_symbol_sample(state, clean) if stored is None else stored, crop, crop)


def commit(track_id, before, after, fast):
    P39.audit_commit(track_id, before, after, fast, True, "CLEAN", 3, 0, 3, "OBJECT_PRESENT", 0.97,
                     {"confidence": 0.8, "overlap": 0.9, "velocity_x": 5.0, "velocity_y": 0.5},
                     (720, 350, 920, 490), np.zeros((1080, 1920, 3), np.uint8))


# --- 1. a normal audited decision -----------------------------------------------------
P39.sample_audit.clear(); P39.track_lineage.clear(); P39.track_born.clear()
P39.track_born[85] = time.time() - 4.0
P39.track_lineage[85] = [12, 37, 85]
for i in range(3):
    sample(85, "NO_X", False, "FAST", now0 + 0.16 * i, 800 + 2 * i, 420)       # unsafe: deferred
for i in range(3):
    sample(85, "X", True, "CLEAN", now0 + 2 + 0.6 * i, 820 + 90 * i, 420)       # clean: stored
commit(85, None, "REJECTED", False)
folders = list(P39.AUDIT_DIR.iterdir())
check("audit folder created with frame, pallet crop and decision.json",
      len(folders) == 1 and all((folders[0] / n).exists() for n in ("frame.jpg", "pallet_crop.jpg", "decision.json")))
data = json.loads((folders[0] / "decision.json").read_text())
check("3 sample crops saved with the requested naming", len(list(P39.SAMPLES_DIR.glob("*_P85_REJECTED_CLEAN_obs*.jpg"))) == 3,
      str(sorted(p.name for p in P39.SAMPLES_DIR.iterdir())))
check("only the 3 stored clean X samples are the votes; 3 unsafe NO_X listed as not stored",
      [s["state"] for s in data["samples_used"]] == ["X"] * 3 and len(data["samples_NOT_stored_unsafe_view_NO_X"]) == 3)
check("movement between samples reported", len(data["movement_between_samples"]) == 2 and data["total_movement_px_first_to_last"] == 180.0,
      str(data["movement_between_samples"]))
check("tracker lineage reported", data["tracker_lineage"] == "P12 -> P37 -> P85")
check("no rule violation when unsafe NO_X was not counted", data["RULE_VIOLATION_unsafe_NO_X_counted"] == [])
check("per-sample p(X)/p(NO_X)/clean/gate recorded", all(k in data["samples_used"][0] for k in ("p_x", "p_no_x", "clean_view", "gate_reason", "stable_frames", "mode")))
log = P39.AUDIT_LOG_FILE.read_text()
check("human readable DECISION COMMIT block written", "DECISION COMMIT" in log and "track=P85" in log and "sample 1:" in log)

# --- 2. the old bug would be caught --------------------------------------------------
P39.sample_audit.clear()
for i in range(3):
    sample(40, "NO_X", False, "FAST", now0 + 0.16 * i, 800 + 2 * i, 420, stored=True)   # Phase 32 behaviour
commit(40, None, "ACCEPTED", True)
data = json.loads(sorted(P39.AUDIT_DIR.iterdir())[-1].joinpath("decision.json").read_text()) if False else None
latest = max(P39.AUDIT_DIR.iterdir(), key=lambda p: p.stat().st_mtime_ns if "P40" in p.name else 0)
data = json.loads((latest / "decision.json").read_text())
check("a fast ACCEPTED from unsafe NO_X would be flagged RULE_VIOLATION", len(data["RULE_VIOLATION_unsafe_NO_X_counted"]) == 3)
check("near-identical samples show a tiny movement ratio", all(m["ratio_of_diagonal"] < 0.05 for m in data["movement_between_samples"]),
      str(data["movement_between_samples"]))

# --- 3. the evidence and lineage follow a bridge ----------------------------------------
P39.sample_audit.clear(); P39.track_lineage.clear()
sample(50, "X", True, "CLEAN", now0, 900, 800)
stores = [dict() for _ in range(7)]
stores[3][50] = deque([("X", .9)])
cache = {50: {"box": (1, 2, 3, 4), "missed": 0, "t_seen": now0}}
P39.move_track_state(50, 60, (5, 6, 7, 8), cache, stores[0], stores[1], stores[3], stores[4], stores[5], stores[6], {})
check("audit samples moved to the new id", 60 in P39.sample_audit and 50 not in P39.sample_audit)
P39.move_track_state(60, 71, (5, 6, 7, 8), {60: dict(cache[60])}, {}, {}, {}, {}, {}, {}, {})
check("lineage grows along the chain", P39.track_lineage[71] == [50, 60, 71], str(P39.track_lineage.get(71)))

# --- 4. clearing the symbol state clears the audit ----------------------------------------
P39.clear_symbol_state(71, {}, {}, {})
check("audit evidence cleared with the symbol state", 71 not in P39.sample_audit)

# --- 5. the audit never raises ---------------------------------------------------------------
try:
    P39.audit_commit(1, None, "ACCEPTED", True, True, "x", 1, 1, 1, "o", 0.5, {}, None, None)
    P39.audit_note_frame(9, {}, True, "x", {})
    P39.audit_record_sample(9, "X", 0.9, 0.9, 0.1, 1.0, 1, 1, 1, (0, 0, 1, 1), "FAST", True, None, None)
    P39.audit_note_tracks([{}])
    check("audit helpers never raise on bad input", True)
except Exception as error:
    check("audit helpers never raise on bad input", False, repr(error))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)

"""
Deterministic simulation tests for crest_rfid_fusion.py.

No camera, PLC, cv2 or GPU needed:   python test_crest_rfid_fusion.py
(also collectable by pytest).

Frame period 1/30 s.  Pallets are 130x120 px boxes.  RFID1 lane y=626,
travel RIGHT->LEFT; RFID2 lane y=775, travel LEFT->RIGHT.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("crest_rfid_fusion", HERE / "crest_rfid_fusion.py")
fusion_mod = importlib.util.module_from_spec(spec)
sys.modules["crest_rfid_fusion"] = fusion_mod
spec.loader.exec_module(fusion_mod)

FusionConfig = fusion_mod.FusionConfig
FusionIdentityManager = fusion_mod.FusionIdentityManager
RFIDEvent = fusion_mod.RFIDEvent

DT = 1.0 / 30.0
SPEED = 120.0          # px/s  (~4 px/frame)
RFID1_Y = 626.0
RFID2_Y = 775.0
T0 = 1000.0


def make_track(track_id, cx, cy, w=130, h=120):
    return {"track_id": track_id, "box": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)}


class Sim:
    def __init__(self, **overrides):
        self.cfg = FusionConfig(log_echo=False, **overrides)
        self.f = FusionIdentityManager(self.cfg)
        self.t = T0
        self.scheduled = []     # (time, station, pallet)

    def rfid_at(self, t, station, pallet):
        self.scheduled.append((t, station, pallet))

    def step(self, tracks):
        for item in [s for s in self.scheduled if s[0] <= self.t]:
            self.scheduled.remove(item)
            self.f.push_rfid_event(RFIDEvent(item[1], item[2], item[0]))
        self.f.update_tracks(self.t, tracks)
        self.f.process_rfid_events(self.t)
        check_invariants(self.f)
        self.t += DT

    def run(self, until, tracks_fn):
        while self.t < until:
            self.step(tracks_fn(self.t))


# --- motion helpers -----------------------------------------------------

def rfid1_x(t, t_start, x0=1300.0):
    return x0 - SPEED * (t - t_start)


def rfid2_x(t, t_start, x0=700.0):
    return x0 + SPEED * (t - t_start)


def center_time_rfid1(t_start, x0=1300.0):
    return t_start + (x0 - 1153.0) / SPEED


def center_time_rfid2(t_start, x0=700.0):
    return t_start + (853.5 - x0) / SPEED


def check_invariants(f):
    owners = {}
    for traj in f.trajectories.values():
        if traj.physical_id is not None:
            assert traj.physical_id not in owners, "duplicate physical owner"
            owners[traj.physical_id] = traj
            assert f.registry[traj.physical_id].trajectory is traj
    for pid, pal in f.registry.items():
        if pal.trajectory is not None:
            assert pal.trajectory.physical_id == pid
            assert pal.trajectory.id in f.trajectories
        else:
            assert pal.visibility_state == "NOT_VISIBLE"
    assert all(t.track_id is None or f.track_to_traj.get(t.track_id) is t for t in f.trajectories.values())


def phys(sim, track_id):
    return sim.f.get_physical_id(track_id)


# =====================================================================
# ACCEPTANCE SCENARIOS
# =====================================================================

def test_scenario1_single_pallet_assigned():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
    sim.run(T0 + 3.0, lambda t: [make_track(57, rfid1_x(t, T0), RFID1_Y)])
    assert phys(sim, 57) == 3
    assert len(sim.f.log.events("PASSAGE_OPEN")) == 1
    assert len(sim.f.log.events("RFID_PASSAGE_MATCH")) == 1
    assert len(sim.f.log.events("PASSAGE_EXIT")) == 1
    match = sim.f.log.events("RFID_PASSAGE_MATCH")[0]
    assert match["resolve_latency"] < 0.3


def test_scenario2_neighbor_does_not_steal():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
    # follower 220 px behind (outside the station while the leader is inside)
    sim.run(T0 + 3.5, lambda t: [
        make_track(57, rfid1_x(t, T0), RFID1_Y),
        make_track(58, rfid1_x(t, T0, 1520.0), RFID1_Y),
    ])
    assert phys(sim, 57) == 3
    assert phys(sim, 58) is None


def test_scenario2b_two_pallets_overlapping_station():
    sim = Sim()
    tA = center_time_rfid1(T0, 1300.0)
    tB = center_time_rfid1(T0, 1340.0)          # 40 px behind -> both inside the station together
    sim.rfid_at(tA, "RFID1", 3)
    sim.rfid_at(tB, "RFID1", 5)
    sim.run(T0 + 4.0, lambda t: [
        make_track(57, rfid1_x(t, T0, 1300.0), RFID1_Y),
        make_track(58, rfid1_x(t, T0, 1340.0), RFID1_Y),
    ])
    assert phys(sim, 57) == 3
    assert phys(sim, 58) == 5
    assert sim.f.log.events("STATION_OVERLAP"), "test must really overlap two pallets"


def test_scenario3_track_id_changes_inside_station():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)

    def tracks(t):
        x = rfid1_x(t, T0)
        if 1140 >= x >= 1118:        # tracker drops the pallet INSIDE the station (after entry)
            return []
        return [make_track(57 if x > 1140 else 92, x, RFID1_Y)]

    sim.run(T0 + 3.0, tracks)
    assert phys(sim, 92) == 3
    opens = sim.f.log.events("PASSAGE_OPEN")
    assert len(opens) == 1, "tracker change inside the gate must stay ONE passage"
    assert len(sim.f.log.events("PASSAGE_TRACK_SWITCH")) == 1
    exit_event = sim.f.log.events("PASSAGE_EXIT")[0]
    assert exit_event["entry_track_id"] == 57 and exit_event["track_id"] == 92
    assert exit_event["track_history"] == [57, 92]


def test_scenario3b_track_id_changes_exactly_at_entry_boundary():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)

    def tracks(t):
        x = rfid1_x(t, T0)
        if 1215 >= x >= 1185:        # lost while still outside, reappears inside
            return []
        return [make_track(57 if x > 1215 else 92, x, RFID1_Y)]

    sim.run(T0 + 3.0, tracks)
    assert phys(sim, 92) == 3
    assert len(sim.f.log.events("PASSAGE_OPEN")) == 1


def test_scenario4_short_track_loss_keeps_identity():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)

    def tracks(t):
        x = rfid1_x(t, T0)
        if 900 >= x >= 860:           # 0.33 s gap far from any station
            return []
        return [make_track(57 if x > 900 else 92, x, RFID1_Y)]

    sim.run(T0 + 5.0, tracks)
    assert phys(sim, 92) == 3
    assert sim.f.registry[3].visibility_state == "ACTIVE"


def test_scenario4b_bridge_hint_from_phase27():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
    hinted = {"done": False}

    def tracks(t):
        x = rfid1_x(t, T0)
        if x > 900:
            return [make_track(57, x, RFID1_Y)]
        if not hinted["done"]:
            hinted["done"] = True
            sim.f.notify_bridge(57, 92)         # Phase 27 bridge fired
        return [make_track(92, x, RFID1_Y)]

    sim.run(T0 + 5.0, tracks)
    assert phys(sim, 92) == 3
    reacquire = sim.f.log.events("IDENTITY_REACQUIRE")
    assert reacquire and reacquire[-1]["reason"] == "state_bridge"


def test_scenario5_removed_pallet_is_unbound_and_not_inherited():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)

    def tracks(t):
        x = rfid1_x(t, T0)
        if x > 900:
            return [make_track(57, x, RFID1_Y)]
        # pallet removed; a different, unknown pallet appears nearby later
        if t > T0 + 4.0:
            return [make_track(200, rfid1_x(t, T0, 1400.0), RFID1_Y)]
        return []

    sim.run(T0 + 6.0, tracks)
    assert sim.f.registry[3].visibility_state == "NOT_VISIBLE"
    assert sim.f.registry[3].trajectory is None
    assert phys(sim, 200) is None
    assert sim.f.log.events("IDENTITY_UNBOUND")


def test_scenario6_reinsertion_unknown_until_rfid():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
    # pallet removed at x<900 (~T0+3.3); it coasts up to bridge_hint_coast_sec (2 s) before unbinding
    sim.run(T0 + 6.0, lambda t: [make_track(57, rfid1_x(t, T0), RFID1_Y)] if rfid1_x(t, T0) > 900 else [])
    # pallet 3 reinserted at an arbitrary place on the bottom lane, outside RFID2
    t_start = sim.t
    assert sim.f.registry[3].visibility_state == "NOT_VISIBLE"
    sim.rfid_at(center_time_rfid2(t_start), "RFID2", 3)
    seen_before_rfid = []

    def tracks(t):
        x = rfid2_x(t, t_start)
        if x < 800:
            seen_before_rfid.append(phys(sim, 120))
        return [make_track(120, x, RFID2_Y)]

    sim.run(t_start + 3.0, tracks)
    assert all(v is None for v in seen_before_rfid) and seen_before_rfid
    assert phys(sim, 120) == 3


def test_scenario7_both_readers_update_one_global_registry():
    sim = Sim()
    plan = [  # (station, pallet, track_id)
        ("RFID1", 3, 10), ("RFID1", 5, 11), ("RFID2", 1, 12), ("RFID2", 2, 13),
        ("RFID2", 4, 14), ("RFID2", 3, 15), ("RFID1", 1, 16), ("RFID1", 3, 17),
    ]
    expected = {}
    for station, pallet, track_id in plan:
        start = sim.t
        if station == "RFID1":
            sim.rfid_at(center_time_rfid1(start), station, pallet)
            sim.run(start + 3.0, lambda t, s=start, tid=track_id: [make_track(tid, rfid1_x(t, s), RFID1_Y)])
            sim.run(sim.t + 0.1, lambda t, s=start, tid=track_id: [make_track(tid, rfid1_x(t, s), RFID1_Y)])
        else:
            sim.rfid_at(center_time_rfid2(start), station, pallet)
            sim.run(start + 3.0, lambda t, s=start, tid=track_id: [make_track(tid, rfid2_x(t, s), RFID2_Y)])
        # next pallet starts after this one left
    # every reading produced an assignment at the time; verify last owners via log
    assigns = sim.f.log.events("IDENTITY_ASSIGN")
    assert [(a["physical_id"], a["track_id"]) for a in assigns] == [(p, t) for _, p, t in plan]
    # trajectories died after leaving: pallets unbound but generations count readings
    assert sim.f.registry[3].generation == 3
    assert sim.f.registry[1].generation == 2


def test_scenario8_stale_owner_loses_physical_id():
    sim = Sim()
    # pallet 3 first confirmed at RFID1 on P57, which then keeps circulating (y=400 lane, visible)
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)

    def phase1(t):
        x = rfid1_x(t, T0)
        return [make_track(57, x, RFID1_Y if x > 1000 else 400.0)]

    sim.run(T0 + 2.5, phase1)
    assert phys(sim, 57) == 3
    # a different trajectory now proves it is pallet 3 at RFID2
    t_start = sim.t
    sim.rfid_at(center_time_rfid2(t_start), "RFID2", 3)
    last_x57 = rfid1_x(sim.t, T0)

    def phase2(t):
        return [
            make_track(57, max(100.0, last_x57 - SPEED * (t - t_start)), 400.0),
            make_track(77, rfid2_x(t, t_start), RFID2_Y),
        ]

    sim.run(t_start + 3.0, phase2)
    assert phys(sim, 77) == 3
    assert phys(sim, 57) is None
    assert sim.f.log.events("IDENTITY_CLEAR")


def test_scenario8b_rfid_overrides_wrong_belief():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 5)
    sim.run(T0 + 2.0, lambda t: [make_track(57, rfid1_x(t, T0), RFID1_Y)])
    assert phys(sim, 57) == 5
    # now the same trajectory passes RFID2... but it is in fact pallet 3 (RFID wins)
    t_start = sim.t
    traj = sim.f.track_to_traj[57]
    sim.rfid_at(center_time_rfid2(t_start), "RFID2", 3)
    # teleport the same track id onto the RFID2 lane
    sim.run(t_start + 3.0, lambda t: [make_track(57, rfid2_x(t, t_start), RFID2_Y)])
    assert traj.physical_id == 3
    assert sim.f.registry[5].trajectory is None
    assert sim.f.registry[3].trajectory is traj
    assert sim.f.log.events("IDENTITY_REMAP")[0]["old_physical"] == 5


def test_scenario9_repeated_reads_collapse():
    sim = Sim()
    tc = center_time_rfid1(T0)
    for k in range(6):
        sim.rfid_at(tc + 0.08 * k, "RFID1", 3)
    sim.run(T0 + 3.0, lambda t: [make_track(57, rfid1_x(t, T0), RFID1_Y)])
    assert phys(sim, 57) == 3
    assert len(sim.f.log.events("RFID_PASSAGE_MATCH")) == 1
    assert len(sim.f.log.events("IDENTITY_ASSIGN")) == 1
    assert len(sim.f.log.events("RFID_DEBOUNCE_DROP")) == 5


def test_scenario9b_later_legitimate_pass_accepted_again():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
    sim.run(T0 + 3.0, lambda t: [make_track(57, rfid1_x(t, T0), RFID1_Y)])
    start = sim.t
    sim.rfid_at(center_time_rfid1(start), "RFID1", 3)
    sim.run(start + 3.0, lambda t: [make_track(60, rfid1_x(t, start), RFID1_Y)])
    assert len(sim.f.log.events("RFID_PASSAGE_MATCH")) == 2
    assert sim.f.registry[3].generation == 2


def test_scenario10_long_run_bounded_state():
    rng = random.Random(7)
    sim = Sim()
    next_id = [1]
    live = {}       # track_id -> (start_time, lane, x0, ends_at)
    horizon = T0 + 1200.0   # 20 simulated minutes
    next_spawn = T0
    biggest = {"traj": 0, "track_map": 0, "passages": 0, "pending": 0, "recent": 0}
    while sim.t < horizon:
        if sim.t >= next_spawn:
            lane = rng.choice(["RFID1", "RFID2"])
            tid = next_id[0]
            next_id[0] += 1
            x0 = 1500.0 if lane == "RFID1" else 500.0
            dur = rng.uniform(2.0, 6.0)
            live[tid] = (sim.t, lane, x0, sim.t + dur)
            pallet = rng.choice([1, 2, 3, 4, 5, 6])
            if rng.random() < 0.8:
                if lane == "RFID1":
                    sim.rfid_at(sim.t + (x0 - 1153.0) / SPEED + rng.uniform(-0.05, 0.1), lane, pallet)
                else:
                    sim.rfid_at(sim.t + (853.5 - x0) / SPEED + rng.uniform(-0.05, 0.1), lane, pallet)
            next_spawn = sim.t + rng.uniform(1.5, 4.0)
        tracks = []
        for tid, (start, lane, x0, ends) in list(live.items()):
            if sim.t > ends:
                del live[tid]                       # manual removal / left the field of view
                continue
            if rng.random() < 0.01:
                continue                            # random one-frame detector miss
            x = rfid1_x(sim.t, start, x0) if lane == "RFID1" else rfid2_x(sim.t, start, x0)
            tracks.append(make_track(tid, x, RFID1_Y if lane == "RFID1" else RFID2_Y))
        sim.step(tracks)
        snap = sim.f.snapshot()
        biggest["traj"] = max(biggest["traj"], snap["trajectories"])
        biggest["track_map"] = max(biggest["track_map"], snap["track_map"])
        biggest["passages"] = max(biggest["passages"], snap["active_passages"])
        biggest["pending"] = max(biggest["pending"], snap["pending_events"])
        biggest["recent"] = max(biggest["recent"], snap["recent_passages"])
    assert biggest["traj"] <= 12 and biggest["track_map"] <= 12, biggest
    assert biggest["passages"] <= 8 and biggest["pending"] <= 6 and biggest["recent"] <= 10, biggest
    snap = sim.f.snapshot()
    assert snap["trajectories"] <= 12


# =====================================================================
# ADVERSARIAL CASES (section 38)
# =====================================================================

def test_A_bbox_jitter_around_entry_line_opens_one_passage():
    rng = random.Random(3)
    sim = Sim()

    def tracks(t):
        x = rfid1_x(t, T0)
        # hover around the entry line for ~1 s with +-4 px noise, then continue
        if 1183 - 40 < x < 1183 + 10:
            x = 1183 + rng.uniform(-4, 4)
        return [make_track(57, x, RFID1_Y + rng.uniform(-3, 3))]

    sim.run(T0 + 2.0, tracks)
    assert len(sim.f.log.events("PASSAGE_OPEN")) <= 1


def test_wrong_direction_crossing_opens_no_passage():
    sim = Sim()
    sim.run(T0 + 3.0, lambda t: [make_track(57, 900.0 + SPEED * (t - T0), RFID1_Y)])
    assert sim.f.log.events("PASSAGE_OPEN") == []


def test_other_lane_does_not_open_passage():
    sim = Sim()
    sim.run(T0 + 3.0, lambda t: [make_track(57, rfid1_x(t, T0), 300.0)])
    assert sim.f.log.events("PASSAGE_OPEN") == []


def test_E_old_event_cannot_overwrite_newer_mapping():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
    sim.run(T0 + 3.0, lambda t: [make_track(57, rfid1_x(t, T0), RFID1_Y)])
    generation = sim.f.registry[3].generation
    sim.f.push_rfid_event(RFIDEvent("RFID1", 3, T0 + 0.1))     # very old event
    sim.f.process_rfid_events(sim.t)
    assert sim.f.registry[3].generation == generation
    assert phys(sim, 57) == 3
    assert sim.f.log.events("RFID_STALE_DROP")


def test_ambiguous_event_is_not_forced():
    sim = Sim()
    tc = center_time_rfid1(T0)
    sim.rfid_at(tc, "RFID1", 3)
    # two pallets 6 px apart: genuinely equally plausible
    sim.run(T0 + 4.0, lambda t: [
        make_track(57, rfid1_x(t, T0, 1300.0), RFID1_Y),
        make_track(58, rfid1_x(t, T0, 1306.0), RFID1_Y),
    ])
    # duplicate-looking tracks are not de-duplicated by this layer (the vision
    # code does that upstream); here the contract is: never guess when ambiguous
    assert sim.f.log.events("RFID_PASSAGE_AMBIGUOUS"), "must be reported as ambiguous"
    assert phys(sim, 57) is None and phys(sim, 58) is None, "must NOT guess"
    assert not sim.f.log.events("IDENTITY_ASSIGN")


def test_I_passage_timeout_and_event_unmatched():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0) + 10.0, "RFID1", 3)          # nobody is there
    sim.run(T0 + 12.5, lambda t: [])
    assert sim.f.log.events("RFID_UNMATCHED")
    assert sim.f.snapshot()["pending_events"] == 0


def test_stuck_pallet_passage_expires():
    sim = Sim()

    def tracks(t):
        x = rfid1_x(t, T0)
        return [make_track(57, max(x, 1150.0), RFID1_Y)]   # stops inside the station

    sim.run(T0 + 6.0, tracks)
    assert sim.f.log.events("PASSAGE_EXPIRED")
    assert sim.f.snapshot()["active_passages"] == 0


def test_L_timing_offset_is_learned_per_station():
    sim = Sim()
    for k in range(7):
        start = sim.t
        sim.rfid_at(center_time_rfid1(start) + 0.15, "RFID1", (k % 6) + 1)
        sim.run(start + 2.6, lambda t, s=start, tid=100 + k: [make_track(tid, rfid1_x(t, s), RFID1_Y)])
    offset1 = sim.f.timing_offset("RFID1")
    assert abs(offset1 - 0.15) < 0.04, offset1
    assert sim.f.timing_offset("RFID2") == 0.0
    late = sim.f.log.events("RFID_PASSAGE_MATCH")[-1]
    assert abs(late["residual"]) < 0.04


def test_rfid_before_vision_entry_is_held_then_matched():
    sim = Sim()
    # antenna triggers 0.2 s BEFORE the vision entry line is crossed
    entry_time = T0 + (1300.0 - 1183.0) / SPEED
    sim.rfid_at(entry_time - 0.2, "RFID1", 4)
    sim.run(T0 + 3.0, lambda t: [make_track(57, rfid1_x(t, T0), RFID1_Y)])
    assert phys(sim, 57) == 4


def test_rfid_thread_failure_does_not_break_vision_side():
    sim = Sim()
    sim.run(T0 + 2.0, lambda t: [make_track(57, rfid1_x(t, T0), RFID1_Y)])
    assert phys(sim, 57) is None          # simply unknown, nothing raised


def test_unknown_station_or_pallet_events_ignored():
    sim = Sim()
    sim.f.push_rfid_event(RFIDEvent("RFID9", 3, T0))
    sim.f.push_rfid_event(RFIDEvent("RFID1", 9, T0))
    sim.step([])
    assert sim.f.log.events("RFID_UNKNOWN_STATION") and sim.f.log.events("RFID_UNKNOWN_PALLET")


def test_station_geometry_validation():
    bad = {"X": {"box": (0, 0, 10, 10), "direction": "LEFT_TO_RIGHT", "entry_x": 50, "exit_x": 10,
                 "y_min": 0, "y_max": 10}}
    try:
        FusionIdentityManager(FusionConfig(log_echo=False), stations=bad)
    except ValueError:
        return
    raise AssertionError("inconsistent entry/exit must be rejected")




def test_identity_survives_a_long_gap_when_the_phase27_bridge_hands_over():
    """Vision state is bridged for up to 2 s; the physical number must travel with it."""
    for hinted in (True, False):
        sim = Sim()
        sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
        gap_start = {}

        def tracks(t):
            x = rfid1_x(t, T0)
            if x > 900:
                return [make_track(57, x, RFID1_Y)]
            gap_start.setdefault("t", t)
            if t - gap_start["t"] < 1.4:                       # blind for 1.4 s
                return []
            if hinted and not gap_start.get("hinted"):
                gap_start["hinted"] = True
                sim.f.notify_bridge(57, 92)                    # Phase 27 bridge fired
            return [make_track(92, x, RFID1_Y)]

        sim.run(T0 + 6.0, tracks)
        if hinted:
            assert phys(sim, 92) == 3, "identity must follow the Phase 27 bridge"
            assert sim.f.registry[3].visibility_state == "ACTIVE"
        else:
            assert phys(sim, 92) is None, "without the bridge a long gap must NOT guess"


def test_late_bridge_hint_takes_over_a_young_unnumbered_trajectory():
    """Phase 31 lets the vision bridge retry, so the hint can arrive after the new id appeared."""
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
    state = {"hint_at": None}

    def tracks(t):
        x = rfid1_x(t, T0)
        if x > 900:
            return [make_track(57, x, RFID1_Y)]
        state.setdefault("t0", t)
        if t - state["t0"] < 0.3:
            return []
        state.setdefault("new_seen", t)
        if t - state["new_seen"] >= 0.15 and state["hint_at"] is None:     # bridge succeeds 5 frames later
            state["hint_at"] = t
            sim.f.notify_bridge(57, 92)
        return [make_track(92, x, RFID1_Y)]

    sim.run(T0 + 6.0, tracks)
    assert phys(sim, 92) == 3
    assert len(sim.f.trajectories) == 1, "the young duplicate trajectory must be merged away"


def test_late_bridge_hint_never_overrides_a_numbered_pallet():
    sim = Sim()
    sim.rfid_at(center_time_rfid1(T0), "RFID1", 3)
    sim.rfid_at(center_time_rfid2(T0), "RFID2", 5)

    def tracks(t):
        out = []
        x = rfid1_x(t, T0)
        if x > 900:
            out.append(make_track(57, x, RFID1_Y))
        out.append(make_track(80, rfid2_x(t, T0), RFID2_Y))       # a different, numbered pallet
        if t > T0 + 4.0 and x <= 900 and not getattr(sim, "hinted", False):
            sim.hinted = True
            sim.f.notify_bridge(57, 80)                            # wrong hint: 80 is established
        return out

    sim.run(T0 + 5.0, tracks)
    assert phys(sim, 80) == 5

def test_pallets_that_travel_beside_the_sensor_box_still_get_their_number():
    """From the real screenshot: pallet boxes are ~114 px tall, centred at y~540 (top lane) and
    y~873 (bottom lane) while the RFID boxes are centred at y=626 / y=775 (the sensors sit beside the track)."""
    for mode, expect_ok in (("overlap", True), ("center", False)):
        sim = Sim(lane_gate_mode=mode, rfid_pre_margin_sec=1.5, rfid_post_margin_sec=1.0,
                  pending_hold_sec=2.5, ambiguity_hold_sec=2.5, event_max_age_sec=5.0)
        sim.rfid_at(center_time_rfid1(T0) - 0.5, "RFID1", 3)
        sim.run(T0 + 3.0, lambda t: [make_track(41, rfid1_x(t, T0), 540.0, w=170, h=114)])
        start = sim.t
        sim.rfid_at(center_time_rfid2(start) - 0.2, "RFID2", 5)
        sim.run(start + 3.0, lambda t: [make_track(52, rfid2_x(t, start), 873.0, w=200, h=150)])
        passages = len(sim.f.log.events("PASSAGE_OPEN"))
        ids = [(a["track_id"], a["physical_id"]) for a in sim.f.log.events("IDENTITY_ASSIGN")]
        print(f"   gate={mode:8s}: passages opened={passages}  numbers assigned (track, pallet)={ids}")
        if expect_ok:
            assert ids == [(41, 3), (52, 5)], ids
        else:
            assert passages == 0 and ids == [], "the old centre gate rejects these pallets"


def test_console_explains_why_no_passage_opened():
    import contextlib, io
    sim = Sim()
    sim.f.log.echo = True
    sim.f.log.echo_events = {"PASSAGE_OPEN"}          # stage-35 style filter: diagnostics must still show
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # (a) pallet far above the station band, crossing its x range
        sim.run(T0 + 3.0, lambda t: [make_track(60, rfid1_x(t, T0), 300.0, w=170, h=114)])
        # (b) pallet on the right lane but moving the WRONG way (left -> right) on the top conveyor
        start = sim.t
        sim.run(start + 3.0, lambda t: [make_track(61, 1000.0 + SPEED * (t - start), 540.0, w=170, h=114)])
        # (c) pallet that is first seen already inside the station
        start2 = sim.t
        sim.run(start2 + 1.0, lambda t: [make_track(62, 1150.0 - SPEED * (t - start2) * 0.3, 540.0, w=170, h=114)])
    out = buf.getvalue()
    print("   console:", [l[:80] for l in out.strip().splitlines()])
    assert "STATION_LANE_MISS" in out and "STATION_NO_ENTRY" in out
    assert out.count("STATION_LANE_MISS") == 1, "once per pass, no spam"
    assert sim.f.log.events("PASSAGE_OPEN") == []


def test_observed_hardware_timing_matches_with_stage34_windows():
    """From the real reader check: RFID1 tag appears ~0.67 s before the pallet is level with the
    box (1.34 s dwell), RFID2 ~0.21 s before (0.42 s dwell); pallets arrive ~3.3 s apart."""
    stage34 = dict(rfid_pre_margin_sec=1.5, rfid_post_margin_sec=1.0, pending_hold_sec=2.5,
                   ambiguity_hold_sec=2.5, event_max_age_sec=5.0)
    cycle = [3, 4, 2, 5, 6, 1]
    for config, label in ((stage34, "stage-34 windows"), ({}, "default windows")):
        sim = Sim(**config)
        matched = 0
        for k in range(12):
            station = "RFID1" if k % 2 == 0 else "RFID2"
            pallet = cycle[k % 6]
            start = sim.t
            if station == "RFID1":
                center = center_time_rfid1(start)
                sim.rfid_at(center - 0.67, station, pallet)
                sim.run(start + 3.3, lambda t, s=start, tid=300 + k: [make_track(tid, rfid1_x(t, s), RFID1_Y)])
            else:
                center = center_time_rfid2(start)
                sim.rfid_at(center - 0.21, station, pallet)
                sim.run(start + 3.3, lambda t, s=start, tid=300 + k: [make_track(tid, rfid2_x(t, s), RFID2_Y)])
        matched = len(sim.f.log.events("IDENTITY_ASSIGN"))
        wrong = [a for a in sim.f.log.events("IDENTITY_ASSIGN")
                 if a["physical_id"] != cycle[(a["track_id"] - 300) % 6]]
        print(f"   {label}: {matched}/12 identities assigned, wrong={len(wrong)}")
        assert not wrong
        if label == "stage-34 windows":
            assert matched == 12
        else:
            assert matched < 12, "the old 0.25 s windows should miss RFID1's early reads"


def test_console_echo_can_be_limited_to_chosen_events():
    import contextlib, io
    log = fusion_mod.EventLog(echo=True, echo_events=("PASSAGE_OPEN",))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        log.log("TRACK_APPEARED", track_id=1)
        log.log("PASSAGE_OPEN", station="RFID1")
    assert "PASSAGE_OPEN" in buf.getvalue() and "TRACK_APPEARED" not in buf.getvalue()
    assert len(log.records) == 2          # nothing is lost from the log itself


# =====================================================================
# RFID READER THREAD (fake pylogix)
# =====================================================================

def _run_reader_with_script(script, max_wait=3.0):
    """script: list of per-poll dicts station -> 4-tuple, or the string 'ERR'."""
    import queue as queue_mod
    import threading
    import time as time_mod
    import types

    rfid1 = fusion_mod.STATION_PLC_TAGS["RFID1"]
    rfid2 = fusion_mod.STATION_PLC_TAGS["RFID2"]
    state = {"i": 0}
    done = threading.Event()

    class Resp:
        def __init__(self, value, status="Success"):
            self.Value, self.Status = value, status

    class FakePLC:
        IPAddress = None
        SocketTimeout = None

        def Read(self, tags):
            i = state["i"]
            if i >= len(script):
                done.set()
                i = len(script) - 1
            else:
                state["i"] += 1
            step = script[i]
            if step == "ERR":
                return [Resp(None, "Connection failed") for _ in tags]
            words = list(step.get("RFID1", (0, 0, 0, 0))) + list(step.get("RFID2", (0, 0, 0, 0)))
            return [Resp(w) for w in words]

        def Close(self):
            pass

    fake = types.ModuleType("pylogix")
    fake.PLC = FakePLC
    sys.modules["pylogix"] = fake

    q = queue_mod.Queue()
    log = fusion_mod.EventLog(echo=False)
    worker = fusion_mod.RFIDReaderWorker(q, log, poll_sec=0.001)
    worker.start()
    done.wait(max_wait)
    time_mod.sleep(0.05)
    worker.stop()
    worker.join(timeout=2.0)
    del sys.modules["pylogix"]
    events = []
    while not q.empty():
        events.append(q.get())
    return events, log, worker


def test_reader_one_event_per_tag_presence_and_rearm():
    tag3 = (-8188, 336, -18499, 21259)
    tag5 = (-8188, 336, -18498, 5934)
    none = (0, 0, 0, 0)
    script = (
        [{"RFID1": none}] * 3
        + [{"RFID1": tag3}] * 6              # tag stays in field: ONE event
        + [{"RFID1": none}] * 2              # leaves -> re-armed
        + [{"RFID1": tag3}] * 3              # same pallet passes again: new event
        + [{"RFID1": tag5}] * 2              # tag switch without NO_TAG: new event
        + [{"RFID2": tag3}] * 2              # other station is independent
        + [{"RFID1": (1, 2, 3, 4)}] * 2      # unknown tag -> logged, no event
    )
    events, log, worker = _run_reader_with_script(script)
    got = [(e.station, e.pallet_id) for e in events]
    assert got == [("RFID1", 3), ("RFID1", 3), ("RFID1", 5), ("RFID2", 3)], got
    assert log.events("RFID_UNKNOWN_TAG")
    stamps = [e.timestamp for e in events]
    assert stamps == sorted(stamps)


def test_reader_survives_plc_errors():
    tag3 = (-8188, 336, -18499, 21259)
    script = ["ERR", "ERR", {"RFID1": tag3}, {"RFID1": tag3}]
    events, log, worker = _run_reader_with_script(script, max_wait=6.0)
    assert [(e.station, e.pallet_id) for e in events] == [("RFID1", 3)]
    assert worker.error_count >= 1 and log.events("RFID_READER_ERROR")


def test_reader_missing_pylogix_disables_only_rfid():
    import queue as queue_mod
    sys.modules["pylogix"] = None            # forces ImportError
    q = queue_mod.Queue()
    log = fusion_mod.EventLog(echo=False)
    worker = fusion_mod.RFIDReaderWorker(q, log)
    worker.start()
    worker.join(timeout=2.0)
    del sys.modules["pylogix"]
    assert not worker.is_alive()
    assert log.events("RFID_READER_DISABLED")


# =====================================================================

def _all_tests():
    return [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failures = 0
    for name, fn in _all_tests():
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
            traceback.print_exc(limit=3)
    print(f"\n{len(_all_tests()) - failures}/{len(_all_tests())} passed")
    sys.exit(1 if failures else 0)

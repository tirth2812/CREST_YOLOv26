"""
CREST RFID <-> VISION PHYSICAL IDENTITY FUSION LAYER
====================================================

Pure-Python identity plumbing.  It does NOT import cv2 / ultralytics at module
import time and it does NOT touch any detector / classifier / state-machine
logic of the Phase 27 vision system.

Concepts
--------
  Pxx (tracker id)      temporary observation id produced by the tracker
  Trajectory            short-term continuity object above Pxx (survives a
                        brief Pxx change)
  StationPassage        one physically valid pass of a trajectory through an
                        RFID station (entry side -> inside -> exit side)
  PhysicalPallet        global registry entry for pallets 1..6 shared by BOTH
                        stations

RFID read  ->  StationPassage  ->  Trajectory  ->  physical pallet id
(never "nearest box").

Threading
---------
  RFIDReaderWorker (background thread)  --queue-->  main thread
  Only the main thread mutates identity state (FusionIdentityManager).

All clocks are time.monotonic() seconds (same process, same clock domain).
"""

from __future__ import annotations

import itertools
import json
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ============================================================
# RFID DECODING (taken unchanged from the project notes)
# ============================================================

RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}

NO_TAG = (0, 0, 0, 0)

PLC_IP = "192.168.82.231"

STATION_PLC_TAGS = {
    "RFID1": [f"IO_Link_Master_ST1_to_ST4:I1.Data[{i}]" for i in (96, 97, 98, 99)],
    "RFID2": [f"IO_Link_Master_ST1_to_ST4:I1.Data[{i}]" for i in (128, 129, 130, 131)],
}


# ============================================================
# STATION GEOMETRY (one generic structure for every station)
# ============================================================

DEFAULT_STATIONS = {
    "RFID1": {
        "box": (1123, 578, 1183, 675),
        "direction": "RIGHT_TO_LEFT",
        "entry_x": 1183,
        "exit_x": 1123,
        "y_min": 578,
        "y_max": 675,
    },
    "RFID2": {
        "box": (823, 734, 884, 815),
        "direction": "LEFT_TO_RIGHT",
        "entry_x": 823,
        "exit_x": 884,
        "y_min": 734,
        "y_max": 815,
    },
}


@dataclass(frozen=True)
class StationConfig:
    name: str
    box: Tuple[int, int, int, int]
    direction: str
    entry_x: float
    exit_x: float
    y_min: float
    y_max: float

    @property
    def sign(self) -> int:
        """+1 when travel is LEFT->RIGHT, -1 when RIGHT->LEFT."""
        return 1 if self.direction == "LEFT_TO_RIGHT" else -1

    # Progress coordinate u = sign * x always INCREASES in the travel direction,
    # so every station can use the same entry/exit logic.
    @property
    def entry_u(self) -> float:
        return self.sign * self.entry_x

    @property
    def exit_u(self) -> float:
        return self.sign * self.exit_x

    @property
    def center_u(self) -> float:
        return 0.5 * (self.entry_u + self.exit_u)

    @staticmethod
    def from_dict(name: str, d: dict) -> "StationConfig":
        if d["direction"] not in ("LEFT_TO_RIGHT", "RIGHT_TO_LEFT"):
            raise ValueError(f"{name}: bad direction {d['direction']!r}")
        st = StationConfig(
            name=name,
            box=tuple(d["box"]),
            direction=d["direction"],
            entry_x=float(d["entry_x"]),
            exit_x=float(d["exit_x"]),
            y_min=float(d["y_min"]),
            y_max=float(d["y_max"]),
        )
        if not st.entry_u < st.exit_u:
            raise ValueError(
                f"{name}: entry_x/exit_x are inconsistent with {st.direction}"
            )
        return st


# ============================================================
# TUNABLES (all in one place)
# ============================================================

@dataclass
class FusionConfig:
    # --- RFID events -------------------------------------------------
    rfid_debounce_sec: float = 0.4          # merge repeated reads of one tag at one station
    event_max_age_sec: float = 2.0          # drop RFID events older than this when processed
    rfid_pre_margin_sec: float = 0.25       # read may precede the vision entry by this much
    rfid_post_margin_sec: float = 0.25      # ... or follow the vision exit by this much
    rfid_resolve_delay_sec: float = 0.15    # wait this long after a read before resolving
    ambiguity_hold_sec: float = 0.70        # max wait for an ambiguous event (hard budget)
    pending_hold_sec: float = 0.70          # max wait for an event with NO candidate passage
    ambiguity_margin_sec: float = 0.10      # best vs second-best cost margin
    offset_max_abs_sec: float = 1.0         # ignore timing-offset samples beyond this
    offset_min_samples: int = 5             # samples before the learned offset is used
    offset_history: int = 40

    # --- station passage geometry -----------------------------------
    boundary_hysteresis_px: float = 8.0     # "clearly outside/inside" band half-width
    y_margin_px: float = 25.0               # lane gate margin around station y range
    # The calibrated box marks the RFID sensor, which sits beside the track, so the
    # pallet CENTRE is ~80-100 px away from the box centre.  "overlap" = the pallet
    # box touches the station's vertical band (+margin); "center" = centre inside it.
    lane_gate_mode: str = "overlap"         # "overlap" | "center" | "both"
    entry_marker_max_age_sec: float = 2.0   # "was clearly outside" memory
    passage_timeout_sec: float = 4.0        # force-close a passage that never exits
    passage_retention_sec: float = 1.5      # keep EXITED passages matchable this long
    station_max_active_passages: int = 0    # 0 = unlimited (only WARN on overlap)

    # --- trajectory continuity --------------------------------------
    track_coast_sec: float = 0.5            # keep a lost trajectory this long
    passage_coast_sec: float = 1.0          # ... longer while it owns an active passage
    max_reacquire_distance_px: float = 80.0
    max_reacquire_y_delta_px: float = 40.0
    bbox_size_ratio_limit: float = 0.60     # min(area)/max(area)
    reacquire_min_speed_px_s: float = 20.0  # direction test only above this speed
    reacquire_direction_tol_px: float = 10.0
    reacquire_unique_margin: float = 0.25   # normalised cost margin vs. runner-up
    bridge_hint_max_distance_px: float = 160.0
    # A lost trajectory (with its physical id) is kept this long so the Phase 27
    # state bridge can still hand it to a new tracker id.  Only the bridge hint
    # may re-link after track_coast_sec; the layer's own motion matching may not.
    bridge_hint_coast_sec: float = 2.0

    # --- RFID -> pallet matching ---------------------------------------
    # "closest" (default): while a tag is in a reader's field a pallet is physically
    # at that reader (a number is only reported when a tag is really there).  The
    # number goes to the pallet whose centre came CLOSEST to the reader, measured
    # along the conveyor (x) in that reader's lane, during the time the tag was
    # read.  Pallet size / height do not matter.
    # "passage": the older entry/exit state machine (kept for reference).
    match_mode: str = "closest"
    closest_pre_sec: float = 0.4            # look this far BEFORE the tag appeared
    closest_post_sec: float = 0.4           # ... and this far AFTER it left
    closest_max_dwell_sec: float = 2.0      # if the "tag left" signal never comes
    closest_max_dx_px: float = 250.0        # nearest pallet must come at least this close
    closest_ambiguity_px: float = 40.0      # two pallets this close to equally near = skip
    # How much LATER than the RFID read the pallet shows up in the camera timeline
    # (camera + inference delay, antenna position).  Measured, not guessed: run with
    # calibrate=True and ONE pallet; see calibration_summary().  {"RFID1": 0.9, ...}
    vision_delay_sec: Optional[dict] = None
    calibrate: bool = False                 # measure vision_delay_sec, assign nothing
    calibration_search_sec: float = 5.0     # how far around the read to look for the pallet

    # --- logging ----------------------------------------------------
    log_echo: bool = True
    log_echo_events: Optional[tuple] = None   # None = print every event; else only these
    log_path: Optional[str] = None


# ============================================================
# STRUCTURED LOG
# ============================================================

class EventLog:
    """Non-blocking structured log: JSONL writer thread + bounded memory ring."""

    ALWAYS_ECHO = (
        "STATION_LANE_MISS", "STATION_NO_ENTRY",
        "RFID_CLOSEST_MATCH", "RFID_NO_PALLET", "RFID_MATCH_AMBIGUOUS",
        "RFID_CALIBRATION_SAMPLE", "RFID_CALIBRATION_SKIP",
        "RFID_READER_OK", "RFID_READER_STATS",
    )
    # vision-side bookkeeping: kept in the log file, never printed
    NEVER_ECHO = (
        "IDENTITY_REACQUIRE", "IDENTITY_COAST_START", "TRACK_APPEARED", "TRACK_LOST",
        "TRACK_DEAD", "PASSAGE_TRACK_SWITCH",
    )

    def __init__(
        self,
        path: Optional[str] = None,
        echo: bool = True,
        keep: int = 5000,
        echo_events: Optional[tuple] = None,
    ):
        self.records: deque = deque(maxlen=keep)
        self.echo = echo
        self.echo_events = set(echo_events) if echo_events else None
        self._q: "queue.Queue" = queue.Queue()
        self._thread = None
        self._path = path
        if path:
            self._thread = threading.Thread(
                target=self._writer, name="crest-fusion-log", daemon=True
            )
            self._thread.start()

    def log(self, event: str, **fields):
        rec = {"event": event}
        for k, v in fields.items():
            rec[k] = round(v, 4) if isinstance(v, float) else v
        self.records.append(rec)
        if self._thread is not None:
            self._q.put(rec)
        if self.echo and event not in self.NEVER_ECHO and (
            self.echo_events is None
            or event in self.echo_events
            or event in self.ALWAYS_ECHO
        ):
            print("[FUSION] " + event + " " + " ".join(f"{k}={v}" for k, v in rec.items() if k != "event"))

    def events(self, name: str) -> List[dict]:
        return [r for r in self.records if r["event"] == name]

    def _writer(self):
        with open(self._path, "a", encoding="utf-8") as handle:
            while True:
                rec = self._q.get()
                if rec is None:
                    break
                handle.write(json.dumps(rec) + "\n")
                handle.flush()

    def close(self):
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=2.0)


# ============================================================
# RFID EVENT + READER THREAD
# ============================================================

@dataclass(frozen=True)
class RFIDEvent:
    station: str
    pallet_id: int
    timestamp: float                     # time.monotonic() right after the PLC read returned
    raw: Tuple[int, ...] = ()


@dataclass(frozen=True)
class RFIDLeave:
    """The tag left the reader field (NO_TAG or a different tag appeared)."""
    station: str
    pallet_id: int
    timestamp: float


class RFIDReaderWorker(threading.Thread):
    """
    Polls the PLC in the background and ONLY emits timestamped RFIDEvent objects.

    One event is emitted when a tag APPEARS (NO_TAG -> tag, or tag A -> tag B).
    While the same tag stays in the field no further event is emitted; a return
    to NO_TAG re-arms the station.  Never touches vision / identity state.
    """

    def __init__(
        self,
        out_queue: "queue.Queue",
        log: EventLog,
        plc_ip: str = PLC_IP,
        station_tags: Optional[dict] = None,
        poll_sec: float = 0.02,
        emit_repeats: bool = False,
        timeout_sec: float = 5.0,
        stats_sec: float = 10.0,
    ):
        super().__init__(name="crest-rfid-reader", daemon=True)
        self.out_queue = out_queue
        self.log = log
        self.plc_ip = plc_ip
        self.station_tags = station_tags or STATION_PLC_TAGS
        self.poll_sec = poll_sec
        self.emit_repeats = emit_repeats
        self.timeout_sec = timeout_sec
        self.stats_sec = stats_sec
        self.stop_event = threading.Event()
        self.last_ok_time = 0.0
        self.error_count = 0
        self.connected = False
        self._last_value: Dict[str, tuple] = {s: NO_TAG for s in self.station_tags}

    def stop(self):
        self.stop_event.set()

    def _read_all(self, comm, flat_tags):
        """(values_by_station | None, set_of_problems).  One round trip for all words."""
        responses = comm.Read(flat_tags)

        if not isinstance(responses, (list, tuple)):
            responses = [responses]

        if len(responses) != len(flat_tags):
            return None, {f"expected {len(flat_tags)} answers, got {len(responses)}"}

        problems = set()
        values = {}
        index = 0
        for station, tags in self.station_tags.items():
            words = []
            for _ in tags:
                response = responses[index]
                index += 1
                if response.Status != "Success":
                    problems.add(str(response.Status))
                elif response.Value is None:
                    problems.add("no value")
                words.append(response.Value)
            values[station] = tuple(words)

        return (None, problems) if problems else (values, problems)

    def run(self):
        try:
            from pylogix import PLC
        except Exception as error:  # vision must keep running without RFID
            self.log.log("RFID_READER_DISABLED", reason=f"pylogix import failed: {error}")
            return

        flat_tags = []
        for station, tags in self.station_tags.items():
            flat_tags.extend(tags)

        comm = None
        failures = 0                       # consecutive failed polls
        was_ok = False
        window_start = time.monotonic()
        window_polls = 0
        window_errors = 0
        last_poll = None
        max_gap = 0.0

        while not self.stop_event.is_set():
            problems = set()
            values = None
            stamp = time.monotonic()

            try:
                if comm is None:
                    comm = PLC()
                    comm.IPAddress = self.plc_ip
                    comm.SocketTimeout = self.timeout_sec

                values, problems = self._read_all(comm, flat_tags)
                stamp = time.monotonic()  # taken as soon as the read returns

            except Exception as error:
                values, problems = None, {f"{type(error).__name__}: {error}"}

            # ---- failed poll: retry at once, reconnect only if it keeps failing
            if values is None:
                failures += 1
                self.error_count += 1
                window_errors += 1
                self.connected = False

                if failures in (1, 5, 20) or failures % 200 == 0:
                    self.log.log(
                        "RFID_READER_ERROR", reason=f"PLC read not successful: {sorted(problems)}",
                        consecutive=failures, count=self.error_count,
                    )

                if failures >= 3:
                    try:
                        if comm is not None:
                            comm.Close()
                    except Exception:
                        pass
                    comm = None
                    self.stop_event.wait(min(1.0, 0.2 * (failures - 2)))
                else:
                    self.stop_event.wait(0.05)
                continue

            # ---- good poll
            if failures or not was_ok:
                self.log.log(
                    "RFID_READER_OK",
                    note=("connected" if not was_ok else f"recovered after {failures} failed polls"),
                )
            failures = 0
            was_ok = True
            self.connected = True
            self.last_ok_time = stamp
            window_polls += 1

            if last_poll is not None:
                max_gap = max(max_gap, stamp - last_poll)
            last_poll = stamp

            for station, raw in values.items():
                self._handle_value(station, raw, stamp)

            if stamp - window_start >= self.stats_sec:
                span = stamp - window_start
                self.log.log(
                    "RFID_READER_STATS", polls_per_sec=round(window_polls / span, 1),
                    longest_gap_sec=round(max_gap, 3), failed_polls=window_errors,
                    note="longest_gap_sec large = the PLC is not being read often enough",
                )
                window_start = stamp
                window_polls = 0
                window_errors = 0
                max_gap = 0.0

            self.stop_event.wait(self.poll_sec)

    def _handle_value(self, station: str, raw: tuple, stamp: float):
        previous = self._last_value[station]
        self._last_value[station] = raw

        # the previous tag left the field (gone, or replaced by another tag)
        if previous != NO_TAG and raw != previous:
            previous_pallet = RFID_MAP.get(previous)
            if previous_pallet is not None:
                self.out_queue.put(RFIDLeave(station=station, pallet_id=previous_pallet, timestamp=stamp))

        if raw == NO_TAG:
            return

        if raw == previous and not self.emit_repeats:
            return

        pallet = RFID_MAP.get(raw)
        if pallet is None:
            self.log.log("RFID_UNKNOWN_TAG", station=station, raw=list(raw), t=stamp)
            return

        self.out_queue.put(RFIDEvent(station=station, pallet_id=pallet, timestamp=stamp, raw=raw))


# ============================================================
# SMALL GEOMETRY / MATH HELPERS
# ============================================================

def load_vision_delay(path) -> dict:
    """{"RFID1": seconds, ...} from a calibration file, or {} if there is none."""
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return {name: float(v["vision_delay_sec"]) for name, v in data.items()}
    except Exception:
        return {}


def box_center(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, (x2 - x1) * (y2 - y1))


def size_ratio(box_a, box_b) -> float:
    a, b = box_area(box_a), box_area(box_b)
    return min(a, b) / max(a, b)


def interpolate_crossing(p0: Tuple[float, float], p1: Tuple[float, float], boundary_u: float) -> float:
    """Time at which u crosses boundary_u on the segment p0=(t0,u0) -> p1=(t1,u1)."""
    t0, u0 = p0
    t1, u1 = p1
    if u1 == u0:
        return t1
    alpha = (boundary_u - u0) / (u1 - u0)
    alpha = min(1.0, max(0.0, alpha))
    return t0 + alpha * (t1 - t0)


def assign_min_cost(rows: List, cols: List, cost: Dict[Tuple, float], skip_cost: float = 1e3) -> Dict:
    """
    Exhaustive min-cost one-to-one assignment for a handful of rows/cols.
    Only pairs present in `cost` are allowed; a row may stay unassigned at
    `skip_cost`.  (Tracks/passages per station are <= a few, so this is cheap.)
    """
    rows = list(rows)[:6]
    best = {"total": math.inf, "pairs": {}}

    def search(i, used, total, pairs):
        if total >= best["total"]:
            return
        if i == len(rows):
            best["total"] = total
            best["pairs"] = dict(pairs)
            return
        row = rows[i]
        options = [(cost[(row, c)], c) for c in cols if (row, c) in cost and c not in used]
        options.sort(key=lambda item: item[0])
        for pair_cost, col in options[:4]:
            used.add(col)
            pairs[row] = col
            search(i + 1, used, total + pair_cost, pairs)
            del pairs[row]
            used.discard(col)
        search(i + 1, used, total + skip_cost, pairs)

    search(0, set(), 0.0, {})
    return best["pairs"]


# ============================================================
# DATA OBJECTS
# ============================================================

class Trajectory:
    """Short-term continuity above the tracker's Pxx id."""

    def __init__(self, trajectory_id: int, track_id: int, box, t: float, stations: List[StationConfig]):
        self.id = trajectory_id
        self.track_id: Optional[int] = track_id      # current/last Pxx
        self.state = "ACTIVE"                         # ACTIVE | COASTING
        self.physical_id: Optional[int] = None
        self.identity_source: Optional[str] = None
        self.box = tuple(box)
        self.center = box_center(box)
        self.velocity = (0.0, 0.0)                    # px / s
        self.first_seen = t
        self.last_seen = t
        self.coast_start: Optional[float] = None
        self.track_id_history: List[int] = [track_id]
        self.station_state = {s.name: {"pre": None, "outside_t": None} for s in stations}
        self.passages: Dict[str, "StationPassage"] = {}
        self.history: deque = deque(maxlen=400)       # (time, centre_x, centre_y)
        self.history.append((t, self.center[0], self.center[1]))

    def observe(self, box, t: float):
        new_center = box_center(box)
        dt = t - self.last_seen
        if 1e-3 < dt <= 0.5:
            inst = ((new_center[0] - self.center[0]) / dt, (new_center[1] - self.center[1]) / dt)
            if self.velocity == (0.0, 0.0):
                self.velocity = inst
            else:
                self.velocity = (0.6 * self.velocity[0] + 0.4 * inst[0], 0.6 * self.velocity[1] + 0.4 * inst[1])
        self.box = tuple(box)
        self.center = new_center
        self.last_seen = t
        self.history.append((t, new_center[0], new_center[1]))


@dataclass
class StationPassage:
    passage_id: int
    station: str
    state: str                                  # ENTERED | INSIDE | EXITED | EXPIRED
    trajectory: Optional[Trajectory]
    entry_track_id: Optional[int]
    current_track_id: Optional[int]
    entry_time: float
    entry_position: Tuple[float, float]
    last_position: Tuple[float, float]
    last_seen_time: float
    exit_time: Optional[float] = None
    center_time: Optional[float] = None
    closed_time: Optional[float] = None
    physical_id: Optional[int] = None
    rfid_timestamp: Optional[float] = None
    rfid_residual: Optional[float] = None
    rfid_confident: bool = False
    offset_learned: bool = False
    track_history: List[int] = field(default_factory=list)
    pre_exit: Optional[Tuple[float, float]] = None
    prev_sample: Optional[Tuple[float, float]] = None
    completed: bool = False

    @property
    def open(self) -> bool:
        return self.state in ("ENTERED", "INSIDE")


@dataclass
class PhysicalPallet:
    physical_id: int
    trajectory: Optional[Trajectory] = None
    current_track_id: Optional[int] = None
    visibility_state: str = "NOT_VISIBLE"       # ACTIVE | COASTING | NOT_VISIBLE
    last_rfid_station: Optional[str] = None
    last_rfid_timestamp: Optional[float] = None
    last_seen_timestamp: Optional[float] = None
    last_position: Optional[Tuple[float, float]] = None
    generation: int = 0


# ============================================================
# FUSION MANAGER
# ============================================================

class FusionIdentityManager:
    """
    Frame-loop usage (main thread only):

        fusion.update_tracks(frame_timestamp, confirmed_tracks)
        fusion.process_rfid_events(time.monotonic())
        phys = fusion.get_physical_id(track_id)
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        stations: Optional[dict] = None,
        log: Optional[EventLog] = None,
        physical_ids=(1, 2, 3, 4, 5, 6),
    ):
        self.cfg = config or FusionConfig()
        station_dict = stations or DEFAULT_STATIONS
        self.stations: List[StationConfig] = [
            StationConfig.from_dict(name, data) for name, data in station_dict.items()
        ]
        self.station_by_name = {s.name: s for s in self.stations}
        self.log = log or EventLog(
            path=self.cfg.log_path,
            echo=self.cfg.log_echo,
            echo_events=self.cfg.log_echo_events,
        )

        self.event_queue: "queue.Queue" = queue.Queue()
        self.registry: Dict[int, PhysicalPallet] = {p: PhysicalPallet(p) for p in physical_ids}

        self.trajectories: Dict[int, Trajectory] = {}
        self.track_to_traj: Dict[int, Trajectory] = {}
        self._next_traj_id = 1
        self._next_passage_id = 1

        self.active_passages: List[StationPassage] = []
        self.recent_passages: deque = deque()

        self.pending: List[RFIDEvent] = []
        self._reads: List[dict] = []              # closest-approach mode: {"event", "leave"}
        self._calibration: Dict[str, List[dict]] = {s.name: [] for s in self.stations}
        self._last_read: Dict[Tuple[str, int], float] = {}
        self._bridge_hints: List[Tuple[int, int]] = []

        self._offsets: Dict[str, deque] = {
            s.name: deque(maxlen=self.cfg.offset_history) for s in self.stations
        }
        self._vision_time = 0.0

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def push_rfid_event(self, event: RFIDEvent):
        """Thread-safe; called by the RFID thread (or tests)."""
        self.event_queue.put(event)

    def notify_bridge(self, old_track_id: int, new_track_id: int):
        """Called from the (unchanged) Phase 27 state bridge."""
        self._bridge_hints.append((old_track_id, new_track_id))

    def get_physical_id(self, track_id: int) -> Optional[int]:
        traj = self.track_to_traj.get(track_id)
        if traj is None:
            return None
        return traj.physical_id

    def get_track_info(self, track_id: int) -> dict:
        traj = self.track_to_traj.get(track_id)
        if traj is None:
            return {}
        return {
            "trajectory": traj.id,
            "physical_id": traj.physical_id,
            "source": traj.identity_source,
            "state": traj.state,
        }

    def timing_offset(self, station: str) -> float:
        samples = self._offsets[station]
        if len(samples) < self.cfg.offset_min_samples:
            return 0.0
        ordered = sorted(samples)
        return ordered[len(ordered) // 2]

    def snapshot(self) -> dict:
        return {
            "trajectories": len(self.trajectories),
            "track_map": len(self.track_to_traj),
            "active_passages": len(self.active_passages),
            "recent_passages": len(self.recent_passages),
            "pending_events": len(self.pending) + len(self._reads),
            "registry": {
                p: (pal.visibility_state, pal.trajectory.id if pal.trajectory else None, pal.generation)
                for p, pal in self.registry.items()
            },
        }

    # ----------------------------------------------------------
    # VISION SIDE
    # ----------------------------------------------------------

    def update_tracks(self, now: float, tracks: List[dict]):
        """`tracks`: iterable of {"track_id": int, "box": (x1,y1,x2,y2)} (post duplicate filter)."""
        self._vision_time = now
        current = {int(t["track_id"]): tuple(t["box"]) for t in tracks}

        self._apply_bridge_hints(now, current)

        updated: List[Trajectory] = []
        new_track_ids: List[int] = []

        for track_id, box in current.items():
            traj = self.track_to_traj.get(track_id)
            if traj is None:
                new_track_ids.append(track_id)
                continue
            if traj.state == "COASTING":
                self._reactivate(traj, track_id, now, "same_track_id")
            traj.observe(box, now)
            traj.track_id = track_id
            updated.append(traj)

        # trajectories whose current Pxx vanished this frame start coasting
        for traj in list(self.trajectories.values()):
            if traj.state == "ACTIVE" and (traj.track_id not in current):
                traj.state = "COASTING"
                traj.coast_start = now
                self.log.log(
                    "TRACK_LOST", t=now, track_id=traj.track_id, trajectory_id=traj.id,
                    physical_id=traj.physical_id, center=_pt(traj.center),
                    passages=sorted(traj.passages),
                )
                if traj.physical_id is not None:
                    self._set_visibility(traj, "COASTING")
                    self.log.log(
                        "IDENTITY_COAST_START", t=now, trajectory_id=traj.id,
                        physical_id=traj.physical_id, track_id=traj.track_id,
                    )

        # reacquire coasting trajectories with brand-new Pxx ids (global assignment)
        coasting = [t for t in self.trajectories.values() if t.state == "COASTING"]
        unmatched_new = list(new_track_ids)
        if coasting and new_track_ids:
            links = self._reacquire(now, coasting, {tid: current[tid] for tid in new_track_ids})
            for track_id, (traj, reason, cost) in links.items():
                self._rebind_track(traj, track_id, now, reason, cost)
                traj.observe(current[track_id], now)
                updated.append(traj)
                unmatched_new.remove(track_id)

        for track_id in unmatched_new:
            traj = Trajectory(self._next_traj_id, track_id, current[track_id], now, self.stations)
            self._next_traj_id += 1
            self.trajectories[traj.id] = traj
            self.track_to_traj[track_id] = traj
            self.log.log(
                "TRACK_APPEARED", t=now, track_id=track_id, trajectory_id=traj.id,
                center=_pt(traj.center),
            )
            updated.append(traj)

        if self.cfg.match_mode == "passage":
            for traj in updated:
                self._update_stations(now, traj)

        self._expire_trajectories(now)
        self._maintain_passages(now)

    # --- bridge hints (from the proven Phase 27 state bridge) -----------

    def _apply_bridge_hints(self, now: float, current: Dict[int, tuple]):
        hints, self._bridge_hints = self._bridge_hints, []
        for old_id, new_id in hints:
            traj = self.track_to_traj.get(old_id)
            if traj is None or new_id not in current:
                continue
            existing = self.track_to_traj.get(new_id)
            if existing is traj:
                continue
            if existing is not None:
                # The Phase 27 bridge may hand the old state over a few frames AFTER
                # the new id appeared (it can retry while the id is young).  Only a
                # young trajectory that has no physical number yet may be taken over.
                if (
                    existing.physical_id is not None
                    or now - existing.first_seen > self.cfg.bridge_hint_coast_sec
                ):
                    self.log.log(
                        "BRIDGE_HINT_REJECTED", t=now, old_track=old_id, new_track=new_id,
                        reason="new id already owns an identity or is established",
                    )
                    continue
                self._kill_trajectory(existing, now, "merged_into_bridged_trajectory")
            new_center = box_center(current[new_id])
            gap = max(0.0, min(now - traj.last_seen, self.cfg.bridge_hint_coast_sec))
            predicted = (
                traj.center[0] + traj.velocity[0] * gap,
                traj.center[1] + traj.velocity[1] * gap,
            )
            distance = min(
                math.hypot(new_center[0] - traj.center[0], new_center[1] - traj.center[1]),
                math.hypot(new_center[0] - predicted[0], new_center[1] - predicted[1]),
            )
            if distance > self.cfg.bridge_hint_max_distance_px:
                self.log.log(
                    "BRIDGE_HINT_REJECTED", t=now, old_track=old_id, new_track=new_id,
                    distance=distance,
                )
                continue
            if traj.state == "COASTING":
                self._reactivate(traj, new_id, now, "state_bridge")
            self._rebind_track(traj, new_id, now, "state_bridge", 0.0)

    # --- continuity ------------------------------------------------------

    def _coast_limit(self, traj: Trajectory) -> float:
        if any(p.open for p in traj.passages.values()):
            return self.cfg.passage_coast_sec
        return self.cfg.track_coast_sec

    def _reacquire_cost(self, traj: Trajectory, box, now: float) -> Optional[float]:
        cfg = self.cfg
        dt = max(0.0, now - traj.last_seen)
        if dt > self._coast_limit(traj):
            return None

        vx, vy = traj.velocity
        predicted = (traj.center[0] + vx * dt, traj.center[1] + vy * dt)
        center = box_center(box)
        dx, dy = center[0] - predicted[0], center[1] - predicted[1]
        distance = math.hypot(dx, dy)

        if distance > cfg.max_reacquire_distance_px or abs(dy) > cfg.max_reacquire_y_delta_px:
            return None

        ratio = size_ratio(traj.box, box)
        if ratio < cfg.bbox_size_ratio_limit:
            return None

        speed = math.hypot(vx, vy)
        if speed >= cfg.reacquire_min_speed_px_s:
            along = ((center[0] - traj.center[0]) * vx + (center[1] - traj.center[1]) * vy) / speed
            if along < -cfg.reacquire_direction_tol_px:
                return None

        return (
            distance / cfg.max_reacquire_distance_px
            + (1.0 - ratio)
            + 0.5 * dt / max(1e-6, self._coast_limit(traj))
        )

    def _reacquire(self, now: float, coasting: List[Trajectory], new_boxes: Dict[int, tuple]):
        cost: Dict[Tuple[int, int], float] = {}
        for traj in coasting:
            for track_id, box in new_boxes.items():
                c = self._reacquire_cost(traj, box, now)
                if c is not None:
                    cost[(traj.id, track_id)] = c

        if not cost:
            return {}

        rows = sorted({k[0] for k in cost})
        cols = sorted({k[1] for k in cost})
        pairs = assign_min_cost(rows, cols, cost, skip_cost=10.0)

        links = {}
        for traj_id, track_id in pairs.items():
            c = cost[(traj_id, track_id)]
            rivals = [
                other for (r, k), other in cost.items()
                if (r == traj_id and k != track_id) or (k == track_id and r != traj_id)
            ]
            if rivals and min(rivals) - c < self.cfg.reacquire_unique_margin:
                self.log.log(
                    "IDENTITY_REACQUIRE_AMBIGUOUS", t=now, trajectory_id=traj_id,
                    new_track=track_id, cost=c, runner_up=min(rivals),
                )
                continue
            links[track_id] = (self.trajectories[traj_id], "motion_continuity", c)
        return links

    def _rebind_track(self, traj: Trajectory, new_track_id: int, now: float, reason: str, cost: float):
        old_track_id = traj.track_id
        if old_track_id is not None and self.track_to_traj.get(old_track_id) is traj:
            del self.track_to_traj[old_track_id]
        traj.track_id = new_track_id
        traj.track_id_history.append(new_track_id)
        self.track_to_traj[new_track_id] = traj
        if traj.state == "COASTING":
            self._reactivate(traj, new_track_id, now, reason)

        for passage in traj.passages.values():
            if passage.open:
                self.log.log(
                    "PASSAGE_TRACK_SWITCH", t=now, passage_id=passage.passage_id,
                    station=passage.station, old_track=old_track_id, new_track=new_track_id,
                    trajectory_id=traj.id, reason=reason,
                )
                passage.current_track_id = new_track_id
                passage.track_history.append(new_track_id)

        self.log.log(
            "IDENTITY_REACQUIRE", t=now, trajectory_id=traj.id, physical_id=traj.physical_id,
            old_track=old_track_id, new_track=new_track_id, reason=reason, cost=cost,
            coast_age=max(0.0, now - traj.last_seen),
        )

    def _reactivate(self, traj: Trajectory, track_id: int, now: float, reason: str):
        traj.state = "ACTIVE"
        traj.coast_start = None
        if traj.physical_id is not None:
            self._set_visibility(traj, "ACTIVE")
            if reason == "same_track_id":
                self.log.log(
                    "IDENTITY_REACQUIRE", t=now, trajectory_id=traj.id,
                    physical_id=traj.physical_id, old_track=track_id, new_track=track_id,
                    reason=reason, coast_age=now - traj.last_seen,
                )

    def _expire_trajectories(self, now: float):
        for traj in list(self.trajectories.values()):
            if traj.state != "COASTING":
                continue
            keep = max(self._coast_limit(traj), self.cfg.bridge_hint_coast_sec)
            if now - traj.last_seen > keep:
                self._kill_trajectory(traj, now, "coast_timeout")

    def _kill_trajectory(self, traj: Trajectory, now: float, reason: str):
        if traj.physical_id is not None:
            physical = traj.physical_id
            self._unbind(traj)
            self.log.log(
                "IDENTITY_UNBOUND", t=now, trajectory_id=traj.id, physical_id=physical,
                track_id=traj.track_id, reason=reason,
            )
        for passage in list(traj.passages.values()):
            if passage.open:
                self._close_passage(passage, now, "EXPIRED", f"trajectory_{reason}")
        self.log.log(
            "TRACK_DEAD", t=now, trajectory_id=traj.id, track_id=traj.track_id, reason=reason,
            track_history=list(traj.track_id_history),
        )
        self.trajectories.pop(traj.id, None)
        if traj.track_id is not None and self.track_to_traj.get(traj.track_id) is traj:
            del self.track_to_traj[traj.track_id]

    # ----------------------------------------------------------
    # STATION PASSAGES
    # ----------------------------------------------------------

    def _lane_ok(self, box, st: StationConfig) -> bool:
        margin = self.cfg.y_margin_px
        cy = box_center(box)[1]
        center_ok = (st.y_min - margin) <= cy <= (st.y_max + margin)
        overlap_ok = (box[3] >= st.y_min - margin) and (box[1] <= st.y_max + margin)
        mode = self.cfg.lane_gate_mode
        if mode == "overlap":
            return overlap_ok
        if mode == "both":
            return center_ok and overlap_ok
        return center_ok

    def _update_stations(self, now: float, traj: Trajectory):
        hyst = self.cfg.boundary_hysteresis_px
        cx, cy = traj.center

        for st in self.stations:
            u = st.sign * cx
            lane = self._lane_ok(traj.box, st)
            passage = traj.passages.get(st.name)

            # --- explain a pallet that is level with a station but opened no passage ---
            in_station_x = st.entry_u <= u <= st.exit_u
            note = traj.station_state[st.name]

            if not in_station_x:
                note["lane_noted"] = False
                note["entry_noted"] = False
            elif passage is None:
                if not lane and not note.get("lane_noted"):
                    note["lane_noted"] = True
                    self.log.log(
                        "STATION_LANE_MISS", t=now, station=st.name, track_id=traj.track_id,
                        box_y=[round(traj.box[1]), round(traj.box[3])],
                        station_y=[st.y_min, st.y_max], margin=self.cfg.y_margin_px,
                        note="pallet is level with the station but outside its vertical band",
                    )
                elif lane and note["outside_t"] is None and not note.get("entry_noted"):
                    note["entry_noted"] = True
                    self.log.log(
                        "STATION_NO_ENTRY", t=now, station=st.name, track_id=traj.track_id,
                        center=_pt(traj.center), direction=st.direction,
                        note="inside the station without entering from its valid side "
                             "(moving the other way, or first seen inside)",
                    )

            if passage is None:
                if not lane:
                    continue
                state = traj.station_state[st.name]

                if u <= st.entry_u:
                    state["pre"] = (now, u)
                if u < st.entry_u - hyst:
                    state["outside_t"] = now

                marker = state["outside_t"]
                if marker is not None and now - marker > self.cfg.entry_marker_max_age_sec:
                    state["outside_t"] = marker = None

                if u > st.entry_u + hyst and marker is not None and state["pre"] is not None:
                    entry_time = interpolate_crossing(state["pre"], (now, u), st.entry_u)
                    state["outside_t"] = None
                    self._open_passage(now, st, traj, entry_time, (cx, cy))
                continue

            # --- active passage ---
            if not lane:
                continue
            passage.last_position = (cx, cy)
            passage.last_seen_time = now
            passage.current_track_id = traj.track_id
            if passage.state == "ENTERED":
                passage.state = "INSIDE"

            if passage.center_time is None and passage.prev_sample is not None:
                if passage.prev_sample[1] < st.center_u <= u:
                    passage.center_time = interpolate_crossing(passage.prev_sample, (now, u), st.center_u)

            if u <= st.exit_u:
                passage.pre_exit = (now, u)

            if u < st.entry_u - hyst:
                passage.prev_sample = (now, u)
                self._close_passage(passage, now, "EXPIRED", "reversed_direction")
                continue

            if u > st.exit_u + hyst and passage.pre_exit is not None:
                passage.exit_time = interpolate_crossing(passage.pre_exit, (now, u), st.exit_u)
                if passage.center_time is None and passage.prev_sample is not None:
                    passage.center_time = interpolate_crossing(
                        passage.prev_sample, (now, u), st.center_u
                    )
                passage.prev_sample = (now, u)
                self._close_passage(passage, now, "EXITED", "exit_crossing")
                continue

            passage.prev_sample = (now, u)

    def _open_passage(self, now: float, st: StationConfig, traj: Trajectory, entry_time: float, position):
        entry_u = st.entry_u
        passage = StationPassage(
            passage_id=self._next_passage_id,
            station=st.name,
            state="ENTERED",
            trajectory=traj,
            entry_track_id=traj.track_id,
            current_track_id=traj.track_id,
            entry_time=entry_time,
            entry_position=position,
            last_position=position,
            last_seen_time=now,
            track_history=[traj.track_id],
            pre_exit=(entry_time, entry_u),
            prev_sample=(entry_time, entry_u),
        )
        self._next_passage_id += 1
        traj.passages[st.name] = passage
        self.active_passages.append(passage)

        self.log.log(
            "PASSAGE_OPEN", t=now, station=st.name, passage_id=passage.passage_id,
            track_id=traj.track_id, trajectory_id=traj.id, physical_id=traj.physical_id,
            center=_pt(position), entry_time=entry_time,
        )

        limit = self.cfg.station_max_active_passages
        concurrent = [p for p in self.active_passages if p.station == st.name and p.open]
        if len(concurrent) > 1:
            self.log.log(
                "STATION_OVERLAP", t=now, station=st.name,
                passages=[p.passage_id for p in concurrent],
                note="more than one pallet inside one RFID station",
            )
        if limit and len(concurrent) > limit:
            self.log.log("PASSAGE_LIMIT_EXCEEDED", t=now, station=st.name, limit=limit)

    def _close_passage(self, passage: StationPassage, now: float, state: str, reason: str):
        passage.state = state
        passage.closed_time = now
        passage.completed = state == "EXITED"
        traj = passage.trajectory
        if traj is not None and traj.passages.get(passage.station) is passage:
            del traj.passages[passage.station]
        if passage in self.active_passages:
            self.active_passages.remove(passage)

        if state == "EXITED":
            self.log.log(
                "PASSAGE_EXIT", t=now, station=passage.station, passage_id=passage.passage_id,
                track_id=passage.current_track_id, entry_track_id=passage.entry_track_id,
                physical_id=passage.physical_id, entry_time=passage.entry_time,
                exit_time=passage.exit_time, center_time=passage.center_time,
                track_history=list(passage.track_history),
            )
            self.recent_passages.append(passage)
            self._learn_offset(passage)
        else:
            self.log.log(
                "PASSAGE_EXPIRED", t=now, station=passage.station, passage_id=passage.passage_id,
                track_id=passage.current_track_id, physical_id=passage.physical_id, reason=reason,
                age=now - passage.entry_time,
            )
            if passage.physical_id is not None:
                # already identified: keep it matchable for a short time
                self.recent_passages.append(passage)

    def _maintain_passages(self, now: float):
        for passage in list(self.active_passages):
            if now - passage.entry_time > self.cfg.passage_timeout_sec:
                self._close_passage(passage, now, "EXPIRED", "timeout")
        cutoff = self.cfg.passage_retention_sec
        while self.recent_passages and now - (self.recent_passages[0].closed_time or 0.0) > cutoff:
            self.recent_passages.popleft()

    # ----------------------------------------------------------
    # RFID SIDE
    # ----------------------------------------------------------

    def process_rfid_events(self, now: float):
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, RFIDLeave):
                self._ingest_leave(event)
            else:
                self._ingest(event, now)
        self._resolve_pending(now)

    def _ingest_leave(self, leave: "RFIDLeave"):
        for read in reversed(self._reads):
            event = read["event"]
            if (
                read["leave"] is None
                and event.station == leave.station
                and event.pallet_id == leave.pallet_id
            ):
                read["leave"] = leave
                return

    def _ingest(self, event: RFIDEvent, now: float):
        age = now - event.timestamp
        self.log.log(
            "RFID_EVENT", t=event.timestamp, station=event.station, physical_id=event.pallet_id,
            raw=list(event.raw), queue_age=age,
        )

        if event.station not in self.station_by_name:
            self.log.log("RFID_UNKNOWN_STATION", station=event.station)
            return
        if event.pallet_id not in self.registry:
            self.log.log("RFID_UNKNOWN_PALLET", physical_id=event.pallet_id)
            return
        if age > self.cfg.event_max_age_sec:
            self.log.log("RFID_STALE_DROP", t=event.timestamp, station=event.station,
                         physical_id=event.pallet_id, age=age, reason="event_max_age")
            return

        key = (event.station, event.pallet_id)
        last = self._last_read.get(key)
        self._last_read[key] = event.timestamp
        if last is not None and abs(event.timestamp - last) <= self.cfg.rfid_debounce_sec:
            self.log.log("RFID_DEBOUNCE_DROP", t=event.timestamp, station=event.station,
                         physical_id=event.pallet_id, since_last=event.timestamp - last)
            return

        if self.cfg.match_mode == "closest":
            self._reads.append({"event": event, "leave": None})
        else:
            self.pending.append(event)

    def _candidate_passages(self, station: str, now: float) -> List[StationPassage]:
        result = [p for p in self.active_passages if p.station == station]
        result += [p for p in self.recent_passages if p.station == station]
        return result

    def _passage_cost(self, event: RFIDEvent, passage: StationPassage) -> Optional[float]:
        """None = physically impossible (hard reject)."""
        cfg = self.cfg
        traj = passage.trajectory
        if traj is None or traj.id not in self.trajectories:
            return None
        if passage.physical_id is not None:
            return None

        t_adj = event.timestamp - self.timing_offset(event.station)
        lo = passage.entry_time - cfg.rfid_pre_margin_sec
        if passage.exit_time is not None:
            hi = passage.exit_time + cfg.rfid_post_margin_sec
        else:
            hi = passage.entry_time + cfg.passage_timeout_sec
        if not (lo <= t_adj <= hi):
            return None

        start = passage.entry_time
        end = passage.exit_time if passage.exit_time is not None else max(self._vision_time, start)
        if t_adj < start:
            distance = start - t_adj
        elif t_adj > end:
            distance = t_adj - end
        else:
            distance = 0.0

        reference = passage.center_time if passage.center_time is not None else passage.entry_time
        return distance + 0.1 * abs(t_adj - reference)

    # ----------------------------------------------------------
    # CLOSEST-APPROACH MATCHING (default)
    # ----------------------------------------------------------

    def _delay(self, station: str) -> float:
        return float((self.cfg.vision_delay_sec or {}).get(station, 0.0))

    def _calibration_sample(self, event: "RFIDEvent", leave: Optional["RFIDLeave"], now: float):
        """ONE pallet on the conveyor: measure when the camera sees it at the reader."""
        cfg = self.cfg
        station = self.station_by_name[event.station]
        station_x = 0.5 * (station.box[0] + station.box[2])

        if leave is None:
            self.log.log("RFID_CALIBRATION_SKIP", station=event.station, reason="no 'tag left' signal")
            return

        start = event.timestamp - cfg.calibration_search_sec
        end = leave.timestamp + cfg.calibration_search_sec

        found = []
        for traj in self.trajectories.values():
            best = None
            for t, cx, cy in traj.history:
                if t < start or t > end or self._station_for_y(cy) != station.name:
                    continue
                d = abs(cx - station_x)
                if best is None or d < best[0]:
                    best = (d, t)
            if best is not None and best[0] <= cfg.closest_max_dx_px:
                found.append((best, traj))

        if len(found) == 0:
            self.log.log(
                "RFID_CALIBRATION_SKIP", station=event.station, physical_id=event.pallet_id,
                reason="NO pallet came near this reader in its lane within "
                       f"+-{cfg.calibration_search_sec:.0f}s of the read: either the reader is on the OTHER "
                       "lane than assumed (RFID1/RFID2 swapped), or the camera/RFID offset is bigger than "
                       "the search window, or nothing was tracked there",
            )
            return

        if len(found) > 1:
            self.log.log(
                "RFID_CALIBRATION_SKIP", station=event.station, physical_id=event.pallet_id,
                reason=f"{len(found)} pallets near the reader - calibrate with exactly ONE pallet on the conveyor",
            )
            return

        (distance, t_closest), traj = found[0]
        middle = 0.5 * (event.timestamp + leave.timestamp)

        x_when_tag_appeared = None
        for t, cx, cy in traj.history:
            if t >= event.timestamp:
                x_when_tag_appeared = cx - station_x
                break

        sample = {
            "delay_sec": t_closest - middle,
            "dwell_sec": leave.timestamp - event.timestamp,
            "closest_px": distance,
            "x_when_tag_appeared_px": x_when_tag_appeared,
        }
        self._calibration[event.station].append(sample)
        self.log.log(
            "RFID_CALIBRATION_SAMPLE", station=event.station, physical_id=event.pallet_id,
            track_id=traj.track_id, delay_sec=sample["delay_sec"], dwell_sec=sample["dwell_sec"],
            closest_px=round(distance),
            x_when_tag_appeared_px=None if x_when_tag_appeared is None else round(x_when_tag_appeared),
            note="delay = when the camera saw the pallet at the reader, minus the middle of the tag read",
        )

    def calibration_summary(self) -> dict:
        summary = {}
        for name, samples in self._calibration.items():
            if not samples:
                continue
            delays = sorted(x["delay_sec"] for x in samples)
            summary[name] = {
                "vision_delay_sec": round(delays[len(delays) // 2], 3),
                "samples": len(delays),
                "min": round(delays[0], 3),
                "max": round(delays[-1], 3),
                "dwell_sec": round(sorted(x["dwell_sec"] for x in samples)[len(samples) // 2], 3),
            }
        return summary

    def write_calibration(self, path) -> dict:
        summary = self.calibration_summary()
        if summary:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
        return summary

    def _station_for_y(self, cy: float) -> str:
        """Which reader's lane a point at height cy belongs to (nearest reader height)."""
        return min(
            self.stations,
            key=lambda st: abs(cy - 0.5 * (st.box[1] + st.box[3])),
        ).name

    def _resolve_closest(self, now: float):
        cfg = self.cfg

        for read in list(self._reads):
            event = read["event"]
            leave = read["leave"]

            if leave is None:
                if now - event.timestamp < cfg.closest_max_dwell_sec:
                    continue                                   # tag may still be in the field
                end_time = event.timestamp + cfg.closest_max_dwell_sec
            else:
                end_time = leave.timestamp

            if cfg.calibrate:
                end_time += cfg.calibration_search_sec
            else:
                end_time += cfg.closest_post_sec + self._delay(event.station)

            # wait until the camera frames up to end_time have been processed
            if self._vision_time < end_time and now - end_time < 1.0:
                continue

            self._reads.remove(read)

            if cfg.calibrate:
                self._calibration_sample(event, leave, now)
                continue

            self._match_closest(
                event,
                event.timestamp - cfg.closest_pre_sec + self._delay(event.station),
                end_time,
                now,
            )

    def _match_closest(self, event: RFIDEvent, start_time: float, end_time: float, now: float):
        cfg = self.cfg
        station = self.station_by_name[event.station]
        station_x = 0.5 * (station.box[0] + station.box[2])

        ranked = []
        for traj in self.trajectories.values():
            nearest = None
            for t, cx, cy in traj.history:
                if t < start_time or t > end_time:
                    continue
                if self._station_for_y(cy) != station.name:
                    continue
                distance = abs(cx - station_x)
                if nearest is None or distance < nearest:
                    nearest = distance
            if nearest is not None:
                ranked.append((nearest, traj))

        ranked.sort(key=lambda item: item[0])

        if not ranked or ranked[0][0] > cfg.closest_max_dx_px:
            self.log.log(
                "RFID_NO_PALLET", t=event.timestamp, station=event.station,
                physical_id=event.pallet_id,
                nearest_px=round(ranked[0][0]) if ranked else None,
                note="a tag was read but no tracked pallet came near the reader",
            )
            return

        best_distance, best = ranked[0]
        if (
            len(ranked) > 1
            and ranked[1][0] <= cfg.closest_max_dx_px
            and ranked[1][0] - best_distance < cfg.closest_ambiguity_px
        ):
            self.log.log(
                "RFID_MATCH_AMBIGUOUS", t=event.timestamp, station=event.station,
                physical_id=event.pallet_id, tracks=[ranked[0][1].track_id, ranked[1][1].track_id],
                distances_px=[round(ranked[0][0]), round(ranked[1][0])],
                action="left_unassigned",
            )
            return

        self.log.log(
            "RFID_CLOSEST_MATCH", t=event.timestamp, station=event.station,
            physical_id=event.pallet_id, track_id=best.track_id, trajectory_id=best.id,
            distance_px=round(best_distance), runner_up_px=round(ranked[1][0]) if len(ranked) > 1 else None,
            resolve_latency=now - event.timestamp,
        )
        self._rfid_assign(best, event.pallet_id, event.timestamp, event.station, 0, now)

    def _resolve_pending(self, now: float):
        if self.cfg.match_mode == "closest":
            self._resolve_closest(now)
            return
        if not self.pending:
            return
        cfg = self.cfg

        for station in {e.station for e in self.pending}:
            events = sorted([e for e in self.pending if e.station == station], key=lambda e: e.timestamp)
            ready = [e for e in events if now - e.timestamp >= cfg.rfid_resolve_delay_sec]
            passages = self._candidate_passages(station, now)

            # an event whose tag is already attached to a live passage is just a repeat
            for event in list(ready):
                duplicate = [p for p in passages if p.physical_id == event.pallet_id
                             and p.trajectory is not None and p.trajectory.id in self.trajectories
                             and abs(event.timestamp - (p.rfid_timestamp or event.timestamp)) <= cfg.passage_timeout_sec]
                if duplicate:
                    self.log.log("RFID_PASSAGE_DUPLICATE", t=event.timestamp, station=station,
                                 physical_id=event.pallet_id, passage_id=duplicate[0].passage_id)
                    self.pending.remove(event)
                    ready.remove(event)

            cost = {}
            for event in ready:
                for passage in passages:
                    c = self._passage_cost(event, passage)
                    if c is not None:
                        cost[(id(event), passage.passage_id)] = c

            by_id = {id(e): e for e in ready}
            by_passage = {p.passage_id: p for p in passages}
            pairs = assign_min_cost(list(by_id), list(by_passage), cost, skip_cost=1e3)
            taken = set(pairs.values())

            for event in ready:
                age = now - event.timestamp
                passage_id = pairs.get(id(event))

                if passage_id is None:
                    if age >= cfg.pending_hold_sec:
                        self.log.log("RFID_UNMATCHED", t=event.timestamp, station=station,
                                     physical_id=event.pallet_id, age=age,
                                     open_passages=[p.passage_id for p in passages if p.open])
                        self.pending.remove(event)
                    continue

                chosen_cost = cost[(id(event), passage_id)]
                rivals = [
                    c for (eid, pid), c in cost.items()
                    if eid == id(event) and pid != passage_id and pid not in taken
                ]
                rival_cost = min(rivals) if rivals else None

                if rival_cost is not None and (rival_cost - chosen_cost) < cfg.ambiguity_margin_sec:
                    if age >= cfg.ambiguity_hold_sec:
                        self.log.log(
                            "RFID_PASSAGE_AMBIGUOUS", t=event.timestamp, station=station,
                            physical_id=event.pallet_id, best_cost=chosen_cost, runner_up=rival_cost,
                            action="left_unassigned",
                        )
                        self.pending.remove(event)
                    continue

                self._commit_rfid(event, by_passage[passage_id], chosen_cost, confident=rival_cost is None, now=now)
                self.pending.remove(event)

    def _commit_rfid(self, event: RFIDEvent, passage: StationPassage, cost: float, confident: bool, now: float):
        offset = self.timing_offset(event.station)
        reference = passage.center_time if passage.center_time is not None else passage.entry_time
        passage.physical_id = event.pallet_id
        passage.rfid_timestamp = event.timestamp
        passage.rfid_confident = confident
        passage.rfid_residual = (event.timestamp - offset) - reference

        self.log.log(
            "RFID_PASSAGE_MATCH", t=event.timestamp, station=event.station,
            passage_id=passage.passage_id, physical_id=event.pallet_id,
            track_id=passage.current_track_id, entry_track_id=passage.entry_track_id,
            state=passage.state, rfid_minus_entry=event.timestamp - passage.entry_time,
            rfid_minus_exit=(event.timestamp - passage.exit_time) if passage.exit_time else None,
            residual=passage.rfid_residual, offset=offset, cost=cost, confident=confident,
            resolve_latency=now - event.timestamp,
        )

        traj = passage.trajectory
        if traj is not None and traj.id in self.trajectories:
            self._rfid_assign(traj, event.pallet_id, event.timestamp, event.station, passage.passage_id, now)

        if passage.state == "EXITED":
            self._learn_offset(passage)

    # ----------------------------------------------------------
    # TIMING OFFSET LEARNING
    # ----------------------------------------------------------

    def _learn_offset(self, passage: StationPassage):
        if passage.offset_learned or passage.physical_id is None or not passage.rfid_confident:
            return
        if passage.center_time is None or passage.rfid_timestamp is None:
            return
        delta = passage.rfid_timestamp - passage.center_time
        if abs(delta) > self.cfg.offset_max_abs_sec:
            return
        passage.offset_learned = True
        self._offsets[passage.station].append(delta)
        self.log.log(
            "RFID_OFFSET_SAMPLE", station=passage.station, delta=delta,
            samples=len(self._offsets[passage.station]),
            median=self.timing_offset(passage.station),
        )

    # ----------------------------------------------------------
    # GLOBAL IDENTITY REGISTRY (atomic, one owner each way)
    # ----------------------------------------------------------

    def _set_visibility(self, traj: Trajectory, state: str):
        pal = self.registry.get(traj.physical_id) if traj.physical_id else None
        if pal is not None and pal.trajectory is traj:
            pal.visibility_state = state
            pal.current_track_id = traj.track_id
            pal.last_seen_timestamp = traj.last_seen
            pal.last_position = traj.center

    def _bind(self, traj: Trajectory, physical_id: int, source: str):
        pal = self.registry[physical_id]
        traj.physical_id = physical_id
        traj.identity_source = source
        pal.trajectory = traj
        pal.current_track_id = traj.track_id
        pal.visibility_state = "ACTIVE" if traj.state == "ACTIVE" else "COASTING"
        pal.last_seen_timestamp = traj.last_seen
        pal.last_position = traj.center

    def _unbind(self, traj: Trajectory):
        physical_id = traj.physical_id
        traj.physical_id = None
        traj.identity_source = None
        if physical_id is None:
            return
        pal = self.registry[physical_id]
        if pal.trajectory is traj:
            pal.trajectory = None
            pal.current_track_id = None
            pal.visibility_state = "NOT_VISIBLE"

    def _rfid_assign(self, traj: Trajectory, physical_id: int, t: float, station: str, passage_id: int, now: float):
        pal = self.registry[physical_id]

        if pal.last_rfid_timestamp is not None and t < pal.last_rfid_timestamp:
            self.log.log("RFID_STALE_DROP", t=t, station=station, physical_id=physical_id,
                         last_rfid=pal.last_rfid_timestamp, reason="older_than_registry")
            return

        pal.generation += 1
        pal.last_rfid_station = station
        pal.last_rfid_timestamp = t
        source = f"{station}:passage{passage_id}"

        if traj.physical_id == physical_id:
            pal.visibility_state = "ACTIVE" if traj.state == "ACTIVE" else "COASTING"
            self.log.log("IDENTITY_VERIFY", t=t, station=station, passage_id=passage_id,
                         trajectory_id=traj.id, track_id=traj.track_id, physical_id=physical_id,
                         generation=pal.generation)
            traj.identity_source = source
            return

        if traj.physical_id is not None:
            previous = traj.physical_id
            self._unbind(traj)
            self.log.log("IDENTITY_REMAP", t=t, station=station, passage_id=passage_id,
                         trajectory_id=traj.id, track_id=traj.track_id,
                         old_physical=previous, new_physical=physical_id, reason="rfid_authoritative")

        owner = pal.trajectory
        if owner is not None and owner is not traj:
            self._unbind(owner)
            self.log.log("IDENTITY_CLEAR", t=t, station=station, physical_id=physical_id,
                         stale_trajectory=owner.id, stale_track=owner.track_id,
                         new_trajectory=traj.id, reason="physical_id_claimed_by_rfid_passage")

        self._bind(traj, physical_id, source)
        self.log.log("IDENTITY_ASSIGN", t=t, station=station, passage_id=passage_id,
                     trajectory_id=traj.id, track_id=traj.track_id, physical_id=physical_id,
                     generation=pal.generation, source=source)

    # ----------------------------------------------------------
    # DEBUG OVERLAY (cv2 imported lazily)
    # ----------------------------------------------------------

    STATION_BOX_COLOR = (255, 0, 255)   # one colour, plain rectangle

    def draw_debug(self, image, now: float, labels: bool = False):
        """
        Draws each RFID station as ONE plain rectangle (no lines, no text).
        labels=True additionally prints the open-passage status lines (debug only).
        """
        import cv2

        for st in self.stations:
            x1, y1, x2, y2 = st.box
            cv2.rectangle(image, (x1, y1), (x2, y2), self.STATION_BOX_COLOR, 2)

        if not labels:
            return

        y = 300
        for passage in self.active_passages:
            traj = passage.trajectory
            dt = f"{passage.rfid_residual * 1000:+.0f}ms" if passage.rfid_residual is not None else "-"
            lines = [
                f"{passage.station} PASSAGE {passage.passage_id}  STATE={passage.state}",
                "  " + " -> ".join(f"P{t}" for t in passage.track_history)
                + f"  RFID={passage.physical_id}  dt={dt}",
                f"  traj={traj.id if traj else '-'} {traj.state if traj else ''}",
            ]
            for line in lines:
                cv2.putText(image, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                y += 20


def _pt(point) -> list:
    return [round(point[0], 1), round(point[1], 1)]

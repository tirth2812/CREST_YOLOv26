# CREST RFID ↔ Vision Physical-Identity Layer

**Existing vision inference/classification behavior unchanged. Only identity-fusion plumbing added.**

| File (in `CODE/`) | Role |
|---|---|
| `crest_rfid_fusion.py` | All fusion logic. Pure Python, no cv2/torch import at load. |
| `29_live_phase27_rfid_fusion.py` | Byte-for-byte copy of `27_live_fast_new_object_final.py` + marked `# [RFID-FUSION]` plumbing. **27 itself is untouched.** |
| `test_crest_rfid_fusion.py` | 30 deterministic simulation tests (no camera/PLC/GPU). |
| `analyze_rfid_fusion_log.py` | Turns a live session log into the timing distributions needed for tuning. |

Note: the repo did **not** contain `27_..._with_rfid_boundaries.py`, so 29 is built from the stable 27. The station boxes/entry/exit lines are drawn by the fusion layer itself (`RFID_DEBUG = True`).

## Architecture

```
frame acquired ─ t=time.monotonic() ──▶ [unchanged Phase 27 inference + tracker + duplicate filter + state bridge]
                                          │ confirmed_tracks (Pxx, box)       │ move_track_state() ──hint──┐
                                          ▼                                   ▼                            │
RFID thread ──RFIDEvent(station,pallet,t)──queue──▶  FusionIdentityManager (main thread only) ◀────────────┘
                                                      ├─ TrajectoryManager   Pxx continuity (coast + global assignment)
                                                      ├─ Station passages    ENTERED → INSIDE → EXITED | EXPIRED
                                                      ├─ RFID matcher        event ↔ passage, one-to-one, timing-offset learned per station
                                                      └─ Identity registry   pallets 1..6, bidirectional one-owner invariant
```

* **Passage, not nearest box.** A trajectory entering through the physically valid entry side (with hysteresis, lane gate and a "was clearly outside" marker) opens a passage. The RFID read is matched to the passage by time window, never by proximity.
* **Generic stations.** Each station is a config entry; the progress coordinate `u = sign·x` makes RFID1 (R→L) and RFID2 (L→R) the same code path.
* **Passage survives Pxx changes.** It is bound to the *trajectory*; the trajectory survives short Pxx breaks (extended coast while it owns an open passage). `PASSAGE_TRACK_SWITCH` is logged.
* **Reuses the proven Phase 27 bridge.** The wrapper around `move_track_state` forwards old→new ids as a *hint* (accepted only if geometrically sane); the layer's own continuity covers breaks the Phase 27 bridge doesn't.
* **RFID is authoritative.** unknown→assign, same→verify, different→remap, id owned elsewhere→clear stale owner; stale/old events never overwrite newer mappings (timestamp + generation). Removed pallets unbind after the short coast; **reinserted pallets stay UNKNOWN until an RFID read.**
* **No guessing.** Ambiguous events are held ≤ `ambiguity_hold_sec` then left unassigned (`RFID_PASSAGE_AMBIGUOUS`).
* **Failure isolation.** Missing `pylogix`, PLC errors, reconnects: only RFID is affected; vision keeps running.

## Integration points (all marked `# [RFID-FUSION]` in file 29)

| What | Where |
|---|---|
| Settings (`RFID_ENABLED`, `RFID_READER_ENABLED`, `RFID_DEBUG`, `RFID_UNKNOWN_LABEL`) | block after `COLOR_UNKNOWN` |
| Bridge hint | wrapper around `move_track_state` (original called unchanged first) |
| RFID thread start/stop | `main()` after `DynamicTracker()` / before `cap.release()` |
| Frame timestamp | immediately after `cap.read()` returns |
| Frame loop | right after `bridge_reacquired_tracks(...)`: `update_tracks(frame_timestamp, confirmed_tracks)` + `process_rfid_events(time.monotonic())` |
| Overlay | only the label *passed to* `draw_pallet` is rewritten (`3 ACCEPTED`); the raw label used for the status counts is unchanged |

Timestamp note: `cap.read()` blocks until a frame arrives, so stamping right after it returns is closer to acquisition than stamping before it (which would include the wait). Driver buffering latency, if any, is absorbed by the learned per-station offset.

## Tunables (`FusionConfig`) — initial value / why

| Constant | Init | Why |
|---|---|---|
| `rfid_debounce_sec` | 0.4 | Merge repeated reports of one tag at one station; later passes still accepted. |
| `event_max_age_sec` | 2.0 | Drop events that sat in the queue too long. |
| `rfid_pre_margin_sec` / `post` | 0.25 / 0.25 | Antenna may fire before/after the drawn box. **Measure (Phase B).** |
| `rfid_resolve_delay_sec` | 0.15 | Let a neighbouring passage appear before committing; sets typical latency (~175 ms in sim). |
| `ambiguity_hold_sec`, `pending_hold_sec` | 0.70, 0.70 | Hard resolution budget (< ~700 ms). |
| `ambiguity_margin_sec` | 0.10 | Best vs runner-up cost margin required to commit. |
| `offset_min_samples`, `offset_history`, `offset_max_abs_sec` | 5, 40, 1.0 | Per-station rolling-median RFID↔vision offset, confident matches only. |
| `boundary_hysteresis_px` | 8 | Clear-outside/inside band; tune from centre jitter. |
| `y_margin_px`, `lane_gate_mode` | 25, `center` | Lane gate around station Y range. |
| `entry_marker_max_age_sec` | 2.0 | "Was clearly outside" memory. |
| `passage_timeout_sec` | 4.0 | Force-close; set to ~3× measured traversal time. |
| `passage_retention_sec` | 1.5 | EXITED passages stay matchable for late reads. |
| `station_max_active_passages` | 0 (off) | Overlap only *warned* (`STATION_OVERLAP`) until you confirm from video whether two pallets can share a station. |
| `track_coast_sec`, `passage_coast_sec` | 0.5, 1.0 | Short only — pallets are removed by hand. |
| `max_reacquire_distance_px`, `_y_delta_px`, `bbox_size_ratio_limit` | 80, 40, 0.6 | Reacquire gates (vs. velocity-predicted position). |
| `reacquire_unique_margin` | 0.25 | Ambiguous reacquisition ⇒ no identity copy. |
| `bridge_hint_max_distance_px` | 160 | Sanity check on Phase 27 bridge hints. |

## Tested (simulation, `python test_crest_rfid_fusion.py` → 30/30)

Acceptance scenarios 1–10, plus these failure modes from spec §38: bbox jitter at the entry line (A), repeated RFID reads (B), tracker change at the entry boundary (C) and inside the gate (D), neighbouring pallet not stealing the read (E), removed pallet contaminating another track (F), reinsertion not guessed (G), old event overwriting a newer mapping (H), duplicate physical owner (I, invariant checked on every simulated frame), passage that never closes (J). Also: two pallets inside one station, RFID read before the vision entry, wrong direction, wrong lane, ambiguous event left unassigned, per-station offset learning, 20-minute churn with bounded dictionaries, and the reader thread (appearance / repeat / re-arm / unknown tag / PLC errors / missing pylogix).
**Not testable offline:** (K) camera-inference timestamp distortion — only the plumbing (stamp immediately after `cap.read()`) was added; and (L) different per-station offsets — the learning logic is tested with synthetic offsets, real values need Phase B.

Mutation check: disabling coasting, ambiguity guard, debounce, lane gate, margins, or age limit each turns the matching tests red.

## NOT validated (needs the real conveyor)

Simulated pallets move at a constant 120 px/s with clean boxes. Real values for conveyor speed, box jitter, antenna trigger point, PLC latency/poll rate, whether pylogix returns signed ints exactly as `RFID_MAP` expects, whether two pallets can occupy a station at once, and GPU frame-rate effects are **unmeasured**. The tracker-ID changes seen live (e.g. P57→P92 inside the gate) are simulated, not recorded. Do not treat the defaults as tuned.

## Live test procedure

1. **Phase A (geometry only):** `RFID_READER_ENABLED=False, RFID_DEBUG=True`. Run pallets; confirm every pass logs `PASSAGE_OPEN`→`PASSAGE_EXIT` (RFID1 right→left, RFID2 left→right), no extra opens, no `PASSAGE_EXPIRED`.
2. **Phase B:** enable the reader; run ~30 clean single-pallet passes per station; `python analyze_rfid_fusion_log.py <log>`; set pre/post margins from `rfid_minus_entry/exit` and check the learned offset.
3. **C–G:** nearby pallets, tracker breaks, removal, reinsertion, long loop (see spec §37). Inspect every `IDENTITY_REMAP`, `RFID_UNMATCHED`, `RFID_PASSAGE_AMBIGUOUS`, `PASSAGE_EXPIRED` in the log.

Run on GPU machine with the `vlm_env` after `pip install pylogix` **only if** `python -c "import pylogix"` fails there.

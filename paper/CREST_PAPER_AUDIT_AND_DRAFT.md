# CREST — Conference Paper: Codebase Audit, Evidence Map and Draft Material

Scope: everything committed in this repository (50 files, commit `8b8accc`). Every number below is
copied from a file in the repo and cites it. Nothing is invented. Items marked **[NOT IN REPO]** could
not be checked and must not be quoted until the source files are produced.

---

## 1. What the repository actually contains (and does not)

| Stage | Folder | Content | Generation |
|---|---|---|---|
| 01 | `01_Pallet_Annotation` | Annotator, 20 annotated images (10 `empty`, 10 `REJECTED`), 6 pallet boxes each = 120 boxes | — |
| 02 | `02_YOLO_Pallet_Detection` | YOLO26n trainer, split, metrics; YOLO11 eval script; packaging script whose embedded README holds the **YOLO11 / Grounding-DINO** numbers | YOLO26 + YOLO11 |
| 03 | `03_Empty_Object_Classifier` | YOLO11 trainers (older), YOLO26 "fair" trainer, splits, metrics, predictions | YOLO11 → YOLO26 |
| 04 | `04_Live_Empty_Object_System` | Despite its name, a **data-collection tool** (freeze, hand-label 1–6, save crops), hard-coded 6 pallets and rectangular ROI zones, YOLO11 | YOLO11 |
| 05 | `05_Dynamic_Pallet_Detection` | Early dynamic tracker (IoU-only, greedy), YOLO11 model | YOLO11 |
| 06 | `06_Live_Empty_Object_Detection_Final` | Per-frame live EMPTY/OBJECT, **no temporal logic**, YOLO11 stack | YOLO11 |
| 07 | `07_Final_Empty_Accepted_Rejected_System/CODE` | `27_live_fast_new_object_final.py` (Phase 27, 3 849 lines) + final tracker `dynamic_live_pallet_detection.py` + `CONFIG/conveyor_boundary.json` | YOLO26 |

**Not in the repository** (`.gitignore` excludes weights, datasets, results and every Stage-07 script except two):

- All three model weights (`pallet_yolo26n_best.pt`, `empty_object_yolo26n_cls_best.pt`, `x_no_x_yolo26n_medium_focus_final_hard_best.pt`).
- **The entire X/NO_X lineage**: no training script, dataset, split, metrics or class-index mapping. The YOLO11 10/10 vs YOLO26 6/10 comparison, the 18/18 result and the "58/60" figure printed by the live script cannot be audited here.
- `live_final_crest_yolo26.py`, `27_live_fast_new_object_final_with_rfid_boundaries.py`, all RFID / pylogix code, the RFID boxes, the CPU "Stage 28" experiments.
- Any live-run log, video, per-frame timing, or scenario-test record. **There is currently no repo evidence of any full-system live result.**

Consequence: the repo supports a precise description of the *method* (Section 2) and the *two* early-stage
model evaluations (Section 4). It does not support any end-to-end accuracy, latency, FPS or ablation claim.

---

## 2. Verified method facts (read directly from `27_live_fast_new_object_final.py` + tracker)

### 2.1 Per-frame data flow (`main()`, lines 2832–3750)

1. Capture 1920×1080 MJPG, 30 FPS requested; camera tried in order 1, 0, 2, 3, 4 (DirectShow, Windows only).
2. Two enhanced copies of the frame: **pallet frame** = gamma 0.60 + CLAHE (clip 2.0, 8×8) on the L channel of LAB; **symbol frame** = gamma 1.15, per-channel gains B 0.80 / G 1.00 / R 0.90, unsharp mask (σ 0.7, weights 1.30 / −0.30).
3. Pallet YOLO is run **twice** — on the raw frame and on the enhanced frame (conf 0.10, imgsz 640, NMS IoU 0.50). Results are merged: same-object if IoU ≥ 0.45, keep the higher confidence (`merge_pallet_detections`). *This dual-view detection is not in the existing project notes and should be described in the paper.*
4. Ring filter (tracker module): box centre inside the outer-minus-inner freehand polygon **and** ≥ 25 % of box area on the ring. Boundary file: outer 20 vertices, inner 23 vertices, normalized.
5. Tracker (`DynamicTracker`): greedy association by cost = centre-distance(to velocity-predicted centre)/box-diagonal + 0.35·(1−IoU); gate IoU ≥ 0.05 **or** distance ≤ 0.75; velocity is an EMA (0.65/0.35) in px/frame; confirmed after 5 hits; **a track is deleted after 5 missed frames**; only currently matched tracks are returned. This is a custom greedy constant-velocity matcher — **not** SORT/ByteTrack/DeepSORT; related work must position it that way.
6. Geometry duplicate suppression: size ratio ≥ 0.60, vertical overlap ≥ 0.70, smaller-box overlap ≥ 0.18, centre ratios ≤ 0.82 (horizontal) / 0.35 (vertical). Tracks that already own symbol state, then cached tracks, then lower ID, are preferred; a suppressed track's state is deleted.
7. State bridge (`bridge_reacquired_tracks`): a *new* ID with no state inherits all state (object/symbol history+decision, sample state, fast-window timers, cache) from a missing ID if the old track was missing ≤ 10 frames, has meaningful state, size ratio ≥ 0.80, centre ratios ≤ 0.45 / 0.35, overlap or IoU minimum, score ≥ 1.10, and the best score beats the runner-up by ≥ 0.25 (ambiguity ⇒ no transfer). Score = 1.8·IoU + 1.4·overlap + 0.55·size − 0.35·h − 0.55·v. It compares against the **last cached box, not a velocity-predicted one**.
8. Memory vs. display: internal state kept 12 frames after loss, drawn only 2 frames after loss, so hidden tracks are never counted or drawn.

### 2.2 EMPTY / OBJECT_PRESENT (`update_object_decision`)

- Classifier: YOLO26n-cls, imgsz 320, on the enhanced pallet crop (3 % padding); min confidence 0.50; history of last 7 samples; samples at most one per 0.18 s per track.
- Initial state: 2 OBJECT votes ⇒ OBJECT_PRESENT; 5 EMPTY votes ⇒ EMPTY.
- EMPTY → OBJECT: in a clean view, 2 consecutive samples ≥ 0.75; in any view, 3 consecutive ≥ 0.85 ("fast"). Weak (< 0.85) object evidence from an unsafe view is not even recorded.
- OBJECT → EMPTY: **only when the clean-view gate is open**, 5 consecutive EMPTY samples ≥ 0.85.
- So the "asymmetric temporal evidence" claim is accurate, with a refinement: appearance can fire anywhere on the loop, **removal can fire only on the clean lower straight**. Removal latency therefore depends on the pallet's lap position, not just on evidence count.

### 2.3 Clean-view gate (`update_dynamic_symbol_gate`) — "motion-aware observation gating"

A view is trusted only if the track is moving **left** (EMA vx ≤ −0.60 px/frame), mostly horizontal (|vx| ≥ 1.25·max(|vy|, 0.25)), detector confidence ≥ 0.20 and ring overlap ≥ 0.30, box geometrically stable versus the previous frame (IoU ≥ 0.50, size ratio ≥ 0.82, centre ratios ≤ 0.22 / 0.20), after a 0.25 s recovery timer, and for ≥ 5 consecutive stable frames. Note that this encodes the **conveyor direction (clockwise) and layout** — it is motion-guided, but it is not geometry-free. Thresholds in px/frame and frames are **frame-rate dependent**; the achieved FPS must be reported.

### 2.4 Symbol path

- Representation: symbol-frame crop → "medium focus" window (x 0.15–0.85, y 0.00–0.67 of the crop) → letterbox to 640×640 with mean-colour padding → YOLO26-cls; sample accepted if confidence ≥ 0.65.
- Normal path: a new sample needs ≥ 0.20 s **and** centre displacement ≥ 0.22 box-diagonal since the previous sample, and an open clean gate. History = last 3 samples. First decision: ≥ 2 of 3 identical (X ⇒ REJECTED, NO_X ⇒ ACCEPTED).
- Established decision flips only on **3 of 3** opposite samples (state-dependent threshold / hysteresis).
- Fast path: on a confirmed EMPTY→OBJECT event, symbol state is cleared, wait 0.35 s, then a 2.5 s window; samples need ≥ 0.16 s and only ≥ 0.05 diagonal displacement, the clean gate is **not** required (detector conf ≥ 0.20, overlap ≥ 0.25 only), decision needs 3/3 identical. If undecided at the end of the window, normal logic takes over.
- **Disclosure needed:** the fast path deliberately relaxes view separation (0.05 vs 0.22 diagonal). Describe it as time-separated rather than "meaningfully different views", otherwise the paper over-claims independence.

### 2.5 Things the notes say that the code does not quite say

| Note / comment | Code reality |
|---|---|
| Comment at lines 179–182: OBJECT_PRESENT "never downgrades back to EMPTY" | Stale. Removal is implemented (5 strong clean EMPTY samples). Fix the comment before releasing the code. |
| "Camera exposure −7 not fundamental" | The Phase-27 file still sets `CAMERA_EXPOSURE = -7.0` and `AUTO_EXPOSURE = 0.25` (lines 106, 377–385). Confirm which camera-fix file was the one actually run. |
| Startup banner "Frozen external validation: 58/60 = 96.67 %, physical-pass majority 20/20" (lines 2713–2721) | Hard-coded text, no source in repo. Do not cite. |
| `is_left_turn_hold_zone` | Dead code referencing undefined `LEFT_TURN_HOLD_X_RATIO` (would raise `NameError`); never called. Remove. |
| `old_new_tracks_match` + `STALE_TRANSFER_*` | Defined, never used (leftover from an earlier bridge). |
| Terminology "REJECTED" in stages 01–03 | In the early data, `REJECTED` = object present with an X mark (annotation JSON: `REJECTED → object_present=true, x_mark_visible=true`). `normalize_object_class` maps any name containing "REJECT" to OBJECT_PRESENT. Keep early-stage class names separate from the final operational states in the paper. |
| Tracker deletion after 5 misses vs. state memory of 12 | Consistent, and it is precisely why the bridge is needed (tracker ID dies at 5 frames, state survives to 12 / bridge to 10). Worth one sentence in the paper. |

---

## 3. Claim classification

### 3.1 Fully supported by repo (can be written as method)

- Three-model, multi-stage architecture and its data flow (Section 2.1).
- Dynamic pallet count (no fixed 6 in the final tracker) and ring-based spatial filtering.
- Geometry-based duplicate suppression and geometry-based short-term state bridging (not ReID).
- Dual raw/enhanced pallet detection, gamma + CLAHE pallet enhancement, separate symbol preprocessing.
- Asymmetric object-state thresholds; state-dependent symbol thresholds (2/3 initial, 3/3 to change); clean-view motion gating; time/displacement-separated sampling; fast new-object path.

### 3.2 Likely true but needs evidence before it is a result

- "Detects a new object within a few seconds" — implied minimum is roughly 0.18 s × 3 samples plus 0.35 s delay plus 3 symbol samples; **measure it**.
- Occlusion preservation, bridge success, no flip on a hand, X↔NO_X change handling, removal detection.
- Real-time operation (no FPS number exists; code runs 2 pallet inferences + 1 classifier call per track per frame).
- Robustness to lighting (the motivation for enhancement; no ablation).

### 3.3 Experimental / unverifiable from repo

- Everything about the X/NO_X classifier (accuracy, YOLO11 vs YOLO26, 18/18, class mapping reversal).
- "Medium-focus" representation chosen by experiment (only the crop constants are visible).
- Live Phase-27 behaviour as observed in the lab (no logs).

### 3.4 Future work only

- RFID–track association, RFID identity rules, RFID-through-bridge. Not in repo, unfinished per notes.

---

## 4. Numbers that exist in the repo, with the limits you must state

### 4.1 Pallet detection

| Result | Value | Source | Caveat |
|---|---|---|---|
| Data | 20 images (10 empty / 10 REJECTED), 6 boxes each; split 10 / 5 / 5 images = 60 / 30 / 30 boxes | `split_used_for_yolo.json`, annotations | Single camera, single day, every image has exactly 6 pallets, X-only objects. Train set is 10 images. |
| YOLO26n held-out | P 0.715, R 0.733, mAP50 0.838, mAP50-95 0.516 | `yolo26_test_metrics.json` | Produced by Ultralytics `val()` (default low-confidence operating point), 5 images. |
| YOLO26n validation | P 0.732, R 0.833, mAP50 0.876, mAP50-95 0.570 | same | 5 images, also used for checkpoint selection. |
| YOLO11n held-out | 30/30 detected, 4 FP, P 0.882, R 1.0, F1 0.9375, mean IoU 0.808 | embedded README in `create_final_yolo_package.py` | Custom IoU-0.5 matching at conf 0.05 / imgsz 960 / NMS 0.60. **Different protocol from the YOLO26 line above, so the two are not comparable.** |
| Grounding-DINO baseline held-out | 25/30, P 0.962, R 0.833, F1 0.893 | same | Same 5 test images. |
| Runtime operating point | conf 0.10, imgsz 640, NMS 0.50 | `27_*.py` | Matches neither evaluation above. |

Do **not** write "YOLO26 improved on YOLO11" for pallet detection; the repo shows the opposite impression under non-matched protocols. Re-evaluate both models with one script, one operating point, and a larger test set that includes other pallet counts and the whole loop.

### 4.2 EMPTY / OBJECT_PRESENT

Two lineages share the same output filename `empty_object_yolo26n_cls_best.pt` (the "fair" script first backs up the older file, then overwrites it). **The model in the live `MODELS` folder is probably the second one — confirm by hash.**

| Run | Data | Result | Caveat |
|---|---|---|---|
| A (`yolo26_empty_object_metrics.json`) | train 57+57, val 15+15, locked held-out 4 images (2+2, WhatsApp photos) | val 30/30, held-out 4/4 | The 4-image test is statistically meaningless (95 % lower bound ≈ 40 % for 4/4). |
| B "fair" (`..._fair_metrics.json`) | train 95+95, val 25+25 (the final YOLO11 dataset, reused for a controlled model-family comparison) | val 50/50 | **No independent test set.** The val set also selects `best.pt`. **Leakage check done in this audit: 6 of the 11 validation capture groups also appear in the training split for each class** (each capture frame was cut into pallets p1…p6 and distributed across splits), so train and val share frames, scenes and lighting. Treat 50/50 as an upper bound, not a generalization estimate. |

Also, the YOLO11 counterpart numbers for run B's comparison are not in the repo, so the "fair comparison" has only one side on record.

### 4.3 X / NO_X

Nothing in the repo. The reported 10/10 (YOLO11) vs 6/10 (first YOLO26) is on 5+5 held-out images and, per your own notes, 18/18 is under a class-mapping question. If you quote these: label as controlled classifier comparison on very small N, give the N, the representation, and the split, and report a confidence interval (10/10 ⇒ ≥ 69 % at 95 %, 6/10 ⇒ 26–88 %). With N = 10 the difference is suggestive, not significant (Fisher p ≈ 0.087).

---

## 5. Experiments needed before submission

Priority order for the strongest paper. Each needs ground truth annotated from recorded video; **record all live sessions** (raw video + the program's console log + per-frame timestamp), so every metric can be recomputed.

1. **Instrumentation (do first, 1 hour):** add per-stage timing and a CSV log (frame time, track id, box, object/symbol observation, decision, gate reason, bridge events). Report mean/p95 end-to-end latency and FPS, GPU model, resolution.
2. **Live scenario matrix** (your Section 33 table), ≥ 10 repetitions per scenario, ≥ 2 pallets per scenario. Metrics: success rate, transition latency (seconds from physical event to displayed state change), count of incorrect state changes per hour of operation, tracking continuity. Support threshold: results stated with exact counts and 95 % Wilson intervals.
3. **Ablation on the same recorded videos** (offline replay, deterministic): (a) single-frame object decision vs. temporal; (b) single-frame symbol vs. 2/3 vs. 3/3 multi-view; (c) no gate vs. motion gate; (d) no bridge vs. bridge; (e) fast path vs. normal path for new-object latency; (f) raw vs. enhanced frame. Primary metric: number of wrong final-state frames and wrong transitions per pallet-lap. This is the experiment that proves "temporal reasoning helps". Refactoring the logic into a function that consumes logged classifier outputs makes (a)–(e) cheap, since no model re-run is needed.
4. **Detector/tracker evaluation:** annotate ≥ 200 frames spanning the whole lap including turns/wire; run YOLO11n and YOLO26n through one evaluation script; report P/R/AP50 and ID switches / track fragmentation with and without bridge.
5. **Classifier evaluation on a subject-disjoint test set:** split EMPTY/OBJECT and X/NO_X **by capture session (and ideally by physical object/day)**, not by crop. Recompute the headline numbers. Verify the shipped weight hashes and class-index → name mapping (print `model.names`) and the ACCEPTED/REJECTED semantic check on a physical card before quoting any X/NO_X number.
6. **Statistics:** with small N, report counts and exact/Wilson intervals, never bare percentages.

What result would support the central claim: fewer wrong final-state frames and wrong transitions with the full pipeline than with single-frame decisions, at an acceptable and stated latency cost.

---

## 6. Draft paper material (conservative; fill brackets only with measured values)

**Research question.** How can a real-time machine-vision system maintain stable inspection decisions for pallets on a continuously moving conveyor despite changing viewpoints, temporary occlusion, tracker discontinuities and physical changes to pallet contents?

**Title (recommended).** CREST: A Real-Time Motion-Aware Multi-Stage Vision Framework for Robust Conveyor Pallet Inspection
Alternatives: *Robust Conveyor Pallet Inspection Using Dynamic Tracking, Temporal State Persistence and Multi-View Verification*; *An Occlusion-Resilient Multi-Stage Vision Framework for Continuous Conveyor Inspection*. No RFID in the title.

**Novelty statement (system-level, to be qualified against a literature review).** CREST combines learned visual classifiers with geometry-based tracking support, state-dependent temporal evidence rules and motion-guided view selection to keep a persistent EMPTY / ACCEPTED / REJECTED state for each pallet of a continuously moving conveyor through short occlusions, tracker-ID discontinuities and physical changes of the pallet's contents.

**Contributions (each maps to code and to an experiment).**
1. A dynamic-count pallet inspection pipeline with conveyor-region filtering and dual raw/enhanced detection (§2.1) — *evidence: detector/tracker evaluation (Exp. 4).*
2. Occlusion-resistant state persistence: display/memory separation, geometry-based duplicate suppression and unique-match state bridging across tracker-ID breaks — *Exp. 2, 3d.*
3. Asymmetric, evidence-based object-state transitions with a fast new-object path — *Exp. 2, 3a, 3e.*
4. Motion-guided clean-view gating and time/displacement-separated multi-view symbol verification with state-dependent (2/3 → 3/3) thresholds — *Exp. 3b, 3c.*
5. (Future work) RFID physical-identity association.

**Abstract (template; bracketed items are placeholders for measured values).**
> Frame-level vision classifiers become unreliable on continuously moving conveyors, where viewpoint, occlusion, illumination and the state of the inspected object all change during operation. This paper presents CREST, an experimental real-time multi-stage inspection prototype for pallets circulating on a laboratory conveyor. A fine-tuned YOLO26n detector, a conveyor-region filter and a custom dynamic tracker supply pallet tracks; each pallet crop is classified as EMPTY or OBJECT_PRESENT and, if occupied, its inspection symbol as X or NO_X, producing the operational states EMPTY, ACCEPTED or REJECTED. Instead of trusting individual predictions, CREST accumulates classifier evidence over time: object appearance is accepted faster than object removal, symbol observations are admitted only from motion-stable views and are separated in time and position, established decisions change only under stronger opposite evidence, and state is carried across short tracker-ID changes by a geometry-based bridge. A fast verification path classifies newly placed objects without waiting for a full conveyor lap. In [N] live trials covering [list scenarios], the prototype [measured result with counts and intervals] at [FPS] frames per second on [GPU]; ablations show [measured effect of temporal logic]. The results indicate that coordinating learned classifiers with temporal and motion-aware reasoning can improve the practical reliability of continuous conveyor inspection; limitations include a small single-site dataset and a single conveyor geometry.

(If no quantitative system results exist at submission time, replace the bracketed sentence with the qualitative wording from your notes and state that quantitative system-level evaluation is ongoing — do not add a percentage.)

**Keywords.** conveyor inspection; real-time visual inspection; YOLO; multi-object tracking; temporal state estimation; occlusion robustness; multi-view verification; industrial machine vision.

**Section plan.** I Introduction · II Related Work · III Architecture · IV Detection and Tracking (ring filter, dual detection, tracker, duplicate suppression, state bridge) · V Temporal Object-State Estimation · VI Motion-Aware Multi-View Symbol Verification · VII Experimental Setup · VIII Results · IX Discussion · X Conclusion. RFID appears only under Future Work.

**Experimental-setup table (fill from the lab, no Windows paths in the manuscript).** Camera model (Intel RealSense RGB stream, 1920×1080, 30 FPS requested, achieved [FPS]); GPU [model]; software [Python/Ultralytics/OpenCV versions]; models and sizes (YOLO26n detector 640 px; YOLO26n-cls 320 px EMPTY/OBJECT; YOLO26n-cls 640 px X/NO_X); datasets (§4 with exact split counts and split unit); thresholds (§2).

**Limitations to state.** Tiny, single-site, single-camera datasets (20 detection images, classifier splits that share capture frames); thresholds hand-tuned and frame-rate dependent; clean-view gate encodes this conveyor's clockwise layout and calibrated boundary; removal detection only on the clean straight; bridge uses last-seen box and may fail for long occlusions or neighbouring pallets; no long-duration test; RFID unfinished; not production-ready.

**Related-work searches to run (no citations provided here; every reference must be verified):** deep-learning conveyor/industrial visual inspection; YOLO in manufacturing inspection; multi-object tracking for industrial scenes (compare against SORT/ByteTrack so the custom tracker choice is justified); temporal smoothing / hysteresis for classification in video; multi-view inspection; occlusion handling and short-term re-association; vision + RFID identification. The gap statement ("persistent decision logic under motion, occlusion, tracker breaks and physical content changes") is a hypothesis until that review is done.

---

## 7. Housekeeping recommended before the paper is finalized

1. Commit (or archive with hashes) the X/NO_X training scripts, split files and metrics, and the three weight files' SHA-256 hashes.
2. Reconcile the duplicate `live_final_crest_yolo26.py` copies by hash; state which file produced the reported runs.
3. Fix the stale comment (lines 179–182), delete dead code (`is_left_turn_hold_zone`, `old_new_tracks_match`), decide whether the −7 exposure remains.
4. Rename `04_Live_Empty_Object_System/live_empty_object_classification.py` (it is a labelling tool) and keep YOLO11-era stages clearly marked as historical.
5. Keep the RFID boundary version and Stage 28 out of the evidence base until association is validated.

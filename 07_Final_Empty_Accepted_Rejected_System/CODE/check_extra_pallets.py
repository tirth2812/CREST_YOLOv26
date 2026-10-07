"""
CREST - EXTRA PALLET CHECK  (READ ONLY, vision files are NOT modified)

Uses the exact same pallet detector, brightening, ring and merge code as
32_live_full_conveyor_consistency_test.py, but only MEASURES:
for every pallet-like box it shows the detector confidence and how much of
the box lies on the conveyor ring, and whether Phase 32 would keep it.

Run it with the same lighting you use for the live test, with the REAL
pallets on the belt.  Let it run ~20 s, press q / Esc.
It prints a summary: real pallets vs extra boxes -> which number separates them.
"""

import importlib.util
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

CODE_DIR = Path(__file__).resolve().parent
STAGE_FILE = CODE_DIR / "32_live_full_conveyor_consistency_test.py"

spec = importlib.util.spec_from_file_location("stage32", STAGE_FILE)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

pallet_stage = stage.load_stage08()
model = YOLO(str(stage.PALLET_MODEL_PATH))
boundary = pallet_stage.load_boundary()
cap, index = stage.open_camera()

for _ in range(30):
    cap.read()

stats = {}  # rounded centre cell -> list of (conf, overlap)
started = time.monotonic()

print("Measuring ... press q or Esc to stop.")

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    pallet_frame = stage.make_pallet_frame(frame)
    ring_mask = pallet_stage.create_ring_mask(pallet_frame.shape, boundary)[0]

    detections = stage.merge_pallet_detections(
        pallet_stage.get_yolo_detections(model, frame),
        pallet_stage.get_yolo_detections(model, pallet_frame),
    )
    kept, dropped = pallet_stage.filter_by_ring(detections, ring_mask)

    output = frame.copy()

    for group, kept_flag in ((kept, True), (dropped, False)):
        for d in group:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            conf = float(d["confidence"])
            overlap = float(d.get("overlap", 0.0))
            colour = (0, 255, 0) if kept_flag else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                output,
                f"conf {conf:.2f} ring {overlap:.2f} {'KEPT' if kept_flag else 'dropped'}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colour,
                2,
            )
            cell = (int((x1 + x2) / 2 // 120), int((y1 + y2) / 2 // 120))
            stats.setdefault(cell, []).append((conf, overlap, kept_flag))

    cv2.imshow("CREST extra pallet check (q = stop)", output)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("SUMMARY  (one row per place in the picture, 120 px cells)")
print("cell(x,y)  frames  conf min/max   ring min/max   kept")
for cell, rows in sorted(stats.items()):
    confs = [r[0] for r in rows]
    overlaps = [r[1] for r in rows]
    kept_n = sum(1 for r in rows if r[2])
    print(
        f"({cell[0]:2d},{cell[1]:2d})   {len(rows):5d}   "
        f"{min(confs):.2f}/{max(confs):.2f}    {min(overlaps):.2f}/{max(overlaps):.2f}     "
        f"{kept_n}/{len(rows)}"
    )
print()
print("Send me this table and say which cell is the carton (top-left) and which is the arrow sign (bottom).")

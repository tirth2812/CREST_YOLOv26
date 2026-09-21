from pathlib import Path
import shutil
import json


# ============================================================
# CREST - FINAL YOLO PALLET LOCALIZER PACKAGE
# ============================================================
#
# PURPOSE:
# Create one clean, understandable folder containing:
#
# 1. Training code
# 2. Test/evaluation code
# 3. Trained YOLO model
# 4. Configuration
# 5. Train/validation/test split
# 6. Test metrics
# 7. Visual debug images
# 8. Current CREST pipeline
# 9. Annotation file
# 10. Full end-to-end explanation
#
# ============================================================


ROOT = Path(
    r"C:\Do_Not_Delete_PLC\original images"
)

TRAINING_DIR = (
    ROOT
    / "01_YOLO_Pallet_Localizer_Training"
)

FINAL_DIR = (
    ROOT
    / "02_YOLO_Pallet_Localizer_Final"
)


# ============================================================
# FINAL FOLDER STRUCTURE
# ============================================================

CODE_DIR = FINAL_DIR / "CODE"
MODEL_DIR = FINAL_DIR / "MODEL"
CONFIG_DIR = FINAL_DIR / "CONFIG_AND_RESULTS"
DEBUG_DIR = FINAL_DIR / "VISUAL_TEST_RESULTS"
DOC_DIR = FINAL_DIR / "DOCUMENTATION"


for folder in [
    CODE_DIR,
    MODEL_DIR,
    CONFIG_DIR,
    DEBUG_DIR,
    DOC_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# HELPER
# ============================================================

def copy_if_exists(
    source: Path,
    destination: Path,
):
    if source.exists():

        shutil.copy2(
            source,
            destination,
        )

        print(
            "COPIED:",
            source.name,
        )

    else:

        print(
            "NOT FOUND:",
            source,
        )


# ============================================================
# COPY CODE FILES
# ============================================================

print("\n" + "=" * 75)
print("COPYING CODE")
print("=" * 75)


copy_if_exists(
    TRAINING_DIR
    / "train_yolo_pallet_localizer.py",

    CODE_DIR
    / "train_yolo_pallet_localizer.py",
)


copy_if_exists(
    TRAINING_DIR
    / "evaluate_yolo_pallet_localizer_test.py",

    CODE_DIR
    / "evaluate_yolo_pallet_localizer_test.py",
)


copy_if_exists(
    TRAINING_DIR
    / "fix_lab_crest_parser.py",

    CODE_DIR
    / "fix_lab_crest_parser.py",
)


copy_if_exists(
    ROOT
    / "crest_pipeline.py",

    CODE_DIR
    / "crest_pipeline.py",
)


# ============================================================
# COPY ANNOTATION FILE
# ============================================================

copy_if_exists(
    ROOT
    / "pallet_annotations.json",

    CONFIG_DIR
    / "pallet_annotations.json",
)


# ============================================================
# COPY TRAINED MODEL
# ============================================================

print("\n" + "=" * 75)
print("COPYING TRAINED MODEL")
print("=" * 75)


copy_if_exists(
    TRAINING_DIR
    / "pallet_yolo11n_best.pt",

    MODEL_DIR
    / "pallet_yolo11n_best.pt",
)


# ============================================================
# COPY CONFIGURATION + RESULTS
# ============================================================

print("\n" + "=" * 75)
print("COPYING CONFIGURATION AND RESULTS")
print("=" * 75)


important_result_files = [
    "yolo_localizer_config.json",
    "split_used_for_yolo.json",
    "yolo_test_metrics.json",
]


for filename in important_result_files:

    copy_if_exists(
        TRAINING_DIR
        / filename,

        CONFIG_DIR
        / filename,
    )


# ============================================================
# COPY VISUAL TEST IMAGES
# ============================================================

print("\n" + "=" * 75)
print("COPYING VISUAL TEST IMAGES")
print("=" * 75)


source_debug = (
    TRAINING_DIR
    / "test_debug_images"
)


if source_debug.exists():

    for image_path in source_debug.iterdir():

        if image_path.is_file():

            shutil.copy2(
                image_path,
                DEBUG_DIR
                / image_path.name,
            )

            print(
                "COPIED DEBUG IMAGE:",
                image_path.name,
            )

else:

    print(
        "Debug image folder not found:",
        source_debug,
    )


# ============================================================
# FULL PROJECT DOCUMENTATION
# ============================================================

README_TEXT = r"""
CREST PROJECT
FINAL YOLO11n PALLET LOCALIZER
===============================================


1. PURPOSE
===============================================

The purpose of this stage of the CREST project is to detect
every physical pallet visible on the conveyor.

The detector does NOT yet decide whether the pallet is:

EMPTY
ACCEPTED
REJECTED

Its only responsibility is:

INPUT IMAGE
    |
    v
FIND EVERY PHYSICAL PALLET
    |
    v
RETURN ONE BOUNDING BOX PER PALLET


The pallet detector will later provide cropped pallet images
to the classification model.


2. FINAL SYSTEM ARCHITECTURE
===============================================

The intended complete CREST system is:

LIVE CAMERA
    |
    v
FULL CAMERA IMAGE
    |
    v
YOLO11n PALLET LOCALIZER
    |
    v
ONE BOX PER PHYSICAL PALLET
    |
    v
PALLET CROP
    |
    v
CLASSIFIER
    |
    +-------------------+
    |                   |
    v                   v
EMPTY              OBJECT PRESENT
                        |
                        v
                  ACCEPTED / REJECTED
                        |
                        v
                ROBOT / PLC DECISION


Current work completed:

YOLO PALLET LOCALIZATION

Future work:

EMPTY / ACCEPTED / REJECTED CLASSIFICATION

PLC integration

Robot integration


3. CAMERA IMAGE POLICY
===============================================

The original camera image is used directly.

IMPORTANT:

FULL SOURCE IMAGE

NO TOP_CROP

An earlier version of the project removed approximately
260 pixels from the top of the image.

That method was abandoned.

The current detector always uses the complete camera frame.


4. ORIGINAL DATASET
===============================================

The dataset contained approximately:

1513 source images


Folders included:

empty

REJECTED

mix


The mix folder is intended mainly for testing.


5. MANUAL PALLET ANNOTATIONS
===============================================

20 source images were manually annotated.

Each image contains:

6 physical pallets


Total manually annotated pallet boxes:

120


Class distribution inside the annotation file:

EMPTY:
60 pallet boxes

REJECTED:
60 pallet boxes


For localization training these classes are NOT treated
differently.

The localization model uses only:

CLASS 0 = PALLET


Therefore:

EMPTY pallet
REJECTED pallet

are both simply:

PALLET


6. ANNOTATION FORMAT
===============================================

The annotations are stored in:

pallet_annotations.json


The CREST JSON structure contains entries such as:

empty/original_0039.jpg

REJECTED/original_0401.jpg


Each image contains a list named:

pallets


Each pallet contains a bounding box:

[x1, y1, x2, y2]


The bounding boxes are already XYXY format.


7. LAB CREST PARSER FIX
===============================================

The version of crest_pipeline.py originally copied to the
lab PC could not parse the CREST annotation JSON.

A parser repair was added.

The repair performs:

1. CREST v2 JSON parsing

2. Correct XYXY bounding-box interpretation

3. Correct Windows/lab image-path resolution

4. Correct source-image grouping

5. Preservation of the original pipeline backup


After fixing the parser, the system correctly found:

20 annotated images

120 pallet boxes

60 EMPTY

60 REJECTED


8. DATA SPLIT
===============================================

The 20 annotated source images were divided into:

TRAIN:
10 source images
60 pallet boxes

VALIDATION:
5 source images
30 pallet boxes

TEST:
5 source images
30 pallet boxes


Important:

All 6 pallet boxes from one source image remain together.

Images are never split at the individual pallet level.

This prevents data leakage.


9. FIRST LOCALIZATION MODEL - GROUNDING DINO
===============================================

The first localization system used:

Grounding DINO

Model:

IDEA-Research/grounding-dino-base


Several text prompts were tested.

Prompts such as:

black conveyor pallet.

black conveyor carrier.

industrial workpiece pallet.

did not perform sufficiently well.


The best prompt found was:

black rectangular pallet.


Final Grounding DINO settings:

Box threshold:
0.12

Text threshold:
0.12

Full camera image:
YES

Top crop:
NO


Multiple ROI views were added to help DINO detect small
pallets.


10. GROUNDING DINO VALIDATION RESULT
===============================================

Validation images:

5

Ground-truth pallets:

30


Grounding DINO result:

True Positives:
27

False Positives:
2

False Negatives:
3


Detected:

27 / 30


Precision:

0.931


Recall:

0.900


F1:

0.915


Mean IoU:

approximately 0.737


11. GROUNDING DINO TEST RESULT
===============================================

Held-out test images:

5

Ground-truth pallets:

30


Grounding DINO detected:

25 / 30


Precision:

0.9615


Recall:

0.8333


F1:

0.8929


Mean IoU:

0.7023


False Positives:

1


False Negatives:

5


The main Grounding DINO problem was:

MISSED PALLETS


Because missing a pallet is dangerous for the final conveyor
inspection system, a supervised detector was trained.


12. WHY YOLO11n WAS TRAINED
===============================================

Grounding DINO was a zero-shot detector.

It was useful for proving that pallets could be localized,
but it still missed too many pallets.

Since 120 manually annotated pallet boxes already existed,
those annotations were used to train a supervised detector.

Selected detector:

YOLO11n


YOLO11n was selected because it is:

small

fast

GPU friendly

suitable for real-time camera inference

easy to integrate with Python/OpenCV


13. YOLO11n TRAINING
===============================================

Model:

YOLO11n


Pretrained weights:

yolo11n.pt


Training class:

0 = pallet


Training input:

10 images

60 pallet boxes


Validation input:

5 images

30 pallet boxes


Test images were NOT used during training.


14. YOLO TRAINING SETTINGS
===============================================

Image size:

960


Epoch limit:

150


Early stopping patience:

30


Batch size:

4


Optimizer:

AdamW


Initial learning rate:

0.001


Weight decay:

0.0005


GPU:

NVIDIA GeForce RTX 5090


PyTorch:

2.11.0 + CUDA


CUDA:

12.8


Training image policy:

FULL SOURCE IMAGE


Top crop:

NONE


15. WINDOWS DATALOADER FIX
===============================================

The first YOLO training attempt failed because:

workers=2


Windows multiprocessing attempted to start multiple copies
of the main Python program.

This caused a DataLoader worker error.


The fix was:

workers=0


This is safe because the training dataset contains only
10 source images.


The RTX 5090 still performs the neural-network computation.


16. YOLO CONFIDENCE SELECTION
===============================================

After training, YOLO inference was run on the validation set
using a very low confidence threshold.

Several confidence thresholds were evaluated.

The selected confidence was:

0.05


Selection prioritized:

1. High recall

2. Precision >= approximately 90%

3. High F1

4. Good box IoU


17. YOLO VALIDATION RESULT
===============================================

Validation images:

5


Ground-truth pallets:

30


YOLO detected:

30 / 30


True Positives:

30


False Positives:

0


False Negatives:

0


Precision:

1.0


Recall:

1.0


F1:

1.0


Mean IoU:

0.8302


This was significantly better than the DINO validation
result.


DINO validation:

27 / 30


YOLO validation:

30 / 30


18. YOLO HELD-OUT TEST
===============================================

The YOLO detector was evaluated on the 5 held-out test
images.

No confidence threshold was changed using test results.

Locked parameters:

Confidence:
0.05

Image size:
960

NMS IoU:
0.60


Test ground-truth pallets:

30


YOLO test result:

True Positives:

30


False Positives:

4


False Negatives:

0


Detected:

30 / 30


Precision:

0.88235


Recall:

1.0


F1:

0.9375


Mean IoU:

0.8084


19. DINO VS YOLO TEST COMPARISON
===============================================

GROUNDING DINO

Detected:
25 / 30

Missed:
5

False Positives:
1

Recall:
0.8333

Precision:
0.9615

F1:
0.8929

Mean IoU:
0.7023


YOLO11n

Detected:
30 / 30

Missed:
0

False Positives:
4

Recall:
1.0

Precision:
0.88235

F1:
0.9375

Mean IoU:
0.8084


20. WHY YOLO IS CURRENTLY PREFERRED
===============================================

For this conveyor system, missing a physical pallet is a
larger problem than occasionally producing an extra
detection.

YOLO achieved:

ZERO MISSED PALLETS

on the held-out test set.


Grounding DINO missed:

5 PALLETS


YOLO also produced better average bounding-box overlap.


Therefore the current preferred pallet detector is:

YOLO11n


21. VISUAL INSPECTION
===============================================

Debug images were created.

In those images:

BLUE BOX

=

manual ground-truth annotation


GREEN BOX

=

YOLO prediction


Visual inspection showed that the YOLO boxes match the
physical pallets very well.

A small number of extra green boxes were visible in some
images.

This is consistent with the measured:

4 false-positive detections


The detector can be improved later using:

more training images

more manually annotated images

negative examples

post-processing

position filtering


For now the result is considered strong enough to continue
building the next stage of the project.


22. FINAL YOLO MODEL
===============================================

The trained model is:

pallet_yolo11n_best.pt


Current locked inference settings:

MODEL:

pallet_yolo11n_best.pt


IMAGE SIZE:

960


CONFIDENCE:

0.05


NMS IOU:

0.60


IMAGE INPUT:

FULL ORIGINAL CAMERA FRAME


TOP CROP:

NONE


23. CURRENT LOCALIZATION PIPELINE
===============================================

FULL CAMERA IMAGE
        |
        v
YOLO11n
        |
        v
RAW PALLET DETECTIONS
        |
        v
BOUNDING BOXES
        |
        v
PALLET CROPS


Current YOLO performance:

Validation:
30 / 30

Test:
30 / 30


24. NEXT PROJECT STAGE
===============================================

The next major stage is pallet classification.

For every YOLO pallet box:

YOLO BOX
   |
   v
CROP PALLET
   |
   v
CLASSIFIER
   |
   +----------+----------+
   |          |          |
   v          v          v
 EMPTY     ACCEPTED    REJECTED


Current available labeled pallet data:

EMPTY:
available

REJECTED:
available

ACCEPTED:
not yet available in the current annotation file


Important:

ACCEPTED examples must be collected/labeled before a
truthful final 3-class classifier can be trained.


25. FUTURE COMPLETE PRODUCTION FLOW
===============================================

CAMERA
   |
   v
CAPTURE FULL FRAME
   |
   v
YOLO11n PALLET LOCALIZER
   |
   v
DETECT ALL PALLETS
   |
   v
FILTER / ORDER PALLETS
   |
   v
CROP EACH PALLET
   |
   v
PALLET CLASSIFIER
   |
   +------------------------------+
   |              |               |
   v              v               v
EMPTY          ACCEPTED        REJECTED
                                  |
                                  v
                           PLC / ROBOT ACTION


26. IMPORTANT FILES
===============================================

CODE\train_yolo_pallet_localizer.py

Training script.


CODE\evaluate_yolo_pallet_localizer_test.py

Held-out YOLO evaluation script.


CODE\crest_pipeline.py

CREST annotation and evaluation utilities.


MODEL\pallet_yolo11n_best.pt

Current trained pallet-localization model.


CONFIG_AND_RESULTS\yolo_localizer_config.json

YOLO inference settings and validation metrics.


CONFIG_AND_RESULTS\yolo_test_metrics.json

Held-out test results.


CONFIG_AND_RESULTS\split_used_for_yolo.json

Exact source-image split used during development.


CONFIG_AND_RESULTS\pallet_annotations.json

Manual pallet annotations.


VISUAL_TEST_RESULTS\

Images showing manual and YOLO bounding boxes.


27. STATUS
===============================================

PALLET DATASET:
DONE

PALLET MANUAL ANNOTATION:
DONE

GROUNDING DINO BASELINE:
DONE

YOLO TRAINING:
DONE

YOLO VALIDATION:
DONE

YOLO HELD-OUT TEST:
DONE

YOLO VISUAL CHECK:
DONE


CURRENT SELECTED LOCALIZER:

YOLO11n


NEXT:

PALLET CLASSIFICATION
AND LIVE CAMERA INTEGRATION
"""


README_FILE = (
    DOC_DIR
    / "README_END_TO_END_YOLO_PALLET_PIPELINE.txt"
)


README_FILE.write_text(
    README_TEXT.strip() + "\n",
    encoding="utf-8",
)


print("\n" + "=" * 75)
print("DOCUMENTATION CREATED")
print("=" * 75)

print(README_FILE)


# ============================================================
# SMALL QUICK SUMMARY
# ============================================================

SUMMARY_TEXT = r"""
CREST YOLO PALLET LOCALIZER - QUICK SUMMARY
============================================

SELECTED MODEL
--------------
YOLO11n

MODEL FILE
----------
MODEL\pallet_yolo11n_best.pt


INPUT
-----
Full original camera image.

NO TOP_CROP.


OUTPUT
------
Bounding box around every physical conveyor pallet.


TRAINING
--------
10 images
60 pallet boxes


VALIDATION
----------
5 images
30 pallets

Detected:
30 / 30

Precision:
1.0

Recall:
1.0

F1:
1.0

Mean IoU:
0.8302


HELD-OUT TEST
-------------
5 images
30 pallets

Detected:
30 / 30

Missed:
0

False Positives:
4

Precision:
0.88235

Recall:
1.0

F1:
0.9375

Mean IoU:
0.8084


INFERENCE SETTINGS
------------------
imgsz = 960

confidence = 0.05

NMS IoU = 0.60


NEXT STAGE
----------
Use YOLO boxes to crop pallets.

Then classify:

EMPTY
ACCEPTED
REJECTED
"""


QUICK_FILE = (
    DOC_DIR
    / "QUICK_SUMMARY.txt"
)


QUICK_FILE.write_text(
    SUMMARY_TEXT.strip() + "\n",
    encoding="utf-8",
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 75)
print("FINAL YOLO PACKAGE CREATED")
print("=" * 75)

print("\nFolder:")
print(FINAL_DIR)

print("\nStructure:")
print(
    r"""
02_YOLO_Pallet_Localizer_Final
|
+-- CODE
|   +-- train_yolo_pallet_localizer.py
|   +-- evaluate_yolo_pallet_localizer_test.py
|   +-- fix_lab_crest_parser.py
|   +-- crest_pipeline.py
|
+-- MODEL
|   +-- pallet_yolo11n_best.pt
|
+-- CONFIG_AND_RESULTS
|   +-- pallet_annotations.json
|   +-- split_used_for_yolo.json
|   +-- yolo_localizer_config.json
|   +-- yolo_test_metrics.json
|
+-- VISUAL_TEST_RESULTS
|   +-- test images with GT + YOLO boxes
|
+-- DOCUMENTATION
    +-- README_END_TO_END_YOLO_PALLET_PIPELINE.txt
    +-- QUICK_SUMMARY.txt
"""
)

print("\nDone.")
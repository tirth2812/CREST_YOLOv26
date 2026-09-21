from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(r"C:\Do_Not_Delete_PLC\original images")

CLASS_DIRS = {
    "EMPTY": ROOT_DIR / "empty",
    "REJECTED": ROOT_DIR / "REJECTED",
}

ANNOTATION_FILE = ROOT_DIR / "pallet_annotations.json"

IMAGES_PER_CLASS = 10
TOTAL_TARGET_IMAGES = 20

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

WINDOW_NAME = "CREST Pallet Annotator - 20 Images"

MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 850
MIN_BOX_SIZE = 12

COLOR_EMPTY = (0, 220, 255)
COLOR_REJECTED = (0, 255, 0)
COLOR_ACTIVE = (255, 255, 255)


def list_class_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ],
        key=lambda path: path.name.casefold(),
    )

    if not images:
        raise RuntimeError(f"No images found in: {folder}")

    return images


def image_feature(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        return None

    small = cv2.resize(
        image,
        (32, 24),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    edges = cv2.Canny(
        (gray * 255).astype(np.uint8),
        50,
        130,
    ).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [8, 8],
        [0, 180, 0, 256],
    ).flatten().astype(np.float32)

    hist /= max(float(hist.sum()), 1.0)

    feature = np.concatenate(
        [
            gray.flatten(),
            edges.flatten(),
            hist,
        ]
    ).astype(np.float32)

    norm = float(np.linalg.norm(feature))

    if norm > 0:
        feature /= norm

    return feature


def select_diverse_images(
    paths: list[Path],
    count: int,
) -> list[Path]:
    if len(paths) <= count:
        return paths

    valid_paths: list[Path] = []
    features: list[np.ndarray] = []

    print(f"Analyzing {len(paths)} images for diversity...")

    for path in paths:
        feature = image_feature(path)

        if feature is not None:
            valid_paths.append(path)
            features.append(feature)

    if len(valid_paths) <= count:
        return valid_paths

    feature_matrix = np.vstack(features)

    centroid = feature_matrix.mean(axis=0)

    start_index = int(
        np.argmin(
            np.sum(
                (feature_matrix - centroid) ** 2,
                axis=1,
            )
        )
    )

    selected_indices = [start_index]

    minimum_distances = np.sum(
        (
            feature_matrix
            - feature_matrix[start_index]
        )
        ** 2,
        axis=1,
    )

    minimum_distances[start_index] = -1.0

    while len(selected_indices) < count:
        next_index = int(
            np.argmax(minimum_distances)
        )

        selected_indices.append(next_index)

        new_distances = np.sum(
            (
                feature_matrix
                - feature_matrix[next_index]
            )
            ** 2,
            axis=1,
        )

        minimum_distances = np.minimum(
            minimum_distances,
            new_distances,
        )

        minimum_distances[selected_indices] = -1.0

    return [
        valid_paths[index]
        for index in selected_indices
    ]


def load_annotations() -> dict:
    if not ANNOTATION_FILE.exists():
        return {
            "version": 2,
            "classes": ["EMPTY", "REJECTED"],
            "selected_images": [],
            "images": {},
        }

    try:
        data = json.loads(
            ANNOTATION_FILE.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read {ANNOTATION_FILE}: {error}"
        ) from error

    data.setdefault("version", 2)
    data.setdefault("classes", ["EMPTY", "REJECTED"])
    data.setdefault("selected_images", [])
    data.setdefault("images", {})

    return data


def save_annotations(data: dict) -> None:
    temporary = ANNOTATION_FILE.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(ANNOTATION_FILE)


def build_or_restore_selection(
    annotations: dict,
) -> list[dict]:
    existing_selection = annotations.get(
        "selected_images",
        [],
    )

    if existing_selection:
        samples = []

        for item in existing_selection:
            path = ROOT_DIR / item["image"]

            if not path.is_file():
                raise FileNotFoundError(
                    f"Previously selected image is missing: {path}"
                )

            samples.append(
                {
                    "path": path,
                    "key": item["image"],
                    "source_class": item["source_class"],
                }
            )

        print(
            f"Restored existing {len(samples)}-image selection."
        )

        return samples

    samples = []

    for class_name, folder in CLASS_DIRS.items():
        all_images = list_class_images(folder)

        selected = select_diverse_images(
            all_images,
            IMAGES_PER_CLASS,
        )

        print()
        print(
            f"{class_name}: selected "
            f"{len(selected)} of {len(all_images)} images"
        )

        for path in selected:
            relative_path = path.relative_to(
                ROOT_DIR
            ).as_posix()

            samples.append(
                {
                    "path": path,
                    "key": relative_path,
                    "source_class": class_name,
                }
            )

    if len(samples) != TOTAL_TARGET_IMAGES:
        print(
            f"WARNING: selected {len(samples)} images "
            f"instead of {TOTAL_TARGET_IMAGES}."
        )

    annotations["selected_images"] = [
        {
            "image": sample["key"],
            "source_class": sample["source_class"],
        }
        for sample in samples
    ]

    save_annotations(annotations)

    print()
    print("20-image selection saved.")

    return samples


def class_values(
    class_name: str,
) -> tuple[bool, bool]:
    if class_name == "EMPTY":
        return False, False

    if class_name == "REJECTED":
        return True, True

    raise ValueError(
        f"Unknown class: {class_name}"
    )


def class_color(
    class_name: str,
) -> tuple[int, int, int]:
    if class_name == "EMPTY":
        return COLOR_EMPTY

    return COLOR_REJECTED


class PalletAnnotator:
    def __init__(
        self,
        samples: list[dict],
        annotations: dict,
    ):
        self.samples = samples
        self.annotations = annotations

        self.index = self.find_resume_index()

        self.image: np.ndarray | None = None
        self.scale = 1.0

        self.current_boxes: list[dict] = []
        self.current_class = "EMPTY"

        self.drawing = False
        self.start_point: tuple[int, int] | None = None
        self.current_point: tuple[int, int] | None = None

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            self.mouse_callback,
        )

    def find_resume_index(self) -> int:
        for index, sample in enumerate(self.samples):
            record = self.annotations["images"].get(
                sample["key"]
            )

            if not record or not record.get(
                "reviewed",
                False,
            ):
                return index

        return 0

    def current_sample(self) -> dict:
        return self.samples[self.index]

    def load_current_image(self) -> None:
        sample = self.current_sample()

        self.image = cv2.imread(
            str(sample["path"]),
            cv2.IMREAD_COLOR,
        )

        if self.image is None:
            raise RuntimeError(
                f"Could not open: {sample['path']}"
            )

        height, width = self.image.shape[:2]

        self.scale = min(
            1.0,
            MAX_DISPLAY_WIDTH / width,
            MAX_DISPLAY_HEIGHT / height,
        )

        existing = self.annotations["images"].get(
            sample["key"],
            {},
        )

        self.current_boxes = [
            dict(annotation)
            for annotation in existing.get(
                "pallets",
                [],
            )
        ]

        self.current_class = sample[
            "source_class"
        ]

        self.drawing = False
        self.start_point = None
        self.current_point = None

    def save_current(
        self,
        reviewed: bool | None = None,
    ) -> None:
        if self.image is None:
            return

        sample = self.current_sample()

        height, width = self.image.shape[:2]

        previous = self.annotations[
            "images"
        ].get(
            sample["key"],
            {},
        )

        if reviewed is None:
            reviewed = previous.get(
                "reviewed",
                False,
            )

        self.annotations["images"][
            sample["key"]
        ] = {
            "image": sample["key"],
            "source_class": sample["source_class"],
            "width": width,
            "height": height,
            "reviewed": reviewed,
            "pallets": self.current_boxes,
        }

        save_annotations(
            self.annotations
        )

    def display_to_original(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int]:
        if self.image is None:
            return 0, 0

        height, width = self.image.shape[:2]

        original_x = int(
            round(
                x / self.scale
            )
        )

        original_y = int(
            round(
                y / self.scale
            )
        )

        original_x = int(
            np.clip(
                original_x,
                0,
                width - 1,
            )
        )

        original_y = int(
            np.clip(
                original_y,
                0,
                height - 1,
            )
        )

        return original_x, original_y

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        parameter,
    ) -> None:
        if self.image is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and self.drawing
        ):
            self.current_point = (x, y)

        elif (
            event == cv2.EVENT_LBUTTONUP
            and self.drawing
        ):
            self.drawing = False

            if self.start_point is None:
                return

            start_x, start_y = (
                self.display_to_original(
                    *self.start_point
                )
            )

            end_x, end_y = (
                self.display_to_original(
                    x,
                    y,
                )
            )

            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)
            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)

            self.start_point = None
            self.current_point = None

            if (
                x2 - x1 < MIN_BOX_SIZE
                or y2 - y1 < MIN_BOX_SIZE
            ):
                return

            (
                object_present,
                x_mark_visible,
            ) = class_values(
                self.current_class
            )

            self.current_boxes.append(
                {
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "class": self.current_class,
                    "object_present": object_present,
                    "x_mark_visible": x_mark_visible,
                }
            )

            self.save_current()

    def render(self) -> np.ndarray:
        if self.image is None:
            raise RuntimeError(
                "No image loaded."
            )

        if self.scale < 1.0:
            display_width = int(
                round(
                    self.image.shape[1]
                    * self.scale
                )
            )

            display_height = int(
                round(
                    self.image.shape[0]
                    * self.scale
                )
            )

            canvas = cv2.resize(
                self.image,
                (
                    display_width,
                    display_height,
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            canvas = self.image.copy()

        for box_index, annotation in enumerate(
            self.current_boxes,
            start=1,
        ):
            x1, y1, x2, y2 = annotation[
                "bbox"
            ]

            sx1 = int(round(x1 * self.scale))
            sy1 = int(round(y1 * self.scale))
            sx2 = int(round(x2 * self.scale))
            sy2 = int(round(y2 * self.scale))

            class_name = annotation[
                "class"
            ]

            color = class_color(
                class_name
            )

            cv2.rectangle(
                canvas,
                (sx1, sy1),
                (sx2, sy2),
                color,
                3,
            )

            cv2.putText(
                canvas,
                f"{box_index}: {class_name}",
                (
                    sx1,
                    max(24, sy1 - 7),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

        if (
            self.drawing
            and self.start_point is not None
            and self.current_point is not None
        ):
            cv2.rectangle(
                canvas,
                self.start_point,
                self.current_point,
                COLOR_ACTIVE,
                2,
            )

        sample = self.current_sample()

        completed = sum(
            bool(
                self.annotations[
                    "images"
                ].get(
                    item["key"],
                    {},
                ).get(
                    "reviewed",
                    False,
                )
            )
            for item in self.samples
        )

        cv2.rectangle(
            canvas,
            (0, 0),
            (
                canvas.shape[1],
                78,
            ),
            (0, 0, 0),
            -1,
        )

        line1 = (
            f"Image {self.index + 1}/20 | "
            f"Completed {completed}/20 | "
            f"{sample['key']} | "
            f"Class: {self.current_class} | "
            f"Boxes: {len(self.current_boxes)}"
        )

        line2 = (
            "Drag=box | E=EMPTY | R=REJECTED | "
            "U=undo | SPACE/N=next | P=previous | Q=save+quit"
        )

        cv2.putText(
            canvas,
            line1,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            line2,
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def next_image(self) -> None:
        if not self.current_boxes:
            print(
                "Draw at least one pallet box before continuing."
            )
            return

        self.save_current(
            reviewed=True
        )

        if self.index >= len(
            self.samples
        ) - 1:
            print()
            print("ALL 20 IMAGES ANNOTATED.")
            print(
                "Saved:",
                ANNOTATION_FILE,
            )
            return

        self.index += 1
        self.load_current_image()

    def previous_image(self) -> None:
        self.save_current()

        if self.index == 0:
            return

        self.index -= 1
        self.load_current_image()

    def run(self) -> None:
        self.load_current_image()

        print()
        print("CREST PALLET ANNOTATION")
        print("========================")
        print("Only 20 images will be shown.")
        print("10 EMPTY + 10 REJECTED.")
        print()
        print(
            "Draw around the COMPLETE physical black pallet/tray."
        )
        print()
        print("E      = EMPTY")
        print("R      = REJECTED")
        print("U      = undo last box")
        print("SPACE  = save + next")
        print("N      = save + next")
        print("P      = previous")
        print("Q      = save + quit")
        print()

        while True:
            cv2.imshow(
                WINDOW_NAME,
                self.render(),
            )

            key = cv2.waitKey(20) & 0xFF

            if key == ord("e"):
                self.current_class = "EMPTY"

            elif key == ord("r"):
                self.current_class = "REJECTED"

            elif key == ord("u"):
                if self.current_boxes:
                    self.current_boxes.pop()
                    self.save_current()

            elif key in (32, ord("n")):
                self.next_image()

            elif key == ord("p"):
                self.previous_image()

            elif key in (ord("q"), 27):
                self.save_current()
                break

        cv2.destroyAllWindows()

        completed = sum(
            bool(
                self.annotations[
                    "images"
                ].get(
                    sample["key"],
                    {},
                ).get(
                    "reviewed",
                    False,
                )
            )
            for sample in self.samples
        )

        print()
        print(
            f"Completed annotations: {completed}/20"
        )
        print(
            "Annotation file:",
            ANNOTATION_FILE,
        )


def main() -> None:
    annotations = load_annotations()

    samples = build_or_restore_selection(
        annotations
    )

    empty_selected = sum(
        sample["source_class"] == "EMPTY"
        for sample in samples
    )

    rejected_selected = sum(
        sample["source_class"] == "REJECTED"
        for sample in samples
    )

    print()
    print(
        "EMPTY selected:",
        empty_selected,
    )

    print(
        "REJECTED selected:",
        rejected_selected,
    )

    print(
        "TOTAL selected:",
        len(samples),
    )

    annotator = PalletAnnotator(
        samples,
        annotations,
    )

    annotator.run()


if __name__ == "__main__":
    main()
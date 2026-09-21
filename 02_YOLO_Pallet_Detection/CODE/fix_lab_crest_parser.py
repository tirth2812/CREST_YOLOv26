from pathlib import Path
import shutil

CREST_PATH = Path(
    r"C:\Do_Not_Delete_PLC\original images\crest_pipeline.py"
)

BACKUP_PATH = Path(
    r"C:\Do_Not_Delete_PLC\original images\01_YOLO_Pallet_Localizer_Training"
    r"\crest_pipeline_BEFORE_LAB_PARSER_FIX.py"
)

text = CREST_PATH.read_text(
    encoding="utf-8"
)

# ============================================================
# BACKUP CURRENT LAB PIPELINE
# ============================================================

if not BACKUP_PATH.exists():
    shutil.copy2(
        CREST_PATH,
        BACKUP_PATH,
    )
    print("Backup created:")
    print(BACKUP_PATH)
else:
    print("Backup already exists:")
    print(BACKUP_PATH)


# ============================================================
# FIX 1
# IMAGE PATH RESOLUTION
# ============================================================

start = text.index(
    "def resolve_image_path("
)

end = text.index(
    "\ndef infer_group_id(",
    start,
)

new_resolver = r'''def resolve_image_path(
    name: str,
    root: Path,
    by_rel: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> Path:
    clean = name.replace("\\", "/").lstrip("./")
    clean_lower = clean.lower()

    # 1. Exact relative path.
    if clean in by_rel:
        return by_rel[clean]

    # 2. Case-insensitive exact relative path.
    for rel_name, path in by_rel.items():
        if rel_name.lower() == clean_lower:
            return path

    # 3. Allow an extra dataset folder before annotation path.
    #
    # Example:
    # annotation:
    #     empty/original_0039.jpg
    #
    # possible dataset path:
    #     new-dataset/empty/original_0039.jpg
    suffix_matches = [
        path
        for rel_name, path in by_rel.items()
        if rel_name.lower().endswith("/" + clean_lower)
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # 4. Use parent folder + basename to disambiguate.
    annotation_path = Path(clean)
    basename = annotation_path.name
    parent_name = (
        annotation_path.parent.name.lower()
        if annotation_path.parent.name
        else ""
    )

    matches = by_name.get(
        basename,
        [],
    )

    if parent_name:
        parent_matches = [
            path
            for path in matches
            if path.parent.name.lower() == parent_name
        ]

        if len(parent_matches) == 1:
            return parent_matches[0]

    # 5. Basename fallback only when unique.
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Annotation references missing image: {name}"
        )

    candidate_text = "\n".join(
        str(path)
        for path in matches[:20]
    )

    raise RuntimeError(
        f"Ambiguous image path '{name}'.\n"
        f"Candidates:\n{candidate_text}"
    )
'''

text = (
    text[:start]
    + new_resolver
    + text[end:]
)

print("Fixed image path resolution.")


# ============================================================
# FIX 2
# EACH MANUALLY ANNOTATED SOURCE IMAGE IS ITS OWN GROUP
#
# Do NOT group every original_XXXX.jpg file together.
# ============================================================

start = text.index(
    "def infer_group_id("
)

end = text.index(
    "\ndef box_from_item(",
    start,
)

new_grouping = r'''def infer_group_id(
    image_path: Path,
    root: Path,
) -> str:
    return str(
        image_path
        .relative_to(root)
        .with_suffix("")
    ).replace("\\", "/")
'''

text = (
    text[:start]
    + new_grouping
    + text[end:]
)

print("Fixed source-image grouping.")


# ============================================================
# FIX 3
# PARSE THE ACTUAL CREST ANNOTATION FORMAT
# ============================================================

start = text.index(
    "def parse_annotations("
)

end = text.index(
    "\ndef apply_labels_csv(",
    start,
)

new_parser = r'''def parse_annotations(
    cfg: Config,
) -> tuple[list[PalletAnnotation], Path]:

    annotation_path = (
        cfg.annotations
        or auto_find_annotations(
            cfg.data_root
        )
    )

    by_rel, by_name = build_image_index(
        cfg.data_root
    )

    data = json.loads(
        annotation_path.read_text(
            encoding="utf-8"
        )
    )

    records: list[PalletAnnotation] = []

    def add_record(
        image_name: str,
        box: list[float],
        label: str | None,
        ann_id: str,
    ) -> None:

        path = resolve_image_path(
            image_name,
            cfg.data_root,
            by_rel,
            by_name,
        )

        records.append(
            PalletAnnotation(
                image_name=str(
                    path.relative_to(
                        cfg.data_root
                    )
                ).replace("\\", "/"),

                image_path=str(path),

                bbox=[
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ],

                label=label,

                group_id=infer_group_id(
                    path,
                    cfg.data_root,
                ),

                annotation_id=ann_id,
            )
        )

    # ========================================================
    # CREST LAB ANNOTATION FORMAT VERSION 2
    #
    # {
    #   "version": 2,
    #   "classes": ["EMPTY", "REJECTED"],
    #   "selected_images": [...],
    #   "images": {
    #       "empty/original_0039.jpg": {
    #           "pallets": [
    #               {
    #                   "bbox": [x1,y1,x2,y2],
    #                   "class": "EMPTY"
    #               }
    #           ]
    #       }
    #   }
    # }
    #
    # IMPORTANT:
    # bbox is ALREADY xyxy.
    # Do NOT convert it as COCO xywh.
    # ========================================================

    if (
        isinstance(data, dict)
        and isinstance(
            data.get("images"),
            dict,
        )
    ):

        selected_labels = {}

        selected_images = data.get(
            "selected_images",
            [],
        )

        if isinstance(
            selected_images,
            list,
        ):
            for selected in selected_images:

                if not isinstance(
                    selected,
                    dict,
                ):
                    continue

                image_name = selected.get(
                    "image"
                )

                if not image_name:
                    continue

                source_label = normalize_label(
                    selected.get(
                        "source_class"
                    )
                )

                if source_label:
                    selected_labels[
                        str(image_name)
                    ] = source_label

        sequence = 0

        for image_name, image_record in data[
            "images"
        ].items():

            if not isinstance(
                image_record,
                dict,
            ):
                continue

            parent_label = (
                item_label(
                    image_record
                )
                or selected_labels.get(
                    str(image_name)
                )
            )

            pallets = image_record.get(
                "pallets",
                [],
            )

            if isinstance(
                pallets,
                dict,
            ):
                pallets = list(
                    pallets.values()
                )

            if not isinstance(
                pallets,
                list,
            ):
                continue

            for pallet_index, pallet in enumerate(
                pallets
            ):

                if not isinstance(
                    pallet,
                    dict,
                ):
                    continue

                box = pallet.get(
                    "bbox"
                )

                if (
                    not isinstance(
                        box,
                        (list, tuple),
                    )
                    or len(box) != 4
                ):
                    continue

                try:
                    xyxy = [
                        float(value)
                        for value in box
                    ]
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                label = (
                    item_label(
                        pallet
                    )
                    or parent_label
                )

                add_record(
                    str(image_name),
                    xyxy,
                    label,
                    (
                        f"crest_v2:"
                        f"{sequence}:"
                        f"{pallet_index}"
                    ),
                )

            sequence += 1

        if not records:
            raise RuntimeError(
                "CREST v2 annotation file was detected, "
                "but no valid pallet boxes were found."
            )

        return (
            records,
            annotation_path,
        )

    # ========================================================
    # COCO FORMAT FALLBACK
    # ========================================================

    if (
        isinstance(data, dict)
        and all(
            key in data
            for key in (
                "images",
                "annotations",
            )
        )
        and isinstance(
            data["images"],
            list,
        )
    ):

        images = {
            item["id"]: item
            for item in data["images"]
            if (
                isinstance(
                    item,
                    dict,
                )
                and "id" in item
            )
        }

        categories = {
            item["id"]: normalize_label(
                item.get(
                    "name"
                )
            )
            for item in data.get(
                "categories",
                [],
            )
            if (
                isinstance(
                    item,
                    dict,
                )
                and "id" in item
            )
        }

        for index, annotation in enumerate(
            data["annotations"]
        ):

            if (
                not isinstance(
                    annotation,
                    dict,
                )
                or annotation.get(
                    "image_id"
                )
                not in images
            ):
                continue

            box = annotation.get(
                "bbox"
            )

            if (
                not isinstance(
                    box,
                    list,
                )
                or len(box) != 4
            ):
                continue

            x, y, width, height = map(
                float,
                box,
            )

            xyxy = [
                x,
                y,
                x + width,
                y + height,
            ]

            label = (
                item_label(
                    annotation
                )
                or categories.get(
                    annotation.get(
                        "category_id"
                    )
                )
            )

            add_record(
                images[
                    annotation["image_id"]
                ].get(
                    "file_name",
                    "",
                ),
                xyxy,
                label,
                str(
                    annotation.get(
                        "id",
                        index,
                    )
                ),
            )

        if records:
            return (
                records,
                annotation_path,
            )

    # ========================================================
    # GENERIC FORMAT FALLBACK
    # ========================================================

    def consume_image_record(
        image_record: dict[str, Any],
        sequence: int,
    ) -> None:

        image_name = next(
            (
                image_record.get(key)
                for key in (
                    "image_name",
                    "file_name",
                    "image",
                    "path",
                )
                if image_record.get(key)
            ),
            None,
        )

        if not image_name:
            return

        parent_label = item_label(
            image_record
        )

        items: Any = None

        for key in (
            "pallets",
            "detections",
            "annotations",
            "boxes",
        ):
            if key in image_record:
                items = image_record[key]
                break

        if items is None:

            single = box_from_item(
                image_record
            )

            if single:
                add_record(
                    str(image_name),
                    single,
                    parent_label,
                    str(
                        image_record.get(
                            "id",
                            sequence,
                        )
                    ),
                )

            return

        if isinstance(
            items,
            dict,
        ):
            items = list(
                items.values()
            )

        if not isinstance(
            items,
            list,
        ):
            return

        for item_index, item in enumerate(
            items
        ):

            box = box_from_item(
                item
            )

            if box is None:
                continue

            label = (
                item_label(
                    item
                )
                if isinstance(
                    item,
                    dict,
                )
                else None
            )

            add_record(
                str(image_name),
                box,
                label or parent_label,
                f"{sequence}:{item_index}",
            )

    if (
        isinstance(
            data,
            dict,
        )
        and isinstance(
            data.get(
                "images"
            ),
            list,
        )
    ):

        for index, image_record in enumerate(
            data["images"]
        ):

            if isinstance(
                image_record,
                dict,
            ):
                consume_image_record(
                    image_record,
                    index,
                )

    elif isinstance(
        data,
        list,
    ):

        for index, image_record in enumerate(
            data
        ):

            if isinstance(
                image_record,
                dict,
            ):
                consume_image_record(
                    image_record,
                    index,
                )

    elif isinstance(
        data,
        dict,
    ):

        for index, (
            image_name,
            value,
        ) in enumerate(
            data.items()
        ):

            if not isinstance(
                value,
                (
                    list,
                    dict,
                ),
            ):
                continue

            if (
                isinstance(
                    value,
                    dict,
                )
                and any(
                    key in value
                    for key in (
                        "pallets",
                        "detections",
                        "annotations",
                        "boxes",
                    )
                )
            ):

                record = dict(
                    value
                )

                record.setdefault(
                    "image_name",
                    image_name,
                )

                consume_image_record(
                    record,
                    index,
                )

                continue

            items = (
                value
                if isinstance(
                    value,
                    list,
                )
                else [value]
            )

            for item_index, item in enumerate(
                items
            ):

                box = box_from_item(
                    item
                )

                if box is None:
                    continue

                label = (
                    item_label(
                        item
                    )
                    if isinstance(
                        item,
                        dict,
                    )
                    else None
                )

                add_record(
                    image_name,
                    box,
                    label,
                    f"{index}:{item_index}",
                )

    if not records:
        raise RuntimeError(
            f"Could not parse any pallet boxes from "
            f"{annotation_path}"
        )

    return (
        records,
        annotation_path,
    )
'''

text = (
    text[:start]
    + new_parser
    + text[end:]
)

print("Added CREST v2 annotation parser.")


# ============================================================
# WRITE UPDATED PIPELINE
# ============================================================

CREST_PATH.write_text(
    text,
    encoding="utf-8",
)

print("\nUpdated:")
print(CREST_PATH)

print("\nChanges:")
print("1. CREST v2 annotation JSON supported")
print("2. bbox interpreted as xyxy")
print("3. Lab image paths resolved correctly")
print("4. Each source image is its own split group")
print("5. Original pipeline backup preserved")
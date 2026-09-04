"""Read the local YOLO-converted VisDrone detection data without copying it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisDroneFrame:
    image_path: Path
    boxes_xyxy: np.ndarray
    class_ids: np.ndarray


def load_visdrone_frame(
    image_path: Path, labels_dir: Path, allowed_class_ids: set[int] | None = None
) -> VisDroneFrame:
    """Load one image's normalized YOLO labels as pixel xyxy boxes."""

    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"missing VisDrone label: {label_path}")
    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[list[float]] = []
    classes: list[int] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        class_id = int(fields[0])
        if allowed_class_ids is not None and class_id not in allowed_class_ids:
            continue
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        boxes.append(
            [
                (center_x - box_width / 2.0) * width,
                (center_y - box_height / 2.0) * height,
                (center_x + box_width / 2.0) * width,
                (center_y + box_height / 2.0) * height,
            ]
        )
        classes.append(class_id)
    box_array = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return VisDroneFrame(image_path, box_array, np.asarray(classes, dtype=int))


def inspect_visdrone_split(
    root: Path, split: str, allowed_class_ids: set[int], maximum_frames: int
) -> tuple[int, int]:
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"invalid VisDrone split under {root / split}")
    images = sorted(images_dir.glob("*.jpg"))[:maximum_frames]
    if len(images) < maximum_frames:
        raise ValueError(f"requested {maximum_frames} frames but found only {len(images)}")
    candidate_count = sum(
        len(load_visdrone_frame(image, labels_dir, allowed_class_ids).class_ids) for image in images
    )
    return len(images), candidate_count


"""Convert local VisDrone horizontal vehicle boxes into YOLO OBB labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


DOTA_NAMES = {
    0: "plane",
    1: "ship",
    2: "storage tank",
    3: "baseball diamond",
    4: "tennis court",
    5: "basketball court",
    6: "ground track field",
    7: "harbor",
    8: "bridge",
    9: "large vehicle",
    10: "small vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer ball field",
    14: "swimming pool",
}
VISDRONE_TO_DOTA = {3: 10, 4: 10, 5: 9, 8: 9}  # car/van -> small; truck/bus -> large
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"}


def convert_label(source: Path, destination: Path) -> int:
    converted: list[str] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label at {source}:{line_number}")
        source_class = int(fields[0])
        if source_class not in VISDRONE_TO_DOTA:
            continue
        cx, cy, width, height = map(float, fields[1:])
        x1, y1 = cx - width / 2.0, cy - height / 2.0
        x2, y2 = cx + width / 2.0, cy + height / 2.0
        # A small number of source boxes extend a fraction of a pixel beyond the image boundary.
        # Ultralytics rejects normalized coordinates outside [0, 1], so clip at conversion time.
        corners = (x1, y1, x2, y1, x2, y2, x1, y2)
        coordinates = tuple(min(1.0, max(0.0, value)) for value in corners)
        values = [str(VISDRONE_TO_DOTA[source_class])]
        values.extend(f"{value:.8f}" for value in coordinates)
        converted.append(" ".join(values))
    destination.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
    return len(converted)


def prepare(source_root: Path, output_root: Path) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for short_name, source_name in SPLITS.items():
        source_images = source_root / source_name / "images"
        source_labels = source_root / source_name / "labels"
        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"missing VisDrone split: {source_root / source_name}")
        image_output = output_root / "images" / short_name
        label_output = output_root / "labels" / short_name
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        image_count = 0
        object_count = 0
        for image in sorted(source_images.glob("*.jpg")):
            link = image_output / image.name
            if not link.exists():
                os.symlink(image.resolve(), link)
            source_label = source_labels / f"{image.stem}.txt"
            destination_label = label_output / f"{image.stem}.txt"
            object_count += convert_label(source_label, destination_label)
            image_count += 1
        counts[short_name] = (image_count, object_count)

    dataset = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": DOTA_NAMES,
    }
    (output_root / "visdrone_obb.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
    )
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/visdrone_obb"))
    arguments = parser.parse_args()
    for split, (images, objects) in prepare(arguments.source, arguments.output).items():
        print(f"{split}: images={images} vehicle_objects={objects}")

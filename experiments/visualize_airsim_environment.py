"""Create reproducible visual summaries of the recorded Cosys-AirSim environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "full_localization"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "full_localization.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "full_localization" / "diagnostics"

OVERVIEW_SAMPLES = (
    ("seq04", 0),
    ("seq05", 80),
    ("seq06", 160),
    ("seq07", 240),
    ("seq08", 320),
    ("seq11", 399),
)
SENSOR_SAMPLES = (("seq04", 0), ("seq11", 200))


def _read_record(data_dir: Path, sequence_id: str, frame_index: int) -> dict:
    manifest = data_dir / sequence_id / "manifest.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == frame_index:
                record = json.loads(line)
                if int(record["frame_index"]) != frame_index:
                    raise ValueError(f"manifest index mismatch in {manifest}")
                return record
    raise IndexError(f"frame {frame_index} is absent from {manifest}")


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _add_boxes(axis: plt.Axes, record: dict, *, include_labels: bool = True) -> None:
    evaluation = record["evaluation_only"]
    for name, box in zip(
        evaluation["candidate_vehicle_names"], evaluation["candidate_gt_boxes_xyxy"]
    ):
        x1, y1, x2, y2 = map(float, box)
        served = name == evaluation["served_target_gt_id"]
        color = "#00e676" if served else "#ffd54f"
        axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.2,
            )
        )
        if include_labels:
            label = "served UE" if served else "distractor"
            axis.text(
                x1,
                max(2.0, y1 - 4.0),
                label,
                color="black",
                fontsize=7.5,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
            )

    for name, pixel in evaluation["antenna_pixels_uv"].items():
        u, v = map(float, pixel)
        color = "#ff1744" if name == evaluation["served_target_gt_id"] else "#29b6f6"
        axis.plot(u, v, marker="+", color=color, markersize=8, markeredgewidth=1.8)


def _vehicle_crop(image: np.ndarray, record: dict, margin_px: int = 35) -> tuple[np.ndarray, tuple]:
    boxes = np.asarray(record["evaluation_only"]["candidate_gt_boxes_xyxy"], dtype=float)
    height, width = image.shape[:2]
    x1 = max(0, int(np.floor(boxes[:, 0].min())) - margin_px)
    y1 = max(0, int(np.floor(boxes[:, 1].min())) - margin_px)
    x2 = min(width, int(np.ceil(boxes[:, 2].max())) + margin_px)
    y2 = min(height, int(np.ceil(boxes[:, 3].max())) + margin_px)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def create_overview(data_dir: Path, config_path: Path, output_dir: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profiles = config["f0"]["sequence_profiles"]
    output_path = output_dir / "airsim_environment_overview.png"

    figure, axes = plt.subplots(2, 3, figsize=(15, 9.2), dpi=160, layout="constrained")
    for axis, (sequence_id, frame_index) in zip(axes.flat, OVERVIEW_SAMPLES):
        record = _read_record(data_dir, sequence_id, frame_index)
        profile = profiles[sequence_id]
        axis.imshow(_load_rgb(record["image_path"]))
        _add_boxes(axis, record, include_labels=False)
        axis.set_title(
            f"{sequence_id} / test / frame {frame_index}\n"
            f"H={profile['altitude_m']:.0f} m, horizontal={profile['horizontal_m']:.0f} m, "
            f"bearing={profile['bearing_deg']:.0f}°",
            fontsize=10,
        )
        axis.axis("off")

    figure.suptitle(
        "Recorded Cosys-AirSim Environment — Representative Test Views",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.002,
        "Green: served ground vehicle | Yellow: distractors | RGB frames used by the visual branch",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def create_sensor_comparison(data_dir: Path, config_path: Path, output_dir: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profiles = config["f0"]["sequence_profiles"]
    output_path = output_dir / "airsim_rgb_segmentation_annotations.png"

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.8), dpi=160, layout="constrained")
    for row, (sequence_id, frame_index) in enumerate(SENSOR_SAMPLES):
        record = _read_record(data_dir, sequence_id, frame_index)
        profile = profiles[sequence_id]
        rgb = _load_rgb(record["image_path"])
        segmentation = _load_rgb(record["segmentation_path"])

        axes[row, 0].imshow(rgb)
        _add_boxes(axes[row, 0], record)
        axes[row, 0].set_title(
            f"RGB + geometric ground truth\n{sequence_id}, frame {frame_index}", fontsize=10
        )

        axes[row, 1].imshow(segmentation)
        _add_boxes(axes[row, 1], record, include_labels=False)
        axes[row, 1].set_title("Instance-segmentation capture", fontsize=10)

        crop, bounds = _vehicle_crop(rgb, record)
        axes[row, 2].imshow(crop)
        x1, y1, _, _ = bounds
        crop_record = json.loads(json.dumps(record))
        evaluation = crop_record["evaluation_only"]
        evaluation["candidate_gt_boxes_xyxy"] = [
            [box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1]
            for box in evaluation["candidate_gt_boxes_xyxy"]
        ]
        evaluation["antenna_pixels_uv"] = {
            name: [pixel[0] - x1, pixel[1] - y1]
            for name, pixel in evaluation["antenna_pixels_uv"].items()
        }
        _add_boxes(axes[row, 2], crop_record)
        axes[row, 2].set_title(
            f"Vehicle-region detail\nH={profile['altitude_m']:.0f} m, "
            f"horizontal={profile['horizontal_m']:.0f} m",
            fontsize=10,
        )

        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle(
        "AirSim Sensor Products Used in the Localization Pipeline",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.002,
        "Boxes are simulator ground truth for inspection only; red/blue crosses mark projected antenna phase centers.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = (
        create_overview(args.data_dir, args.config, args.output_dir),
        create_sensor_comparison(args.data_dir, args.config, args.output_dir),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

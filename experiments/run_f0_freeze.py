"""Validate and freeze the formal AirSim splits and LoRA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("F0 freeze requires mode: full_localization")
    phase = config["f0"]
    frame_count = int(phase["frames_per_sequence"])
    maximum_delta_s = float(phase["maximum_rgb_segmentation_delta_s"])
    minimum_box_area = float(phase["minimum_candidate_box_area_px2"])
    root = Path(phase["output_root"])
    sequence_rows = []
    observed: set[str] = set()
    for split, sequence_ids in phase["sequence_splits"].items():
        for sequence_id in sequence_ids:
            if sequence_id in observed:
                raise AssertionError(f"duplicate sequence across splits: {sequence_id}")
            observed.add(sequence_id)
            manifest = root / sequence_id / "manifest.jsonl"
            metadata = root / sequence_id / "metadata.json"
            if not manifest.is_file() or not metadata.is_file():
                raise FileNotFoundError(f"incomplete sequence: {sequence_id}")
            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if len(records) != frame_count:
                raise AssertionError(f"{sequence_id}: {len(records)} != {frame_count}")
            maximum_observed_delta_s = 0.0
            minimum_observed_box_area = math.inf
            for index, record in enumerate(records):
                if record["sequence_id"] != sequence_id or record["split"] != split:
                    raise AssertionError(f"{sequence_id}: split or ID mismatch")
                if record["frame_index"] != index:
                    raise AssertionError(f"{sequence_id}: non-contiguous frame indices")
                if not Path(record["image_path"]).is_file():
                    raise FileNotFoundError(record["image_path"])
                if not Path(record["segmentation_path"]).is_file():
                    raise FileNotFoundError(record["segmentation_path"])
                if not _all_finite(record):
                    raise AssertionError(f"{sequence_id}/{index}: non-finite manifest value")
                pair_delta_s = float(record["image_pair_delta_s"])
                maximum_observed_delta_s = max(maximum_observed_delta_s, pair_delta_s)
                if pair_delta_s > maximum_delta_s:
                    raise AssertionError(
                        f"{sequence_id}/{index}: RGB/segmentation delta {pair_delta_s}s "
                        f"> {maximum_delta_s}s"
                    )
                required_gt = {
                    "served_target_gt_id",
                    "antenna_phase_centers_world_m",
                    "candidate_gt_boxes_xyxy",
                }
                if not required_gt <= set(record["evaluation_only"]):
                    raise AssertionError(f"{sequence_id}: missing evaluation-only GT")
                evaluation = record["evaluation_only"]
                names = evaluation["candidate_vehicle_names"]
                boxes = evaluation["candidate_gt_boxes_xyxy"]
                for box in boxes:
                    area = (float(box[2]) - float(box[0])) * (
                        float(box[3]) - float(box[1])
                    )
                    minimum_observed_box_area = min(minimum_observed_box_area, area)
                    if area < minimum_box_area:
                        raise AssertionError(
                            f"{sequence_id}/{index}: candidate box area {area}px^2 "
                            f"< {minimum_box_area}px^2"
                        )
                served_box = boxes[names.index("ServedVehicle")]
                served_pixel = evaluation["antenna_pixels_uv"]["ServedVehicle"]
                if not (
                    served_box[0] <= served_pixel[0] <= served_box[2]
                    and served_box[1] <= served_pixel[1] <= served_box[3]
                ):
                    raise AssertionError(
                        f"{sequence_id}/{index}: served antenna projects outside GT box"
                    )
            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "frames": len(records),
                    "manifest": str(manifest),
                    "manifest_sha256": _sha256(manifest),
                    "metadata_sha256": _sha256(metadata),
                    "maximum_image_pair_delta_s": maximum_observed_delta_s,
                    "minimum_candidate_box_area_px2": minimum_observed_box_area,
                }
            )
    if len(observed) != 12 or sum(row["frames"] for row in sequence_rows) != 4800:
        raise AssertionError("formal dataset must contain 12 complete sequences and 4800 frames")

    detector_record = Path(config["outputs"]["root"]) / "detector_training.json"
    if not detector_record.is_file():
        raise FileNotFoundError(detector_record)
    detector = json.loads(detector_record.read_text(encoding="utf-8"))
    checkpoint = Path(phase["detector"]["merged_checkpoint"])
    if not checkpoint.is_file() or detector["checkpoint_sha256"] != _sha256(checkpoint):
        raise AssertionError("formal detector checkpoint hash mismatch")
    if detector["selection_metric"] != "validation mAP50-95":
        raise AssertionError("formal detector was not selected by validation mAP50-95")
    if int(detector["training_images"]) <= 0 or int(detector["validation_images"]) <= 0:
        raise AssertionError("formal detector split counts are invalid")

    output_root = Path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    freeze_json = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "sequences": sequence_rows,
        "detector": detector,
    }
    (output_root / "f0_data_freeze.json").write_text(
        json.dumps(freeze_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sequence_lines = "\n".join(
        f"- `{row['sequence_id']}` ({row['split']}, {row['frames']} frames): "
        f"`{row['manifest_sha256']}`"
        for row in sequence_rows
    )
    Path(config["outputs"]["data_freeze"]).write_text(
        f"""# FULL localization data freeze

Status: **PASS**

The formal dataset contains 12 disjoint 400-frame Cosys-AirSim sequences. Splits are made only at
complete-sequence boundaries. RGB, camera/UAV pose and evaluation-only target records are
synchronized in each manifest.

## Frozen sequences

{sequence_lines}

## Frozen detector

- Checkpoint: `{checkpoint}`
- SHA-256: `{detector['checkpoint_sha256']}`
- LoRA rank / alpha / dropout: `{phase['detector']['rank']} / {phase['detector']['alpha']} / {phase['detector']['dropout']}`
- Training / validation images: `{detector['training_images']} / {detector['validation_images']}`
- Selection metric: `{detector['selection_metric']}`
- Validation mAP50-95: `{float(detector['map50_95']):.6f}`

AirSim test sequences were not used for detector checkpoint selection.
""",
        encoding="utf-8",
    )
    print("F0 GATE: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)

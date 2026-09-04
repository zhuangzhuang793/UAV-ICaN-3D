"""Fit and freeze FULL C0/C1 antenna-pixel calibration without test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.vision import (
    FrozenVisualCalibration,
    best_box_match,
    fit_visual_calibrations,
    mean_pixel_nll,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split(config: dict, sequence_ids: list[str], expected_split: str) -> list[dict]:
    root = Path(config["f0"]["output_root"])
    test_ids = set(config["f0"]["sequence_splits"]["test"])
    if test_ids.intersection(sequence_ids):
        raise RuntimeError("F1 refuses to read a final-test sequence")
    records: list[dict] = []
    for sequence_id in sequence_ids:
        path = root / sequence_id / "manifest.jsonl"
        sequence_records = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(sequence_records) != int(config["f0"]["frames_per_sequence"]):
            raise RuntimeError(f"incomplete F1 sequence: {sequence_id}")
        if any(record["split"] != expected_split for record in sequence_records):
            raise RuntimeError(f"unexpected split label in {sequence_id}")
        records.extend(sequence_records)
    return records


def _run_detector(records: list[dict], config: dict, cache_path: Path) -> list[dict]:
    from ultralytics import YOLO

    phase = config["f1"]
    checkpoint = Path(config["f0"]["detector"]["merged_checkpoint"])
    model = YOLO(str(checkpoint))
    predictions = model.predict(
        source=[record["image_path"] for record in records],
        device=str(phase["detector_device"]),
        imgsz=int(phase["detector_image_size_px"]),
        batch=int(phase["detector_batch_size"]),
        conf=float(phase["detector_confidence"]),
        iou=float(phase["detector_nms_iou"]),
        classes=[int(value) for value in phase["detector_vehicle_class_ids"]],
        agnostic_nms=True,
        verbose=False,
        stream=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    detections: list[dict] = []
    with cache_path.open("w", encoding="utf-8") as destination:
        for record, prediction in zip(records, predictions, strict=True):
            if prediction.obb is None:
                boxes_xyxy = np.empty((0, 4), dtype=float)
                boxes_xywhr = np.empty((0, 5), dtype=float)
                classes = np.empty(0, dtype=int)
                confidence = np.empty(0, dtype=float)
            else:
                boxes_xyxy = prediction.obb.xyxy.cpu().numpy().astype(float)
                boxes_xywhr = prediction.obb.xywhr.cpu().numpy().astype(float)
                classes = prediction.obb.cls.cpu().numpy().astype(int)
                confidence = prediction.obb.conf.cpu().numpy().astype(float)
            item = {
                "sequence_id": record["sequence_id"],
                "frame_index": record["frame_index"],
                "timestamp_s": record["timestamp_s"],
                "boxes_xyxy": boxes_xyxy.tolist(),
                "boxes_xywhr": boxes_xywhr.tolist(),
                "class_ids": classes.tolist(),
                "confidences": confidence.tolist(),
            }
            destination.write(json.dumps(item, separators=(",", ":")) + "\n")
            detections.append(item)
    return detections


def _matched_samples(
    records: list[dict], detections: list[dict], minimum_iou: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    boxes: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    for record, detection in zip(records, detections, strict=True):
        gt = record["evaluation_only"]
        names = gt["candidate_vehicle_names"]
        served_gt_box = gt["candidate_gt_boxes_xyxy"][names.index("ServedVehicle")]
        detected_xyxy = np.asarray(detection["boxes_xyxy"], dtype=float).reshape(-1, 4)
        match = best_box_match(served_gt_box, detected_xyxy, minimum_iou)
        if match is None:
            continue
        detected_xywhr = np.asarray(detection["boxes_xywhr"], dtype=float).reshape(-1, 5)
        boxes.append(detected_xywhr[match])
        truth.append(np.asarray(gt["antenna_pixels_uv"]["ServedVehicle"], dtype=float))
        intrinsics = record["camera_intrinsics"]
        current_size = (int(intrinsics["width_px"]), int(intrinsics["height_px"]))
        if image_size is not None and image_size != current_size:
            raise RuntimeError("F1 requires one fixed camera image size")
        image_size = current_size
    if image_size is None or len(boxes) < 3:
        raise RuntimeError("F1 has too few matched served-vehicle detections")
    return np.asarray(boxes), np.asarray(truth), image_size


def _metrics(
    model: FrozenVisualCalibration,
    boxes: np.ndarray,
    truth: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, float]:
    errors = model.corrected_pixels(boxes, image_size) - truth
    return {
        "pixel_nll": mean_pixel_nll(errors, model.covariance_uv),
        "pixel_rmse": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "u_bias_px": float(np.mean(errors[:, 0])),
        "v_bias_px": float(np.mean(errors[:, 1])),
    }


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("F1 requires mode: full_localization")
    phase = config["f1"]
    calibration_ids = list(phase["calibration_sequences"])
    validation_ids = list(phase["validation_sequences"])
    if calibration_ids != list(config["f0"]["sequence_splits"]["calibration"]):
        raise RuntimeError("F1 calibration IDs differ from the F0 freeze")
    if validation_ids != list(config["f0"]["sequence_splits"]["validation"]):
        raise RuntimeError("F1 validation IDs differ from the F0 freeze")

    calibration_records = _load_split(config, calibration_ids, "calibration")
    validation_records = _load_split(config, validation_ids, "validation")
    output_root = Path(config["outputs"]["root"])
    calibration_cache = output_root / "f1_calibration_detections.jsonl"
    validation_cache = output_root / "f1_validation_detections.jsonl"
    calibration_detections = _run_detector(calibration_records, config, calibration_cache)
    validation_detections = _run_detector(validation_records, config, validation_cache)
    minimum_iou = float(phase["calibration_minimum_iou"])
    calibration_boxes, calibration_truth, image_size = _matched_samples(
        calibration_records, calibration_detections, minimum_iou
    )
    validation_boxes, validation_truth, validation_image_size = _matched_samples(
        validation_records, validation_detections, minimum_iou
    )
    if validation_image_size != image_size:
        raise RuntimeError("calibration and validation camera sizes differ")
    c0, c1 = fit_visual_calibrations(
        calibration_boxes,
        calibration_truth,
        image_size,
        float(phase["c1_ridge_alpha"]),
        float(phase["covariance_floor_px2"]),
    )
    candidate_metrics = {
        "C0": _metrics(c0, validation_boxes, validation_truth, image_size),
        "C1": _metrics(c1, validation_boxes, validation_truth, image_size),
    }
    improvement = candidate_metrics["C0"]["pixel_nll"] - candidate_metrics["C1"][
        "pixel_nll"
    ]
    selected = c1 if improvement > float(phase["negligible_nll_improvement"]) else c0
    eigenvalues = np.linalg.eigvalsh(selected.covariance_uv)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
        raise AssertionError("selected F1 covariance is not finite positive definite")

    checkpoint = Path(config["f0"]["detector"]["merged_checkpoint"])
    result = {
        "status": "PASS",
        "selected_model": selected.to_dict(),
        "candidate_models": {"C0": c0.to_dict(), "C1": c1.to_dict()},
        "validation_metrics": candidate_metrics,
        "nll_c0_minus_c1": improvement,
        "negligible_nll_improvement": float(phase["negligible_nll_improvement"]),
        "calibration_sequences": calibration_ids,
        "validation_sequences": validation_ids,
        "calibration_frames": len(calibration_records),
        "validation_frames": len(validation_records),
        "calibration_matches": int(calibration_boxes.shape[0]),
        "validation_matches": int(validation_boxes.shape[0]),
        "image_size_px": list(image_size),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "calibration_detection_cache_sha256": _sha256(calibration_cache),
        "validation_detection_cache_sha256": _sha256(validation_cache),
        "final_test_frames_read": 0,
    }
    result_path = output_root / "f1_visual_calibration.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_root / "f1_visual_model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["candidate", "selected", "pixel_nll", "pixel_rmse", "u_bias_px", "v_bias_px"],
        )
        writer.writeheader()
        for name in ("C0", "C1"):
            writer.writerow(
                {"candidate": name, "selected": int(selected.kind == name), **candidate_metrics[name]}
            )

    Path(config["outputs"]["visual_calibration"]).write_text(
        f"""# FULL visual calibration

Status: **PASS**

The model was fitted on the 800-frame calibration split (`{', '.join(calibration_ids)}`) and
selected by mean pixel NLL on the independent 800-frame validation split
(`{', '.join(validation_ids)}`). No held-out test frame was read.

## Selection

- Selected model: **{selected.kind}**
- Calibration / validation GT-matched detections: `{calibration_boxes.shape[0]} / {validation_boxes.shape[0]}`
- C0 validation pixel NLL / RMSE: `{candidate_metrics['C0']['pixel_nll']:.6f} / {candidate_metrics['C0']['pixel_rmse']:.3f} px`
- C1 validation pixel NLL / RMSE: `{candidate_metrics['C1']['pixel_nll']:.6f} / {candidate_metrics['C1']['pixel_rmse']:.3f} px`
- C0-minus-C1 NLL: `{improvement:.6f}` nats/frame
- Negligible-improvement threshold: `{float(phase['negligible_nll_improvement']):.6f}` nats/frame

## Frozen measurement model

- OBB-center-minus-antenna bias: `{(-selected.correction_intercept_uv).tolist()}` px
- Full anisotropic covariance: `{selected.covariance_uv.tolist()}` px²
- Covariance eigenvalues: `{eigenvalues.tolist()}` px²
- Formal detector SHA-256: `{result['checkpoint_sha256']}`
- Machine-readable artifact: `{result_path}`

C0 is a constant 2-D correction. C1 is ridge regression over normalized OBB center, log size,
log aspect ratio, and doubled-angle sine/cosine features; it outputs only `[delta_u, delta_v]`.
Both covariances are estimated only from calibration residuals. The independently projected
vehicle-pose plus fixed-FRD-lever-arm antenna pixel is the ground truth; detector box centers are
never treated as antenna ground truth.
""",
        encoding="utf-8",
    )
    print(f"calibration_matches={calibration_boxes.shape[0]}/800")
    print(f"validation_matches={validation_boxes.shape[0]}/800")
    print(f"C0_pixel_nll={candidate_metrics['C0']['pixel_nll']:.6f}")
    print(f"C1_pixel_nll={candidate_metrics['C1']['pixel_nll']:.6f}")
    print(f"selected={selected.kind}")
    print("F1 GATE: PASS")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)

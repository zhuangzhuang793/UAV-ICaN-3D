"""Compare image-only CV/Hungarian tracking with RF-gated association."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import chi2

from experiments.run_f1_visual_calibration import _run_detector
from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body
from uav_ican_3d.localization import estimate_position_map, predict_rf_observation, wrap_angle
from uav_ican_3d.types import RFObservation
from uav_ican_3d.vision import (
    FrozenVisualCalibration,
    ImageCVHungarianTracker,
    associate_candidates,
    best_box_match,
    box_iou,
    mahalanobis_squared,
    project_joint_rf_belief_to_image,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform(values: dict, rotation_key: str, translation_key: str) -> RigidTransform:
    return RigidTransform(
        np.asarray(values[rotation_key], dtype=float),
        np.asarray(values[translation_key], dtype=float),
    )


def _camera(record: dict) -> PinholeCamera:
    values = record["camera_intrinsics"]
    return PinholeCamera(
        float(values["fx_px"]),
        float(values["fy_px"]),
        float(values["cx_px"]),
        float(values["cy_px"]),
    )


def _load_test_records(config: dict) -> list[dict]:
    phase0 = config["f0"]
    expected = list(phase0["sequence_splits"]["test"])
    if expected != list(config["f7"]["test_sequences"]):
        raise RuntimeError("held-out test IDs differ between F0 and F7")
    root = Path(phase0["output_root"])
    records: list[dict] = []
    for sequence_id in expected:
        sequence_records = [
            json.loads(line)
            for line in (root / sequence_id / "manifest.jsonl").read_text().splitlines()
            if line
        ]
        if len(sequence_records) != int(phase0["frames_per_sequence"]):
            raise RuntimeError(f"incomplete test sequence: {sequence_id}")
        if any(record["split"] != "test" for record in sequence_records):
            raise RuntimeError(f"non-test record in {sequence_id}")
        records.extend(sequence_records)
    return records


def _candidate_pixels(
    detection: dict, model: FrozenVisualCalibration, record: dict
) -> np.ndarray:
    boxes = np.asarray(detection["boxes_xywhr"], dtype=float).reshape(-1, 5)
    intrinsics = record["camera_intrinsics"]
    return model.corrected_pixels(
        boxes, (int(intrinsics["width_px"]), int(intrinsics["height_px"]))
    )


def _rf_belief(record: dict, phase: dict, rng: np.random.Generator):
    actual_world_body = _transform(record["transform_world_body"], "R_WB", "t_W_B")
    body_camera = _transform(record["transform_body_camera"], "R_BC", "t_B_C")
    target = np.asarray(
        record["evaluation_only"]["antenna_phase_centers_world_m"]["ServedVehicle"],
        dtype=float,
    )
    angle_std = np.deg2rad(float(phase["rf_aoa_std_deg"]))
    rf_covariance = np.diag(
        [float(phase["rf_range_std_m"]) ** 2, angle_std**2, angle_std**2]
    )
    pose_covariance = np.diag(
        [float(phase["pose_position_std_m"]) ** 2] * 3
        + [np.deg2rad(float(phase["pose_attitude_std_deg"])) ** 2] * 3
    )
    nominal_world_body = perturb_world_body(
        actual_world_body, rng.multivariate_normal(np.zeros(6), pose_covariance)
    )
    measurement = predict_rf_observation(target, actual_world_body)
    measurement += rng.multivariate_normal(np.zeros(3), rf_covariance)
    measurement[1:] = wrap_angle(measurement[1:])
    observation = RFObservation(*measurement, rf_covariance, float(record["timestamp_s"]))
    estimate = estimate_position_map(
        observation, nominal_world_body, RigidTransform.identity(), pose_covariance
    )
    belief = project_joint_rf_belief_to_image(
        estimate.position_world_m,
        estimate.joint_covariance,
        nominal_world_body,
        body_camera,
        _camera(record),
        pose_delta_mean=estimate.pose_delta,
    )
    return belief


def _offline_subset(
    record: dict,
    detection: dict,
    candidates: np.ndarray,
    belief,
    minimum_iou: float,
    gate_threshold: float,
) -> tuple[str, int]:
    """Evaluation-only split construction; never called by an online associator."""

    evaluation = record["evaluation_only"]
    names = evaluation["candidate_vehicle_names"]
    boxes = evaluation["candidate_gt_boxes_xyxy"]
    detected_boxes = np.asarray(detection["boxes_xyxy"], dtype=float).reshape(-1, 4)
    qualifying = 0
    for name, gt_box in zip(names, boxes, strict=True):
        if name == "ServedVehicle":
            continue
        match = best_box_match(gt_box, detected_boxes, minimum_iou)
        if match is not None:
            distance = float(mahalanobis_squared(candidates[[match]], belief)[0])
            if distance <= gate_threshold:
                qualifying += 1
    return ("hard" if qualifying else "easy"), qualifying


def _is_correct(record: dict, detection: dict, selected: int | None, minimum_iou: float) -> int:
    if selected is None:
        return 0
    evaluation = record["evaluation_only"]
    names = evaluation["candidate_vehicle_names"]
    served_box = evaluation["candidate_gt_boxes_xyxy"][names.index("ServedVehicle")]
    detected_boxes = np.asarray(detection["boxes_xyxy"], dtype=float).reshape(-1, 4)
    if selected >= len(detected_boxes):
        raise AssertionError("association returned an invalid candidate index")
    return int(box_iou(served_box, detected_boxes[[selected]])[0] >= minimum_iou)


def _accuracy(rows: list[dict], method: str, subset: str) -> float:
    values = [row[f"{method}_correct"] for row in rows if row["subset"] == subset]
    if not values:
        raise RuntimeError(f"F4 produced an empty {subset} subset")
    return float(np.mean(values))


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("F4 requires mode: full_localization")
    phase = config["f4"]
    output_root = Path(config["outputs"]["root"])
    calibration_artifact = output_root / "f1_visual_calibration.json"
    calibration_values = json.loads(calibration_artifact.read_text())
    if calibration_values["status"] != "PASS":
        raise RuntimeError("F4 requires a frozen F1 calibration")
    calibration = FrozenVisualCalibration.from_dict(calibration_values["selected_model"])
    records = _load_test_records(config)
    if len(records) != 3200:
        raise RuntimeError("F4 requires all 3200 held-out frames")
    cache_path = output_root / "f4_test_detections.jsonl"
    detector_config = dict(config)
    detector_config["f1"] = dict(
        config["f1"],
        detector_device=int(phase["detector_device"]),
        detector_batch_size=int(phase["detector_batch_size"]),
    )
    detections = _run_detector(records, detector_config, cache_path)

    rng = np.random.default_rng(int(phase["rf_seed"]))
    candidates_by_frame: list[np.ndarray] = []
    beliefs = []
    subset_rows: list[dict] = []
    minimum_iou = float(phase["subset_candidate_minimum_iou"])
    rf_gate_threshold = float(chi2.ppf(float(phase["rf_gate_confidence"]), df=2))
    for record, detection in zip(records, detections, strict=True):
        candidates = _candidate_pixels(detection, calibration, record)
        belief = _rf_belief(record, phase, rng)
        subset, qualifying = _offline_subset(
            record, detection, candidates, belief, minimum_iou, rf_gate_threshold
        )
        candidates_by_frame.append(candidates)
        beliefs.append(belief)
        subset_rows.append(
            {
                "sequence_id": record["sequence_id"],
                "frame_index": record["frame_index"],
                "subset": subset,
                "qualifying_nonserved_distractors": qualifying,
            }
        )

    subset_path = output_root / "f4_subset_freeze.jsonl"
    subset_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in subset_rows),
        encoding="utf-8",
    )
    subset_hash_before_evaluation = _sha256(subset_path)

    trackers: dict[str, ImageCVHungarianTracker] = {}
    rows: list[dict] = []
    failures: list[dict] = []
    for record, detection, candidates, belief, subset_row in zip(
        records, detections, candidates_by_frame, beliefs, subset_rows, strict=True
    ):
        sequence_id = record["sequence_id"]
        if sequence_id not in trackers:
            trackers[sequence_id] = ImageCVHungarianTracker(
                float(phase["visual_tracker_process_std_px"]),
                float(phase["visual_tracker_measurement_std_px"]),
                float(phase["visual_tracker_gate_confidence"]),
                int(phase["visual_tracker_max_missed_frames"]),
            )
        confidences = np.asarray(detection["confidences"], dtype=float)
        visual_selected = trackers[sequence_id].step(
            candidates, confidences, float(record["timestamp_s"])
        )
        proposed_selected = associate_candidates(
            candidates, belief, float(phase["rf_gate_confidence"])
        ).selected_index
        evaluation = record["evaluation_only"]
        names = evaluation["candidate_vehicle_names"]
        served_box = evaluation["candidate_gt_boxes_xyxy"][names.index("ServedVehicle")]
        detected_boxes = np.asarray(detection["boxes_xyxy"], dtype=float).reshape(-1, 4)
        oracle_selected = best_box_match(served_box, detected_boxes, minimum_iou)
        row = {
            "sequence_id": sequence_id,
            "frame_index": int(record["frame_index"]),
            "subset": subset_row["subset"],
            "detections": len(candidates),
            "visual_selected": -1 if visual_selected is None else visual_selected,
            "proposed_selected": -1 if proposed_selected is None else proposed_selected,
            "oracle_selected": -1 if oracle_selected is None else oracle_selected,
            "visual_correct": _is_correct(record, detection, visual_selected, minimum_iou),
            "proposed_correct": _is_correct(record, detection, proposed_selected, minimum_iou),
            "oracle_correct": int(oracle_selected is not None),
            "proposed_visual_update": int(proposed_selected is not None),
        }
        rows.append(row)
        if not row["proposed_correct"]:
            failures.append(
                {
                    **row,
                    "image_path": record["image_path"],
                    "boxes_xyxy": detection["boxes_xyxy"],
                    "candidate_pixels_uv": candidates.tolist(),
                    "rf_image_mean_uv": belief.mean_uv.tolist(),
                    "rf_image_covariance_uv": belief.covariance_uv.tolist(),
                }
            )
    if _sha256(subset_path) != subset_hash_before_evaluation:
        raise AssertionError("F4 subset changed after association evaluation")

    frame_csv = output_root / "f4_association_frames.csv"
    with frame_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failure_dir = output_root / "failure_cases"
    failure_dir.mkdir(exist_ok=True)
    (failure_dir / "f4_incorrect_associations.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in failures),
        encoding="utf-8",
    )
    metrics = {
        f"{method}_{subset}_accuracy": _accuracy(rows, method, subset)
        for method in ("visual", "proposed", "oracle")
        for subset in ("easy", "hard")
    }
    metrics.update(
        {
            "status": "PASS"
            if metrics["proposed_hard_accuracy"] > metrics["visual_hard_accuracy"]
            else "FAIL",
            "frames": len(rows),
            "easy_frames": sum(row["subset"] == "easy" for row in rows),
            "hard_frames": sum(row["subset"] == "hard" for row in rows),
            "proposed_visual_update_rate": float(
                np.mean([row["proposed_visual_update"] for row in rows])
            ),
            "incorrect_proposed_associations": len(failures),
            "subset_rule": str(phase["hard_subset_rule"]),
            "subset_freeze_sha256": subset_hash_before_evaluation,
            "detector_cache_sha256": _sha256(cache_path),
            "online_gt_fields": [],
        }
    )
    (output_root / "f4_association_summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics["status"] != "PASS":
        raise AssertionError("F4 GATE: FAIL")
    print("F4 GATE: PASS")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)

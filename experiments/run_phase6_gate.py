"""Run real DOTA-pretrained visual detections through joint localization Quick Gate 6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body
from uav_ican_3d.localization import estimate_position_map, predict_rf_observation, wrap_angle
from uav_ican_3d.types import RFObservation, VisualObservation
from uav_ican_3d.vision import (
    associate_candidates,
    best_box_match,
    box_iou,
    calibrate_bbox_center_to_ue_pixel,
    calibrated_box_centers,
    project_joint_rf_belief_to_image,
)


def _load_manifest(path: Path, frame_count: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing synchronized data {path}; run generate_cosys_phase5_data.py first"
        )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) < frame_count:
        raise ValueError(f"manifest has {len(records)} frames, requires {frame_count}")
    return records[:frame_count]


def _run_detector(records: list[dict], phase: dict) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(phase["detector_model"]))
    predictions = model.predict(
        source=[record["image_path"] for record in records],
        device=str(phase["detector_device"]),
        imgsz=int(phase["detector_image_size_px"]),
        conf=float(phase["detector_confidence"]),
        iou=float(phase["detector_nms_iou"]),
        classes=[int(value) for value in phase["detector_vehicle_class_ids"]],
        agnostic_nms=True,
        verbose=False,
        stream=False,
    )
    detections = []
    for record, prediction in zip(records, predictions, strict=True):
        if prediction.obb is None:
            boxes = np.empty((0, 4), dtype=float)
            classes = np.empty(0, dtype=int)
            confidences = np.empty(0, dtype=float)
        else:
            boxes = prediction.obb.xyxy.cpu().numpy().astype(float)
            classes = prediction.obb.cls.cpu().numpy().astype(int)
            confidences = prediction.obb.conf.cpu().numpy().astype(float)
        detections.append(
            {
                "timestamp_s": float(record["timestamp_s"]),
                "boxes_xyxy": boxes.tolist(),
                "class_ids": classes.tolist(),
                "confidences": confidences.tolist(),
            }
        )
    cache_path = Path(phase["detector_cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in detections),
        encoding="utf-8",
    )
    return detections


def _transform(record: dict, rotation_key: str, translation_key: str) -> RigidTransform:
    return RigidTransform(
        np.asarray(record[rotation_key], dtype=float),
        np.asarray(record[translation_key], dtype=float),
    )


def _camera(record: dict) -> PinholeCamera:
    values = record["camera_intrinsics"]
    return PinholeCamera(
        float(values["fx_px"]),
        float(values["fy_px"]),
        float(values["cx_px"]),
        float(values["cy_px"]),
    )


def _noise_covariances(phase: dict) -> tuple[np.ndarray, np.ndarray]:
    angle_std = np.deg2rad(float(phase["rf_aoa_std_deg"]))
    rf_covariance = np.diag(
        [float(phase["rf_range_std_m"]) ** 2, angle_std**2, angle_std**2]
    )
    pose_covariance = np.diag(
        [float(phase["pose_position_std_m"]) ** 2] * 3
        + [np.deg2rad(float(phase["pose_attitude_std_deg"])) ** 2] * 3
    )
    return rf_covariance, pose_covariance


def _rf_estimate(
    record: dict,
    phase: dict,
    rng: np.random.Generator,
) -> tuple[object, RFObservation, RigidTransform, RigidTransform, PinholeCamera]:
    true_world_body = _transform(record["transform_world_body"], "R_WB", "t_W_B")
    body_camera = _transform(record["transform_body_camera"], "R_BC", "t_B_C")
    antenna = np.asarray(record["antenna_phase_center_world_m"], dtype=float)
    camera = _camera(record)
    rf_covariance, pose_covariance = _noise_covariances(phase)
    nominal_world_body = perturb_world_body(
        true_world_body, rng.multivariate_normal(np.zeros(6), pose_covariance)
    )
    measurement = predict_rf_observation(antenna, true_world_body)
    measurement += rng.multivariate_normal(np.zeros(3), rf_covariance)
    measurement[1:] = wrap_angle(measurement[1:])
    observation = RFObservation(
        float(measurement[0]),
        float(measurement[1]),
        float(measurement[2]),
        rf_covariance,
        float(record["timestamp_s"]),
    )
    estimate = estimate_position_map(
        observation,
        nominal_world_body,
        RigidTransform.identity(),
        pose_covariance,
    )
    return estimate, observation, nominal_world_body, body_camera, camera


def _write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    lines = [",".join(columns)]
    lines.extend(",".join(str(row[column]) for column in columns) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 6 gate refuses to run unless mode is 'quick'")
    phase = config["phase6"]
    calibration_count = int(phase["calibration_frames"])
    evaluation_count = int(phase["evaluation_frames"])
    records = _load_manifest(
        Path(phase["synchronized_manifest"]), calibration_count + evaluation_count
    )
    detections = _run_detector(records, phase)
    calibration_records = records[:calibration_count]
    calibration_detections = detections[:calibration_count]
    minimum_iou = float(phase["calibration_minimum_iou"])
    calibration = calibrate_bbox_center_to_ue_pixel(
        [np.asarray(item["boxes_xyxy"], dtype=float) for item in calibration_detections],
        [np.asarray(record["served_gt_box_xyxy"], dtype=float) for record in calibration_records],
        [np.asarray(record["true_ue_pixel_uv"], dtype=float) for record in calibration_records],
        minimum_iou=minimum_iou,
        covariance_floor_px2=float(phase["visual_covariance_floor_px2"]),
    )
    if np.linalg.eigvalsh(calibration.covariance_uv)[0] <= 0.0:
        raise AssertionError("calibrated visual covariance is not positive definite")

    rng = np.random.default_rng(int(config["seed"]) + 6)
    rows: list[dict[str, float | int]] = []
    first_outlier_fallback = False
    for local_index, (record, detection) in enumerate(
        zip(records[calibration_count:], detections[calibration_count:], strict=True)
    ):
        rf_estimate, rf_observation, nominal_world_body, body_camera, camera = _rf_estimate(
            record, phase, rng
        )
        antenna = np.asarray(record["antenna_phase_center_world_m"], dtype=float)
        boxes = np.asarray(detection["boxes_xyxy"], dtype=float).reshape(-1, 4)
        candidates = calibrated_box_centers(boxes, calibration)
        rf_belief = project_joint_rf_belief_to_image(
            rf_estimate.position_world_m,
            rf_estimate.joint_covariance,
            nominal_world_body,
            body_camera,
            camera,
            pose_delta_mean=rf_estimate.pose_delta,
        )
        association = associate_candidates(
            candidates, rf_belief, confidence=float(phase["association_confidence"])
        )
        selected = association.selected_index
        selected_correct = int(
            selected is not None
            and box_iou(record["served_gt_box_xyxy"], boxes[[selected]])[0] >= minimum_iou
        )
        fused_estimate = rf_estimate
        visual_update = 0
        if selected is not None:
            visual_observation = VisualObservation(
                candidates[selected],
                calibration.covariance_uv,
                float(record["timestamp_s"]),
            )
            try:
                fused_estimate = estimate_position_map(
                    rf_observation,
                    nominal_world_body,
                    RigidTransform.identity(),
                    _noise_covariances(phase)[1],
                    camera=camera,
                    transform_body_camera=body_camera,
                    visual_observation=visual_observation,
                    initial_position_world_m=rf_estimate.position_world_m,
                )
                visual_update = 1
            except RuntimeError:
                fused_estimate = rf_estimate

        rf_error = rf_estimate.position_world_m - antenna
        fused_error = fused_estimate.position_world_m - antenna
        gt_match = best_box_match(record["served_gt_box_xyxy"], boxes, minimum_iou)
        rows.append(
            {
                "frame": calibration_count + local_index,
                "detections": len(boxes),
                "served_detected": int(gt_match is not None),
                "association_correct": selected_correct,
                "visual_update": visual_update,
                "rf_error_3d_m": float(np.linalg.norm(rf_error)),
                "fused_error_3d_m": float(np.linalg.norm(fused_error)),
                "rf_error_z_m": float(abs(rf_error[2])),
                "fused_error_z_m": float(abs(fused_error[2])),
            }
        )
        if local_index == 0:
            outlier = rf_belief.mean_uv + 20.0 * np.sqrt(
                np.diag(rf_belief.covariance_uv)
            )
            rejected = associate_candidates(
                outlier.reshape(1, 2),
                rf_belief,
                confidence=float(phase["association_confidence"]),
            )
            first_outlier_fallback = rejected.selected_index is None

    _write_rows(Path(phase["results_csv"]), rows)
    rf_rmse = float(np.sqrt(np.mean([row["rf_error_3d_m"] ** 2 for row in rows])))
    fused_rmse = float(np.sqrt(np.mean([row["fused_error_3d_m"] ** 2 for row in rows])))
    rf_z_rmse = float(np.sqrt(np.mean([row["rf_error_z_m"] ** 2 for row in rows])))
    fused_z_rmse = float(np.sqrt(np.mean([row["fused_error_z_m"] ** 2 for row in rows])))
    visual_updates = int(sum(row["visual_update"] for row in rows))
    association_correct = int(sum(row["association_correct"] for row in rows))
    gate = phase["gate"]
    gain_3d = (rf_rmse - fused_rmse) / rf_rmse
    gain_z = (rf_z_rmse - fused_z_rmse) / rf_z_rmse
    passed = (
        calibration.matched_samples >= int(gate["minimum_calibration_matches"])
        and visual_updates >= int(gate["minimum_evaluation_visual_updates"])
        and gain_3d >= float(gate["minimum_3d_rmse_gain"])
        and gain_z >= float(gate["minimum_z_rmse_gain"])
        and first_outlier_fallback
    )
    status = "PASS" if passed else "FAIL"
    Path("docs/PHASE6_DECISION.md").write_text(
        f"""# Phase 6 decision

Status: **{status}**

YOLO11n-OBB pretrained on the aerial DOTA task supplied real small/large-vehicle detections. The
first {calibration_count} synchronized frames calibrated bbox-center-to-antenna bias and a full
anisotropic pixel covariance from {calibration.matched_samples} GT matches. No detector confidence
was reinterpreted as pixel variance.

- Calibration bias `[u, v]`: `{calibration.bias_uv.tolist()}` px
- Calibration covariance: `{calibration.covariance_uv.tolist()}` px²
- Evaluation visual updates: `{visual_updates}/{evaluation_count}`
- Correct RF-guided detector associations: `{association_correct}/{evaluation_count}`
- RF-only / RF+real-Vision 3-D RMSE: `{rf_rmse:.3f} / {fused_rmse:.3f}` m
- RF-only / RF+real-Vision Z-RMSE: `{rf_z_rmse:.3f} / {fused_z_rmse:.3f}` m
- Relative 3-D / Z gains: `{gain_3d:.3f} / {gain_z:.3f}`
- Deliberate far visual outlier rejected before fusion: `{first_outlier_fallback}`

The detector center is bias-corrected against the independently projected antenna pixel and uses
the empirical residual covariance; it is not directly declared to be the antenna phase center.
Mahalanobis pre-fusion gating falls back to RF-only when a visual candidate is an outlier.
""",
        encoding="utf-8",
    )
    print(f"calibration_matches={calibration.matched_samples}/{calibration_count}")
    print(f"visual_bias_uv_px={calibration.bias_uv.tolist()}")
    print(f"visual_covariance_px2={calibration.covariance_uv.tolist()}")
    print(f"evaluation_visual_updates={visual_updates}/{evaluation_count}")
    print(f"correct_associations={association_correct}/{evaluation_count}")
    print(f"rf_only_rmse_3d_m={rf_rmse:.3f} rf_real_vision_rmse_3d_m={fused_rmse:.3f}")
    print(f"rf_only_z_rmse_m={rf_z_rmse:.3f} rf_real_vision_z_rmse_m={fused_z_rmse:.3f}")
    print(f"relative_gain_3d={gain_3d:.3f} relative_gain_z={gain_z:.3f}")
    print(f"outlier_fallback={first_outlier_fallback}")
    print(f"PHASE 6 QUICK GATE: {status}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)

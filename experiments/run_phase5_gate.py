"""Run the synchronized RF-guided ground-truth association Quick Gate 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body
from uav_ican_3d.localization import (
    estimate_position_map,
    joint_target_pose_information,
    predict_rf_observation,
    rf_observation_and_shared_pose_jacobians,
    wrap_angle,
)
from uav_ican_3d.simulation import complementarity_scene
from uav_ican_3d.types import RFObservation
from uav_ican_3d.vision import (
    ImageBelief,
    associate_candidates,
    inspect_visdrone_split,
    project_joint_rf_belief_to_image,
)


REQUIRED_MANIFEST_FIELDS = {
    "timestamp_s",
    "image_path",
    "camera_intrinsics",
    "transform_world_body",
    "transform_body_camera",
    "vehicle_reference_world_m",
    "antenna_lever_arm_vehicle_m",
    "antenna_phase_center_world_m",
    "candidate_gt_boxes_xyxy",
    "served_candidate_index",
    "served_gt_box_xyxy",
    "segmentation_timestamp_s",
    "transform_world_vehicle",
    "true_ue_pixel_uv",
}


def _rigid_transform(record: dict, rotation_key: str, translation_key: str) -> RigidTransform:
    return RigidTransform(
        np.asarray(record[rotation_key], dtype=float),
        np.asarray(record[translation_key], dtype=float),
    )


def _camera(record: dict) -> PinholeCamera:
    intrinsics = record["camera_intrinsics"]
    return PinholeCamera(
        float(intrinsics["fx_px"]),
        float(intrinsics["fy_px"]),
        float(intrinsics["cx_px"]),
        float(intrinsics["cy_px"]),
    )


def _validate_synchronized_manifest(path: Path, maximum_frames: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("synchronized manifest is empty")
    for line_number, record in enumerate(records[:maximum_frames], start=1):
        missing = REQUIRED_MANIFEST_FIELDS - record.keys()
        if missing:
            raise ValueError(f"manifest line {line_number} misses fields: {sorted(missing)}")
        if not Path(record["image_path"]).is_file():
            raise FileNotFoundError(f"manifest image does not exist: {record['image_path']}")
        if abs(float(record["segmentation_timestamp_s"]) - float(record["timestamp_s"])) > 0.05:
            raise ValueError(f"manifest line {line_number} image pair is not synchronized")
        world_body = _rigid_transform(record["transform_world_body"], "R_WB", "t_W_B")
        body_camera = _rigid_transform(record["transform_body_camera"], "R_BC", "t_B_C")
        world_vehicle = _rigid_transform(record["transform_world_vehicle"], "R_WV", "t_W_V")
        antenna = np.asarray(record["antenna_phase_center_world_m"], dtype=float)
        lever_arm = np.asarray(record["antenna_lever_arm_vehicle_m"], dtype=float)
        if not np.allclose(world_vehicle.apply_point(lever_arm), antenna, atol=1e-6):
            raise ValueError(f"manifest line {line_number} violates antenna lever-arm geometry")
        true_pixel = np.asarray(record["true_ue_pixel_uv"], dtype=float)
        projected = _camera(record).project_world(
            antenna, world_body.compose(body_camera)
        )
        if not np.allclose(projected, true_pixel, atol=1e-5):
            raise ValueError(f"manifest line {line_number} violates camera projection geometry")
        box = np.asarray(record["served_gt_box_xyxy"], dtype=float)
        if not (box[0] <= true_pixel[0] <= box[2] and box[1] <= true_pixel[1] <= box[3]):
            raise ValueError(f"manifest line {line_number} has UE pixel outside served GT box")
    return records[:maximum_frames]


def _geometry_component_smoke(config: dict) -> tuple[float, int | None]:
    phase3 = config["phase3"]
    phase5 = config["phase5"]
    target, world_body, body_array, body_camera, camera = complementarity_scene(
        phase3, 60.0, 60.0
    )
    _, rf_p, rf_xi = rf_observation_and_shared_pose_jacobians(target, world_body, body_array)
    angle_std = np.deg2rad(2.0)
    rf_covariance = np.diag([1.0, angle_std**2, angle_std**2])
    pose_covariance = np.diag([0.2**2] * 3 + [np.deg2rad(0.5) ** 2] * 3)
    rf_information = joint_target_pose_information(
        rf_p, rf_xi, rf_covariance, pose_covariance
    )
    joint_covariance = np.linalg.solve(rf_information, np.eye(9))
    belief = project_joint_rf_belief_to_image(
        target,
        joint_covariance,
        world_body,
        body_camera,
        camera,
        covariance_floor_px2=float(phase5["covariance_floor_px2"]),
    )
    candidates = np.vstack((belief.mean_uv, belief.mean_uv + np.array([200.0, 150.0])))
    association = associate_candidates(
        candidates, belief, confidence=float(phase5["gating_confidence"])
    )
    return float(np.linalg.det(belief.covariance_uv)), association.selected_index


def _write_blocker(
    path: Path, visdrone_frames: int, visdrone_candidates: int, manifest: Path
) -> None:
    path.write_text(
        f"""# Phase 5 blocker

Status: **BLOCKED**

The RF image-belief projection and Mahalanobis association components passed their geometry smoke
test. The local VisDrone adapter also read {visdrone_frames} frames containing
{visdrone_candidates} car/van/truck/bus GT candidates.

VisDrone cannot satisfy the Phase 5 GT-first gate because it has no synchronized UAV pose, camera
pose/calibration, vehicle 3-D reference, UE antenna lever arm, or RF observation. The synchronized
Cosys-AirSim export expected below is missing or malformed.

Expected manifest: `{manifest}`

Do not infer 3-D ground truth by selecting a VisDrone bbox and placing a synthetic vehicle on its
back-projected ray. That would make the projection test circular. Regenerate the small simulator
set with `experiments/generate_cosys_phase5_data.py`; detector inference and Phase 6 must not begin
before the gate is resolved.
""",
        encoding="utf-8",
    )


def _evaluate(records: list[dict], config: dict) -> list[dict[str, float | int]]:
    phase = config["phase5"]
    rng = np.random.default_rng(int(config["seed"]) + 5)
    angle_std = np.deg2rad(float(phase["rf_aoa_std_deg"]))
    rf_covariance = np.diag(
        [float(phase["rf_range_std_m"]) ** 2, angle_std**2, angle_std**2]
    )
    pose_covariance = np.diag(
        [float(phase["pose_position_std_m"]) ** 2] * 3
        + [np.deg2rad(float(phase["pose_attitude_std_deg"])) ** 2] * 3
    )
    confidence = float(phase["gating_confidence"])
    rows: list[dict[str, float | int]] = []
    for frame_index, record in enumerate(records):
        true_world_body = _rigid_transform(
            record["transform_world_body"], "R_WB", "t_W_B"
        )
        body_camera = _rigid_transform(
            record["transform_body_camera"], "R_BC", "t_B_C"
        )
        antenna = np.asarray(record["antenna_phase_center_world_m"], dtype=float)
        true_pixel = np.asarray(record["true_ue_pixel_uv"], dtype=float)
        camera = _camera(record)

        pose_error = rng.multivariate_normal(np.zeros(6), pose_covariance)
        nominal_world_body = perturb_world_body(true_world_body, pose_error)
        rf_vector = predict_rf_observation(antenna, true_world_body)
        rf_vector += rng.multivariate_normal(np.zeros(3), rf_covariance)
        rf_vector[1:] = wrap_angle(rf_vector[1:])
        observation = RFObservation(
            range_m=float(rf_vector[0]),
            azimuth_rad=float(rf_vector[1]),
            elevation_rad=float(rf_vector[2]),
            covariance=rf_covariance,
            timestamp_s=float(record["timestamp_s"]),
        )
        estimate = estimate_position_map(
            observation,
            nominal_world_body,
            RigidTransform.identity(),
            pose_covariance,
        )
        belief = project_joint_rf_belief_to_image(
            estimate.position_world_m,
            estimate.joint_covariance,
            nominal_world_body,
            body_camera,
            camera,
            pose_delta_mean=estimate.pose_delta,
            covariance_floor_px2=float(phase["covariance_floor_px2"]),
        )
        true_distance = float(
            (true_pixel - belief.mean_uv)
            @ np.linalg.solve(belief.covariance_uv, true_pixel - belief.mean_uv)
        )
        boxes = np.asarray(record["candidate_gt_boxes_xyxy"], dtype=float)
        centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        guided = associate_candidates(centers, belief, confidence=confidence)
        served_index = int(record["served_candidate_index"])
        image_center = np.array(
            [record["camera_intrinsics"]["cx_px"], record["camera_intrinsics"]["cy_px"]],
            dtype=float,
        )
        unguided_index = int(np.argmin(np.linalg.norm(centers - image_center, axis=1)))
        rows.append(
            {
                "frame": frame_index,
                "true_mahalanobis_squared": true_distance,
                "covered": int(true_distance <= guided.threshold),
                "guided_correct": int(guided.selected_index == served_index),
                "unguided_correct": int(unguided_index == served_index),
                "guided_selected": -1 if guided.selected_index is None else guided.selected_index,
                "served_index": served_index,
                "belief_u_px": float(belief.mean_uv[0]),
                "belief_v_px": float(belief.mean_uv[1]),
                "true_u_px": float(true_pixel[0]),
                "true_v_px": float(true_pixel[1]),
            }
        )
    return rows


def _write_results(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    lines = [",".join(columns)]
    lines.extend(
        ",".join(str(row[column]) for column in columns)
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(
    path: Path,
    frame_count: int,
    target_confidence: float,
    coverage: float,
    unguided_rate: float,
    guided_rate: float,
    passed: bool,
) -> None:
    status = "PASS" if passed else "FAIL"
    path.write_text(
        f"""# Phase 5 decision

Status: **{status}**

The synchronized QUICK set contains {frame_count} Cosys-AirSim 3.4.1 / UE 5.8 frames. RGB,
external-camera pose/calibration, three vehicle poses, and instance-segmentation GT boxes were
captured while simulation time was paused. The served UE phase center is computed from the vehicle
reference and a fixed FRD lever arm; it is not inferred by back-projecting a box.

- Target confidence: `{target_confidence:.3f}`
- Empirical true-UE pixel coverage: `{coverage:.3f}`
- Unguided image-center baseline association rate: `{unguided_rate:.3f}`
- RF-guided Mahalanobis association rate: `{guided_rate:.3f}`
- Absolute guided gain: `{guided_rate - unguided_rate:.3f}`

GT boxes are used as detector candidates in this gate. Candidate reference pixels remain the
explicitly marked `TEMPORARY_APPROXIMATION` bbox centers; the independently projected antenna
pixel is used for the confidence-coverage test. A detector/keypoint model may replace the GT
candidates in Phase 6 without changing the RF image-belief geometry.
""",
        encoding="utf-8",
    )


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 5 gate refuses to run unless mode is 'quick'")
    phase = config["phase5"]
    determinant, selected = _geometry_component_smoke(config)
    if determinant <= 0.0 or selected != 0:
        raise AssertionError("RF image-belief geometry component smoke test failed")
    frames, candidates = inspect_visdrone_split(
        Path(phase["visdrone_root"]),
        str(phase["visdrone_split"]),
        {int(value) for value in phase["vehicle_class_ids"]},
        int(phase["inspection_frames"]),
    )
    manifest = Path(phase["synchronized_manifest"])
    print(f"image_belief_covariance_determinant={determinant:.6e}")
    print(f"visdrone_frames_inspected={frames} vehicle_candidates={candidates}")
    try:
        synchronized_records = _validate_synchronized_manifest(
            manifest, int(phase["inspection_frames"])
        )
    except (FileNotFoundError, ValueError) as error:
        _write_blocker(Path("docs/PHASE5_BLOCKER.md"), frames, candidates, manifest)
        print(f"synchronized_data_error={error}")
        print("PHASE 5 QUICK GATE: BLOCKED")
        raise SystemExit(2) from error
    rows = _evaluate(synchronized_records, config)
    _write_results(Path(phase["results_csv"]), rows)
    coverage = float(np.mean([row["covered"] for row in rows]))
    guided_rate = float(np.mean([row["guided_correct"] for row in rows]))
    unguided_rate = float(np.mean([row["unguided_correct"] for row in rows]))
    confidence = float(phase["gating_confidence"])
    gate = phase["gate"]
    passed = (
        abs(coverage - confidence) <= float(gate["coverage_tolerance"])
        and guided_rate >= float(gate["minimum_guided_association_rate"])
        and guided_rate - unguided_rate >= float(gate["minimum_guided_gain"])
    )
    _write_decision(
        Path("docs/PHASE5_DECISION.md"),
        len(rows),
        confidence,
        coverage,
        unguided_rate,
        guided_rate,
        passed,
    )
    print(f"synchronized_frames_validated={len(synchronized_records)}")
    print(f"true_ue_pixel_coverage={coverage:.3f} target={confidence:.3f}")
    print(f"unguided_association_rate={unguided_rate:.3f}")
    print(f"rf_guided_association_rate={guided_rate:.3f}")
    print(f"guided_absolute_gain={guided_rate - unguided_rate:.3f}")
    if not passed:
        print("PHASE 5 QUICK GATE: FAIL")
        raise SystemExit(1)
    print("PHASE 5 QUICK GATE: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)

"""Run the Phase 2 camera/shared-pose EFIM quick gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body
from uav_ican_3d.localization import (
    equivalent_target_information,
    joint_target_pose_information,
    position_bounds,
    predict_rf_observation,
    rf_observation_and_shared_pose_jacobians,
    wrap_angle,
)


def _geometry(config: dict) -> tuple[np.ndarray, RigidTransform, RigidTransform, RigidTransform]:
    geometry = config["geometry"]
    target = np.asarray(geometry["ue_position_world_m"], dtype=float)
    world_body = RigidTransform(
        np.asarray(geometry["rotation_world_body"], dtype=float),
        np.asarray(geometry["uav_position_world_m"], dtype=float),
    )
    body_array = RigidTransform(
        np.asarray(geometry["array_rotation_body"], dtype=float),
        np.asarray(geometry["array_translation_body_m"], dtype=float),
    )
    body_camera = RigidTransform(
        np.asarray(geometry["camera_rotation_body"], dtype=float),
        np.asarray(geometry["camera_translation_body_m"], dtype=float),
    )
    return target, world_body, body_array, body_camera


def _pose_numeric_jacobian(
    measurement, world_body: RigidTransform, step: float, angular_rows: slice | None = None
) -> np.ndarray:
    columns = []
    for index in range(6):
        offset = np.zeros(6)
        offset[index] = step
        difference = measurement(perturb_world_body(world_body, offset)) - measurement(
            perturb_world_body(world_body, -offset)
        )
        if angular_rows is not None:
            difference[angular_rows] = wrap_angle(difference[angular_rows])
        columns.append(difference / (2.0 * step))
    return np.column_stack(columns)


def _position_numeric_jacobian(measurement, position: np.ndarray, step: float) -> np.ndarray:
    columns = []
    for index in range(3):
        offset = np.zeros(3)
        offset[index] = step
        columns.append((measurement(position + offset) - measurement(position - offset)) / (2.0 * step))
    return np.column_stack(columns)


def _relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    return float(np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric))


def _bounds(
    rf_p: np.ndarray,
    rf_xi: np.ndarray,
    rf_covariance: np.ndarray,
    pose_covariance: np.ndarray,
    rank_tolerance: float,
    camera_p: np.ndarray | None = None,
    camera_xi: np.ndarray | None = None,
    camera_covariance: np.ndarray | None = None,
):
    joint = joint_target_pose_information(
        rf_p,
        rf_xi,
        rf_covariance,
        pose_covariance,
        camera_p,
        camera_xi,
        camera_covariance,
    )
    efim = equivalent_target_information(joint, rank_tolerance)
    return joint, efim, position_bounds(efim, rank_tolerance)


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 2 gate refuses to run unless mode is 'quick'")
    phase = config["phase2"]
    target, world_body, body_array, body_camera = _geometry(phase)
    camera = PinholeCamera(**config["phase0"]["camera"])
    geometry_tolerance = float(config["phase1"]["geometry_tolerance_m"])

    _, rf_p, rf_xi = rf_observation_and_shared_pose_jacobians(
        target, world_body, body_array, geometry_tolerance
    )
    _, camera_p, camera_xi = camera.observation_and_shared_pose_jacobians(
        target, world_body, body_camera
    )
    step = float(phase["finite_difference_step"])
    rf_measure_pose = lambda pose: predict_rf_observation(
        target, pose.compose(body_array), geometry_tolerance
    )
    camera_measure_pose = lambda pose: camera.project_world(target, pose.compose(body_camera))
    rf_pose_numeric = _pose_numeric_jacobian(rf_measure_pose, world_body, step, slice(1, 3))
    camera_pose_numeric = _pose_numeric_jacobian(camera_measure_pose, world_body, step)
    camera_position_numeric = _position_numeric_jacobian(
        lambda position: camera.project_world(position, world_body.compose(body_camera)), target, step
    )
    errors = {
        "rf_pose": _relative_error(rf_xi, rf_pose_numeric),
        "camera_pose": _relative_error(camera_xi, camera_pose_numeric),
        "camera_position": _relative_error(camera_p, camera_position_numeric),
    }
    worst_error = max(errors.values())
    if worst_error > float(phase["jacobian_relative_tolerance"]):
        raise AssertionError(f"shared-pose/camera Jacobian mismatch: {errors}")

    nominal_rf = config["phase1"]["nominal_noise"]
    rf_covariance = np.diag(
        [
            float(nominal_rf["range_std_m"]) ** 2,
            np.deg2rad(float(nominal_rf["azimuth_std_deg"])) ** 2,
            np.deg2rad(float(nominal_rf["elevation_std_deg"])) ** 2,
        ]
    )
    position_variance = float(phase["pose_position_std_m"]) ** 2
    attitude_variance = np.deg2rad(float(phase["pose_attitude_std_deg"])) ** 2
    pose_covariance = np.diag([position_variance] * 3 + [attitude_variance] * 3)
    camera_std = float(phase["camera_noise_std_px"])
    camera_covariance = np.eye(2) * camera_std**2
    lower_camera_covariance = np.eye(2) * float(phase["lower_camera_noise_std_px"]) ** 2
    rank_tolerance = float(phase["fim_rank_relative_tolerance"])

    rf_joint, rf_efim, rf_bounds = _bounds(
        rf_p, rf_xi, rf_covariance, pose_covariance, rank_tolerance
    )
    camera_off_joint, camera_off_efim, camera_off_bounds = _bounds(
        rf_p, rf_xi, rf_covariance, pose_covariance, rank_tolerance
    )
    if not np.allclose(camera_off_efim, rf_efim, rtol=1e-12, atol=1e-12):
        raise AssertionError("camera-off EFIM does not reduce to RF-only")
    joint, joint_efim, joint_bounds = _bounds(
        rf_p,
        rf_xi,
        rf_covariance,
        pose_covariance,
        rank_tolerance,
        camera_p,
        camera_xi,
        camera_covariance,
    )
    _, _, lower_camera_bounds = _bounds(
        rf_p,
        rf_xi,
        rf_covariance,
        pose_covariance,
        rank_tolerance,
        camera_p,
        camera_xi,
        lower_camera_covariance,
    )
    if lower_camera_bounds.peb_3d_m > joint_bounds.peb_3d_m + 1e-12:
        raise AssertionError("joint PEB increased when camera pixel noise decreased")

    pose_scale = float(phase["larger_pose_covariance_scale"])
    _, _, larger_pose_bounds = _bounds(
        rf_p,
        rf_xi,
        rf_covariance,
        pose_covariance * pose_scale,
        rank_tolerance,
        camera_p,
        camera_xi,
        camera_covariance,
    )
    if larger_pose_bounds.peb_3d_m + 1e-12 < joint_bounds.peb_3d_m:
        raise AssertionError("joint PEB decreased when shared pose covariance increased")

    marginal_target_covariance = np.linalg.solve(joint, np.eye(9))[:3, :3]
    schur_target_covariance = np.linalg.solve(joint_efim, np.eye(3))
    marginal_error = float(
        np.linalg.norm(marginal_target_covariance - schur_target_covariance)
        / np.linalg.norm(marginal_target_covariance)
    )
    if marginal_error > 1e-9:
        raise AssertionError("Schur covariance and joint marginal target block disagree")

    print(f"mode=quick jacobian_errors={errors}")
    print(f"camera_off_vs_rf_efim_max_error={np.max(np.abs(camera_off_efim-rf_efim)):.3e}")
    print(f"rf_only_PEB3D={rf_bounds.peb_3d_m:.6f}m")
    print(f"joint_PEB3D={joint_bounds.peb_3d_m:.6f}m")
    print(f"lower_camera_noise_joint_PEB3D={lower_camera_bounds.peb_3d_m:.6f}m")
    print(f"larger_pose_covariance_joint_PEB3D={larger_pose_bounds.peb_3d_m:.6f}m")
    print(f"schur_marginal_relative_error={marginal_error:.3e}")
    print("PHASE 2 QUICK GATE: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)


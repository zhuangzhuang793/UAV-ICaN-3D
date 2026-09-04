"""Run small observation-level MAP Monte Carlo experiments for Phase 4."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.geometry import perturb_world_body
from uav_ican_3d.localization import estimate_position_map, predict_rf_observation
from uav_ican_3d.simulation import complementarity_scene
from uav_ican_3d.types import RFObservation, VisualObservation


@dataclass(frozen=True)
class GeometryMetrics:
    altitude_m: float
    horizontal_distance_m: float
    trials: int
    rf_rmse_3d_m: float
    joint_rmse_3d_m: float
    rf_z_rmse_m: float
    joint_z_rmse_m: float
    rf_mean_nees: float
    joint_mean_nees: float


def _covariances(phase: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle_std = np.deg2rad(float(phase["aoa_std_deg"]))
    rf = np.diag([float(phase["range_std_m"]) ** 2, angle_std**2, angle_std**2])
    camera = np.eye(2) * float(phase["camera_noise_std_px"]) ** 2
    pose = np.diag(
        [float(phase["pose_position_std_m"]) ** 2] * 3
        + [np.deg2rad(float(phase["pose_attitude_std_deg"])) ** 2] * 3
    )
    return rf, camera, pose


def _metrics(errors: list[np.ndarray], covariances: list[np.ndarray]) -> tuple[float, float, float]:
    error_array = np.asarray(errors)
    rmse_3d = float(np.sqrt(np.mean(np.sum(error_array**2, axis=1))))
    z_rmse = float(np.sqrt(np.mean(error_array[:, 2] ** 2)))
    nees = [float(error @ np.linalg.solve(covariance, error)) for error, covariance in zip(errors, covariances)]
    return rmse_3d, z_rmse, float(np.mean(nees))


def _run_geometry(
    phase3: dict,
    phase4: dict,
    altitude_m: float,
    distance_m: float,
    rng: np.random.Generator,
) -> GeometryMetrics:
    target, nominal_world_body, body_array, body_camera, camera = complementarity_scene(
        phase3, altitude_m, distance_m
    )
    rf_covariance, camera_covariance, pose_covariance = _covariances(phase4)
    optimizer = phase4["optimizer"]
    rf_errors: list[np.ndarray] = []
    joint_errors: list[np.ndarray] = []
    rf_estimate_covariances: list[np.ndarray] = []
    joint_estimate_covariances: list[np.ndarray] = []
    trials = int(phase4["monte_carlo_trials"])
    for trial in range(trials):
        actual_pose_delta = rng.multivariate_normal(np.zeros(6), pose_covariance)
        actual_world_body = perturb_world_body(nominal_world_body, actual_pose_delta)
        rf_mean = predict_rf_observation(target, actual_world_body.compose(body_array))
        pixel_mean = camera.project_world(target, actual_world_body.compose(body_camera))
        rf_vector = rng.multivariate_normal(rf_mean, rf_covariance)
        rf_vector[1:] = (rf_vector[1:] + np.pi) % (2.0 * np.pi) - np.pi
        pixel = rng.multivariate_normal(pixel_mean, camera_covariance)
        timestamp = float(trial)
        rf_observation = RFObservation(*rf_vector, rf_covariance, timestamp)
        visual_observation = VisualObservation(pixel, camera_covariance, timestamp)
        kwargs = dict(
            max_function_evaluations=int(optimizer["max_function_evaluations"]),
            gradient_tolerance=float(optimizer["gradient_tolerance"]),
            parameter_tolerance=float(optimizer["parameter_tolerance"]),
            cost_tolerance=float(optimizer["cost_tolerance"]),
        )
        rf_estimate = estimate_position_map(
            rf_observation,
            nominal_world_body,
            body_array,
            pose_covariance,
            **kwargs,
        )
        joint_estimate = estimate_position_map(
            rf_observation,
            nominal_world_body,
            body_array,
            pose_covariance,
            camera,
            body_camera,
            visual_observation,
            **kwargs,
        )
        for estimate in (rf_estimate, joint_estimate):
            if np.linalg.eigvalsh(estimate.position_covariance)[0] <= 0.0:
                raise AssertionError("MAP position covariance is not positive definite")
        rf_errors.append(rf_estimate.position_world_m - target)
        joint_errors.append(joint_estimate.position_world_m - target)
        rf_estimate_covariances.append(rf_estimate.position_covariance)
        joint_estimate_covariances.append(joint_estimate.position_covariance)
    rf_rmse, rf_z_rmse, rf_nees = _metrics(rf_errors, rf_estimate_covariances)
    joint_rmse, joint_z_rmse, joint_nees = _metrics(
        joint_errors, joint_estimate_covariances
    )
    return GeometryMetrics(
        altitude_m,
        distance_m,
        trials,
        rf_rmse,
        joint_rmse,
        rf_z_rmse,
        joint_z_rmse,
        rf_nees,
        joint_nees,
    )


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 4 gate refuses to run unless mode is 'quick'")
    phase3 = config["phase3"]
    phase4 = config["phase4"]
    trials = int(phase4["monte_carlo_trials"])
    if not 20 <= trials <= 50:
        raise ValueError("Phase 4 QUICK Monte Carlo trials must be between 20 and 50")
    rng = np.random.default_rng(int(config["seed"]) + 4)
    metrics = [
        _run_geometry(phase3, phase4, float(altitude), float(distance), rng)
        for altitude, distance in phase4["representative_geometries_m"]
    ]
    output = Path("results/phase4_quick_monte_carlo.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(GeometryMetrics.__dataclass_fields__))
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.__dict__)

    gate = phase4["gate"]
    for metric in metrics:
        print(metric)
        if metric.joint_rmse_3d_m >= metric.rf_rmse_3d_m:
            raise AssertionError("joint empirical 3-D RMSE did not improve")
        if metric.joint_z_rmse_m >= metric.rf_z_rmse_m:
            raise AssertionError("joint empirical Z RMSE did not improve")
        for nees in (metric.rf_mean_nees, metric.joint_mean_nees):
            if not float(gate["nees_mean_min"]) <= nees <= float(gate["nees_mean_max"]):
                raise AssertionError(f"mean NEES has an obvious order-of-magnitude error: {nees}")
    print("PHASE 4 QUICK GATE: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)

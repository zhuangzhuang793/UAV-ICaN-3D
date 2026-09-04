"""Run the Phase 1 RF measurement and RF-only FIM quick gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.geometry import RigidTransform
from uav_ican_3d.localization import (
    SingularGeometryError,
    fisher_information,
    position_bounds,
    predict_rf_observation,
    rf_position_jacobian,
    wrap_angle,
)


def _finite_difference_jacobian(position: np.ndarray, step: float, tolerance: float) -> np.ndarray:
    columns = []
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        high = predict_rf_observation(position + offset, RigidTransform.identity(), tolerance)
        low = predict_rf_observation(position - offset, RigidTransform.identity(), tolerance)
        difference = high - low
        difference[1:] = wrap_angle(difference[1:])
        columns.append(difference / (2.0 * step))
    return np.column_stack(columns)


def _measurement_covariance(range_std_m: float, angle_std_deg: float) -> np.ndarray:
    angle_std_rad = np.deg2rad(angle_std_deg)
    return np.diag([range_std_m**2, angle_std_rad**2, angle_std_rad**2])


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 1 gate refuses to run unless mode is 'quick'")
    phase = config["phase1"]
    trials = int(phase["random_trials"])
    if not 1 <= trials <= 10:
        raise ValueError("phase1.random_trials must be between 1 and 10")

    rng = np.random.default_rng(int(config["seed"]) + 1)
    step = float(phase["finite_difference_step_m"])
    geometry_tolerance = float(phase["geometry_tolerance_m"])
    worst_relative_error = 0.0
    representative_position = None
    representative_jacobian = None
    for _ in range(trials):
        position = rng.uniform([-100.0, -100.0, -50.0], [100.0, 100.0, 50.0])
        if np.hypot(position[0], position[1]) < 5.0:
            position[0] += 10.0
        analytic = rf_position_jacobian(position, RigidTransform.identity(), geometry_tolerance)
        numeric = _finite_difference_jacobian(position, step, geometry_tolerance)
        relative_error = float(np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric))
        worst_relative_error = max(worst_relative_error, relative_error)
        representative_position = position
        representative_jacobian = analytic

    tolerance = float(phase["jacobian_relative_tolerance"])
    if worst_relative_error > tolerance:
        raise AssertionError(
            f"analytic RF Jacobian relative error {worst_relative_error} exceeds {tolerance}"
        )

    nominal = phase["nominal_noise"]
    nominal_covariance = np.diag(
        [
            float(nominal["range_std_m"]) ** 2,
            np.deg2rad(float(nominal["azimuth_std_deg"])) ** 2,
            np.deg2rad(float(nominal["elevation_std_deg"])) ** 2,
        ]
    )
    increased_range_covariance = nominal_covariance.copy()
    increased_range_covariance[0, 0] = float(phase["increased_range_std_m"]) ** 2
    increased_aoa_covariance = _measurement_covariance(
        float(nominal["range_std_m"]), float(phase["increased_aoa_std_deg"])
    )

    assert representative_position is not None and representative_jacobian is not None
    rank_tolerance = float(phase["fim_rank_relative_tolerance"])
    nominal_bounds = position_bounds(
        fisher_information(representative_jacobian, nominal_covariance), rank_tolerance
    )
    range_bounds = position_bounds(
        fisher_information(representative_jacobian, increased_range_covariance), rank_tolerance
    )
    aoa_bounds = position_bounds(
        fisher_information(representative_jacobian, increased_aoa_covariance), rank_tolerance
    )
    if range_bounds.peb_3d_m + 1e-12 < nominal_bounds.peb_3d_m:
        raise AssertionError("PEB decreased when range noise increased")
    if aoa_bounds.peb_3d_m + 1e-12 < nominal_bounds.peb_3d_m:
        raise AssertionError("PEB decreased when AoA noise increased")

    singular_detected = False
    try:
        predict_rf_observation(
            np.array([0.0, 0.0, 20.0]), RigidTransform.identity(), geometry_tolerance
        )
    except SingularGeometryError:
        singular_detected = True
    if not singular_detected:
        raise AssertionError("array-axis singular geometry was not detected")

    print(f"mode=quick trials={trials}")
    print(f"worst_jacobian_relative_error={worst_relative_error:.3e}")
    print(
        "nominal_bounds="
        f"PEB3D:{nominal_bounds.peb_3d_m:.6f}m "
        f"PEBXY:{nominal_bounds.peb_xy_m:.6f}m ZEB:{nominal_bounds.zeb_m:.6f}m"
    )
    print(f"increased_range_noise_PEB3D={range_bounds.peb_3d_m:.6f}m")
    print(f"increased_aoa_noise_PEB3D={aoa_bounds.peb_3d_m:.6f}m")
    print("singular_array_axis_detected=true")
    print("PHASE 1 QUICK GATE: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)


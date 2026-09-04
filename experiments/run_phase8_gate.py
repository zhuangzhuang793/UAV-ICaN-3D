"""Run one short two-timescale belief-prediction-control QUICK episode."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from uav_ican_3d.control import (
    AngularBelief,
    propagate_position_to_angles,
    robust_mpc_step,
    sample_trajectory_distribution,
    select_robust_beam,
)
from uav_ican_3d.geometry import RigidTransform
from uav_ican_3d.prediction import BeliefTrajectoryPredictor


def _load_predictor(checkpoint_path: Path) -> BeliefTrajectoryPredictor:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"missing {checkpoint_path}; run experiments/run_phase7_gate.py first"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BeliefTrajectoryPredictor(
        int(checkpoint["history_steps"]),
        int(checkpoint["horizon_steps"]),
        int(checkpoint["hidden_size"]),
        float(checkpoint["dt_s"]),
        use_covariance=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _kalman_step(
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement_position: np.ndarray,
    measurement_covariance: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    transition = np.block(
        [[np.eye(3), np.eye(3) * dt_s], [np.zeros((3, 3)), np.eye(3)]]
    )
    process_covariance = np.diag([0.02] * 3 + [0.08] * 3)
    predicted_mean = transition @ mean
    predicted_covariance = transition @ covariance @ transition.T + process_covariance
    observation = np.column_stack((np.eye(3), np.zeros((3, 3))))
    innovation_covariance = (
        observation @ predicted_covariance @ observation.T + measurement_covariance
    )
    gain = predicted_covariance @ observation.T @ np.linalg.inv(innovation_covariance)
    updated_mean = predicted_mean + gain @ (measurement_position - observation @ predicted_mean)
    updated_covariance = (np.eye(6) - gain @ observation) @ predicted_covariance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return updated_mean, updated_covariance


def _advance_vehicle(
    position: np.ndarray, heading: float, speed_mps: float, turn_rate_rps: float, dt_s: float
) -> tuple[np.ndarray, float]:
    velocity = speed_mps * np.array([np.cos(heading), np.sin(heading), 0.0])
    return position + velocity * dt_s, heading + turn_rate_rps * dt_s


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 8 gate refuses to run unless mode is 'quick'")
    phase = config["phase8"]
    predictor = _load_predictor(Path(phase["predictor_checkpoint"]))
    rng = np.random.default_rng(int(config["seed"]) + 8)
    dt_s = float(phase["slow_dt_s"])
    measurement_std = float(phase["position_measurement_std_m"])
    measurement_covariance = np.eye(3) * measurement_std**2
    vehicle_position = np.array([20.0, 5.0, 0.0])
    vehicle_heading = 0.25
    vehicle_speed = 6.0
    vehicle_turn_rate = 0.12
    uav_position = np.array([0.0, 0.0, 50.0])
    filter_mean = np.concatenate(
        (vehicle_position + rng.normal(0.0, measurement_std, 3), np.zeros(3))
    )
    filter_covariance = np.diag([measurement_std**2] * 3 + [4.0] * 3)
    history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=predictor.history_steps)

    for _ in range(predictor.history_steps):
        vehicle_position, vehicle_heading = _advance_vehicle(
            vehicle_position, vehicle_heading, vehicle_speed, vehicle_turn_rate, dt_s
        )
        measurement = vehicle_position + rng.multivariate_normal(
            np.zeros(3), measurement_covariance
        )
        filter_mean, filter_covariance = _kalman_step(
            filter_mean, filter_covariance, measurement, measurement_covariance, dt_s
        )
        history.append((filter_mean.copy(), filter_covariance.copy()))

    azimuth_grid = np.deg2rad(np.asarray(phase["azimuth_grid_deg"], dtype=float))
    elevation_grid = np.deg2rad(np.asarray(phase["elevation_grid_deg"], dtype=float))
    rows: list[dict[str, float | int]] = []
    uncertainty_beam_change = False
    low_probe = None
    high_probe = None
    for slow_step in range(int(phase["episode_steps"])):
        vehicle_position, vehicle_heading = _advance_vehicle(
            vehicle_position, vehicle_heading, vehicle_speed, vehicle_turn_rate, dt_s
        )
        measurement = vehicle_position + rng.multivariate_normal(
            np.zeros(3), measurement_covariance
        )
        filter_mean, filter_covariance = _kalman_step(
            filter_mean, filter_covariance, measurement, measurement_covariance, dt_s
        )
        history.append((filter_mean.copy(), filter_covariance.copy()))
        mean_tensor = torch.from_numpy(
            np.stack([item[0] for item in history]).astype(np.float32)
        ).unsqueeze(0)
        covariance_tensor = torch.from_numpy(
            np.stack([item[1] for item in history]).astype(np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            predicted_mean, predicted_std = predictor(mean_tensor, covariance_tensor)
        predicted_mean_numpy = predicted_mean[0].numpy().astype(float)
        predicted_std_numpy = predicted_std[0].numpy().astype(float)
        trajectory_samples = sample_trajectory_distribution(
            predicted_mean_numpy,
            predicted_std_numpy,
            int(phase["trajectory_samples"]),
            rng,
        )
        decision = robust_mpc_step(
            uav_position,
            trajectory_samples,
            dt_s,
            float(phase["maximum_uav_speed_mps"]),
            tuple(float(value) for value in phase["altitude_bounds_m"]),
        )
        uav_position = uav_position + decision.velocity_world_mps * dt_s
        world_array = RigidTransform(
            np.diag([1.0, -1.0, -1.0]), uav_position.copy()
        )
        angular_belief = propagate_position_to_angles(
            filter_mean[:3], filter_covariance[:3, :3], world_array
        )
        if slow_step == 0:
            low = AngularBelief(
                angular_belief.mean_azimuth_elevation_rad,
                angular_belief.covariance_rad2 * 0.05,
            )
            high = AngularBelief(
                angular_belief.mean_azimuth_elevation_rad,
                angular_belief.covariance_rad2
                * float(phase["beam_high_covariance_probe_scale"]),
            )
            low_probe = select_robust_beam(
                low,
                azimuth_grid,
                elevation_grid,
                np.random.default_rng(int(config["seed"]) + 80),
                int(phase["beam_angle_samples"]),
            )
            high_probe = select_robust_beam(
                high,
                azimuth_grid,
                elevation_grid,
                np.random.default_rng(int(config["seed"]) + 81),
                int(phase["beam_angle_samples"]),
            )
            uncertainty_beam_change = (
                low_probe.aperture_size != high_probe.aperture_size
                or not np.isclose(low_probe.azimuth_rad, high_probe.azimuth_rad)
                or not np.isclose(low_probe.elevation_rad, high_probe.elevation_rad)
            )
        for fast_index in range(int(phase["fast_beam_updates_per_step"])):
            selection = select_robust_beam(
                angular_belief,
                azimuth_grid,
                elevation_grid,
                rng,
                int(phase["beam_angle_samples"]),
            )
            rows.append(
                {
                    "timestamp_s": slow_step * dt_s + fast_index * dt_s / 3.0,
                    "slow_step": slow_step,
                    "fast_step": fast_index,
                    "belief_error_m": float(np.linalg.norm(filter_mean[:3] - vehicle_position)),
                    "prediction_mean_std_m": float(np.mean(predicted_std_numpy)),
                    "uav_x_m": float(uav_position[0]),
                    "uav_y_m": float(uav_position[1]),
                    "uav_z_m": float(uav_position[2]),
                    "uav_speed_mps": float(np.linalg.norm(decision.velocity_world_mps)),
                    "mpc_score": decision.score,
                    "expected_rate": decision.expected_rate,
                    "localization_quality": decision.localization_quality,
                    "beam_azimuth_rad": selection.azimuth_rad,
                    "beam_elevation_rad": selection.elevation_rad,
                    "beam_aperture": selection.aperture_size,
                    "beam_robust_gain": selection.robust_gain,
                }
            )

    numeric_values = np.array(
        [[float(value) for value in row.values()] for row in rows], dtype=float
    )
    altitude_bounds = tuple(float(value) for value in phase["altitude_bounds_m"])
    constraint_ok = bool(
        all(
            row["uav_speed_mps"] <= float(phase["maximum_uav_speed_mps"]) + 1e-9
            and altitude_bounds[0] <= row["uav_z_m"] <= altitude_bounds[1]
            for row in rows
        )
    )
    gate = phase["gate"]
    passed = (
        len({row["slow_step"] for row in rows}) >= int(gate["minimum_slow_updates"])
        and len(rows) >= int(gate["minimum_fast_updates"])
        and np.all(np.isfinite(numeric_values))
        and constraint_ok
        and (
            uncertainty_beam_change
            or not bool(gate["require_uncertainty_beam_change"])
        )
    )
    result_path = Path(phase["results_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    result_path.write_text(
        ",".join(columns)
        + "\n"
        + "\n".join(",".join(str(row[column]) for column in columns) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    assert low_probe is not None and high_probe is not None
    status = "PASS" if passed else "FAIL"
    Path("docs/PHASE8_DECISION.md").write_text(
        f"""# Phase 8 decision

Status: **{status}**

One {int(phase['episode_steps'])}-step slow loop consumed Kalman belief histories with the saved
Phase 7 predictor, sampled complete future trajectories, and selected feasible robust-MPC actions.
Each slow update drove {int(phase['fast_beam_updates_per_step'])} uncertainty-aware beam updates on
a physical 4x4 UPA codebook with 4x4 narrow and 2x2 broad tapers.

- Slow / fast updates: `{len({row['slow_step'] for row in rows})} / {len(rows)}`
- All values finite and UAV constraints satisfied: `{constraint_ok}`
- Low-covariance beam: aperture `{low_probe.aperture_size}`, angles
  `{[low_probe.azimuth_rad, low_probe.elevation_rad]}` rad
- High-covariance beam: aperture `{high_probe.aperture_size}`, angles
  `{[high_probe.azimuth_rad, high_probe.elevation_rad]}` rad
- Beam selection changed with angular covariance: `{uncertainty_beam_change}`

The controller receives only filtered beliefs and predicted distributions. Environment truth is
used to generate noisy measurements and log error, not as predictor, MPC, or beam-selector input.
""",
        encoding="utf-8",
    )
    print(f"slow_updates={len({row['slow_step'] for row in rows})}")
    print(f"fast_beam_updates={len(rows)}")
    print(f"constraints_ok={constraint_ok} all_finite={np.all(np.isfinite(numeric_values))}")
    print(
        f"low_cov_beam=({low_probe.aperture_size}, {low_probe.azimuth_rad:.3f}, "
        f"{low_probe.elevation_rad:.3f})"
    )
    print(
        f"high_cov_beam=({high_probe.aperture_size}, {high_probe.azimuth_rad:.3f}, "
        f"{high_probe.elevation_rad:.3f})"
    )
    print(f"uncertainty_changed_beam={uncertainty_beam_change}")
    print(f"PHASE 8 QUICK GATE: {status}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)

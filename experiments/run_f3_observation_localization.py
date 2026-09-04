"""Paired nonlinear-MAP Monte Carlo over the eight frozen F2 geometries."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import chi2, spearmanr

from uav_ican_3d.geometry import perturb_world_body
from uav_ican_3d.localization import estimate_position_map, predict_rf_observation, wrap_angle
from uav_ican_3d.simulation import complementarity_scene
from uav_ican_3d.types import RFObservation, VisualObservation


@dataclass(frozen=True)
class TrialResult:
    cell_id: int
    trial: int
    seed: int
    rf_error_x_m: float
    rf_error_y_m: float
    rf_error_z_m: float
    joint_error_x_m: float
    joint_error_y_m: float
    joint_error_z_m: float
    rf_nees: float
    joint_nees: float
    rf_covered_95: int
    joint_covered_95: int


def _trial_metrics(error: np.ndarray, covariance: np.ndarray, threshold: float) -> tuple[float, int]:
    nees = float(error @ np.linalg.solve(covariance, error))
    return nees, int(nees <= threshold)


def _run_geometry(payload: tuple[dict, dict, dict, dict, int]) -> tuple[dict, list[TrialResult]]:
    phase2, phase3, cell, visual_model, seed = payload
    cell_id = int(cell["cell_id"])
    altitude = float(cell["altitude_m"])
    distance = float(cell["horizontal_distance_m"])
    target, nominal_world_body, body_array, body_camera, camera = complementarity_scene(
        phase2, altitude, distance
    )
    rf_parameters = visual_model["rf_parameters"]
    angle_std = np.deg2rad(float(rf_parameters["aoa_std_deg"]))
    rf_covariance = np.diag(
        [float(rf_parameters["range_std_m"]) ** 2, angle_std**2, angle_std**2]
    )
    pose_covariance = np.diag(
        [float(rf_parameters["pose_position_std_m"]) ** 2] * 3
        + [np.deg2rad(float(rf_parameters["attitude_std_deg"])) ** 2] * 3
    )
    camera_covariance = np.asarray(visual_model["camera_covariance_uv_px2"], dtype=float)
    optimizer = phase3["optimizer"]
    optimizer_kwargs = {
        "max_function_evaluations": int(optimizer["max_function_evaluations"]),
        "gradient_tolerance": float(optimizer["gradient_tolerance"]),
        "parameter_tolerance": float(optimizer["parameter_tolerance"]),
        "cost_tolerance": float(optimizer["cost_tolerance"]),
    }
    rng = np.random.default_rng(seed)
    threshold = float(chi2.ppf(float(phase3["coverage_confidence"]), df=3))
    trials: list[TrialResult] = []
    for trial in range(int(phase3["monte_carlo_trials"])):
        actual_delta = rng.multivariate_normal(np.zeros(6), pose_covariance)
        actual_world_body = perturb_world_body(nominal_world_body, actual_delta)
        rf_mean = predict_rf_observation(target, actual_world_body.compose(body_array))
        rf_vector = rf_mean + rng.multivariate_normal(np.zeros(3), rf_covariance)
        rf_vector[1:] = wrap_angle(rf_vector[1:])
        pixel_mean = camera.project_world(target, actual_world_body.compose(body_camera))
        pixel = rng.multivariate_normal(pixel_mean, camera_covariance)
        timestamp = float(trial)
        rf_observation = RFObservation(*rf_vector, rf_covariance, timestamp)
        visual_observation = VisualObservation(pixel, camera_covariance, timestamp)
        rf_estimate = estimate_position_map(
            rf_observation,
            nominal_world_body,
            body_array,
            pose_covariance,
            **optimizer_kwargs,
        )
        joint_estimate = estimate_position_map(
            rf_observation,
            nominal_world_body,
            body_array,
            pose_covariance,
            camera,
            body_camera,
            visual_observation,
            initial_position_world_m=rf_estimate.position_world_m,
            **optimizer_kwargs,
        )
        rf_error = rf_estimate.position_world_m - target
        joint_error = joint_estimate.position_world_m - target
        rf_nees, rf_covered = _trial_metrics(
            rf_error, rf_estimate.position_covariance, threshold
        )
        joint_nees, joint_covered = _trial_metrics(
            joint_error, joint_estimate.position_covariance, threshold
        )
        trials.append(
            TrialResult(
                cell_id,
                trial,
                seed,
                *map(float, rf_error),
                *map(float, joint_error),
                rf_nees,
                joint_nees,
                rf_covered,
                joint_covered,
            )
        )

    rf_errors = np.asarray([[row.rf_error_x_m, row.rf_error_y_m, row.rf_error_z_m] for row in trials])
    joint_errors = np.asarray(
        [[row.joint_error_x_m, row.joint_error_y_m, row.joint_error_z_m] for row in trials]
    )
    rf_rmse = float(np.sqrt(np.mean(np.sum(rf_errors**2, axis=1))))
    joint_rmse = float(np.sqrt(np.mean(np.sum(joint_errors**2, axis=1))))
    rf_z_rmse = float(np.sqrt(np.mean(rf_errors[:, 2] ** 2)))
    joint_z_rmse = float(np.sqrt(np.mean(joint_errors[:, 2] ** 2)))
    summary = {
        "cell_id": cell_id,
        "altitude_m": altitude,
        "horizontal_distance_m": distance,
        "trials": len(trials),
        "f2_gain_peb": float(cell["f2_gain_peb"]),
        "f2_gain_zeb": float(cell["f2_gain_zeb"]),
        "rf_rmse_3d_m": rf_rmse,
        "joint_rmse_3d_m": joint_rmse,
        "empirical_gain_3d": (rf_rmse - joint_rmse) / rf_rmse,
        "rf_z_rmse_m": rf_z_rmse,
        "joint_z_rmse_m": joint_z_rmse,
        "empirical_gain_z": (rf_z_rmse - joint_z_rmse) / rf_z_rmse,
        "rf_mean_nees": float(np.mean([row.rf_nees for row in trials])),
        "joint_mean_nees": float(np.mean([row.joint_nees for row in trials])),
        "rf_coverage_95": float(np.mean([row.rf_covered_95 for row in trials])),
        "joint_coverage_95": float(np.mean([row.joint_covered_95 for row in trials])),
    }
    return summary, trials


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("F3 requires mode: full_localization")
    phase = config["f3"]
    if int(phase["monte_carlo_trials"]) != 500:
        raise RuntimeError("F3 requires exactly 500 trials per geometry")
    output_root = Path(config["outputs"]["root"])
    protocol = json.loads((output_root / "f2_representative_geometries.json").read_text())
    if protocol["status"] != "FROZEN_BEFORE_F3" or len(protocol["cells"]) != 8:
        raise RuntimeError("F3 geometry protocol was not frozen by F2")
    calibration = json.loads((output_root / "f1_visual_calibration.json").read_text())
    model_inputs = {
        "rf_parameters": protocol["rf_parameters"],
        "camera_covariance_uv_px2": calibration["selected_model"]["covariance_uv_px2"],
    }
    payloads = [
        (
            config["f2"],
            phase,
            cell,
            model_inputs,
            int(config["seed"]) + 3000 + int(cell["cell_id"]),
        )
        for cell in protocol["cells"]
    ]
    with ProcessPoolExecutor(max_workers=int(phase["parallel_workers"])) as executor:
        outputs = list(executor.map(_run_geometry, payloads))
    summaries = [item[0] for item in outputs]
    trials = [row for _, geometry_trials in outputs for row in geometry_trials]
    summaries.sort(key=lambda row: int(row["cell_id"]))
    trials.sort(key=lambda row: (row.cell_id, row.trial))
    _write_rows(output_root / "f3_geometry_summary.csv", summaries)
    _write_rows(output_root / "f3_paired_trials.csv", [asdict(row) for row in trials])

    peb_correlation = float(
        spearmanr(
            [row["f2_gain_peb"] for row in summaries],
            [row["empirical_gain_3d"] for row in summaries],
        ).statistic
    )
    zeb_correlation = float(
        spearmanr(
            [row["f2_gain_zeb"] for row in summaries],
            [row["empirical_gain_z"] for row in summaries],
        ).statistic
    )
    improved_3d = sum(row["empirical_gain_3d"] > 0.0 for row in summaries)
    improved_z = sum(row["empirical_gain_z"] > 0.0 for row in summaries)
    gate = phase["gate"]
    nees_min, nees_max = map(float, gate["nees_mean_range"])
    coverage_min, coverage_max = map(float, gate["noncatastrophic_coverage_range"])
    passed = (
        improved_3d >= int(gate["minimum_improved_geometries"])
        and improved_z >= int(gate["minimum_improved_geometries"])
        and peb_correlation >= float(gate["minimum_gain_spearman"])
        and zeb_correlation >= float(gate["minimum_gain_spearman"])
        and all(
            nees_min <= row["rf_mean_nees"] <= nees_max
            and nees_min <= row["joint_mean_nees"] <= nees_max
            and coverage_min <= row["rf_coverage_95"] <= coverage_max
            and coverage_min <= row["joint_coverage_95"] <= coverage_max
            for row in summaries
        )
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "geometry_count": len(summaries),
        "trials_per_geometry": int(phase["monte_carlo_trials"]),
        "paired_trial_count": len(trials),
        "improved_3d_geometries": improved_3d,
        "improved_z_geometries": improved_z,
        "peb_empirical_gain_spearman": peb_correlation,
        "zeb_empirical_gain_spearman": zeb_correlation,
        "rf_mean_nees_range": [
            min(row["rf_mean_nees"] for row in summaries),
            max(row["rf_mean_nees"] for row in summaries),
        ],
        "joint_mean_nees_range": [
            min(row["joint_mean_nees"] for row in summaries),
            max(row["joint_mean_nees"] for row in summaries),
        ],
        "rf_coverage_range": [
            min(row["rf_coverage_95"] for row in summaries),
            max(row["rf_coverage_95"] for row in summaries),
        ],
        "joint_coverage_range": [
            min(row["joint_coverage_95"] for row in summaries),
            max(row["joint_coverage_95"] for row in summaries),
        ],
    }
    (output_root / "f3_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("F3 GATE: FAIL")
    print("F3 GATE: PASS")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)

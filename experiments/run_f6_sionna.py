"""Run strict-firewall Sionna LoS sanity and mild-multipath stress tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from experiments.run_f5_analytic_waveform import _estimate_only_waveforms, _waveform_components
from uav_ican_3d.geometry import RigidTransform, perturb_world_body
from uav_ican_3d.localization import (
    SionnaPathChannel,
    estimate_position_map,
    synthesize_sionna_waveform,
    trace_sionna_channel,
    wrap_angle,
)
from uav_ican_3d.simulation import complementarity_scene
from uav_ican_3d.types import RFObservation, VisualObservation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truth(cell: dict, target_height_m: float) -> np.ndarray:
    distance = float(cell["horizontal_distance_m"])
    vertical_down = float(cell["altitude_m"]) - target_height_m
    return np.array(
        [np.hypot(distance, vertical_down), 0.0, np.arctan2(vertical_down, distance)]
    )


def _trace(config: dict, cell: dict, reflections: bool, seed: int) -> SionnaPathChannel:
    phase = config["f6"]
    return trace_sionna_channel(
        Path(phase["scene_asset"]),
        float(config["system"]["carrier_hz"]),
        [float(cell["horizontal_distance_m"]), 0.0, float(phase["target_height_m"])],
        [0.0, 0.0, float(cell["altitude_m"])],
        max_depth=(
            int(phase["mild_multipath_reflection_order"])
            if reflections
            else int(phase["los_max_depth"])
        ),
        reflections=reflections,
        samples_per_source=int(phase["samples_per_source"]),
        max_paths_per_source=int(phase["max_paths_per_source"]),
        synthetic_array=bool(phase["synthetic_array"]),
        seed=seed,
    )


def _benchmark(
    config: dict, cells: list[dict], reflections: bool, seed_offset: int
) -> tuple[list[dict], list[dict], list[dict]]:
    phase5 = config["f5"]
    phase6 = config["f6"]
    waveform, estimator = _waveform_components(config)
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    channel_rows: list[dict] = []
    for cell in cells:
        trace_seed = int(config["seed"]) + seed_offset + int(cell["cell_id"])
        channel = _trace(config, cell, reflections, trace_seed)
        truth = _truth(cell, float(phase6["target_height_m"]))
        channel_rows.append(
            {
                "cell_id": int(cell["cell_id"]),
                "reflections": int(reflections),
                "path_count": int(channel.delays_s.size),
                "minimum_delay_s": float(np.min(channel.delays_s)),
                "maximum_delay_s": float(np.max(channel.delays_s)),
                "delay_spread_s": float(np.max(channel.delays_s) - np.min(channel.delays_s)),
            }
        )
        for snr_index, snr_db in enumerate(map(float, phase5["snr_db"])):
            seed = trace_seed * 100 + snr_index
            rng = np.random.default_rng(seed)
            errors: list[np.ndarray] = []
            count = int(phase5["waveform_realizations"])
            batch_size = int(phase5["estimator_batch_size"])
            for start in range(0, count, batch_size):
                current = min(batch_size, count - start)
                received = np.stack(
                    [
                        synthesize_sionna_waveform(channel, snr_db, waveform, rng)
                        for _ in range(current)
                    ]
                )
                estimates = _estimate_only_waveforms(estimator, received)
                batch_errors = estimates - truth
                batch_errors[:, 1] = wrap_angle(batch_errors[:, 1])
                errors.extend(batch_errors)
            error_array = np.asarray(errors)
            bias = np.mean(error_array, axis=0)
            covariance = np.cov(error_array, rowvar=False, ddof=1)
            if not np.all(np.isfinite(covariance)):
                raise AssertionError("Sionna waveform error covariance is non-finite")
            for realization, error in enumerate(error_array):
                detail_rows.append(
                    {
                        "cell_id": int(cell["cell_id"]),
                        "altitude_m": float(cell["altitude_m"]),
                        "horizontal_distance_m": float(cell["horizontal_distance_m"]),
                        "snr_db": snr_db,
                        "realization": realization,
                        "seed": seed,
                        "range_error_m": float(error[0]),
                        "azimuth_error_deg": float(np.rad2deg(error[1])),
                        "elevation_error_deg": float(np.rad2deg(error[2])),
                    }
                )
            summary_rows.append(
                {
                    "cell_id": int(cell["cell_id"]),
                    "snr_db": snr_db,
                    "path_count": int(channel.delays_s.size),
                    "range_rmse_m": float(np.sqrt(np.mean(error_array[:, 0] ** 2))),
                    "azimuth_rmse_deg": float(
                        np.rad2deg(np.sqrt(np.mean(error_array[:, 1] ** 2)))
                    ),
                    "elevation_rmse_deg": float(
                        np.rad2deg(np.sqrt(np.mean(error_array[:, 2] ** 2)))
                    ),
                    "range_bias_m": float(bias[0]),
                    "azimuth_bias_deg": float(np.rad2deg(bias[1])),
                    "elevation_bias_deg": float(np.rad2deg(bias[2])),
                    **{
                        f"cov_{row}{column}": float(covariance[row, column])
                        for row in range(3)
                        for column in range(3)
                    },
                }
            )
            print(
                f"{'MP' if reflections else 'LOS'} cell={cell['cell_id']} "
                f"paths={channel.delays_s.size} snr={snr_db:g} "
                f"range={summary_rows[-1]['range_rmse_m']:.3f}m "
                f"az={summary_rows[-1]['azimuth_rmse_deg']:.3f}deg "
                f"el={summary_rows[-1]['elevation_rmse_deg']:.3f}deg",
                flush=True,
            )
    return detail_rows, summary_rows, channel_rows


def _aggregate(detail_rows: list[dict], snr_levels: list[float]) -> list[dict]:
    output: list[dict] = []
    for snr_db in snr_levels:
        rows = [row for row in detail_rows if np.isclose(row["snr_db"], snr_db)]
        output.append(
            {
                "snr_db": snr_db,
                "waveforms": len(rows),
                "range_rmse_m": float(
                    np.sqrt(np.mean([row["range_error_m"] ** 2 for row in rows]))
                ),
                "azimuth_rmse_deg": float(
                    np.sqrt(np.mean([row["azimuth_error_deg"] ** 2 for row in rows]))
                ),
                "elevation_rmse_deg": float(
                    np.sqrt(np.mean([row["elevation_error_deg"] ** 2 for row in rows]))
                ),
                "range_bias_m": float(np.mean([row["range_error_m"] for row in rows])),
                "azimuth_bias_deg": float(np.mean([row["azimuth_error_deg"] for row in rows])),
                "elevation_bias_deg": float(np.mean([row["elevation_error_deg"] for row in rows])),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _localization_stress(
    config: dict, cells: list[dict], multipath_errors: list[dict]
) -> tuple[list[dict], dict[str, float]]:
    output_root = Path(config["outputs"]["root"])
    phase5 = config["f5"]
    phase6 = config["f6"]
    f5 = json.loads((output_root / "f5_summary.json").read_text())
    rf_bias = np.asarray(f5["nominal_bias_range_azimuth_elevation"], dtype=float)
    rf_covariance = np.asarray(
        f5["nominal_covariance_range_azimuth_elevation"], dtype=float
    )
    calibration = json.loads((output_root / "f1_visual_calibration.json").read_text())
    camera_covariance = np.asarray(
        calibration["selected_model"]["covariance_uv_px2"], dtype=float
    )
    protocol = json.loads((output_root / "f2_representative_geometries.json").read_text())
    parameters = protocol["rf_parameters"]
    pose_covariance = np.diag(
        [float(parameters["pose_position_std_m"]) ** 2] * 3
        + [np.deg2rad(float(parameters["attitude_std_deg"])) ** 2] * 3
    )
    nominal_snr = float(phase5["nominal_snr_db"])
    rows: list[dict] = []
    rng = np.random.default_rng(int(config["seed"]) + 66000)
    for cell in cells:
        target0, actual_world_body, body_array, body_camera, camera = complementarity_scene(
            config["f2"], float(cell["altitude_m"]), float(cell["horizontal_distance_m"])
        )
        target = target0.copy()
        target[2] = float(phase6["target_height_m"])
        truth = _truth(cell, float(phase6["target_height_m"]))
        cell_errors = [
            row
            for row in multipath_errors
            if int(row["cell_id"]) == int(cell["cell_id"])
            and np.isclose(row["snr_db"], nominal_snr)
        ]
        for error_row in cell_errors:
            error = np.array(
                [
                    error_row["range_error_m"],
                    np.deg2rad(error_row["azimuth_error_deg"]),
                    np.deg2rad(error_row["elevation_error_deg"]),
                ]
            )
            measurement = truth + error - rf_bias
            measurement[1:] = wrap_angle(measurement[1:])
            nominal_world_body = perturb_world_body(
                actual_world_body, rng.multivariate_normal(np.zeros(6), pose_covariance)
            )
            timestamp = float(error_row["realization"])
            observation = RFObservation(*measurement, rf_covariance, timestamp)
            visual = VisualObservation(
                rng.multivariate_normal(
                    camera.project_world(target, actual_world_body.compose(body_camera)),
                    camera_covariance,
                ),
                camera_covariance,
                timestamp,
            )
            rf_estimate = estimate_position_map(
                observation, nominal_world_body, body_array, pose_covariance
            )
            joint_estimate = estimate_position_map(
                observation,
                nominal_world_body,
                body_array,
                pose_covariance,
                camera,
                body_camera,
                visual,
                initial_position_world_m=rf_estimate.position_world_m,
            )
            rf_error = rf_estimate.position_world_m - target
            joint_error = joint_estimate.position_world_m - target
            rows.append(
                {
                    "cell_id": int(cell["cell_id"]),
                    "realization": int(error_row["realization"]),
                    "rf_error_3d_m": float(np.linalg.norm(rf_error)),
                    "joint_error_3d_m": float(np.linalg.norm(joint_error)),
                    "rf_error_z_m": float(abs(rf_error[2])),
                    "joint_error_z_m": float(abs(joint_error[2])),
                }
            )
    rf_rmse = float(np.sqrt(np.mean([row["rf_error_3d_m"] ** 2 for row in rows])))
    joint_rmse = float(np.sqrt(np.mean([row["joint_error_3d_m"] ** 2 for row in rows])))
    rf_z_rmse = float(np.sqrt(np.mean([row["rf_error_z_m"] ** 2 for row in rows])))
    joint_z_rmse = float(np.sqrt(np.mean([row["joint_error_z_m"] ** 2 for row in rows])))
    return rows, {
        "paired_trials": len(rows),
        "rf_rmse_3d_m": rf_rmse,
        "joint_rmse_3d_m": joint_rmse,
        "gain_3d": (rf_rmse - joint_rmse) / rf_rmse,
        "rf_z_rmse_m": rf_z_rmse,
        "joint_z_rmse_m": joint_z_rmse,
        "gain_z": (rf_z_rmse - joint_z_rmse) / rf_z_rmse,
    }


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("F6 requires mode: full_localization")
    phase5 = config["f5"]
    phase6 = config["f6"]
    if not phase6["path_truth_firewall"]:
        raise RuntimeError("Sionna path-truth firewall must remain enabled")
    output_root = Path(config["outputs"]["root"])
    protocol = json.loads((output_root / "f2_representative_geometries.json").read_text())
    cells = protocol["cells"]
    snr_levels = list(map(float, phase5["snr_db"]))

    los_detail, los_summary, los_channels = _benchmark(config, cells, False, 61000)
    los_curve = _aggregate(los_detail, snr_levels)
    _write_csv(output_root / "f6a_sionna_los_errors.csv", los_detail)
    _write_csv(output_root / "f6a_sionna_los_summary.csv", los_summary)
    _write_csv(output_root / "f6a_sionna_los_snr_curve.csv", los_curve)
    _write_csv(output_root / "f6a_sionna_los_channels.csv", los_channels)
    with (output_root / "f5_waveform_snr_curve.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        analytic_curve = {
            float(row["snr_db"]): row for row in csv.DictReader(source)
        }
    correlations = {
        metric: float(
            spearmanr(snr_levels, [row[metric] for row in los_curve]).statistic
        )
        for metric in ("range_rmse_m", "azimuth_rmse_deg", "elevation_rmse_deg")
    }
    nominal_snr = float(phase5["nominal_snr_db"])
    los_nominal = next(row for row in los_curve if np.isclose(row["snr_db"], nominal_snr))
    analytic_nominal = analytic_curve[nominal_snr]
    ratios = {
        metric: float(los_nominal[metric]) / float(analytic_nominal[metric])
        for metric in ("range_rmse_m", "azimuth_rmse_deg", "elevation_rmse_deg")
    }
    ratio_min, ratio_max = map(float, phase6["gate_a"]["nominal_rmse_ratio_range"])
    correlation_threshold = float(
        phase6["gate_a"]["minimum_error_snr_spearman_magnitude"]
    )
    pass_a = (
        all(value <= -correlation_threshold for value in correlations.values())
        and all(ratio_min <= value <= ratio_max for value in ratios.values())
        and all(row["path_count"] == 1 for row in los_channels)
    )
    gate_a = {
        "status": "PASS" if pass_a else "FAIL",
        "sionna_version": __import__("sionna.rt", fromlist=["__version__"]).__version__,
        "paths_per_geometry": [row["path_count"] for row in los_channels],
        "snr_error_spearman": correlations,
        "nominal_sionna_to_analytic_rmse_ratio": ratios,
        "path_truth_passed_to_online_estimator": False,
        "online_estimator_inputs": ["complex_received_waveform", "known_reference"],
    }
    (output_root / "f6a_gate.json").write_text(
        json.dumps(gate_a, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(gate_a, indent=2, sort_keys=True), flush=True)
    if not pass_a:
        raise AssertionError("F6-A GATE: FAIL")
    print("F6-A GATE: PASS; enabling reflections", flush=True)

    multipath_detail, multipath_summary, multipath_channels = _benchmark(
        config, cells, True, 62000
    )
    multipath_curve = _aggregate(multipath_detail, snr_levels)
    _write_csv(output_root / "f6b_sionna_multipath_errors.csv", multipath_detail)
    _write_csv(output_root / "f6b_sionna_multipath_summary.csv", multipath_summary)
    _write_csv(output_root / "f6b_sionna_multipath_snr_curve.csv", multipath_curve)
    _write_csv(output_root / "f6b_sionna_multipath_channels.csv", multipath_channels)
    localization_rows, localization = _localization_stress(config, cells, multipath_detail)
    _write_csv(output_root / "f6b_localization_trials.csv", localization_rows)
    reflected_paths = sum(max(0, row["path_count"] - 1) for row in multipath_channels)
    pass_b = (
        reflected_paths > 0
        and localization["gain_3d"] > 0.0
        and localization["gain_z"] > 0.0
    )
    result = {
        "status": "PASS" if pass_b else "FAIL",
        "gate_a": gate_a,
        "multipath_paths_per_geometry": [row["path_count"] for row in multipath_channels],
        "total_reflected_paths": reflected_paths,
        "multipath_nominal_waveform": next(
            row for row in multipath_curve if np.isclose(row["snr_db"], nominal_snr)
        ),
        "localization": localization,
        "new_nlos_mitigation": False,
        "path_truth_passed_to_online_estimator": False,
        "scene_sha256": _sha256(Path(phase6["scene_asset"])),
    }
    (output_root / "f6_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not pass_b:
        raise AssertionError("F6-B GATE: FAIL")
    print("F6-B GATE: PASS")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)

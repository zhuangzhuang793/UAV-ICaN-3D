"""Validate communication-waveform delay/AoA and one real-vision fusion subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body
from uav_ican_3d.localization import (
    SRSWaveformConfig,
    WaveformRFEstimator,
    estimate_position_map,
    predict_rf_observation,
    simulate_los_srs,
    wrap_angle,
)
from uav_ican_3d.types import RFObservation, VisualObservation
from uav_ican_3d.vision import (
    associate_candidates,
    box_iou,
    calibrate_bbox_center_to_ue_pixel,
    calibrated_box_centers,
    project_joint_rf_belief_to_image,
)


def _grid(specification: list[float], radians: bool = False) -> np.ndarray:
    start, stop, step = (float(value) for value in specification)
    values = np.arange(start, stop + step * 0.5, step)
    return np.deg2rad(values) if radians else values


def _waveform_components(phase: dict) -> tuple[SRSWaveformConfig, WaveformRFEstimator]:
    config = SRSWaveformConfig(
        carrier_hz=float(phase["carrier_hz"]),
        subcarrier_count=int(phase["subcarrier_count"]),
        subcarrier_spacing_hz=float(phase["subcarrier_spacing_hz"]),
        upa_rows=int(phase["upa_shape"][0]),
        upa_columns=int(phase["upa_shape"][1]),
    )
    estimator = WaveformRFEstimator(
        config,
        _grid(phase["range_grid_m"]),
        _grid(phase["azimuth_grid_deg"], radians=True),
        _grid(phase["elevation_grid_deg"], radians=True),
    )
    return config, estimator


def _waveform_monte_carlo(
    phase: dict,
    config: SRSWaveformConfig,
    estimator: WaveformRFEstimator,
    seed: int,
) -> tuple[list[dict[str, float | int]], dict[float, np.ndarray], dict[float, float]]:
    truth_config = phase["calibration_truth"]
    truth = np.array(
        [
            float(truth_config["range_m"]),
            np.deg2rad(float(truth_config["azimuth_deg"])),
            np.deg2rad(float(truth_config["elevation_deg"])),
        ]
    )
    rows = []
    residuals_by_snr = {}
    metric_by_snr = {}
    for snr_index, snr_db in enumerate(float(value) for value in phase["snr_db"]):
        residuals = []
        for trial in range(int(phase["monte_carlo_trials"])):
            rng = np.random.default_rng(seed + 1000 * snr_index + trial)
            received = simulate_los_srs(*truth, snr_db, config, rng)
            estimate = estimator.estimate(received).vector
            residual = estimate - truth
            residual[1:] = wrap_angle(residual[1:])
            residuals.append(residual)
            rows.append(
                {
                    "snr_db": snr_db,
                    "trial": trial,
                    "range_error_m": float(residual[0]),
                    "azimuth_error_deg": float(np.rad2deg(residual[1])),
                    "elevation_error_deg": float(np.rad2deg(residual[2])),
                }
            )
        residual_matrix = np.asarray(residuals)
        residuals_by_snr[snr_db] = residual_matrix
        equivalent_position_errors = np.column_stack(
            (
                residual_matrix[:, 0],
                truth[0] * residual_matrix[:, 1],
                truth[0] * residual_matrix[:, 2],
            )
        )
        metric_by_snr[snr_db] = float(
            np.sqrt(np.mean(np.sum(equivalent_position_errors**2, axis=1)))
        )
    return rows, residuals_by_snr, metric_by_snr


def _transform(record: dict, rotation_key: str, translation_key: str) -> RigidTransform:
    return RigidTransform(np.asarray(record[rotation_key]), np.asarray(record[translation_key]))


def _camera(record: dict) -> PinholeCamera:
    values = record["camera_intrinsics"]
    return PinholeCamera(
        values["fx_px"], values["fy_px"], values["cx_px"], values["cy_px"]
    )


def _load_json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _empirical_covariance(
    residuals: np.ndarray, phase: dict
) -> tuple[np.ndarray, np.ndarray]:
    bias = np.mean(residuals, axis=0)
    covariance = np.cov(residuals, rowvar=False)
    floor = phase["covariance_floor"]
    covariance += np.diag(
        [
            float(floor["range_std_m"]) ** 2,
            np.deg2rad(float(floor["angle_std_deg"])) ** 2,
            np.deg2rad(float(floor["angle_std_deg"])) ** 2,
        ]
    )
    return bias, covariance


def _pipeline_subset(
    config: dict,
    waveform_config: SRSWaveformConfig,
    estimator: WaveformRFEstimator,
    rf_bias: np.ndarray,
    rf_covariance: np.ndarray,
) -> tuple[list[dict[str, float | int]], float, float, float, float]:
    phase6 = config["phase6"]
    phase9 = config["phase9"]
    calibration_count = int(phase6["calibration_frames"])
    evaluation_count = int(phase9["pipeline_evaluation_frames"])
    records = _load_json_lines(Path(phase6["synchronized_manifest"]))
    detections = _load_json_lines(Path(phase6["detector_cache"]))
    calibration = calibrate_bbox_center_to_ue_pixel(
        [np.asarray(item["boxes_xyxy"]).reshape(-1, 4) for item in detections[:calibration_count]],
        [record["served_gt_box_xyxy"] for record in records[:calibration_count]],
        [record["true_ue_pixel_uv"] for record in records[:calibration_count]],
        float(phase6["calibration_minimum_iou"]),
        float(phase6["visual_covariance_floor_px2"]),
    )
    pose_covariance = np.diag(
        [float(phase6["pose_position_std_m"]) ** 2] * 3
        + [np.deg2rad(float(phase6["pose_attitude_std_deg"])) ** 2] * 3
    )
    rng = np.random.default_rng(int(config["seed"]) + 900)
    rows = []
    selected_pixel_history: list[np.ndarray] = []
    start = calibration_count
    stop = start + evaluation_count
    for frame_index, (record, detection) in enumerate(
        zip(records[start:stop], detections[start:stop], strict=True), start=start
    ):
        true_world_body = _transform(record["transform_world_body"], "R_WB", "t_W_B")
        body_camera = _transform(record["transform_body_camera"], "R_BC", "t_B_C")
        antenna = np.asarray(record["antenna_phase_center_world_m"])
        truth = predict_rf_observation(antenna, true_world_body)
        received = simulate_los_srs(
            *truth,
            float(phase9["nominal_snr_db"]),
            waveform_config,
            rng,
        )
        measurement = estimator.estimate(received).vector - rf_bias
        measurement[1:] = wrap_angle(measurement[1:])
        observation = RFObservation(
            measurement[0],
            measurement[1],
            measurement[2],
            rf_covariance,
            record["timestamp_s"],
        )
        nominal_world_body = perturb_world_body(
            true_world_body, rng.multivariate_normal(np.zeros(6), pose_covariance)
        )
        rf_estimate = estimate_position_map(
            observation, nominal_world_body, RigidTransform.identity(), pose_covariance
        )
        camera = _camera(record)
        belief = project_joint_rf_belief_to_image(
            rf_estimate.position_world_m,
            rf_estimate.joint_covariance,
            nominal_world_body,
            body_camera,
            camera,
            pose_delta_mean=rf_estimate.pose_delta,
        )
        boxes = np.asarray(detection["boxes_xyxy"]).reshape(-1, 4)
        candidates = calibrated_box_centers(boxes, calibration)
        association = associate_candidates(
            candidates, belief, float(phase6["association_confidence"])
        )
        selected_index = None
        gated = np.flatnonzero(association.squared_distances <= association.threshold)
        if gated.size:
            if len(selected_pixel_history) >= 2:
                expected_pixel = (
                    2.0 * selected_pixel_history[-1] - selected_pixel_history[-2]
                )
            elif selected_pixel_history:
                expected_pixel = selected_pixel_history[-1]
            else:
                expected_pixel = belief.mean_uv
            temporal_cost = np.sum((candidates[gated] - expected_pixel) ** 2, axis=1) / (
                float(phase9["temporal_association_std_px"]) ** 2
            )
            confidences = np.asarray(detection["confidences"], dtype=float)[gated]
            confidence_cost = -float(phase9["detector_confidence_weight"]) * np.log(
                np.maximum(confidences, 1e-6)
            )
            scores = association.squared_distances[gated] + temporal_cost + confidence_cost
            selected_index = int(gated[int(np.argmin(scores))])
        fused_estimate = rf_estimate
        updated = 0
        association_correct = 0
        if selected_index is not None:
            selected_pixel_history.append(candidates[selected_index].copy())
            association_correct = int(
                box_iou(record["served_gt_box_xyxy"], boxes[[selected_index]])[0]
                >= float(phase6["calibration_minimum_iou"])
            )
            visual = VisualObservation(
                candidates[selected_index],
                calibration.covariance_uv,
                record["timestamp_s"],
            )
            fused_estimate = estimate_position_map(
                observation,
                nominal_world_body,
                RigidTransform.identity(),
                pose_covariance,
                camera,
                body_camera,
                visual,
                rf_estimate.position_world_m,
            )
            updated = 1
        rf_error = rf_estimate.position_world_m - antenna
        fused_error = fused_estimate.position_world_m - antenna
        rows.append(
            {
                "frame": frame_index,
                "visual_update": updated,
                "association_correct": association_correct,
                "rf_error_3d_m": float(np.linalg.norm(rf_error)),
                "fused_error_3d_m": float(np.linalg.norm(fused_error)),
                "rf_error_z_m": float(abs(rf_error[2])),
                "fused_error_z_m": float(abs(fused_error[2])),
            }
        )
    rf_rmse = float(np.sqrt(np.mean([row["rf_error_3d_m"] ** 2 for row in rows])))
    fused_rmse = float(np.sqrt(np.mean([row["fused_error_3d_m"] ** 2 for row in rows])))
    rf_z_rmse = float(np.sqrt(np.mean([row["rf_error_z_m"] ** 2 for row in rows])))
    fused_z_rmse = float(np.sqrt(np.mean([row["fused_error_z_m"] ** 2 for row in rows])))
    return rows, rf_rmse, fused_rmse, rf_z_rmse, fused_z_rmse


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    path.write_text(
        ",".join(columns)
        + "\n"
        + "\n".join(",".join(str(row[column]) for column in columns) for row in rows)
        + "\n"
    )


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 9 gate refuses to run unless mode is 'quick'")
    phase = config["phase9"]
    waveform_config, estimator = _waveform_components(phase)
    rows, residuals, metrics = _waveform_monte_carlo(
        phase, waveform_config, estimator, int(config["seed"]) + 9
    )
    nominal_snr = float(phase["nominal_snr_db"])
    rf_bias, rf_covariance = _empirical_covariance(residuals[nominal_snr], phase)
    pipeline_rows, rf_rmse, fused_rmse, rf_z_rmse, fused_z_rmse = _pipeline_subset(
        config, waveform_config, estimator, rf_bias, rf_covariance
    )
    _write_csv(Path(phase["results_csv"]), rows)
    _write_csv(Path(phase["pipeline_results_csv"]), pipeline_rows)
    Path(phase["calibration_json"]).write_text(
        json.dumps(
            {
                "snr_db": nominal_snr,
                "bias_range_azimuth_elevation": rf_bias.tolist(),
                "covariance_range_azimuth_elevation": rf_covariance.tolist(),
                "units": ["m", "rad", "rad"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    snr_values = [float(value) for value in phase["snr_db"]]
    error_values = [metrics[value] for value in snr_values]
    high_low_improvement = (error_values[0] - error_values[-1]) / error_values[0]
    gain_3d = (rf_rmse - fused_rmse) / rf_rmse
    gain_z = (rf_z_rmse - fused_z_rmse) / rf_z_rmse
    gate = phase["gate"]
    passed = (
        all(first > second for first, second in zip(error_values, error_values[1:]))
        and high_low_improvement >= float(gate["minimum_high_low_error_improvement"])
        and np.linalg.eigvalsh(rf_covariance)[0] > 0.0
        and gain_3d >= float(gate["minimum_pipeline_3d_rmse_gain"])
        and gain_z >= float(gate["minimum_pipeline_z_rmse_gain"])
    )
    status = "PASS" if passed else "FAIL"
    Path("docs/PHASE9_DECISION.md").write_text(
        f"""# Phase 9 decision

Status: **{status}**

A 3.5 GHz single-antenna uplink transmitted a known QPSK SRS-like reference to the same 4x4
half-wavelength UPA. A grid maximum-likelihood delay estimator used received subcarrier phases;
a separate 2-D spatial maximum-likelihood stage estimated azimuth and elevation. Ground-truth
path values enter only the channel simulator and evaluation residuals, never the estimator API.

- SNR / equivalent position RMSE: `{dict(zip(snr_values, error_values, strict=True))}` m
- High-to-low SNR error improvement: `{high_low_improvement:.3f}`
- Nominal empirical RF bias `[range, az, el]`: `{rf_bias.tolist()}`
- Nominal empirical RF covariance: `{rf_covariance.tolist()}`
- Waveform RF-only / RF+real-Vision 3-D RMSE: `{rf_rmse:.3f} / {fused_rmse:.3f}` m
- Waveform RF-only / RF+real-Vision Z-RMSE: `{rf_z_rmse:.3f} / {fused_z_rmse:.3f}` m
- Pipeline relative 3-D / Z gains: `{gain_3d:.3f} / {gain_z:.3f}`

The empirical waveform-estimator covariance replaces the hand-set RF covariance in the pipeline
subset. The visual inputs are cached real YOLO11n-OBB detections from Phase 6.
""",
        encoding="utf-8",
    )
    print(f"snr_error_metric_m={dict(zip(snr_values, error_values, strict=True))}")
    print(f"high_low_error_improvement={high_low_improvement:.3f}")
    print(f"empirical_rf_bias={rf_bias.tolist()}")
    print(f"empirical_rf_covariance={rf_covariance.tolist()}")
    print(f"waveform_rf_rmse_3d_m={rf_rmse:.3f} fused_rmse_3d_m={fused_rmse:.3f}")
    print(f"waveform_rf_z_rmse_m={rf_z_rmse:.3f} fused_z_rmse_m={fused_z_rmse:.3f}")
    print(f"pipeline_gain_3d={gain_3d:.3f} pipeline_gain_z={gain_z:.3f}")
    print(f"PHASE 9 QUICK GATE: {status}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)

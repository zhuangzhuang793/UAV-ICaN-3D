"""Run the post-F7 RF realism ladder without modifying frozen F7 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from experiments.run_f5_analytic_waveform import _estimate_only_waveforms, _waveform_components
from experiments.run_f7_final_localization import (
    _association_correct,
    _camera,
    _candidate_pixels,
    _load_detections,
    _optimizer_options,
    _transform,
)
from uav_ican_3d.geometry import RigidTransform, perturb_world_body
from uav_ican_3d.localization import estimate_position_map, predict_rf_observation, wrap_angle
from uav_ican_3d.localization.rf_realism import (
    LinkBudget,
    correlated_shadowing_db,
    draw_sequence_state,
    link_budget_for_reference_snr,
    simulate_impaired_los_srs,
)
from uav_ican_3d.types import RFObservation, VisualObservation
from uav_ican_3d.vision import (
    FrozenVisualCalibration,
    associate_candidates,
    project_joint_rf_belief_to_image,
)


PRIMARY_METHODS = ("rf_only_full_3d", "rf_vision_shared_pose")
CONDITION_FLAGS = {
    "link_budget": (False, False, False),
    "synchronization": (True, False, False),
    "array_calibration": (True, True, False),
    "interference": (True, True, True),
}


def _load_sequence(base: dict, sequence_id: str) -> list[dict]:
    path = Path(base["f0"]["output_root"]) / sequence_id / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected = int(base["f0"]["frames_per_sequence"])
    if len(records) != expected:
        raise RuntimeError(f"{sequence_id} has {len(records)} records, expected {expected}")
    return records


def _budget(settings: dict, waveform) -> LinkBudget:
    values = settings["link_budget"]
    return link_budget_for_reference_snr(
        float(values["reference_range_m"]),
        float(values["reference_snr_db"]),
        waveform,
        tx_gain_dbi=float(values["tx_gain_dbi"]),
        rx_gain_dbi=float(values["rx_gain_dbi"]),
        receiver_noise_figure_db=float(values["receiver_noise_figure_db"]),
        system_loss_db=float(values["system_loss_db"]),
    )


def _truths(records: list[dict]) -> tuple[list[RigidTransform], list[np.ndarray], np.ndarray]:
    bodies: list[RigidTransform] = []
    targets: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    for record in records:
        body = _transform(record["transform_world_body"], "R_WB", "t_W_B")
        target = np.asarray(
            record["evaluation_only"]["antenna_phase_centers_world_m"]["ServedVehicle"],
            dtype=float,
        )
        bodies.append(body)
        targets.append(target)
        observations.append(predict_rf_observation(target, body))
    return bodies, targets, np.asarray(observations)


def _simulate_sequence(
    base: dict,
    settings: dict,
    condition: str,
    sequence_id: str,
    seed: int,
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    waveform, estimator = _waveform_components(base)
    _, _, truth = _truths(records)
    sync, array, interference = CONDITION_FLAGS[condition]
    state_rng = np.random.default_rng(seed + 100_000 * int(sequence_id[-2:]))
    shadow_rng = np.random.default_rng(seed + 200_000 + 100_000 * int(sequence_id[-2:]))
    waveform_rng = np.random.default_rng(seed + 400_000 + 100_000 * int(sequence_id[-2:]))
    synchronization = settings["synchronization"]
    calibration = settings["array_calibration"]
    state = draw_sequence_state(
        state_rng,
        clock_bias_std_m=float(synchronization["clock_bias_std_m"]),
        clock_drift_std_mps=float(synchronization["clock_drift_std_mps"]),
        residual_cfo_std_hz=float(synchronization["residual_cfo_std_hz"]),
        array_amplitude_std_db=float(calibration["amplitude_std_db"]),
        array_phase_std_deg=float(calibration["phase_std_deg"]),
    )
    budget = _budget(settings, waveform)
    link = settings["link_budget"]
    shadowing = correlated_shadowing_db(
        len(records),
        float(link["shadowing_std_db"]),
        float(link["shadowing_frame_correlation"]),
        shadow_rng,
    )
    snr = np.asarray(budget.snr_db(truth[:, 0], waveform)) + shadowing
    snr = np.clip(snr, float(link["snr_floor_db"]), float(link["snr_ceiling_db"]))
    timestamps = np.asarray([float(record["timestamp_s"]) for record in records])
    elapsed = timestamps - timestamps[0]
    range_rate = np.gradient(truth[:, 0], timestamps)
    waveforms = np.stack(
        [
            simulate_impaired_los_srs(
                float(truth[index, 0]),
                float(truth[index, 1]),
                float(truth[index, 2]),
                float(snr[index]),
                waveform,
                waveform_rng,
                state=state,
                elapsed_s=float(elapsed[index]),
                range_rate_mps=float(range_rate[index]),
                enable_synchronization_errors=sync,
                enable_array_errors=array,
                phase_noise_std_deg=float(synchronization["phase_noise_std_deg"]),
                enable_interference=interference,
                interference_probability=float(
                    settings["interference"]["occupancy_probability"]
                ),
                interference_to_noise_db=float(
                    settings["interference"]["interference_to_noise_db"]
                ),
            )
            for index in range(len(records))
        ]
    )
    estimates = _estimate_only_waveforms(estimator, waveforms)
    return truth, estimates, snr


def _calibrate_condition(base: dict, settings: dict, condition: str) -> dict:
    errors: list[np.ndarray] = []
    snrs: list[np.ndarray] = []
    protocol = settings["protocol"]
    for sequence_id in protocol["calibration_sequences"]:
        records = _load_sequence(base, sequence_id)
        for seed in protocol["calibration_seeds"]:
            truth, estimates, snr = _simulate_sequence(
                base, settings, condition, sequence_id, int(seed), records
            )
            error = estimates - truth
            error[:, 1] = wrap_angle(error[:, 1])
            errors.append(error)
            snrs.append(snr)
    values = np.concatenate(errors)
    bias = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance += np.diag(np.asarray(protocol["covariance_floor"], dtype=float))
    return {
        "condition": condition,
        "samples": int(values.shape[0]),
        "mean_snr_db": float(np.mean(np.concatenate(snrs))),
        "bias_range_azimuth_elevation": bias.tolist(),
        "covariance_range_azimuth_elevation": covariance.tolist(),
        "range_rmse_m": float(np.sqrt(np.mean(values[:, 0] ** 2))),
        "azimuth_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(values[:, 1] ** 2)))),
        "elevation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(values[:, 2] ** 2)))),
    }


def _run_test_task(payload: tuple) -> list[dict]:
    (
        base,
        settings,
        condition,
        sequence_id,
        seed,
        records,
        detections,
        calibration_values,
        rf_calibration,
        subset_by_frame,
    ) = payload
    visual_calibration = FrozenVisualCalibration.from_dict(calibration_values)
    bodies, targets, _ = _truths(records)
    truth, estimates, snrs = _simulate_sequence(
        base, settings, condition, sequence_id, int(seed), records
    )
    rf_bias = np.asarray(rf_calibration["bias_range_azimuth_elevation"], dtype=float)
    rf_covariance = np.asarray(
        rf_calibration["covariance_range_azimuth_elevation"], dtype=float
    )
    body_array = RigidTransform.identity()
    angle_std = np.deg2rad(float(base["f2"]["nominal_attitude_std_deg"]))
    pose_covariance = np.diag(
        [float(base["f2"]["pose_position_std_m"]) ** 2] * 3 + [angle_std**2] * 3
    )
    pose_rng = np.random.default_rng(int(seed) + 50_000 + 1000 * int(sequence_id[-2:]))
    options = _optimizer_options(base)
    rows: list[dict] = []
    for index, (record, detection, actual_body, target) in enumerate(
        zip(records, detections, bodies, targets, strict=True)
    ):
        nominal_body = perturb_world_body(
            actual_body, pose_rng.multivariate_normal(np.zeros(6), pose_covariance)
        )
        measurement = estimates[index] - rf_bias
        measurement[1:] = wrap_angle(measurement[1:])
        observation = RFObservation(
            *measurement, rf_covariance, float(record["timestamp_s"])
        )
        body_camera = _transform(record["transform_body_camera"], "R_BC", "t_B_C")
        candidates = _candidate_pixels(record, detection, visual_calibration)
        rf_only = estimate_position_map(
            observation,
            nominal_body,
            body_array,
            pose_covariance,
            **options,
        )
        camera = _camera(record)
        try:
            belief = project_joint_rf_belief_to_image(
                rf_only.position_world_m,
                rf_only.joint_covariance,
                nominal_body,
                body_camera,
                camera,
                pose_delta_mean=rf_only.pose_delta,
            )
            association = associate_candidates(
                candidates, belief, float(base["f7"]["association_confidence"])
            )
            selected_index = association.selected_index
            rf_projection_valid = 1
        except ValueError as error:
            if "positive depth" not in str(error):
                raise
            selected_index = None
            rf_projection_valid = 0
        if selected_index is None:
            shared = rf_only
            fusion_optimizer_valid = 0
        else:
            visual = VisualObservation(
                candidates[selected_index],
                visual_calibration.covariance_uv,
                observation.timestamp_s,
            )
            try:
                shared = estimate_position_map(
                    observation,
                    nominal_body,
                    body_array,
                    pose_covariance,
                    camera,
                    body_camera,
                    visual,
                    initial_position_world_m=rf_only.position_world_m,
                    **options,
                )
                fusion_optimizer_valid = 1
            except ValueError as error:
                if "positive depth" not in str(error):
                    raise
                shared = rf_only
                fusion_optimizer_valid = 0
        estimates_by_method = {
            "rf_only_full_3d": rf_only,
            "rf_vision_shared_pose": shared,
        }
        correct = _association_correct(record, detection, selected_index)
        rf_error = estimates[index] - truth[index]
        rf_error[1:] = wrap_angle(rf_error[1:])
        for method in PRIMARY_METHODS:
            result = estimates_by_method[method]
            position_error = result.position_world_m - target
            rows.append(
                {
                    "condition": condition,
                    "sequence_id": sequence_id,
                    "frame_index": int(record["frame_index"]),
                    "waveform_seed": int(seed),
                    "method": method,
                    "subset": subset_by_frame[(sequence_id, int(record["frame_index"]))],
                    "range_m": float(truth[index, 0]),
                    "snr_db": float(snrs[index]),
                    "rf_range_error_m": float(rf_error[0]),
                    "rf_azimuth_error_deg": float(np.rad2deg(rf_error[1])),
                    "rf_elevation_error_deg": float(np.rad2deg(rf_error[2])),
                    "association_correct": correct,
                    "rf_projection_valid": rf_projection_valid,
                    "fusion_optimizer_valid": fusion_optimizer_valid,
                    "visual_update": int(selected_index is not None and fusion_optimizer_valid),
                    "error_x_m": float(position_error[0]),
                    "error_y_m": float(position_error[1]),
                    "error_z_m": float(position_error[2]),
                    "error_3d_m": float(np.linalg.norm(position_error)),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        # Checkpoints written before the optimizer-fallback audit completed only
        # contain successful optimizations, so the old visual-update flag is the
        # exact value of this newly explicit diagnostic.
        row.setdefault("fusion_optimizer_valid", row["visual_update"])
    return rows


def _summarize(rows: list[dict], condition: str, settings: dict) -> dict:
    output: dict[str, object] = {"condition": condition, "methods": {}}
    rf_rows = [row for row in rows if row["method"] == PRIMARY_METHODS[0]]
    output["rf_waveform"] = {
        "mean_snr_db": float(np.mean([float(row["snr_db"]) for row in rf_rows])),
        "range_rmse_m": float(
            np.sqrt(np.mean([float(row["rf_range_error_m"]) ** 2 for row in rf_rows]))
        ),
        "azimuth_rmse_deg": float(
            np.sqrt(np.mean([float(row["rf_azimuth_error_deg"]) ** 2 for row in rf_rows]))
        ),
        "elevation_rmse_deg": float(
            np.sqrt(np.mean([float(row["rf_elevation_error_deg"]) ** 2 for row in rf_rows]))
        ),
    }
    for method in PRIMARY_METHODS:
        selected = [row for row in rows if row["method"] == method]
        error = np.asarray([float(row["error_3d_m"]) for row in selected])
        z_error = np.asarray([float(row["error_z_m"]) for row in selected])
        output["methods"][method] = {
            "samples": len(selected),
            "rmse_3d_m": float(np.sqrt(np.mean(error**2))),
            "z_rmse_m": float(np.sqrt(np.mean(z_error**2))),
            "p95_error_3d_m": float(np.percentile(error, 95.0)),
            "association_accuracy": float(
                np.mean([int(row["association_correct"]) for row in selected])
            ),
            "rf_projection_valid_rate": float(
                np.mean([int(row["rf_projection_valid"]) for row in selected])
            ),
            "fusion_optimizer_valid_rate": float(
                np.mean([int(row["fusion_optimizer_valid"]) for row in selected])
            ),
            "visual_update_rate": float(
                np.mean([int(row["visual_update"]) for row in selected])
            ),
        }
    rf = output["methods"]["rf_only_full_3d"]
    joint = output["methods"]["rf_vision_shared_pose"]
    output["gain_3d"] = (rf["rmse_3d_m"] - joint["rmse_3d_m"]) / rf["rmse_3d_m"]
    output["gain_z"] = (rf["z_rmse_m"] - joint["z_rmse_m"]) / rf["z_rmse_m"]

    sequence_ids = settings["protocol"]["test_sequences"]
    by_sequence: dict[tuple[str, str], np.ndarray] = {}
    for sequence_id in sequence_ids:
        for method in PRIMARY_METHODS:
            by_sequence[(sequence_id, method)] = np.asarray(
                [
                    float(row["error_3d_m"]) ** 2
                    for row in rows
                    if row["sequence_id"] == sequence_id and row["method"] == method
                ]
            )
    rng = np.random.default_rng(int(settings["protocol"]["bootstrap_seed"]))
    improvements = []
    for _ in range(int(settings["protocol"]["bootstrap_resamples"])):
        sampled = rng.choice(sequence_ids, len(sequence_ids), replace=True)
        rf_squared = np.concatenate(
            [by_sequence[(sequence_id, PRIMARY_METHODS[0])] for sequence_id in sampled]
        )
        joint_squared = np.concatenate(
            [by_sequence[(sequence_id, PRIMARY_METHODS[1])] for sequence_id in sampled]
        )
        rf_rmse = np.sqrt(np.mean(rf_squared))
        joint_rmse = np.sqrt(np.mean(joint_squared))
        improvements.append((rf_rmse - joint_rmse) / rf_rmse)
    output["gain_3d_sequence_bootstrap_95ci"] = [
        float(np.percentile(improvements, 2.5)),
        float(np.percentile(improvements, 97.5)),
    ]
    output["distance_bins"] = _bin_summary(
        rows, settings["protocol"]["distance_bins_m"], "range_m"
    )
    output["snr_bins"] = _bin_summary(rows, settings["protocol"]["snr_bins_db"], "snr_db")
    return output


def _bin_summary(rows: list[dict], edges: list[float], field: str) -> list[dict]:
    result = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        values = [
            row for row in rows if lower <= float(row[field]) < upper
        ]
        if not values:
            continue
        item: dict[str, object] = {"lower": lower, "upper": upper, "samples": len(values) // 2}
        for method in PRIMARY_METHODS:
            selected = [float(row["error_3d_m"]) for row in values if row["method"] == method]
            item[method] = float(np.sqrt(np.mean(np.square(selected))))
        rf = item[PRIMARY_METHODS[0]]
        joint = item[PRIMARY_METHODS[1]]
        item["gain_3d"] = (rf - joint) / rf
        result.append(item)
    return result


def run(settings_path: Path, selected_conditions: list[str] | None = None) -> dict:
    settings = yaml.safe_load(settings_path.read_text())
    base = yaml.safe_load(Path(settings["base_config"]).read_text())
    output_root = Path(settings["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    conditions = selected_conditions or list(settings["protocol"]["conditions"])
    invalid = set(conditions) - set(CONDITION_FLAGS)
    if invalid:
        raise ValueError(f"unknown conditions: {sorted(invalid)}")

    waveform, _ = _waveform_components(base)
    budget = _budget(settings, waveform)
    calibrations: dict[str, dict] = {}
    for condition in conditions:
        path = output_root / f"calibration_{condition}.json"
        if path.exists():
            calibration = json.loads(path.read_text())
        else:
            calibration = _calibrate_condition(base, settings, condition)
            path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
        calibrations[condition] = calibration
        print(f"calibrated {condition}: {calibration['samples']} waveforms", flush=True)

    test_records = {
        sequence_id: _load_sequence(base, sequence_id)
        for sequence_id in settings["protocol"]["test_sequences"]
    }
    flat_records = [
        record
        for sequence_id in settings["protocol"]["test_sequences"]
        for record in test_records[sequence_id]
    ]
    detections = _load_detections(
        Path(base["outputs"]["root"]) / "f4_test_detections.jsonl", flat_records
    )
    test_detections = {
        sequence_id: [row for row in detections if row["sequence_id"] == sequence_id]
        for sequence_id in settings["protocol"]["test_sequences"]
    }
    visual_artifact = json.loads(
        (Path(base["outputs"]["root"]) / "f1_visual_calibration.json").read_text()
    )
    subset_rows = [
        json.loads(line)
        for line in (Path(base["outputs"]["root"]) / "f4_subset_freeze.jsonl").read_text().splitlines()
        if line
    ]
    subset_by_frame = {
        (row["sequence_id"], int(row["frame_index"])): row["subset"] for row in subset_rows
    }
    all_summaries = []
    for condition in conditions:
        task_paths = []
        payloads = []
        for sequence_id in settings["protocol"]["test_sequences"]:
            for seed in settings["protocol"]["test_seeds"]:
                task_path = checkpoint_root / f"{condition}_{sequence_id}_{int(seed)}.csv"
                task_paths.append(task_path)
                if not task_path.exists():
                    payloads.append(
                        (
                            base,
                            settings,
                            condition,
                            sequence_id,
                            int(seed),
                            test_records[sequence_id],
                            test_detections[sequence_id],
                            visual_artifact["selected_model"],
                            calibrations[condition],
                            subset_by_frame,
                        )
                    )
        if payloads:
            with ProcessPoolExecutor(
                max_workers=int(settings["protocol"]["parallel_workers"])
            ) as executor:
                futures = {
                    executor.submit(_run_test_task, payload): (
                        payload[3], payload[4]
                    )
                    for payload in payloads
                }
                for future in as_completed(futures):
                    sequence_id, seed = futures[future]
                    rows = future.result()
                    task_path = checkpoint_root / f"{condition}_{sequence_id}_{seed}.csv"
                    _write_csv(task_path, rows)
                    print(f"complete {condition} {sequence_id} seed={seed}", flush=True)
        rows = []
        for task_path in task_paths:
            rows.extend(_read_csv(task_path))
        rows.sort(
            key=lambda row: (
                row["sequence_id"],
                int(row["waveform_seed"]),
                int(row["frame_index"]),
                PRIMARY_METHODS.index(row["method"]),
            )
        )
        _write_csv(output_root / f"frames_{condition}.csv", rows)
        summary = _summarize(rows, condition, settings)
        (output_root / f"summary_{condition}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    frozen_f7 = json.loads(
        (Path(base["outputs"]["root"]) / "f7_summary.json").read_text()
    )
    frozen_f5 = json.loads(
        (Path(base["outputs"]["root"]) / "f5_summary.json").read_text()
    )
    with (Path(base["outputs"]["root"]) / "f5_waveform_snr_curve.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        nominal_waveform = next(
            row
            for row in csv.DictReader(source)
            if np.isclose(float(row["snr_db"]), float(frozen_f5["nominal_snr_db"]))
        )
    frozen_baseline = {
        "condition": "ideal_fixed_0db",
        "rf_waveform": {
            "mean_snr_db": float(frozen_f5["nominal_snr_db"]),
            "range_rmse_m": float(nominal_waveform["range_rmse_m"]),
            "azimuth_rmse_deg": float(nominal_waveform["azimuth_rmse_deg"]),
            "elevation_rmse_deg": float(nominal_waveform["elevation_rmse_deg"]),
        },
        "methods": {
            method: frozen_f7["methods"][method] for method in PRIMARY_METHODS
        },
        "gain_3d": float(frozen_f7["primary_3d_rmse_reduction"]),
        "gain_z": float(frozen_f7["primary_z_rmse_reduction"]),
        "gain_3d_sequence_bootstrap_95ci": [
            float(frozen_f7["bootstrap"]["improvement_95ci_lower"]),
            float(frozen_f7["bootstrap"]["improvement_95ci_upper"]),
        ],
    }
    comparison_rows = []
    for summary in [frozen_baseline, *all_summaries]:
        rf = summary["methods"][PRIMARY_METHODS[0]]
        joint = summary["methods"][PRIMARY_METHODS[1]]
        waveform_summary = summary["rf_waveform"]
        comparison_rows.append(
            {
                "condition": summary["condition"],
                "mean_snr_db": waveform_summary["mean_snr_db"],
                "rf_range_rmse_m": waveform_summary["range_rmse_m"],
                "rf_azimuth_rmse_deg": waveform_summary["azimuth_rmse_deg"],
                "rf_elevation_rmse_deg": waveform_summary["elevation_rmse_deg"],
                "rf_only_3d_rmse_m": rf["rmse_3d_m"],
                "rf_vision_3d_rmse_m": joint["rmse_3d_m"],
                "gain_3d": summary["gain_3d"],
                "rf_only_z_rmse_m": rf["z_rmse_m"],
                "rf_vision_z_rmse_m": joint["z_rmse_m"],
                "gain_z": summary["gain_z"],
                "gain_3d_ci_lower": summary["gain_3d_sequence_bootstrap_95ci"][0],
                "gain_3d_ci_upper": summary["gain_3d_sequence_bootstrap_95ci"][1],
            }
        )
    _write_csv(output_root / "rf_realism_comparison.csv", comparison_rows)
    result = {
        "status": "COMPLETE",
        "frozen_baseline": frozen_baseline,
        "conditions": all_summaries,
        "calibrations": calibrations,
        "link_budget": {
            **settings["link_budget"],
            "derived_tx_power_total_dbm": budget.tx_power_total_dbm,
            "noise_power_per_subcarrier_dbm": budget.noise_power_per_subcarrier_dbm(waveform),
        },
        "frozen_visual_calibration_reused": True,
        "frozen_detection_cache_reused": True,
        "held_out_test_frames": len(flat_records),
        "test_seeds": settings["protocol"]["test_seeds"],
        "sionna_reported_separately": True,
    }
    (output_root / "analytic_ladder_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rf_realism_ladder.yaml")
    )
    parser.add_argument("--conditions", nargs="*", choices=sorted(CONDITION_FLAGS))
    arguments = parser.parse_args()
    run(arguments.config, arguments.conditions or None)

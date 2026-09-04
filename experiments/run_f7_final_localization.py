"""Run the frozen waveform plus real-Vision F7 localization experiment."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import chi2

from experiments.run_f5_analytic_waveform import _estimate_only_waveforms, _waveform_components
from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body
from uav_ican_3d.localization import (
    localize_online,
    predict_rf_observation,
    simulate_los_srs,
    wrap_angle,
)
from uav_ican_3d.types import RFObservation
from uav_ican_3d.vision import FrozenVisualCalibration, box_iou


METHODS = (
    "rf_only_full_3d",
    "rf_vision_shared_pose",
    "rf_vision_no_pose_uncertainty",
    "range_azimuth_vision",
)


def _transform(values: dict, rotation_key: str, translation_key: str) -> RigidTransform:
    return RigidTransform(
        np.asarray(values[rotation_key], dtype=float),
        np.asarray(values[translation_key], dtype=float),
    )


def _camera(record: dict) -> PinholeCamera:
    values = record["camera_intrinsics"]
    return PinholeCamera(
        float(values["fx_px"]),
        float(values["fy_px"]),
        float(values["cx_px"]),
        float(values["cy_px"]),
    )


def _load_records(config: dict) -> list[dict]:
    expected = list(config["f7"]["test_sequences"])
    if expected != list(config["f0"]["sequence_splits"]["test"]):
        raise RuntimeError("F7 IDs differ from the F0 held-out test split")
    records: list[dict] = []
    root = Path(config["f0"]["output_root"])
    for sequence_id in expected:
        sequence = [
            json.loads(line)
            for line in (root / sequence_id / "manifest.jsonl").read_text().splitlines()
            if line
        ]
        if len(sequence) != int(config["f0"]["frames_per_sequence"]):
            raise RuntimeError(f"incomplete test sequence: {sequence_id}")
        if any(record["split"] != "test" for record in sequence):
            raise RuntimeError(f"non-test record in {sequence_id}")
        records.extend(sequence)
    if len(records) != 3200:
        raise RuntimeError("F7 requires all 3200 held-out frames")
    return records


def _load_detections(path: Path, records: list[dict]) -> list[dict]:
    detections = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(detections) != len(records):
        raise RuntimeError("detector cache and held-out records have different lengths")
    for record, detection in zip(records, detections, strict=True):
        key = (record["sequence_id"], int(record["frame_index"]))
        detected_key = (detection["sequence_id"], int(detection["frame_index"]))
        if key != detected_key:
            raise RuntimeError("detector cache order does not match held-out manifests")
    return detections


def _candidate_pixels(
    record: dict, detection: dict, calibration: FrozenVisualCalibration
) -> np.ndarray:
    boxes = np.asarray(detection["boxes_xywhr"], dtype=float).reshape(-1, 5)
    intrinsics = record["camera_intrinsics"]
    return calibration.corrected_pixels(
        boxes, (int(intrinsics["width_px"]), int(intrinsics["height_px"]))
    )


def _association_correct(record: dict, detection: dict, selected: int | None) -> int:
    if selected is None:
        return 0
    evaluation = record["evaluation_only"]
    names = evaluation["candidate_vehicle_names"]
    served_box = evaluation["candidate_gt_boxes_xyxy"][names.index("ServedVehicle")]
    detected_boxes = np.asarray(detection["boxes_xyxy"], dtype=float).reshape(-1, 4)
    return int(box_iou(served_box, detected_boxes[[selected]])[0] >= 0.10)


def _optimizer_options(config: dict) -> dict[str, float | int]:
    values = config["f3"]["optimizer"]
    return {
        "max_function_evaluations": int(values["max_function_evaluations"]),
        "gradient_tolerance": float(values["gradient_tolerance"]),
        "parameter_tolerance": float(values["parameter_tolerance"]),
        "cost_tolerance": float(values["cost_tolerance"]),
    }


def _run_sequence_seed(payload: tuple) -> tuple[list[dict], list[dict]]:
    config, sequence_id, waveform_seed, records, detections, subset_by_frame, calibration_values, f5 = payload
    phase = config["f7"]
    calibration = FrozenVisualCalibration.from_dict(calibration_values)
    waveform, estimator = _waveform_components(config)
    body_array = RigidTransform.identity()
    angle_std = np.deg2rad(float(config["f2"]["nominal_attitude_std_deg"]))
    pose_covariance = np.diag(
        [float(config["f2"]["pose_position_std_m"]) ** 2] * 3 + [angle_std**2] * 3
    )
    rf_covariance = np.asarray(
        f5["nominal_covariance_range_azimuth_elevation"], dtype=float
    )
    rf_bias = np.asarray(f5["nominal_bias_range_azimuth_elevation"], dtype=float)
    waveform_rng = np.random.default_rng(int(waveform_seed) + 1000 * int(sequence_id[-2:]))
    pose_rng = np.random.default_rng(int(waveform_seed) + 50000 + 1000 * int(sequence_id[-2:]))
    truths: list[np.ndarray] = []
    actual_bodies: list[RigidTransform] = []
    targets: list[np.ndarray] = []
    for record in records:
        actual_body = _transform(record["transform_world_body"], "R_WB", "t_W_B")
        target = np.asarray(
            record["evaluation_only"]["antenna_phase_centers_world_m"]["ServedVehicle"],
            dtype=float,
        )
        truth = predict_rf_observation(target, actual_body.compose(body_array))
        truths.append(truth)
        actual_bodies.append(actual_body)
        targets.append(target)

    estimates: list[np.ndarray] = []
    batch_size = int(phase["waveform_batch_size"])
    for start in range(0, len(records), batch_size):
        stop = min(len(records), start + batch_size)
        received = np.stack(
            [
                simulate_los_srs(
                    float(truths[index][0]),
                    float(truths[index][1]),
                    float(truths[index][2]),
                    float(phase["waveform_snr_db"]),
                    waveform,
                    waveform_rng,
                )
                for index in range(start, stop)
            ]
        )
        estimates.extend(_estimate_only_waveforms(estimator, received))

    rows: list[dict] = []
    failures: list[dict] = []
    threshold = float(chi2.ppf(float(phase["coverage_confidence"]), df=3))
    options = _optimizer_options(config)
    for record, detection, actual_body, target, estimate_vector in zip(
        records, detections, actual_bodies, targets, estimates, strict=True
    ):
        nominal_body = perturb_world_body(
            actual_body, pose_rng.multivariate_normal(np.zeros(6), pose_covariance)
        )
        measurement = np.asarray(estimate_vector, dtype=float) - rf_bias
        measurement[1:] = wrap_angle(measurement[1:])
        rf_observation = RFObservation(
            *measurement, rf_covariance, float(record["timestamp_s"])
        )
        body_camera = _transform(record["transform_body_camera"], "R_BC", "t_B_C")
        candidates = _candidate_pixels(record, detection, calibration)
        online = localize_online(
            rf_observation,
            nominal_body,
            body_array,
            pose_covariance,
            _camera(record),
            body_camera,
            candidates,
            calibration.covariance_uv,
            float(phase["association_confidence"]),
            options,
        )
        selected = online.primary_selected_index
        correct = _association_correct(record, detection, selected)
        subset = subset_by_frame[(sequence_id, int(record["frame_index"]))]
        for method in METHODS:
            result = online.estimates[method]
            error = result.position_world_m - target
            nees = float(error @ np.linalg.solve(result.position_covariance, error))
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": int(record["frame_index"]),
                    "waveform_seed": int(waveform_seed),
                    "method": method,
                    "subset": subset,
                    "visual_update": int(
                        selected is not None if method != "range_azimuth_vision" else True
                    ),
                    "association_correct": correct,
                    "selected_index": (
                        -1
                        if selected is None and method != "range_azimuth_vision"
                        else (
                            online.reduced_array_selected_index
                            if method == "range_azimuth_vision"
                            else int(selected)
                        )
                    ),
                    "estimate_x_m": float(result.position_world_m[0]),
                    "estimate_y_m": float(result.position_world_m[1]),
                    "estimate_z_m": float(result.position_world_m[2]),
                    "error_x_m": float(error[0]),
                    "error_y_m": float(error[1]),
                    "error_z_m": float(error[2]),
                    "error_3d_m": float(np.linalg.norm(error)),
                    "absolute_z_error_m": float(abs(error[2])),
                    "position_nees": nees,
                    "covered_95": int(nees <= threshold),
                    "cov_xx_m2": float(result.position_covariance[0, 0]),
                    "cov_yy_m2": float(result.position_covariance[1, 1]),
                    "cov_zz_m2": float(result.position_covariance[2, 2]),
                }
            )
        if not correct:
            failures.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": int(record["frame_index"]),
                    "waveform_seed": int(waveform_seed),
                    "subset": subset,
                    "image_path": record["image_path"],
                    "boxes_xyxy": detection["boxes_xyxy"],
                    "candidate_pixels_uv": candidates.tolist(),
                    "selected_index": -1 if selected is None else int(selected),
                    "rf_image_mean_uv": online.rf_image_mean_uv.tolist(),
                    "rf_image_covariance_uv": online.rf_image_covariance_uv.tolist(),
                }
            )
    return rows, failures


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _method_summary(rows: list[dict], method: str) -> dict[str, float | int | str]:
    selected = [row for row in rows if row["method"] == method]
    errors = np.asarray([row["error_3d_m"] for row in selected])
    z_errors = np.asarray([row["error_z_m"] for row in selected])
    return {
        "method": method,
        "samples": len(selected),
        "rmse_3d_m": float(np.sqrt(np.mean(errors**2))),
        "z_rmse_m": float(np.sqrt(np.mean(z_errors**2))),
        "p95_error_3d_m": float(np.percentile(errors, 95.0)),
        "mean_position_nees": float(np.mean([row["position_nees"] for row in selected])),
        "coverage_95": float(np.mean([row["covered_95"] for row in selected])),
        "visual_update_rate": float(np.mean([row["visual_update"] for row in selected])),
    }


def _bootstrap_primary(rows: list[dict], config: dict) -> dict[str, float | int | str]:
    phase = config["f7"]
    sequence_ids = list(phase["test_sequences"])
    squared: dict[tuple[str, str], np.ndarray] = {}
    for sequence_id in sequence_ids:
        for method in ("rf_only_full_3d", "rf_vision_shared_pose"):
            squared[(sequence_id, method)] = np.asarray(
                [
                    row["error_3d_m"] ** 2
                    for row in rows
                    if row["sequence_id"] == sequence_id and row["method"] == method
                ]
            )
    rng = np.random.default_rng(int(phase["bootstrap_seed"]))
    improvements = np.empty(int(phase["bootstrap_resamples"]), dtype=float)
    for index in range(improvements.size):
        sampled = rng.choice(sequence_ids, size=len(sequence_ids), replace=True)
        rf = np.concatenate([squared[(sequence_id, "rf_only_full_3d")] for sequence_id in sampled])
        joint = np.concatenate(
            [squared[(sequence_id, "rf_vision_shared_pose")] for sequence_id in sampled]
        )
        rf_rmse = float(np.sqrt(np.mean(rf)))
        joint_rmse = float(np.sqrt(np.mean(joint)))
        improvements[index] = (rf_rmse - joint_rmse) / rf_rmse
    return {
        "unit": str(phase["bootstrap_unit"]),
        "resamples": improvements.size,
        "seed": int(phase["bootstrap_seed"]),
        "improvement_95ci_lower": float(np.percentile(improvements, 2.5)),
        "improvement_95ci_upper": float(np.percentile(improvements, 97.5)),
    }


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase = config["f7"]
    if config.get("mode") != "full_localization":
        raise RuntimeError("F7 requires mode: full_localization")
    if phase["waveform_model"] != "analytic_los" or len(phase["rf_seeds"]) != 5:
        raise RuntimeError("F7 requires the frozen analytic waveform model and five RF seeds")
    output_root = Path(config["outputs"]["root"])
    if not Path(config["outputs"]["final_freeze"]).exists():
        raise RuntimeError("write FULL_LOCALIZATION_FREEZE.md before the F7 final test")
    records = _load_records(config)
    detections = _load_detections(output_root / "f4_test_detections.jsonl", records)
    calibration_artifact = json.loads((output_root / "f1_visual_calibration.json").read_text())
    calibration_values = calibration_artifact["selected_model"]
    f5 = json.loads((output_root / "f5_summary.json").read_text())
    if f5["status"] != "PASS" or not np.isclose(
        float(phase["waveform_snr_db"]), float(f5["nominal_snr_db"])
    ):
        raise RuntimeError("F7 waveform settings differ from the frozen F5 nominal model")
    subset_rows = [
        json.loads(line)
        for line in (output_root / "f4_subset_freeze.jsonl").read_text().splitlines()
        if line
    ]
    subset_by_frame = {
        (row["sequence_id"], int(row["frame_index"])): row["subset"] for row in subset_rows
    }
    records_by_sequence = {
        sequence_id: [row for row in records if row["sequence_id"] == sequence_id]
        for sequence_id in phase["test_sequences"]
    }
    detections_by_sequence = {
        sequence_id: [row for row in detections if row["sequence_id"] == sequence_id]
        for sequence_id in phase["test_sequences"]
    }
    payloads = [
        (
            config,
            sequence_id,
            int(seed),
            records_by_sequence[sequence_id],
            detections_by_sequence[sequence_id],
            subset_by_frame,
            calibration_values,
            f5,
        )
        for sequence_id in phase["test_sequences"]
        for seed in phase["rf_seeds"]
    ]
    rows: list[dict] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=int(phase["parallel_workers"])) as executor:
        futures = {executor.submit(_run_sequence_seed, payload): payload[1:3] for payload in payloads}
        for future in as_completed(futures):
            task_rows, task_failures = future.result()
            rows.extend(task_rows)
            failures.extend(task_failures)
            sequence_id, seed = futures[future]
            print(f"F7 complete sequence={sequence_id} waveform_seed={seed}", flush=True)
    rows.sort(key=lambda row: (row["sequence_id"], row["waveform_seed"], row["frame_index"], METHODS.index(row["method"])))
    failures.sort(key=lambda row: (row["sequence_id"], row["waveform_seed"], row["frame_index"]))
    if len(rows) != 3200 * 5 * len(METHODS):
        raise AssertionError("F7 did not produce every frame/seed/method result")
    _write_csv(output_root / "f7_frame_results.csv", rows)
    failure_root = output_root / "failure_cases"
    failure_root.mkdir(exist_ok=True)
    (failure_root / "f7_incorrect_associations.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in failures),
        encoding="utf-8",
    )
    summaries = [_method_summary(rows, method) for method in METHODS]
    _write_csv(output_root / "full_localization_main_results.csv", summaries[:2])
    _write_csv(output_root / "full_localization_ablations.csv", summaries[2:])
    summary_by_method = {row["method"]: row for row in summaries}
    rf = summary_by_method["rf_only_full_3d"]
    joint = summary_by_method["rf_vision_shared_pose"]
    bootstrap = _bootstrap_primary(rows, config)
    result = {
        "status": "COMPLETE",
        "held_out_frames": 3200,
        "waveform_seeds": list(map(int, phase["rf_seeds"])),
        "paired_primary_samples": 16000,
        "waveform_model": str(phase["waveform_model"]),
        "waveform_snr_db": float(phase["waveform_snr_db"]),
        "methods": summary_by_method,
        "primary_3d_rmse_reduction": (
            float(rf["rmse_3d_m"]) - float(joint["rmse_3d_m"])
        )
        / float(rf["rmse_3d_m"]),
        "primary_z_rmse_reduction": (
            float(rf["z_rmse_m"]) - float(joint["z_rmse_m"])
        )
        / float(rf["z_rmse_m"]),
        "bootstrap": bootstrap,
        "association": {
            subset: float(
                np.mean(
                    [
                        row["association_correct"]
                        for row in rows
                        if row["method"] == "rf_vision_shared_pose" and row["subset"] == subset
                    ]
                )
            )
            for subset in ("easy", "hard")
        },
        "incorrect_primary_associations": len(failures),
        "online_ground_truth_fields": [],
        "sionna_path_truth_passed_to_estimator": False,
    }
    (output_root / "f7_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)

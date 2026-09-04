"""Run the FULL RF-quality and shared-pose EFIM experiment families."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/uav_ican_full_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/uav_ican_full_cache")
import matplotlib.pyplot as plt

from uav_ican_3d.localization import (
    equivalent_target_information,
    fisher_information,
    joint_target_pose_information,
    position_bounds,
    rf_observation_and_shared_pose_jacobians,
)
from uav_ican_3d.simulation import complementarity_scene


@dataclass(frozen=True)
class F2Result:
    family: str
    level: str
    altitude_m: float
    horizontal_distance_m: float
    range_std_m: float
    aoa_std_deg: float
    attitude_std_deg: float
    position_covariance_scale: float
    visual_available: int
    vision_only_rank: int
    pixel_u: float
    pixel_v: float
    peb_rf_m: float
    peb_joint_m: float
    zeb_rf_m: float
    zeb_joint_m: float
    gain_peb: float
    gain_zeb: float


def _pose_covariance(phase: dict, attitude_std_deg: float, position_scale: float) -> np.ndarray:
    position_variance = float(phase["pose_position_std_m"]) ** 2 * position_scale
    attitude_variance = np.deg2rad(attitude_std_deg) ** 2
    return np.diag([position_variance] * 3 + [attitude_variance] * 3)


def _vision_rank(
    camera_p: np.ndarray,
    camera_xi: np.ndarray,
    camera_covariance: np.ndarray,
    pose_covariance: np.ndarray,
) -> int:
    jacobian = np.column_stack((camera_p, camera_xi))
    information = fisher_information(jacobian, camera_covariance)
    information[3:, 3:] += np.linalg.solve(pose_covariance, np.eye(6))
    efim = equivalent_target_information(information)
    eigenvalues = np.linalg.eigvalsh(efim)
    tolerance = max(float(eigenvalues[-1]), 1.0) * 1e-10
    return int(np.count_nonzero(eigenvalues > tolerance))


def evaluate_cell(
    phase: dict,
    camera_covariance: np.ndarray,
    family: str,
    level: str,
    altitude_m: float,
    distance_m: float,
    range_std_m: float,
    aoa_std_deg: float,
    attitude_std_deg: float,
    position_scale: float = 1.0,
) -> F2Result:
    target, world_body, body_array, body_camera, camera = complementarity_scene(
        phase, altitude_m, distance_m
    )
    _, rf_p, rf_xi = rf_observation_and_shared_pose_jacobians(
        target, world_body, body_array
    )
    pixel, camera_p, camera_xi = camera.observation_and_shared_pose_jacobians(
        target, world_body, body_camera
    )
    camera_config = phase["camera"]
    visual_available = bool(
        0.0 <= pixel[0] < float(camera_config["width_px"])
        and 0.0 <= pixel[1] < float(camera_config["height_px"])
    )
    angle_std = np.deg2rad(aoa_std_deg)
    rf_covariance = np.diag([range_std_m**2, angle_std**2, angle_std**2])
    pose_covariance = _pose_covariance(phase, attitude_std_deg, position_scale)
    rf_information = joint_target_pose_information(
        rf_p, rf_xi, rf_covariance, pose_covariance
    )
    if visual_available:
        joint_information = joint_target_pose_information(
            rf_p,
            rf_xi,
            rf_covariance,
            pose_covariance,
            camera_p,
            camera_xi,
            camera_covariance,
        )
    else:
        joint_information = rf_information
    rf_bounds = position_bounds(equivalent_target_information(rf_information))
    joint_bounds = position_bounds(equivalent_target_information(joint_information))
    return F2Result(
        family,
        level,
        altitude_m,
        distance_m,
        range_std_m,
        aoa_std_deg,
        attitude_std_deg,
        position_scale,
        int(visual_available),
        _vision_rank(camera_p, camera_xi, camera_covariance, pose_covariance),
        float(pixel[0]),
        float(pixel[1]),
        rf_bounds.peb_3d_m,
        joint_bounds.peb_3d_m,
        rf_bounds.zeb_m,
        joint_bounds.zeb_m,
        float((rf_bounds.peb_3d_m - joint_bounds.peb_3d_m) / rf_bounds.peb_3d_m),
        float((rf_bounds.zeb_m - joint_bounds.zeb_m) / rf_bounds.zeb_m),
    )


def _write_csv(path: Path, rows: list[F2Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(F2Result.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _reference_rows(rows: list[F2Result]) -> list[F2Result]:
    return [row for row in rows if row.family == "reference"]


def _write_map(
    rows: list[F2Result], phase: dict, metric: str, label: str, output: Path
) -> None:
    reference = _reference_rows(rows)
    altitudes = list(map(float, phase["altitude_m"]))
    distances = list(map(float, phase["horizontal_distance_m"]))
    lookup = {(row.altitude_m, row.horizontal_distance_m): getattr(row, metric) for row in reference}
    matrix = np.asarray(
        [[lookup[(altitude, distance)] for distance in distances] for altitude in altitudes]
    )
    figure, axis = plt.subplots(figsize=(8.2, 4.7))
    image = axis.imshow(matrix * 100.0, origin="lower", aspect="auto", vmin=0.0)
    axis.set_xticks(range(len(distances)), [f"{value:g}" for value in distances])
    axis.set_yticks(range(len(altitudes)), [f"{value:g}" for value in altitudes])
    axis.set_xlabel("Horizontal distance (m)")
    axis.set_ylabel("UAV altitude (m)")
    axis.set_title(label)
    figure.colorbar(image, ax=axis, label="Relative gain (%)")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _representative_protocol(rows: list[F2Result], phase: dict) -> list[dict[str, float]]:
    available = sorted(
        (row for row in _reference_rows(rows) if row.visual_available),
        key=lambda row: (row.gain_peb, row.altitude_m, row.horizontal_distance_m),
    )
    count = 8
    bins = np.array_split(np.arange(len(available)), count)
    selected: list[F2Result] = []
    altitude_span = max(map(float, phase["altitude_m"])) - min(map(float, phase["altitude_m"]))
    distance_span = max(map(float, phase["horizontal_distance_m"])) - min(
        map(float, phase["horizontal_distance_m"])
    )
    for indices in bins:
        candidates = [available[int(index)] for index in indices]
        if not selected:
            chosen = candidates[0]
        else:
            chosen = max(
                candidates,
                key=lambda row: min(
                    ((row.altitude_m - prior.altitude_m) / altitude_span) ** 2
                    + ((row.horizontal_distance_m - prior.horizontal_distance_m) / distance_span) ** 2
                    for prior in selected
                ),
            )
        selected.append(chosen)
    return [
        {
            "cell_id": index,
            "altitude_m": row.altitude_m,
            "horizontal_distance_m": row.horizontal_distance_m,
            "f2_gain_peb": row.gain_peb,
            "f2_gain_zeb": row.gain_zeb,
        }
        for index, row in enumerate(selected)
    ]


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("F2 requires mode: full_localization")
    phase = config["f2"]
    calibration = json.loads(
        (Path(config["outputs"]["root"]) / "f1_visual_calibration.json").read_text()
    )
    if calibration["status"] != "PASS" or calibration["final_test_frames_read"] != 0:
        raise RuntimeError("F2 requires a leakage-free frozen F1 model")
    camera_covariance = np.asarray(
        calibration["selected_model"]["covariance_uv_px2"], dtype=float
    )
    altitudes = list(map(float, phase["altitude_m"]))
    distances = list(map(float, phase["horizontal_distance_m"]))
    geometries = [(altitude, distance) for altitude in altitudes for distance in distances]
    nominal_range = float(phase["nominal_range_std_m"])
    nominal_aoa = float(phase["nominal_aoa_std_deg"])
    nominal_attitude = float(phase["nominal_attitude_std_deg"])
    rows: list[F2Result] = []
    for altitude, distance in geometries:
        rows.append(
            evaluate_cell(
                phase,
                camera_covariance,
                "reference",
                "nominal",
                altitude,
                distance,
                nominal_range,
                nominal_aoa,
                nominal_attitude,
            )
        )
        for index, (range_std, aoa_std) in enumerate(
            zip(phase["range_std_m"], phase["aoa_std_deg"], strict=True)
        ):
            rows.append(
                evaluate_cell(
                    phase,
                    camera_covariance,
                    "rf_quality",
                    f"paired_log_level_{index}",
                    altitude,
                    distance,
                    float(range_std),
                    float(aoa_std),
                    nominal_attitude,
                )
            )
        for index, attitude_std in enumerate(phase["attitude_std_deg"]):
            rows.append(
                evaluate_cell(
                    phase,
                    camera_covariance,
                    "pose_quality",
                    f"attitude_log_level_{index}",
                    altitude,
                    distance,
                    nominal_range,
                    nominal_aoa,
                    float(attitude_std),
                )
            )
        rows.append(
            evaluate_cell(
                phase,
                camera_covariance,
                "position_sensitivity",
                "position_covariance_x4",
                altitude,
                distance,
                nominal_range,
                nominal_aoa,
                nominal_attitude,
                float(phase["position_covariance_sensitivity_scale"]),
            )
        )

    if not all(np.all(np.isfinite(list(asdict(row).values())[2:])) for row in rows):
        raise AssertionError("F2 produced a non-finite numerical result")
    output_root = Path(config["outputs"]["root"])
    dense_csv = output_root / "f2_efim_dense.csv"
    map_csv = output_root / "f2_reference_gain_maps.csv"
    _write_csv(dense_csv, rows)
    _write_csv(map_csv, _reference_rows(rows))
    _write_map(rows, phase, "gain_peb", "FULL shared-pose PEB gain", output_root / "f2_peb_gain_map.pdf")
    _write_map(rows, phase, "gain_zeb", "FULL shared-pose ZEB gain", output_root / "f2_zeb_gain_map.pdf")

    gate = phase["gate"]
    strong = [
        row
        for row in _reference_rows(rows)
        if row.visual_available
        and row.gain_peb >= float(gate["minimum_reference_peb_gain"])
        and row.gain_zeb >= float(gate["minimum_reference_zeb_gain"])
    ]
    robust = [
        row
        for row in rows
        if row.family == "pose_quality"
        and np.isclose(row.attitude_std_deg, max(map(float, phase["attitude_std_deg"])))
        and row.visual_available
        and row.gain_peb >= float(gate["minimum_robust_gain_at_max_attitude"])
        and row.gain_zeb >= float(gate["minimum_robust_gain_at_max_attitude"])
    ]
    passed = (
        len(strong) >= int(gate["minimum_reference_cells"])
        and len({row.altitude_m for row in strong}) >= int(gate["minimum_altitudes_spanned"])
        and len({row.horizontal_distance_m for row in strong})
        >= int(gate["minimum_distances_spanned"])
        and len(robust) >= int(gate["minimum_robust_cells_at_max_attitude"])
        and all(row.vision_only_rank < 3 for row in rows)
        and min(row.gain_peb for row in rows) >= -1e-9
        and min(row.gain_zeb for row in rows) >= -1e-9
    )
    if not passed:
        raise AssertionError(
            f"F2 GATE: FAIL strong_reference={len(strong)} robust_max_attitude={len(robust)}"
        )
    representatives = _representative_protocol(rows, phase)
    protocol_path = output_root / "f2_representative_geometries.json"
    protocol_path.write_text(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_F3",
                "selection_rule": "one cell from each nominal PEB-gain octile; within each octile choose maximum normalized spatial separation from prior cells",
                "rf_parameters": {
                    "range_std_m": nominal_range,
                    "aoa_std_deg": nominal_aoa,
                    "attitude_std_deg": nominal_attitude,
                    "pose_position_std_m": float(phase["pose_position_std_m"]),
                },
                "cells": representatives,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    result = {
        "status": "PASS",
        "evaluated_cells": len(rows),
        "reference_cells": len(_reference_rows(rows)),
        "visual_available_reference_cells": sum(row.visual_available for row in _reference_rows(rows)),
        "strong_reference_cells": len(strong),
        "robust_max_attitude_cells": len(robust),
        "reference_peb_gain_range": [
            min(row.gain_peb for row in _reference_rows(rows)),
            max(row.gain_peb for row in _reference_rows(rows)),
        ],
        "reference_zeb_gain_range": [
            min(row.gain_zeb for row in _reference_rows(rows)),
            max(row.gain_zeb for row in _reference_rows(rows)),
        ],
        "camera_covariance_uv_px2": camera_covariance.tolist(),
        "families": {
            "reference": 70,
            "rf_quality_paired_levels": 280,
            "pose_quality": 350,
            "position_covariance_x4": 70,
        },
        "representative_protocol": str(protocol_path),
    }
    (output_root / "f2_efim_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"evaluated_cells={len(rows)}")
    print(f"strong_reference_cells={len(strong)}")
    print(f"robust_max_attitude_cells={len(robust)}")
    print(f"reference_PEB_gain_range={result['reference_peb_gain_range']}")
    print(f"reference_ZEB_gain_range={result['reference_zeb_gain_range']}")
    print("F2 GATE: PASS")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)

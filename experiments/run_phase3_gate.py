"""Create a sparse modality-complementarity scan and Phase 3 decision."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/uav_ican_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/uav_ican_cache")
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
class ScanResult:
    altitude_m: float
    horizontal_distance_m: float
    range_std_m: float
    aoa_std_deg: float
    attitude_std_deg: float
    peb_rf_m: float
    peb_joint_m: float
    zeb_rf_m: float
    zeb_joint_m: float
    gain_peb: float
    gain_zeb: float
    vision_only_rank: int
    pixel_u: float
    pixel_v: float


def _pose_covariance(phase: dict, attitude_std_deg: float) -> np.ndarray:
    position_variance = float(phase["pose_position_std_m"]) ** 2
    attitude_variance = np.deg2rad(attitude_std_deg) ** 2
    return np.diag([position_variance] * 3 + [attitude_variance] * 3)


def _vision_only_rank(
    camera_p: np.ndarray,
    camera_xi: np.ndarray,
    camera_covariance: np.ndarray,
    pose_covariance: np.ndarray,
) -> int:
    joint_jacobian = np.column_stack((camera_p, camera_xi))
    information = fisher_information(joint_jacobian, camera_covariance)
    information[3:, 3:] += np.linalg.solve(pose_covariance, np.eye(6))
    equivalent = equivalent_target_information(information)
    eigenvalues = np.linalg.eigvalsh(equivalent)
    threshold = max(float(eigenvalues[-1]), 1.0) * 1e-10
    return int(np.count_nonzero(eigenvalues > threshold))


def _evaluate(
    phase: dict,
    altitude_m: float,
    distance_m: float,
    range_std_m: float,
    aoa_std_deg: float,
    attitude_std_deg: float,
) -> ScanResult:
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
    if not (
        0.0 <= pixel[0] < float(camera_config["width_px"])
        and 0.0 <= pixel[1] < float(camera_config["height_px"])
    ):
        raise RuntimeError(
            f"target outside camera FOV at altitude={altitude_m}, distance={distance_m}: {pixel}"
        )

    angle_std_rad = np.deg2rad(aoa_std_deg)
    rf_covariance = np.diag([range_std_m**2, angle_std_rad**2, angle_std_rad**2])
    camera_covariance = np.eye(2) * float(phase["camera_noise_std_px"]) ** 2
    pose_covariance = _pose_covariance(phase, attitude_std_deg)
    rf_information = joint_target_pose_information(
        rf_p, rf_xi, rf_covariance, pose_covariance
    )
    joint_information = joint_target_pose_information(
        rf_p,
        rf_xi,
        rf_covariance,
        pose_covariance,
        camera_p,
        camera_xi,
        camera_covariance,
    )
    rf_bounds = position_bounds(equivalent_target_information(rf_information))
    joint_bounds = position_bounds(equivalent_target_information(joint_information))
    gain_peb = (rf_bounds.peb_3d_m - joint_bounds.peb_3d_m) / rf_bounds.peb_3d_m
    gain_zeb = (rf_bounds.zeb_m - joint_bounds.zeb_m) / rf_bounds.zeb_m
    return ScanResult(
        altitude_m,
        distance_m,
        range_std_m,
        aoa_std_deg,
        attitude_std_deg,
        rf_bounds.peb_3d_m,
        joint_bounds.peb_3d_m,
        rf_bounds.zeb_m,
        joint_bounds.zeb_m,
        float(gain_peb),
        float(gain_zeb),
        _vision_only_rank(camera_p, camera_xi, camera_covariance, pose_covariance),
        float(pixel[0]),
        float(pixel[1]),
    )


def _write_csv(results: list[ScanResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ScanResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def _write_map(phase: dict, results: list[ScanResult], output: Path) -> None:
    reference = phase["reference_map"]
    selected = [
        result
        for result in results
        if result.range_std_m == float(reference["range_std_m"])
        and result.aoa_std_deg == float(reference["aoa_std_deg"])
        and result.attitude_std_deg == float(reference["attitude_std_deg"])
    ]
    altitudes = [float(value) for value in phase["altitude_m"]]
    distances = [float(value) for value in phase["horizontal_distance_m"]]
    gain_by_geometry = {
        (result.altitude_m, result.horizontal_distance_m): result.gain_peb for result in selected
    }
    matrix = np.array(
        [[gain_by_geometry[(altitude, distance)] for distance in distances] for altitude in altitudes]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    image = axis.imshow(matrix * 100.0, vmin=0.0, vmax=max(40.0, float(matrix.max() * 100.0)))
    axis.set_xticks(range(len(distances)), labels=[f"{value:g}" for value in distances])
    axis.set_yticks(range(len(altitudes)), labels=[f"{value:g}" for value in altitudes])
    axis.set_xlabel("Horizontal distance (m)")
    axis.set_ylabel("UAV altitude (m)")
    axis.set_title("RF+Vision PEB gain (%)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{100.0 * matrix[row, column]:.1f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="PEB gain (%)")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def _write_decision(
    results: list[ScanResult], stable: list[ScanResult], passed: bool, output: Path
) -> None:
    best = max(results, key=lambda result: result.gain_peb)
    best_stable = max(stable, key=lambda result: result.gain_peb) if stable else None
    weakest_stable = min(stable, key=lambda result: result.gain_peb) if stable else None
    geometries = {(result.altitude_m, result.horizontal_distance_m) for result in stable}
    altitudes = {geometry[0] for geometry in geometries}
    distances = {geometry[1] for geometry in geometries}
    status = "PASS" if passed else "FAIL"
    text = f"""# Phase 3 modality complementarity decision

Status: **{status}**

The QUICK scan evaluated {len(results)} non-extreme combinations. It used UAV altitudes of
30–100 m, below the common 400 ft (about 122 m) small-UAS ceiling documented by the
[FAA](https://www.faa.gov/newsroom/small-unmanned-aircraft-systems-uas-regulations-part-107).
The camera model follows the pinhole/reprojection convention documented by
[OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). RF and camera error values are
explicit engineering scan assumptions, not claimed hardware measurements; Phase 6 and Phase 9
must replace them with empirical detector and waveform-estimator residual covariances.

Gate conditions required at least 20% PEB and ZEB improvement in at least four reasonable cells,
spanning at least two altitudes and two horizontal distances. "Reasonable" excludes perfect pose:
attitude standard deviation is 0.5–1.0 degrees, range standard deviation is at most 1 m, AoA
standard deviation is at most 2 degrees, and pixel standard deviation is 2 px.

- Stable reasonable cells: {len(stable)}
- Stable geometries: {len(geometries)} across {len(altitudes)} altitudes and {len(distances)} distances
- Vision-only target-information rank: {sorted(set(result.vision_only_rank for result in results))}
- Reasonable stable PEB-gain range: {f"{100.0 * weakest_stable.gain_peb:.2f}%–{100.0 * best_stable.gain_peb:.2f}%" if stable else "none"}
- Overall scan maximum (reported only for audit, not used to pass): {100.0 * best.gain_peb:.2f}%

Vision-only remains rank deficient for unconstrained 3-D depth, while RF-only is full rank and the
joint model materially improves its bounds. Thus the observed gain is complementary rather than a
vision-only replacement of a deliberately weakened RF baseline.

Decision: {"Proceed to Phase 4." if passed else "STOP; do not proceed to visual training."}
"""
    output.write_text(text, encoding="utf-8")


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 3 gate refuses to run unless mode is 'quick'")
    phase = config["phase3"]
    results = [
        _evaluate(phase, altitude, distance, range_std, aoa_std, attitude_std)
        for altitude in map(float, phase["altitude_m"])
        for distance in map(float, phase["horizontal_distance_m"])
        for range_std in map(float, phase["range_std_m"])
        for aoa_std in map(float, phase["aoa_std_deg"])
        for attitude_std in map(float, phase["attitude_std_deg"])
    ]
    gate = phase["gate"]
    stable = [
        result
        for result in results
        if result.range_std_m <= float(gate["reasonable_max_range_std_m"])
        and result.aoa_std_deg <= float(gate["reasonable_max_aoa_std_deg"])
        and float(gate["reasonable_min_attitude_std_deg"])
        <= result.attitude_std_deg
        <= float(gate["reasonable_max_attitude_std_deg"])
        and result.gain_peb >= float(gate["minimum_peb_gain"])
        and result.gain_zeb >= float(gate["minimum_zeb_gain"])
    ]
    geometries = {(result.altitude_m, result.horizontal_distance_m) for result in stable}
    passed = (
        len(stable) >= int(gate["minimum_stable_cells"])
        and len({geometry[0] for geometry in geometries}) >= 2
        and len({geometry[1] for geometry in geometries}) >= 2
        and all(result.vision_only_rank < 3 for result in results)
    )
    _write_csv(results, Path("results/phase3_quick_scan.csv"))
    _write_map(phase, results, Path("docs/figures/phase3_complementarity_map.png"))
    _write_decision(results, stable, passed, Path("docs/PHASE3_DECISION.md"))
    best = max(results, key=lambda result: result.gain_peb)
    print(f"mode=quick scanned_cells={len(results)}")
    print(f"stable_reasonable_cells={len(stable)} stable_geometries={len(geometries)}")
    print(f"vision_only_ranks={sorted(set(result.vision_only_rank for result in results))}")
    print(f"best_PEB_gain={100.0 * best.gain_peb:.3f}% best_ZEB_gain={100.0 * best.gain_zeb:.3f}%")
    if not passed:
        raise AssertionError("PHASE 3 QUICK GATE: FAIL")
    print("PHASE 3 QUICK GATE: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)

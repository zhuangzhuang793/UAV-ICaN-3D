"""Run available Phase 5 checks and block unless synchronized GT frames exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.localization import joint_target_pose_information, rf_observation_and_shared_pose_jacobians
from uav_ican_3d.simulation import complementarity_scene
from uav_ican_3d.vision import (
    ImageBelief,
    associate_candidates,
    inspect_visdrone_split,
    project_joint_rf_belief_to_image,
)


REQUIRED_MANIFEST_FIELDS = {
    "timestamp_s",
    "image_path",
    "camera_intrinsics",
    "transform_world_body",
    "transform_body_camera",
    "vehicle_reference_world_m",
    "antenna_lever_arm_vehicle_m",
    "served_gt_box_xyxy",
}


def _validate_synchronized_manifest(path: Path, maximum_frames: int) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("synchronized manifest is empty")
    for line_number, record in enumerate(records[:maximum_frames], start=1):
        missing = REQUIRED_MANIFEST_FIELDS - record.keys()
        if missing:
            raise ValueError(f"manifest line {line_number} misses fields: {sorted(missing)}")
        if not Path(record["image_path"]).is_file():
            raise FileNotFoundError(f"manifest image does not exist: {record['image_path']}")
    return min(len(records), maximum_frames)


def _geometry_component_smoke(config: dict) -> tuple[float, int | None]:
    phase3 = config["phase3"]
    phase5 = config["phase5"]
    target, world_body, body_array, body_camera, camera = complementarity_scene(
        phase3, 60.0, 60.0
    )
    _, rf_p, rf_xi = rf_observation_and_shared_pose_jacobians(target, world_body, body_array)
    angle_std = np.deg2rad(2.0)
    rf_covariance = np.diag([1.0, angle_std**2, angle_std**2])
    pose_covariance = np.diag([0.2**2] * 3 + [np.deg2rad(0.5) ** 2] * 3)
    rf_information = joint_target_pose_information(
        rf_p, rf_xi, rf_covariance, pose_covariance
    )
    joint_covariance = np.linalg.solve(rf_information, np.eye(9))
    belief = project_joint_rf_belief_to_image(
        target,
        joint_covariance,
        world_body,
        body_camera,
        camera,
        covariance_floor_px2=float(phase5["covariance_floor_px2"]),
    )
    candidates = np.vstack((belief.mean_uv, belief.mean_uv + np.array([200.0, 150.0])))
    association = associate_candidates(
        candidates, belief, confidence=float(phase5["gating_confidence"])
    )
    return float(np.linalg.det(belief.covariance_uv)), association.selected_index


def _write_blocker(path: Path, visdrone_frames: int, visdrone_candidates: int, manifest: Path) -> None:
    path.write_text(
        f"""# Phase 5 blocker

Status: **BLOCKED**

The RF image-belief projection and Mahalanobis association components passed their geometry smoke
test. The local VisDrone adapter also read {visdrone_frames} frames containing
{visdrone_candidates} car/van/truck/bus GT candidates.

VisDrone cannot satisfy the Phase 5 GT-first gate because it has no synchronized UAV pose, camera
pose/calibration, vehicle 3-D reference, UE antenna lever arm, or RF observation. No local
AirSim/Cosys-AirSim scene or alternate synchronized dataset was found.

Expected manifest: `{manifest}`

Do not infer 3-D ground truth by selecting a VisDrone bbox and placing a synthetic vehicle on its
back-projected ray. That would make the projection test circular. Provide a genuinely synchronized
small scene following `data/README.md`, or authorize setup of a compatible simulator. Detector
inference and Phase 6 must not begin before this gate is resolved.
""",
        encoding="utf-8",
    )


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 5 gate refuses to run unless mode is 'quick'")
    phase = config["phase5"]
    determinant, selected = _geometry_component_smoke(config)
    if determinant <= 0.0 or selected != 0:
        raise AssertionError("RF image-belief geometry component smoke test failed")
    frames, candidates = inspect_visdrone_split(
        Path(phase["visdrone_root"]),
        str(phase["visdrone_split"]),
        {int(value) for value in phase["vehicle_class_ids"]},
        int(phase["inspection_frames"]),
    )
    manifest = Path(phase["synchronized_manifest"])
    print(f"image_belief_covariance_determinant={determinant:.6e}")
    print(f"visdrone_frames_inspected={frames} vehicle_candidates={candidates}")
    try:
        synchronized_frames = _validate_synchronized_manifest(
            manifest, int(phase["inspection_frames"])
        )
    except (FileNotFoundError, ValueError) as error:
        _write_blocker(Path("docs/PHASE5_BLOCKER.md"), frames, candidates, manifest)
        print(f"synchronized_data_error={error}")
        print("PHASE 5 QUICK GATE: BLOCKED")
        raise SystemExit(2) from error
    print(f"synchronized_frames_validated={synchronized_frames}")
    raise NotImplementedError(
        "manifest exists; implement GT coverage and guided-vs-unguided evaluation before PASS"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)


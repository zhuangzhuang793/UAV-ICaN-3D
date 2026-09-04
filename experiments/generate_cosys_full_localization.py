"""Generate sequence-isolated synchronized Cosys-AirSim data for FULL localization."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    from experiments.generate_cosys_phase5_data import (
        ANTENNA_LEVER_ARM_FRD_M,
        VEHICLE_NAMES,
        _bbox_for_color,
        _global_camera_pose_ned,
        _quaternion,
        _set_instance_ids,
        _transform_record,
        _vector,
    )
except ModuleNotFoundError:  # Direct execution places experiments/ rather than the repo on sys.path.
    from generate_cosys_phase5_data import (
        ANTENNA_LEVER_ARM_FRD_M,
        VEHICLE_NAMES,
        _bbox_for_color,
        _global_camera_pose_ned,
        _quaternion,
        _set_instance_ids,
        _transform_record,
        _vector,
    )
from uav_ican_3d.geometry import PinholeCamera
from uav_ican_3d.simulation import (
    airsim_camera_to_project_frames,
    ned_frd_pose_to_enu_transform,
)


def _sequence_split(phase: dict, sequence_id: str) -> str:
    matches = [name for name, values in phase["sequence_splits"].items() if sequence_id in values]
    if len(matches) != 1:
        raise ValueError(f"sequence {sequence_id} must occur in exactly one split, got {matches}")
    return matches[0]


def _trajectory(
    frame_index: int,
    frame_count: int,
    profile: dict,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, float]], np.ndarray, tuple[float, float, float]]:
    """Return vehicle poses and an external-camera pose in AirSim home-local NED."""

    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=5)
    angle = 2.0 * np.pi * frame_index / frame_count
    served = np.array(
        [
            5.0 * np.sin(angle + phase[0]),
            4.0 * np.cos(0.8 * angle + phase[1]),
            0.88,
        ]
    )
    hard_fraction = float(profile["hard_fraction"])
    hard = np.cos(2.0 * angle + phase[2]) > np.cos(np.pi * hard_fraction)
    near_scale = 1.0 if hard else 3.5
    distractor_a = served + np.array(
        [
            near_scale * (2.0 + 0.5 * np.sin(1.4 * angle + phase[3])),
            near_scale * (1.2 + 0.4 * np.cos(1.1 * angle + phase[4])),
            0.0,
        ]
    )
    distractor_b = served + np.array(
        [
            -near_scale * (1.8 + 0.4 * np.cos(1.3 * angle + phase[4])),
            near_scale * (1.4 + 0.5 * np.sin(0.9 * angle + phase[3])),
            0.0,
        ]
    )
    trajectories = {
        "ServedVehicle": (served, 0.30 * np.sin(angle + phase[2])),
        "DistractorA": (distractor_a, 0.45 + 0.20 * np.cos(angle + phase[3])),
        "DistractorB": (distractor_b, -0.55 + 0.18 * np.sin(angle + phase[4])),
    }

    altitude = float(profile["altitude_m"]) * (1.0 + 0.06 * np.sin(angle + phase[3]))
    horizontal = float(profile["horizontal_m"]) * (
        1.0 + 0.08 * np.cos(0.7 * angle + phase[4])
    )
    bearing = np.deg2rad(float(profile["bearing_deg"])) + 0.20 * np.sin(
        0.6 * angle + phase[2]
    )
    camera_position = served + np.array(
        [-horizontal * np.cos(bearing), -horizontal * np.sin(bearing), -altitude]
    )
    camera_yaw = bearing
    camera_pitch = -np.arctan2(altitude, horizontal)
    return trajectories, camera_position, (0.0, camera_pitch, camera_yaw)


def _complete_sequence(path: Path, frame_count: int) -> bool:
    manifest = path / "manifest.jsonl"
    if not manifest.is_file():
        return False
    records = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != frame_count:
        return False
    return all(
        (path / "images" / f"frame_{index:04d}.png").is_file()
        and (path / "segmentation" / f"frame_{index:04d}.png").is_file()
        for index in range(frame_count)
    )


def generate_sequence(
    output_dir: Path,
    sequence_id: str,
    split: str,
    profile: dict,
    frame_count: int,
    seed: int,
    host: str,
    port: int,
) -> Path:
    try:
        import cosysairsim as airsim
    except ImportError as error:
        raise RuntimeError("install the Cosys-AirSim 3.4.1 client wheel first") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    segmentation_dir = output_dir / "segmentation"
    images_dir.mkdir(exist_ok=True)
    segmentation_dir.mkdir(exist_ok=True)
    client = airsim.CarClient(ip=host, port=port, timeout_value=20)
    if not client.ping():
        raise RuntimeError(f"Cosys-AirSim RPC did not respond at {host}:{port}")
    missing = set(VEHICLE_NAMES) - set(client.listVehicles())
    if missing:
        raise RuntimeError(f"Cosys-AirSim scene misses vehicles: {sorted(missing)}")
    instance_colors = _set_instance_ids(client, airsim)
    records: list[dict[str, object]] = []

    try:
        for frame_index in range(frame_count):
            trajectories, camera_position, camera_angles = _trajectory(
                frame_index, frame_count, profile, seed
            )
            client.simPause(False)
            for vehicle_name, (position, yaw) in trajectories.items():
                client.simSetVehiclePose(
                    airsim.Pose(
                        airsim.Vector3r(*position.tolist()),
                        airsim.euler_to_quaternion(0.0, 0.0, float(yaw)),
                    ),
                    True,
                    vehicle_name=vehicle_name,
                )
            client.simSetCameraPose(
                "uav",
                airsim.Pose(
                    airsim.Vector3r(*camera_position.tolist()),
                    airsim.euler_to_quaternion(*camera_angles),
                ),
                vehicle_name="ServedVehicle",
            )
            time.sleep(0.04)
            client.simPause(True)

            camera_info = client.simGetCameraInfo("uav", vehicle_name="ServedVehicle")
            camera_position_ned, camera_quaternion = _global_camera_pose_ned(client, camera_info)
            world_body, body_camera = airsim_camera_to_project_frames(
                camera_position_ned, camera_quaternion
            )
            width_px, height_px = 640, 480
            focal_px = 0.5 * width_px / np.tan(0.5 * np.deg2rad(float(camera_info.fov)))
            camera = PinholeCamera(focal_px, focal_px, width_px / 2.0, height_px / 2.0)
            responses = client.simGetImages(
                [
                    airsim.ImageRequest("uav", airsim.ImageType.Scene, False, True),
                    airsim.ImageRequest("uav", airsim.ImageType.Segmentation, False, True),
                ],
                vehicle_name="ServedVehicle",
            )
            if len(responses) != 2 or any(
                response.width != width_px or response.height != height_px for response in responses
            ):
                raise RuntimeError("Cosys-AirSim returned malformed image responses")
            scene_path = (images_dir / f"frame_{frame_index:04d}.png").resolve()
            segmentation_path = (
                segmentation_dir / f"frame_{frame_index:04d}.png"
            ).resolve()
            scene_path.write_bytes(bytes(responses[0].image_data_uint8))
            segmentation_path.write_bytes(bytes(responses[1].image_data_uint8))
            segmentation_rgb = np.asarray(
                Image.open(io.BytesIO(bytes(responses[1].image_data_uint8))).convert("RGB")
            )

            transforms: dict[str, object] = {}
            antenna_positions: dict[str, list[float]] = {}
            antenna_pixels: dict[str, list[float]] = {}
            boxes: dict[str, list[float]] = {}
            for vehicle_name in VEHICLE_NAMES:
                pose = client.simGetObjectPose(vehicle_name)
                world_vehicle = ned_frd_pose_to_enu_transform(
                    _vector(pose.position), _quaternion(pose.orientation)
                )
                antenna_world = world_vehicle.apply_point(ANTENNA_LEVER_ARM_FRD_M)
                antenna_pixel = camera.project_world(
                    antenna_world, world_body.compose(body_camera)
                )
                transforms[vehicle_name] = {
                    "R_WV": world_vehicle.rotation.tolist(),
                    "t_W_V": world_vehicle.translation.tolist(),
                }
                antenna_positions[vehicle_name] = antenna_world.tolist()
                antenna_pixels[vehicle_name] = antenna_pixel.tolist()
                boxes[vehicle_name] = _bbox_for_color(
                    segmentation_rgb, instance_colors[vehicle_name]
                )

            candidate_names = list(VEHICLE_NAMES)
            shift = (frame_index + seed) % len(candidate_names)
            candidate_names = candidate_names[shift:] + candidate_names[:shift]
            served_index = candidate_names.index("ServedVehicle")
            record = {
                "sequence_id": sequence_id,
                "split": split,
                "frame_index": frame_index,
                "timestamp_s": float(responses[0].time_stamp) * 1e-9,
                "segmentation_timestamp_s": float(responses[1].time_stamp) * 1e-9,
                "image_pair_delta_s": abs(
                    float(responses[1].time_stamp - responses[0].time_stamp) * 1e-9
                ),
                "image_path": str(scene_path),
                "segmentation_path": str(segmentation_path),
                "camera_intrinsics": {
                    "fx_px": float(camera.fx_px),
                    "fy_px": float(camera.fy_px),
                    "cx_px": float(camera.cx_px),
                    "cy_px": float(camera.cy_px),
                    "width_px": width_px,
                    "height_px": height_px,
                },
                "transform_world_body": _transform_record(world_body),
                "transform_body_camera": {
                    "R_BC": body_camera.rotation.tolist(),
                    "t_B_C": body_camera.translation.tolist(),
                },
                "antenna_lever_arm_vehicle_m": ANTENNA_LEVER_ARM_FRD_M.tolist(),
                "evaluation_only": {
                    "served_target_gt_id": "ServedVehicle",
                    "vehicle_transforms": transforms,
                    "antenna_phase_centers_world_m": antenna_positions,
                    "antenna_pixels_uv": antenna_pixels,
                    "candidate_gt_boxes_xyxy": [boxes[name] for name in candidate_names],
                    "candidate_vehicle_names": candidate_names,
                    "served_candidate_index": served_index,
                },
                "source": "Cosys-AirSim 3.4.1 / UE 5.8 Blocks",
                "pose_capture_mode": "simulation paused before pose and image RPC batch",
            }
            records.append(record)
            if (frame_index + 1) % 25 == 0 or frame_index + 1 == frame_count:
                print(f"{sequence_id}: captured={frame_index + 1}/{frame_count}", flush=True)
    finally:
        client.simPause(False)

    manifest = output_dir / "manifest.jsonl"
    temporary_manifest = output_dir / "manifest.jsonl.tmp"
    temporary_manifest.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest)
    metadata = {
        "sequence_id": sequence_id,
        "split": split,
        "seed": seed,
        "profile": profile,
        "frame_count": frame_count,
        "world_frame": "ENU",
        "body_frame": "FRD, coincident with AirSim external camera",
        "camera_frame": "OpenCV x-right/y-down/z-forward",
        "vehicle_frame": "FRD",
        "airsim_source_frame": "NED",
        "antenna_lever_arm_vehicle_m": ANTENNA_LEVER_ARM_FRD_M.tolist(),
        "gt_scope": "evaluation_only",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run(config_path: Path, sequence_ids: list[str] | None, output_root: Path | None) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("FULL generator requires mode: full_localization")
    phase = config["f0"]
    profiles = phase["sequence_profiles"]
    selected = sequence_ids or list(profiles)
    unknown = set(selected) - set(profiles)
    if unknown:
        raise ValueError(f"unknown sequences: {sorted(unknown)}")
    root = output_root or Path(phase["output_root"])
    frame_count = int(phase["frames_per_sequence"])
    base_seed = int(config["seed"])
    for sequence_id in selected:
        sequence_dir = root / sequence_id
        if _complete_sequence(sequence_dir, frame_count):
            print(f"{sequence_id}: already complete, skipping")
            continue
        manifest = generate_sequence(
            sequence_dir,
            sequence_id,
            _sequence_split(phase, sequence_id),
            profiles[sequence_id],
            frame_count,
            base_seed + int(sequence_id[-2:]),
            str(phase["airsim_host"]),
            int(phase["airsim_port"]),
        )
        print(f"{sequence_id}: manifest={manifest.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.sequences, arguments.output_root)

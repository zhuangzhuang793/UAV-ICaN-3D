"""Generate the small synchronized Cosys-AirSim dataset required by Quick Gate 5."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from uav_ican_3d.geometry import PinholeCamera
from uav_ican_3d.simulation import (
    ROTATION_ENU_NED,
    airsim_camera_to_project_frames,
    ned_frd_pose_to_enu_transform,
    quaternion_xyzw_to_rotation,
)


VEHICLE_NAMES = ("ServedVehicle", "DistractorA", "DistractorB")
SEGMENTATION_IDS = {"ServedVehicle": 202, "DistractorA": 200, "DistractorB": 201}
ANTENNA_LEVER_ARM_FRD_M = np.array([0.0, 0.0, -1.2])


def _vector(value: object) -> np.ndarray:
    return np.array([value.x_val, value.y_val, value.z_val], dtype=float)


def _quaternion(value: object) -> np.ndarray:
    return np.array([value.x_val, value.y_val, value.z_val, value.w_val], dtype=float)


def _global_camera_pose_ned(client: object, camera_info: object) -> tuple[np.ndarray, np.ndarray]:
    """Convert AirSim's home-local external-camera pose to the level's common NED frame."""

    vehicle_local = client.simGetVehiclePose(vehicle_name="ServedVehicle")
    vehicle_global = client.simGetObjectPose("ServedVehicle")
    rotation_home_local = quaternion_xyzw_to_rotation(_quaternion(vehicle_local.orientation))
    rotation_global_vehicle = quaternion_xyzw_to_rotation(_quaternion(vehicle_global.orientation))
    rotation_global_home = rotation_global_vehicle @ rotation_home_local.T
    translation_global_home = _vector(vehicle_global.position) - (
        rotation_global_home @ _vector(vehicle_local.position)
    )
    rotation_home_camera = quaternion_xyzw_to_rotation(_quaternion(camera_info.pose.orientation))
    rotation_global_camera = rotation_global_home @ rotation_home_camera
    translation_global_camera = (
        rotation_global_home @ _vector(camera_info.pose.position) + translation_global_home
    )
    quaternion_placeholder = _rotation_to_quaternion_xyzw(rotation_global_camera)
    return translation_global_camera, quaternion_placeholder


def _rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Stable rotation-to-quaternion conversion used only at the simulator boundary."""

    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        diagonal_term = (
            1.0
            + matrix[index, index]
            - matrix[next_index, next_index]
            - matrix[last_index, last_index]
        )
        scale = 2.0 * np.sqrt(diagonal_term)
        quaternion = np.zeros(4)
        quaternion[index] = 0.25 * scale
        quaternion[next_index] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        quaternion[last_index] = (matrix[last_index, index] + matrix[index, last_index]) / scale
        quaternion[3] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
    return quaternion / np.linalg.norm(quaternion)


def _transform_record(transform: object) -> dict[str, list[float] | list[list[float]]]:
    return {
        "R_WB": transform.rotation.tolist(),
        "t_W_B": transform.translation.tolist(),
    }


def _bbox_for_color(segmentation_rgb: np.ndarray, color_rgb: np.ndarray) -> list[float]:
    mask = np.all(segmentation_rgb == color_rgb.reshape(1, 1, 3), axis=2)
    rows, columns = np.nonzero(mask)
    if rows.size < 40:
        raise RuntimeError(f"segmentation color {color_rgb.tolist()} has only {rows.size} pixels")
    return [
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    ]


def _set_instance_ids(client: object, airsim: object) -> dict[str, np.ndarray]:
    objects = client.simListInstanceSegmentationObjects()
    colors: dict[str, np.ndarray] = {}
    color_map = airsim.load_colormap()
    for vehicle_name in VEHICLE_NAMES:
        roots = [
            name
            for name in objects
            if name.startswith(f"{vehicle_name}_") and "camera" not in name.lower()
        ]
        if len(roots) != 1:
            raise RuntimeError(f"expected one segmentation root for {vehicle_name}, got {roots}")
        object_id = SEGMENTATION_IDS[vehicle_name]
        if not client.simSetSegmentationObjectID(roots[0], object_id, False):
            raise RuntimeError(f"failed to set segmentation ID for {roots[0]}")
        colors[vehicle_name] = np.asarray(color_map[object_id], dtype=np.uint8)
    return colors


def _trajectory(frame_index: int, frame_count: int) -> dict[str, tuple[np.ndarray, float]]:
    angle = 2.0 * np.pi * frame_index / frame_count
    return {
        "ServedVehicle": (
            np.array([2.5 * np.sin(angle), 1.8 * np.cos(angle), 0.88]),
            0.25 * np.sin(angle),
        ),
        "DistractorA": (
            np.array([1.3 * np.cos(angle + 0.4), 1.1 * np.sin(angle + 0.4), 0.88]),
            0.44 + 0.15 * np.cos(angle),
        ),
        "DistractorB": (
            np.array([1.0 * np.sin(angle - 0.8), 1.4 * np.cos(angle - 0.8), 0.88]),
            -0.52 + 0.12 * np.sin(angle),
        ),
    }


def generate(output_dir: Path, frame_count: int, host: str, port: int) -> Path:
    try:
        import cosysairsim as airsim
    except ImportError as error:
        raise RuntimeError("install the Cosys-AirSim 3.4.1 client wheel first") from error

    if frame_count < 10:
        raise ValueError("Gate 5 requires at least ten synchronized frames")
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
            client.simPause(False)
            for vehicle_name, (position, yaw) in _trajectory(frame_index, frame_count).items():
                pose = airsim.Pose(
                    airsim.Vector3r(*position.tolist()),
                    airsim.euler_to_quaternion(0.0, 0.0, float(yaw)),
                )
                client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
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
            if len(responses) != 2 or any(response.width != width_px for response in responses):
                raise RuntimeError("Cosys-AirSim returned malformed image responses")
            scene_path = (images_dir / f"frame_{frame_index:03d}.png").resolve()
            segmentation_path = (segmentation_dir / f"frame_{frame_index:03d}.png").resolve()
            scene_path.write_bytes(bytes(responses[0].image_data_uint8))
            segmentation_path.write_bytes(bytes(responses[1].image_data_uint8))
            segmentation_rgb = np.asarray(
                Image.open(io.BytesIO(bytes(responses[1].image_data_uint8))).convert("RGB")
            )

            vehicle_transforms = {}
            antenna_positions = {}
            antenna_pixels = {}
            boxes = {}
            for vehicle_name in VEHICLE_NAMES:
                vehicle_pose = client.simGetObjectPose(vehicle_name)
                world_vehicle = ned_frd_pose_to_enu_transform(
                    _vector(vehicle_pose.position), _quaternion(vehicle_pose.orientation)
                )
                antenna_world = world_vehicle.apply_point(ANTENNA_LEVER_ARM_FRD_M)
                antenna_pixel = camera.project_world(
                    antenna_world, world_body.compose(body_camera)
                )
                vehicle_transforms[vehicle_name] = world_vehicle
                antenna_positions[vehicle_name] = antenna_world
                antenna_pixels[vehicle_name] = antenna_pixel
                boxes[vehicle_name] = _bbox_for_color(
                    segmentation_rgb, instance_colors[vehicle_name]
                )
                x_min, y_min, x_max, y_max = boxes[vehicle_name]
                if not (x_min <= antenna_pixel[0] <= x_max and y_min <= antenna_pixel[1] <= y_max):
                    raise RuntimeError(
                        f"projected antenna for {vehicle_name} lies outside its GT box: "
                        f"pixel={antenna_pixel.tolist()} box={boxes[vehicle_name]}"
                    )

            candidate_names = list(VEHICLE_NAMES)
            shift = frame_index % len(candidate_names)
            candidate_names = candidate_names[shift:] + candidate_names[:shift]
            served_index = candidate_names.index("ServedVehicle")
            served_vehicle = vehicle_transforms["ServedVehicle"]
            record = {
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
                "transform_world_vehicle": {
                    "R_WV": served_vehicle.rotation.tolist(),
                    "t_W_V": served_vehicle.translation.tolist(),
                },
                "vehicle_reference_world_m": served_vehicle.translation.tolist(),
                "antenna_lever_arm_vehicle_m": ANTENNA_LEVER_ARM_FRD_M.tolist(),
                "antenna_phase_center_world_m": antenna_positions["ServedVehicle"].tolist(),
                "true_ue_pixel_uv": antenna_pixels["ServedVehicle"].tolist(),
                "served_gt_box_xyxy": boxes["ServedVehicle"],
                "candidate_gt_boxes_xyxy": [boxes[name] for name in candidate_names],
                "candidate_vehicle_names": candidate_names,
                "served_candidate_index": served_index,
                "source": "Cosys-AirSim 3.4.1 / UE 5.8 Blocks, instance-segmentation GT",
                "pose_capture_mode": "simulation paused before pose and image RPC batch",
            }
            records.append(record)
            print(
                f"captured={frame_index + 1}/{frame_count} "
                f"served_pixel={np.round(antenna_pixels['ServedVehicle'], 2).tolist()}"
            )
    finally:
        client.simPause(False)

    manifest = output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    metadata = {
        "generator": str(Path(__file__).resolve()),
        "frame_count": frame_count,
        "world_frame": "ENU",
        "body_frame": "FRD, coincident with AirSim external camera",
        "camera_frame": "OpenCV x-right/y-down/z-forward",
        "vehicle_frame": "FRD",
        "airsim_source_frame": "NED",
        "antenna_lever_arm_vehicle_m": ANTENNA_LEVER_ARM_FRD_M.tolist(),
        "gt_boxes": "exact instance-segmentation colors assigned through the simulator RPC",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest={manifest.resolve()} records={len(records)}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/synchronized_quick"))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    arguments = parser.parse_args()
    generate(arguments.output_dir, arguments.frames, arguments.host, arguments.port)

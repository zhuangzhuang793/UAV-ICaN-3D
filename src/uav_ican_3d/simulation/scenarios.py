"""Small deterministic geometries shared by quick-gate experiments."""

from __future__ import annotations

import numpy as np

from uav_ican_3d.geometry import PinholeCamera, RigidTransform


def complementarity_scene(
    phase_config: dict, altitude_m: float, horizontal_distance_m: float
) -> tuple[np.ndarray, RigidTransform, RigidTransform, RigidTransform, PinholeCamera]:
    """Construct the fixed-mount Phase 3/4 radio-visual geometry."""

    target_world = np.array([horizontal_distance_m, 0.0, 0.0])
    transform_world_body = RigidTransform(
        np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, altitude_m])
    )
    transform_body_array = RigidTransform.identity()
    transform_body_camera = RigidTransform(
        np.asarray(phase_config["camera_rotation_body"], dtype=float), np.zeros(3)
    )
    camera_config = phase_config["camera"]
    camera = PinholeCamera(
        camera_config["fx_px"],
        camera_config["fy_px"],
        camera_config["cx_px"],
        camera_config["cy_px"],
    )
    return (
        target_world,
        transform_world_body,
        transform_body_array,
        transform_body_camera,
        camera,
    )


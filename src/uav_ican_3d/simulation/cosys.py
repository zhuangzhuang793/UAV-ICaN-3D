"""Coordinate adapters for Cosys-AirSim's local NED API.

Cosys-AirSim reports vehicle and camera poses in a North-East-Down world and uses
Forward-Right-Down camera coordinates.  The research code uses ENU for the world and OpenCV
coordinates for projection, so the two fixed basis changes live here instead of being repeated in
data-generation scripts.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_ican_3d.geometry import RigidTransform


FloatArray = NDArray[np.float64]

ROTATION_ENU_NED = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float
)
"""Map a free vector from AirSim NED into the project ENU world."""

ROTATION_AIRSIM_CAMERA_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
)
"""Map OpenCV camera coordinates into AirSim camera FRD coordinates."""


def quaternion_xyzw_to_rotation(quaternion_xyzw: ArrayLike) -> FloatArray:
    """Return the active local-to-parent rotation for an ``[x, y, z, w]`` quaternion."""

    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must be a finite vector with shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion_xyzw must have nonzero norm")
    x_coord, y_coord, z_coord, scalar = quaternion / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y_coord**2 + z_coord**2),
                2.0 * (x_coord * y_coord - z_coord * scalar),
                2.0 * (x_coord * z_coord + y_coord * scalar),
            ],
            [
                2.0 * (x_coord * y_coord + z_coord * scalar),
                1.0 - 2.0 * (x_coord**2 + z_coord**2),
                2.0 * (y_coord * z_coord - x_coord * scalar),
            ],
            [
                2.0 * (x_coord * z_coord - y_coord * scalar),
                2.0 * (y_coord * z_coord + x_coord * scalar),
                1.0 - 2.0 * (x_coord**2 + y_coord**2),
            ],
        ],
        dtype=float,
    )


def ned_frd_pose_to_enu_transform(
    position_ned_m: ArrayLike, quaternion_xyzw: ArrayLike
) -> RigidTransform:
    """Convert an AirSim NED/FRD pose into a project ``T_WB`` transform."""

    position = np.asarray(position_ned_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_ned_m must be a finite vector with shape (3,)")
    rotation_ned_body = quaternion_xyzw_to_rotation(quaternion_xyzw)
    return RigidTransform(
        ROTATION_ENU_NED @ rotation_ned_body,
        ROTATION_ENU_NED @ position,
    )


def airsim_camera_to_project_frames(
    position_ned_m: ArrayLike, quaternion_xyzw: ArrayLike
) -> tuple[RigidTransform, RigidTransform]:
    """Represent an AirSim camera pose as project ``T_WB`` and fixed ``T_BC``.

    The virtual UAV body is chosen coincident with the AirSim camera FRD frame.  This is a valid
    calibrated rig with zero translation and makes the frame conversion explicit.
    """

    world_body = ned_frd_pose_to_enu_transform(position_ned_m, quaternion_xyzw)
    body_camera = RigidTransform(ROTATION_AIRSIM_CAMERA_OPENCV, np.zeros(3))
    return world_body, body_camera

"""Three-dimensional equivalent-range and 2-D AoA measurement model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_ican_3d.geometry import RigidTransform, shared_sensor_relative_geometry


FloatArray = NDArray[np.float64]


class SingularGeometryError(ValueError):
    """Raised when a requested measurement is not locally observable."""


def _observation_from_relative(
    relative: FloatArray, geometry_tolerance_m: float
) -> FloatArray:
    x_coord, y_coord, z_coord = relative
    horizontal_range = float(np.hypot(x_coord, y_coord))
    range_m = float(np.linalg.norm(relative))
    if range_m <= geometry_tolerance_m:
        raise SingularGeometryError("range and angles are undefined at the array origin")
    if horizontal_range <= geometry_tolerance_m:
        raise SingularGeometryError("azimuth is undefined on the array z-axis")
    return np.array(
        [range_m, np.arctan2(y_coord, x_coord), np.arctan2(z_coord, horizontal_range)],
        dtype=float,
    )


def _jacobian_from_relative(relative: FloatArray, geometry_tolerance_m: float) -> FloatArray:
    x_coord, y_coord, z_coord = relative
    horizontal_squared = float(x_coord * x_coord + y_coord * y_coord)
    horizontal_range = float(np.sqrt(horizontal_squared))
    range_squared = float(horizontal_squared + z_coord * z_coord)
    range_m = float(np.sqrt(range_squared))
    if range_m <= geometry_tolerance_m:
        raise SingularGeometryError("RF Jacobian is undefined at the array origin")
    if horizontal_range <= geometry_tolerance_m:
        raise SingularGeometryError("azimuth Jacobian is undefined on the array z-axis")
    return np.array(
        [
            [x_coord / range_m, y_coord / range_m, z_coord / range_m],
            [-y_coord / horizontal_squared, x_coord / horizontal_squared, 0.0],
            [
                -z_coord * x_coord / (range_squared * horizontal_range),
                -z_coord * y_coord / (range_squared * horizontal_range),
                horizontal_range / range_squared,
            ],
        ],
        dtype=float,
    )


def _relative_position_array(
    position_world: ArrayLike, transform_world_array: RigidTransform
) -> FloatArray:
    position = np.asarray(position_world, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_world must be a finite vector with shape (3,)")
    return transform_world_array.inverse().apply_point(position)


def predict_rf_observation(
    position_world: ArrayLike,
    transform_world_array: RigidTransform,
    geometry_tolerance_m: float = 1e-9,
) -> FloatArray:
    """Return ``[range_m, azimuth_rad, elevation_rad]`` for a UE position."""

    relative = _relative_position_array(position_world, transform_world_array)
    return _observation_from_relative(relative, geometry_tolerance_m)


def rf_position_jacobian(
    position_world: ArrayLike,
    transform_world_array: RigidTransform,
    geometry_tolerance_m: float = 1e-9,
) -> FloatArray:
    """Analytic Jacobian of RF observation with respect to world UE position."""

    relative = _relative_position_array(position_world, transform_world_array)
    jacobian_relative = _jacobian_from_relative(relative, geometry_tolerance_m)
    rotation_array_world = transform_world_array.rotation.T
    return jacobian_relative @ rotation_array_world


def rf_observation_and_shared_pose_jacobians(
    position_world: ArrayLike,
    transform_world_body: RigidTransform,
    transform_body_array: RigidTransform,
    geometry_tolerance_m: float = 1e-9,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return RF prediction, ``H_RF,p``, and shared ``H_RF,xi``."""

    relative, derivative_point, derivative_pose = shared_sensor_relative_geometry(
        position_world, transform_world_body, transform_body_array
    )
    relative_jacobian = _jacobian_from_relative(relative, geometry_tolerance_m)
    return (
        _observation_from_relative(relative, geometry_tolerance_m),
        relative_jacobian @ derivative_point,
        relative_jacobian @ derivative_pose,
    )


def wrap_angle(angle_rad: ArrayLike) -> FloatArray:
    """Wrap angle residuals to ``[-pi, pi)``."""

    angle = np.asarray(angle_rad, dtype=float)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

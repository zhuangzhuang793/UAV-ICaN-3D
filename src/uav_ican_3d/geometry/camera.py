"""OpenCV pinhole-camera projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .transforms import RigidTransform, shared_sensor_relative_geometry


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PinholeCamera:
    """Undistorted pinhole intrinsics for an OpenCV camera frame.

    Camera coordinates use x right, y down, and z forward. Projection is only
    defined for points with strictly positive camera-frame depth.
    """

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self) -> None:
        values = np.array([self.fx_px, self.fy_px, self.cx_px, self.cy_px], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("camera intrinsics must be finite")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("camera focal lengths must be positive")

    @property
    def matrix(self) -> FloatArray:
        return np.array(
            [[self.fx_px, 0.0, self.cx_px], [0.0, self.fy_px, self.cy_px], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def project_camera(self, point_camera: ArrayLike) -> FloatArray:
        point = np.asarray(point_camera, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point_camera must be a finite vector with shape (3,)")
        if point[2] <= 0.0:
            raise ValueError("camera projection requires positive depth")
        return np.array(
            [
                self.fx_px * point[0] / point[2] + self.cx_px,
                self.fy_px * point[1] / point[2] + self.cy_px,
            ],
            dtype=float,
        )

    def project_world(self, point_world: ArrayLike, transform_world_camera: RigidTransform) -> FloatArray:
        """Project a world point using ``T_WC`` (camera frame to world frame)."""

        point_camera = transform_world_camera.inverse().apply_point(point_world)
        return self.project_camera(point_camera)

    def projection_jacobian(self, point_camera: ArrayLike) -> FloatArray:
        """Jacobian of pixel projection with respect to a camera-frame point."""

        point = np.asarray(point_camera, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point_camera must be a finite vector with shape (3,)")
        x_coord, y_coord, depth = point
        if depth <= 0.0:
            raise ValueError("camera projection requires positive depth")
        return np.array(
            [
                [self.fx_px / depth, 0.0, -self.fx_px * x_coord / depth**2],
                [0.0, self.fy_px / depth, -self.fy_px * y_coord / depth**2],
            ],
            dtype=float,
        )

    def observation_and_shared_pose_jacobians(
        self,
        point_world: ArrayLike,
        transform_world_body: RigidTransform,
        transform_body_camera: RigidTransform,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return pixel prediction, ``H_cam,p``, and shared ``H_cam,xi``."""

        point_camera, derivative_point, derivative_pose = shared_sensor_relative_geometry(
            point_world, transform_world_body, transform_body_camera
        )
        projection_jacobian = self.projection_jacobian(point_camera)
        return (
            self.project_camera(point_camera),
            projection_jacobian @ derivative_point,
            projection_jacobian @ derivative_pose,
        )

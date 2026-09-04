"""OpenCV pinhole-camera projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .transforms import RigidTransform


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


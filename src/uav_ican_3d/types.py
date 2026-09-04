"""Validated data interfaces shared by localization, vision, prediction, and control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry.transforms import RigidTransform


FloatArray = NDArray[np.float64]


def _finite_vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return array.copy()


def _covariance(value: ArrayLike, size: int, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix with shape ({size}, {size})")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(matrix).min() < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix.copy()


def _timestamp(value: float) -> float:
    timestamp = float(value)
    if not np.isfinite(timestamp):
        raise ValueError("timestamp_s must be finite")
    return timestamp


@dataclass(frozen=True)
class PoseBelief:
    """Nominal UAV body pose ``T_WB`` with a 6-D shared pose covariance.

    The tangent ordering is ``[position_W_m, rotation_B_rad]``. The rotation
    perturbation convention is fixed in ``docs/COORDINATES.md`` before it is
    used by estimators in later phases.
    """

    transform_world_body: RigidTransform
    covariance: FloatArray
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "covariance", _covariance(self.covariance, 6, "covariance"))
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s))


@dataclass(frozen=True)
class RFObservation:
    """Equivalent range, azimuth, and elevation observation in SI/radians."""

    range_m: float
    azimuth_rad: float
    elevation_rad: float
    covariance: FloatArray
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        measurement = np.array([self.range_m, self.azimuth_rad, self.elevation_rad])
        if not np.all(np.isfinite(measurement)) or self.range_m < 0.0:
            raise ValueError("RF measurement must be finite and range_m must be nonnegative")
        object.__setattr__(self, "covariance", _covariance(self.covariance, 3, "covariance"))
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s))

    @property
    def vector(self) -> FloatArray:
        return np.array([self.range_m, self.azimuth_rad, self.elevation_rad], dtype=float)


@dataclass(frozen=True)
class VisualObservation:
    """Pixel observation in ``[u, v]`` order with a full 2-D covariance."""

    pixel_uv: FloatArray
    covariance: FloatArray
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixel_uv", _finite_vector(self.pixel_uv, 2, "pixel_uv"))
        object.__setattr__(self, "covariance", _covariance(self.covariance, 2, "covariance"))
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s))


@dataclass(frozen=True)
class BeliefState:
    """Six-state UE belief ``[x, y, z, vx, vy, vz]`` in world ENU coordinates."""

    mean: FloatArray
    covariance: FloatArray
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _finite_vector(self.mean, 6, "mean"))
        object.__setattr__(self, "covariance", _covariance(self.covariance, 6, "covariance"))
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s))


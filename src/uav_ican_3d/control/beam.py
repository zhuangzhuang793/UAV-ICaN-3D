"""Uncertainty-aware 3-D beam selection for a 4x4 half-wavelength UPA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_ican_3d.geometry import RigidTransform
from uav_ican_3d.localization import predict_rf_observation, rf_position_jacobian


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AngularBelief:
    mean_azimuth_elevation_rad: FloatArray
    covariance_rad2: FloatArray


@dataclass(frozen=True)
class BeamSelection:
    azimuth_rad: float
    elevation_rad: float
    aperture_size: int
    robust_gain: float


def propagate_position_to_angles(
    position_world_m: ArrayLike,
    position_covariance_world_m2: ArrayLike,
    transform_world_array: RigidTransform,
) -> AngularBelief:
    """Propagate a position belief to azimuth/elevation using the analytic RF Jacobian."""

    covariance = np.asarray(position_covariance_world_m2, dtype=float)
    if covariance.shape != (3, 3):
        raise ValueError("position covariance must have shape (3, 3)")
    observation = predict_rf_observation(position_world_m, transform_world_array)
    jacobian = rf_position_jacobian(position_world_m, transform_world_array)[1:]
    angular_covariance = jacobian @ covariance @ jacobian.T
    angular_covariance = 0.5 * (angular_covariance + angular_covariance.T)
    return AngularBelief(observation[1:].copy(), angular_covariance)


def _array_gain(
    sample_angles: FloatArray,
    steering_azimuth: float,
    steering_elevation: float,
    aperture_size: int,
) -> FloatArray:
    rows, columns = np.meshgrid(
        np.arange(aperture_size), np.arange(aperture_size), indexing="ij"
    )
    element_y = rows.reshape(-1)
    element_z = columns.reshape(-1)
    sample_y = np.cos(sample_angles[:, 1]) * np.sin(sample_angles[:, 0])
    sample_z = np.sin(sample_angles[:, 1])
    steering_y = np.cos(steering_elevation) * np.sin(steering_azimuth)
    steering_z = np.sin(steering_elevation)
    phase = np.pi * (
        (sample_y[:, None] - steering_y) * element_y[None, :]
        + (sample_z[:, None] - steering_z) * element_z[None, :]
    )
    response = np.sum(np.exp(1j * phase), axis=1)
    return np.abs(response) ** 2 / aperture_size**2


def select_robust_beam(
    belief: AngularBelief,
    azimuth_grid_rad: ArrayLike,
    elevation_grid_rad: ArrayLike,
    rng: np.random.Generator,
    sample_count: int = 256,
    lower_quantile: float = 0.10,
) -> BeamSelection:
    """Select a narrow 4x4 or broad 2x2 beam by lower-tail array gain."""

    if sample_count < 16 or not 0.0 < lower_quantile < 0.5:
        raise ValueError("invalid robust beam sampling configuration")
    samples = rng.multivariate_normal(
        belief.mean_azimuth_elevation_rad,
        belief.covariance_rad2 + np.eye(2) * 1e-12,
        size=sample_count,
    )
    best: BeamSelection | None = None
    for aperture_size in (2, 4):
        for azimuth in np.asarray(azimuth_grid_rad, dtype=float):
            for elevation in np.asarray(elevation_grid_rad, dtype=float):
                gains = _array_gain(samples, azimuth, elevation, aperture_size)
                robust_gain = float(np.quantile(gains, lower_quantile))
                candidate = BeamSelection(
                    float(azimuth), float(elevation), aperture_size, robust_gain
                )
                if best is None or candidate.robust_gain > best.robust_gain:
                    best = candidate
    assert best is not None
    return best

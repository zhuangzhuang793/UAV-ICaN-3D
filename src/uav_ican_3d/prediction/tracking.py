"""Constant-velocity belief tracking between localization and prediction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from uav_ican_3d.types import BeliefState


def update_position_belief(
    prior: BeliefState,
    measured_position_world_m: ArrayLike,
    measurement_covariance_world_m2: ArrayLike,
    timestamp_s: float,
    process_position_variance: float = 0.02,
    process_velocity_variance: float = 0.08,
) -> BeliefState:
    """Predict and update a 6-D constant-velocity Gaussian belief from a position belief."""

    dt_s = float(timestamp_s - prior.timestamp_s)
    if dt_s <= 0.0:
        raise ValueError("timestamp must increase")
    measurement = np.asarray(measured_position_world_m, dtype=float)
    measurement_covariance = np.asarray(measurement_covariance_world_m2, dtype=float)
    if measurement.shape != (3,) or measurement_covariance.shape != (3, 3):
        raise ValueError("position measurement and covariance have wrong shapes")
    transition = np.block(
        [[np.eye(3), np.eye(3) * dt_s], [np.zeros((3, 3)), np.eye(3)]]
    )
    process_covariance = np.diag(
        [process_position_variance] * 3 + [process_velocity_variance] * 3
    )
    predicted_mean = transition @ prior.mean
    predicted_covariance = transition @ prior.covariance @ transition.T + process_covariance
    observation = np.column_stack((np.eye(3), np.zeros((3, 3))))
    innovation_covariance = (
        observation @ predicted_covariance @ observation.T + measurement_covariance
    )
    gain = predicted_covariance @ observation.T @ np.linalg.inv(innovation_covariance)
    updated_mean = predicted_mean + gain @ (measurement - observation @ predicted_mean)
    updated_covariance = (np.eye(6) - gain @ observation) @ predicted_covariance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return BeliefState(updated_mean, updated_covariance, timestamp_s)

"""Observation-level nonlinear MAP estimator for target position and shared UAV pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body
from uav_ican_3d.types import RFObservation, VisualObservation

from .rf import predict_rf_observation, wrap_angle


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MAPEstimate:
    position_world_m: FloatArray
    pose_delta: FloatArray
    joint_covariance: FloatArray
    position_covariance: FloatArray
    cost: float
    optimality: float
    function_evaluations: int


def position_from_rf_observation(
    observation: RFObservation, transform_world_array_nominal: RigidTransform
) -> FloatArray:
    """Invert the nominal spherical RF model to produce a GT-free initializer."""

    range_m, azimuth, elevation = observation.vector
    cosine_elevation = np.cos(elevation)
    relative_array = np.array(
        [
            range_m * cosine_elevation * np.cos(azimuth),
            range_m * cosine_elevation * np.sin(azimuth),
            range_m * np.sin(elevation),
        ]
    )
    return transform_world_array_nominal.apply_point(relative_array)


def _cholesky(covariance: ArrayLike, size: int, name: str) -> FloatArray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (size, size) or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be a symmetric ({size}, {size}) matrix")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error


def estimate_position_map(
    rf_observation: RFObservation,
    nominal_world_body: RigidTransform,
    transform_body_array: RigidTransform,
    pose_covariance: ArrayLike,
    camera: PinholeCamera | None = None,
    transform_body_camera: RigidTransform | None = None,
    visual_observation: VisualObservation | None = None,
    initial_position_world_m: ArrayLike | None = None,
    max_function_evaluations: int = 250,
    gradient_tolerance: float = 1e-10,
    parameter_tolerance: float = 1e-10,
    cost_tolerance: float = 1e-10,
) -> MAPEstimate:
    """Estimate target position and the single shared pose perturbation.

    Supplying a visual observation requires both camera intrinsics and its body extrinsic. Ground
    truth is deliberately absent from this API.
    """

    camera_inputs = (camera, transform_body_camera, visual_observation)
    if any(value is not None for value in camera_inputs) and any(
        value is None for value in camera_inputs
    ):
        raise ValueError("camera, transform_body_camera, and visual_observation are all-or-none")
    rf_cholesky = _cholesky(rf_observation.covariance, 3, "RF covariance")
    pose_cholesky = _cholesky(pose_covariance, 6, "pose covariance")
    camera_cholesky = (
        _cholesky(visual_observation.covariance, 2, "camera covariance")
        if visual_observation is not None
        else None
    )

    nominal_world_array = nominal_world_body.compose(transform_body_array)
    if initial_position_world_m is None:
        initial_position = position_from_rf_observation(
            rf_observation, nominal_world_array
        )
    else:
        initial_position = np.asarray(initial_position_world_m, dtype=float)
        if initial_position.shape != (3,) or not np.all(np.isfinite(initial_position)):
            raise ValueError("initial_position_world_m must be a finite vector with shape (3,)")
    initial_state = np.concatenate((initial_position, np.zeros(6)))

    def residual(state: FloatArray) -> FloatArray:
        position = state[:3]
        pose_delta = state[3:]
        world_body = perturb_world_body(nominal_world_body, pose_delta)
        predicted_rf = predict_rf_observation(position, world_body.compose(transform_body_array))
        rf_residual = predicted_rf - rf_observation.vector
        rf_residual[1:] = wrap_angle(rf_residual[1:])
        components = [np.linalg.solve(rf_cholesky, rf_residual)]
        if visual_observation is not None:
            assert camera is not None and transform_body_camera is not None
            assert camera_cholesky is not None
            predicted_pixel = camera.project_world(
                position, world_body.compose(transform_body_camera)
            )
            components.append(
                np.linalg.solve(camera_cholesky, predicted_pixel - visual_observation.pixel_uv)
            )
        components.append(np.linalg.solve(pose_cholesky, pose_delta))
        return np.concatenate(components)

    solution = least_squares(
        residual,
        initial_state,
        method="trf",
        x_scale="jac",
        max_nfev=max_function_evaluations,
        gtol=gradient_tolerance,
        xtol=parameter_tolerance,
        ftol=cost_tolerance,
    )
    if not solution.success:
        raise RuntimeError(f"MAP optimizer failed: {solution.message}")
    normal_matrix = solution.jac.T @ solution.jac
    eigenvalues = np.linalg.eigvalsh(normal_matrix)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        raise RuntimeError(f"MAP local Hessian is singular: {eigenvalues.tolist()}")
    joint_covariance = np.linalg.solve(normal_matrix, np.eye(9))
    joint_covariance = 0.5 * (joint_covariance + joint_covariance.T)
    return MAPEstimate(
        position_world_m=solution.x[:3].copy(),
        pose_delta=solution.x[3:].copy(),
        joint_covariance=joint_covariance,
        position_covariance=joint_covariance[:3, :3].copy(),
        cost=float(solution.cost),
        optimality=float(solution.optimality),
        function_evaluations=int(solution.nfev),
    )


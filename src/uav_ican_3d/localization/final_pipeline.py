"""Ground-truth-free online localization boundary for the formal F7 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_ican_3d.geometry import PinholeCamera, RigidTransform
from uav_ican_3d.types import RFObservation, VisualObservation
from uav_ican_3d.vision import associate_candidates, project_joint_rf_belief_to_image

from .map_estimator import MAPEstimate, estimate_position_fixed_pose, estimate_position_map
from .rf import predict_rf_observation, wrap_angle


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OnlineLocalizationResult:
    """Online estimates and association diagnostics; no evaluation truth is represented."""

    estimates: dict[str, MAPEstimate]
    primary_selected_index: int | None
    reduced_array_selected_index: int
    rf_image_mean_uv: FloatArray
    rf_image_covariance_uv: FloatArray


def _visual_ray_initializer(
    pixel_uv: ArrayLike,
    range_m: float,
    nominal_world_body: RigidTransform,
    transform_body_array: RigidTransform,
    transform_body_camera: RigidTransform,
    camera: PinholeCamera,
) -> FloatArray:
    """Intersect a candidate camera ray with the RF range sphere."""

    pixel = np.asarray(pixel_uv, dtype=float)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("pixel_uv must be a finite vector with shape (2,)")
    world_camera = nominal_world_body.compose(transform_body_camera)
    world_array = nominal_world_body.compose(transform_body_array)
    ray_camera = np.array(
        [
            (pixel[0] - camera.cx_px) / camera.fx_px,
            (pixel[1] - camera.cy_px) / camera.fy_px,
            1.0,
        ]
    )
    ray_world = world_camera.apply_vector(ray_camera)
    ray_world /= np.linalg.norm(ray_world)
    offset = world_camera.translation - world_array.translation
    linear = float(ray_world @ offset)
    discriminant = linear**2 - (float(offset @ offset) - float(range_m) ** 2)
    if discriminant >= 0.0:
        roots = (-linear - np.sqrt(discriminant), -linear + np.sqrt(discriminant))
        positive = [root for root in roots if root > 0.0]
        distance = max(positive) if positive else max(1.0, -linear)
    else:
        distance = max(1.0, -linear)
    return world_camera.translation + distance * ray_world


def _select_reduced_array_candidate(
    candidate_pixels_uv: FloatArray,
    rf_observation: RFObservation,
    nominal_world_body: RigidTransform,
    transform_body_array: RigidTransform,
    transform_body_camera: RigidTransform,
    camera: PinholeCamera,
) -> tuple[int, FloatArray]:
    """Select using only range, azimuth, and each detector candidate's visual ray."""

    initializers = [
        _visual_ray_initializer(
            pixel,
            rf_observation.range_m,
            nominal_world_body,
            transform_body_array,
            transform_body_camera,
            camera,
        )
        for pixel in candidate_pixels_uv
    ]
    world_array = nominal_world_body.compose(transform_body_array)
    scores = []
    for position in initializers:
        predicted = predict_rf_observation(position, world_array)
        residual = predicted[:2] - rf_observation.vector[:2]
        residual[1] = wrap_angle(residual[1])
        covariance = rf_observation.covariance[:2, :2]
        scores.append(float(residual @ np.linalg.solve(covariance, residual)))
    selected = int(np.argmin(scores))
    return selected, initializers[selected]


def localize_online(
    rf_observation: RFObservation,
    nominal_world_body: RigidTransform,
    transform_body_array: RigidTransform,
    pose_covariance: ArrayLike,
    camera: PinholeCamera,
    transform_body_camera: RigidTransform,
    candidate_pixels_uv: ArrayLike,
    visual_covariance_uv: ArrayLike,
    association_confidence: float,
    optimizer_options: dict[str, float | int] | None = None,
) -> OnlineLocalizationResult:
    """Run the frozen F7 methods using online observations only."""

    candidates = np.asarray(candidate_pixels_uv, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or candidates.shape[0] == 0:
        raise ValueError("candidate_pixels_uv must have finite shape (N, 2), N > 0")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidate_pixels_uv must be finite")
    options = {} if optimizer_options is None else dict(optimizer_options)
    rf_only = estimate_position_map(
        rf_observation,
        nominal_world_body,
        transform_body_array,
        pose_covariance,
        **options,
    )
    belief = project_joint_rf_belief_to_image(
        rf_only.position_world_m,
        rf_only.joint_covariance,
        nominal_world_body,
        transform_body_camera,
        camera,
        pose_delta_mean=rf_only.pose_delta,
    )
    association = associate_candidates(candidates, belief, association_confidence)
    world_array = nominal_world_body.compose(transform_body_array)
    world_camera = nominal_world_body.compose(transform_body_camera)
    if association.selected_index is None:
        shared = rf_only
        fixed = estimate_position_fixed_pose(rf_observation, world_array, **options)
    else:
        visual = VisualObservation(
            candidates[association.selected_index],
            visual_covariance_uv,
            rf_observation.timestamp_s,
        )
        shared = estimate_position_map(
            rf_observation,
            nominal_world_body,
            transform_body_array,
            pose_covariance,
            camera,
            transform_body_camera,
            visual,
            initial_position_world_m=rf_only.position_world_m,
            **options,
        )
        fixed = estimate_position_fixed_pose(
            rf_observation,
            world_array,
            camera,
            world_camera,
            visual,
            initial_position_world_m=rf_only.position_world_m,
            **options,
        )

    reduced_index, reduced_initial = _select_reduced_array_candidate(
        candidates,
        rf_observation,
        nominal_world_body,
        transform_body_array,
        transform_body_camera,
        camera,
    )
    reduced_visual = VisualObservation(
        candidates[reduced_index], visual_covariance_uv, rf_observation.timestamp_s
    )
    reduced = estimate_position_map(
        rf_observation,
        nominal_world_body,
        transform_body_array,
        pose_covariance,
        camera,
        transform_body_camera,
        reduced_visual,
        initial_position_world_m=reduced_initial,
        rf_component_indices=(0, 1),
        **options,
    )
    return OnlineLocalizationResult(
        estimates={
            "rf_only_full_3d": rf_only,
            "rf_vision_shared_pose": shared,
            "rf_vision_no_pose_uncertainty": fixed,
            "range_azimuth_vision": reduced,
        },
        primary_selected_index=association.selected_index,
        reduced_array_selected_index=reduced_index,
        rf_image_mean_uv=belief.mean_uv,
        rf_image_covariance_uv=belief.covariance_uv,
    )

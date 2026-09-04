"""RF-belief projection and served-target Mahalanobis association."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImageBelief:
    mean_uv: FloatArray
    covariance_uv: FloatArray


@dataclass(frozen=True)
class AssociationResult:
    selected_index: int | None
    squared_distances: FloatArray
    threshold: float


def project_joint_rf_belief_to_image(
    position_world_m: ArrayLike,
    target_pose_covariance: ArrayLike,
    nominal_world_body: RigidTransform,
    transform_body_camera: RigidTransform,
    camera: PinholeCamera,
    pose_delta_mean: ArrayLike | None = None,
    covariance_floor_px2: float = 1e-9,
) -> ImageBelief:
    """Project an RF joint belief over ``[target position, shared pose]`` into pixels.

    Keeping the complete 9-D covariance preserves target/pose cross-correlation created by the RF
    estimator. Discarding that cross block can double-count or misstate platform uncertainty.
    """

    position = np.asarray(position_world_m, dtype=float)
    covariance = np.asarray(target_pose_covariance, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_world_m must be a finite vector with shape (3,)")
    if covariance.shape != (9, 9) or not np.allclose(
        covariance, covariance.T, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("target_pose_covariance must be a symmetric (9, 9) matrix")
    if np.linalg.eigvalsh(covariance)[0] < -1e-10:
        raise ValueError("target_pose_covariance must be positive semidefinite")
    delta = np.zeros(6) if pose_delta_mean is None else np.asarray(pose_delta_mean, dtype=float)
    if delta.shape != (6,) or not np.all(np.isfinite(delta)):
        raise ValueError("pose_delta_mean must be a finite vector with shape (6,)")
    world_body = perturb_world_body(nominal_world_body, delta)
    pixel, camera_p, camera_xi = camera.observation_and_shared_pose_jacobians(
        position, world_body, transform_body_camera
    )
    projection_jacobian = np.column_stack((camera_p, camera_xi))
    image_covariance = projection_jacobian @ covariance @ projection_jacobian.T
    image_covariance = 0.5 * (image_covariance + image_covariance.T)
    image_covariance += np.eye(2) * float(covariance_floor_px2)
    if np.linalg.eigvalsh(image_covariance)[0] <= 0.0:
        raise ValueError("projected image covariance is not positive definite")
    return ImageBelief(pixel, image_covariance)


def mahalanobis_squared(candidate_uv: ArrayLike, image_belief: ImageBelief) -> FloatArray:
    candidates = np.asarray(candidate_uv, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or not np.all(np.isfinite(candidates)):
        raise ValueError("candidate_uv must be a finite matrix with shape (N, 2)")
    residuals = candidates - image_belief.mean_uv
    solved = np.linalg.solve(image_belief.covariance_uv, residuals.T).T
    return np.einsum("ni,ni->n", residuals, solved)


def associate_candidates(
    candidate_uv: ArrayLike, image_belief: ImageBelief, confidence: float = 0.95
) -> AssociationResult:
    """Select the minimum-distance candidate only when it lies inside the RF confidence ellipse."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    distances = mahalanobis_squared(candidate_uv, image_belief)
    threshold = float(chi2.ppf(confidence, df=2))
    if distances.size == 0:
        return AssociationResult(None, distances, threshold)
    best = int(np.argmin(distances))
    return AssociationResult(best if distances[best] <= threshold else None, distances, threshold)


def temporary_bbox_center_reference(boxes_xyxy: ArrayLike) -> FloatArray:
    """TEMPORARY_APPROXIMATION: use bbox centers until vehicle antenna keypoints exist."""

    boxes = np.asarray(boxes_xyxy, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.all(np.isfinite(boxes)):
        raise ValueError("boxes_xyxy must be a finite matrix with shape (N, 4)")
    if np.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError("bbox maximum coordinates must not be below minimum coordinates")
    return 0.5 * (boxes[:, :2] + boxes[:, 2:])


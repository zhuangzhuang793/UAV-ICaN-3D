"""Joint target/shared-pose information and equivalent target information."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fim import fisher_information
from .rf import SingularGeometryError


FloatArray = NDArray[np.float64]


def joint_target_pose_information(
    rf_position_jacobian: ArrayLike,
    rf_pose_jacobian: ArrayLike,
    rf_covariance: ArrayLike,
    pose_covariance: ArrayLike,
    camera_position_jacobian: ArrayLike | None = None,
    camera_pose_jacobian: ArrayLike | None = None,
    camera_covariance: ArrayLike | None = None,
) -> FloatArray:
    """Build the 9-D information for ``Theta=[p_W, delta_xi]``."""

    rf_p = np.asarray(rf_position_jacobian, dtype=float)
    rf_xi = np.asarray(rf_pose_jacobian, dtype=float)
    if rf_p.shape != (3, 3) or rf_xi.shape != (3, 6):
        raise ValueError("RF Jacobians must have shapes (3, 3) and (3, 6)")
    joint_rf = np.column_stack((rf_p, rf_xi))
    information = fisher_information(joint_rf, rf_covariance)

    pose_prior = np.asarray(pose_covariance, dtype=float)
    if pose_prior.shape != (6, 6) or not np.all(np.isfinite(pose_prior)):
        raise ValueError("pose_covariance must be a finite (6, 6) matrix")
    try:
        pose_information = np.linalg.solve(pose_prior, np.eye(6))
    except np.linalg.LinAlgError as error:
        raise ValueError("pose_covariance must be positive definite") from error
    if np.linalg.eigvalsh(0.5 * (pose_prior + pose_prior.T))[0] <= 0.0:
        raise ValueError("pose_covariance must be positive definite")
    information[3:, 3:] += pose_information

    camera_arguments = (camera_position_jacobian, camera_pose_jacobian, camera_covariance)
    if any(argument is not None for argument in camera_arguments):
        if any(argument is None for argument in camera_arguments):
            raise ValueError("all camera Jacobians and covariance must be supplied together")
        camera_p = np.asarray(camera_position_jacobian, dtype=float)
        camera_xi = np.asarray(camera_pose_jacobian, dtype=float)
        if camera_p.shape != (2, 3) or camera_xi.shape != (2, 6):
            raise ValueError("camera Jacobians must have shapes (2, 3) and (2, 6)")
        information += fisher_information(
            np.column_stack((camera_p, camera_xi)), camera_covariance
        )
    return 0.5 * (information + information.T)


def equivalent_target_information(
    joint_information: ArrayLike, rank_relative_tolerance: float = 1e-12
) -> FloatArray:
    """Schur-complement the shared pose nuisance out of a 9-D joint FIM."""

    information = np.asarray(joint_information, dtype=float)
    if information.shape != (9, 9) or not np.all(np.isfinite(information)):
        raise ValueError("joint_information must be a finite (9, 9) matrix")
    target_target = information[:3, :3]
    target_pose = information[:3, 3:]
    pose_pose = information[3:, 3:]
    pose_eigenvalues = np.linalg.eigvalsh(pose_pose)
    if pose_eigenvalues[-1] <= 0.0 or pose_eigenvalues[0] <= (
        pose_eigenvalues[-1] * rank_relative_tolerance
    ):
        raise SingularGeometryError("shared-pose information block is singular")
    equivalent = target_target - target_pose @ np.linalg.solve(pose_pose, target_pose.T)
    return 0.5 * (equivalent + equivalent.T)


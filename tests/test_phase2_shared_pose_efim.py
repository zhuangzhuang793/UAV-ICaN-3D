import numpy as np
import pytest

from uav_ican_3d.geometry import PinholeCamera, RigidTransform, perturb_world_body
from uav_ican_3d.localization import (
    SingularGeometryError,
    equivalent_target_information,
    joint_target_pose_information,
    position_bounds,
    rf_observation_and_shared_pose_jacobians,
)


def _scene():
    world_body = RigidTransform(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 50.0]))
    body_array = RigidTransform.identity()
    body_camera = RigidTransform(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([0.3, 0.0, 0.1]),
    )
    return np.array([20.0, 5.0, 0.0]), world_body, body_array, body_camera


def test_pose_perturbation_is_body_right_rotation_and_world_translation() -> None:
    _, world_body, _, _ = _scene()
    delta = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 1e-3])
    perturbed = perturb_world_body(world_body, delta)
    assert np.allclose(perturbed.translation, world_body.translation + delta[:3])
    relative_rotation = world_body.rotation.T @ perturbed.rotation
    assert np.isclose(relative_rotation[1, 0], np.sin(1e-3))


def test_camera_position_jacobian_has_rank_two() -> None:
    target, world_body, _, body_camera = _scene()
    camera = PinholeCamera(400.0, 420.0, 320.0, 240.0)
    _, camera_p, camera_xi = camera.observation_and_shared_pose_jacobians(
        target, world_body, body_camera
    )
    assert camera_p.shape == (2, 3)
    assert camera_xi.shape == (2, 6)
    assert np.linalg.matrix_rank(camera_p) == 2


def test_camera_only_free_3d_target_is_rank_deficient() -> None:
    target, world_body, _, body_camera = _scene()
    camera = PinholeCamera(400.0, 420.0, 320.0, 240.0)
    _, camera_p, camera_xi = camera.observation_and_shared_pose_jacobians(
        target, world_body, body_camera
    )
    camera_joint = np.zeros((9, 9))
    joint_jacobian = np.column_stack((camera_p, camera_xi))
    camera_joint += joint_jacobian.T @ joint_jacobian
    camera_joint[3:, 3:] += np.eye(6)
    camera_efim = equivalent_target_information(camera_joint)
    with pytest.raises(SingularGeometryError):
        position_bounds(camera_efim)


def test_joint_information_accepts_full_covariances() -> None:
    target, world_body, body_array, body_camera = _scene()
    _, rf_p, rf_xi = rf_observation_and_shared_pose_jacobians(target, world_body, body_array)
    camera = PinholeCamera(400.0, 420.0, 320.0, 240.0)
    _, camera_p, camera_xi = camera.observation_and_shared_pose_jacobians(
        target, world_body, body_camera
    )
    rf_covariance = np.array([[0.25, 0.001, 0.0], [0.001, 0.01, 0.002], [0.0, 0.002, 0.02]])
    camera_covariance = np.array([[4.0, 0.5], [0.5, 5.0]])
    pose_covariance = np.eye(6) * 0.01
    joint = joint_target_pose_information(
        rf_p, rf_xi, rf_covariance, pose_covariance, camera_p, camera_xi, camera_covariance
    )
    bounds = position_bounds(equivalent_target_information(joint))
    assert joint.shape == (9, 9)
    assert bounds.peb_3d_m > 0.0


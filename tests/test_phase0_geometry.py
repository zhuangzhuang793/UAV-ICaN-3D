import numpy as np
import pytest

from uav_ican_3d.geometry import PinholeCamera, RigidTransform, is_rotation_matrix
from uav_ican_3d.types import BeliefState, PoseBelief, RFObservation, VisualObservation


def test_world_body_round_trip_for_hand_constructed_pose() -> None:
    rotation_world_body = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform_world_body = RigidTransform(rotation_world_body, np.array([10.0, 20.0, 30.0]))
    point_body = np.array([2.0, 3.0, -4.0])

    point_world = transform_world_body.apply_point(point_body)
    assert np.allclose(point_world, np.array([7.0, 22.0, 26.0]))
    assert np.allclose(transform_world_body.inverse().apply_point(point_world), point_body)


def test_transform_composition() -> None:
    transform_world_body = RigidTransform(np.eye(3), np.array([10.0, 0.0, 0.0]))
    transform_body_camera = RigidTransform(np.eye(3), np.array([1.0, 2.0, 3.0]))
    transform_world_camera = transform_world_body.compose(transform_body_camera)
    assert np.allclose(transform_world_camera.apply_point(np.zeros(3)), [11.0, 2.0, 3.0])


def test_hand_computed_camera_projection() -> None:
    camera = PinholeCamera(fx_px=400.0, fy_px=420.0, cx_px=320.0, cy_px=240.0)
    assert np.allclose(camera.project_camera([1.0, 2.0, 10.0]), [360.0, 324.0])


def test_camera_rejects_nonpositive_depth() -> None:
    camera = PinholeCamera(fx_px=400.0, fy_px=420.0, cx_px=320.0, cy_px=240.0)
    with pytest.raises(ValueError, match="positive depth"):
        camera.project_camera([1.0, 2.0, 0.0])


def test_rotation_validation_rejects_reflection() -> None:
    reflection = np.diag([1.0, 1.0, -1.0])
    assert not is_rotation_matrix(reflection)
    with pytest.raises(ValueError, match="proper rotation"):
        RigidTransform(reflection, np.zeros(3))


def test_shared_data_types_accept_full_covariances() -> None:
    pose_covariance = np.eye(6)
    pose_covariance[0, 1] = pose_covariance[1, 0] = 0.1
    PoseBelief(RigidTransform.identity(), pose_covariance)

    rf_covariance = np.eye(3)
    rf_covariance[0, 2] = rf_covariance[2, 0] = 0.2
    RFObservation(10.0, 0.1, -0.2, rf_covariance)

    VisualObservation(np.array([320.0, 240.0]), np.array([[4.0, 1.0], [1.0, 9.0]]))
    BeliefState(np.zeros(6), np.eye(6))


def test_data_types_reject_nonsymmetric_covariance() -> None:
    covariance = np.eye(3)
    covariance[0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        RFObservation(10.0, 0.0, 0.0, covariance)


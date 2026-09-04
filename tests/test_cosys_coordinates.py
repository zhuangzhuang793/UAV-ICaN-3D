import numpy as np

from uav_ican_3d.geometry import PinholeCamera
from uav_ican_3d.simulation import (
    ROTATION_AIRSIM_CAMERA_OPENCV,
    ROTATION_ENU_NED,
    airsim_camera_to_project_frames,
    quaternion_xyzw_to_rotation,
)


def test_cosys_basis_changes_are_proper_rotations() -> None:
    for rotation in (ROTATION_ENU_NED, ROTATION_AIRSIM_CAMERA_OPENCV):
        assert np.allclose(rotation.T @ rotation, np.eye(3))
        assert np.isclose(np.linalg.det(rotation), 1.0)


def test_quaternion_conversion_matches_minus_ninety_degree_pitch() -> None:
    half = np.sqrt(0.5)
    rotation = quaternion_xyzw_to_rotation([0.0, -half, 0.0, half])
    expected = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(rotation, expected)


def test_air_camera_projection_matches_frd_pinhole_equations() -> None:
    half = np.sqrt(0.5)
    position_ned = np.array([-3.1, -13.4, -28.1])
    world_body, body_camera = airsim_camera_to_project_frames(
        position_ned, [0.0, -half, 0.0, half]
    )
    target_ned = np.array([10.0, 0.0, -0.3])
    target_enu = ROTATION_ENU_NED @ target_ned
    camera = PinholeCamera(320.0, 320.0, 320.0, 240.0)

    pixel = camera.project_world(target_enu, world_body.compose(body_camera))
    relative_ned = target_ned - position_ned
    relative_air_camera = quaternion_xyzw_to_rotation(
        [0.0, -half, 0.0, half]
    ).T @ relative_ned
    expected = np.array(
        [
            320.0 + 320.0 * relative_air_camera[1] / relative_air_camera[0],
            240.0 + 320.0 * relative_air_camera[2] / relative_air_camera[0],
        ]
    )
    assert np.allclose(pixel, expected)

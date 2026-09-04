import numpy as np

from uav_ican_3d.localization import (
    estimate_position_fixed_pose,
    estimate_position_map,
    predict_rf_observation,
)
from uav_ican_3d.simulation import complementarity_scene
from uav_ican_3d.types import RFObservation, VisualObservation


def _observations():
    import yaml

    with open("configs/quick.yaml", encoding="utf-8") as stream:
        phase3 = yaml.safe_load(stream)["phase3"]
    target, world_body, body_array, body_camera, camera = complementarity_scene(
        phase3, 60.0, 60.0
    )
    rf_covariance = np.diag([1.0, np.deg2rad(2.0) ** 2, np.deg2rad(2.0) ** 2])
    camera_covariance = np.eye(2) * 4.0
    rf = RFObservation(
        *predict_rf_observation(target, world_body.compose(body_array)), rf_covariance
    )
    visual = VisualObservation(
        camera.project_world(target, world_body.compose(body_camera)), camera_covariance
    )
    return target, world_body, body_array, body_camera, camera, rf, visual


def test_noiseless_joint_map_recovers_target_without_ground_truth_initializer() -> None:
    target, world_body, body_array, body_camera, camera, rf, visual = _observations()
    pose_covariance = np.diag([0.2**2] * 3 + [np.deg2rad(0.5) ** 2] * 3)
    estimate = estimate_position_map(
        rf, world_body, body_array, pose_covariance, camera, body_camera, visual
    )
    assert np.allclose(estimate.position_world_m, target, atol=1e-8)
    assert np.allclose(estimate.pose_delta, np.zeros(6), atol=1e-8)


def test_rf_only_map_covariance_is_positive_definite() -> None:
    _, world_body, body_array, _, _, rf, _ = _observations()
    pose_covariance = np.diag([0.2**2] * 3 + [np.deg2rad(0.5) ** 2] * 3)
    estimate = estimate_position_map(rf, world_body, body_array, pose_covariance)
    assert np.linalg.eigvalsh(estimate.position_covariance)[0] > 0.0
    assert estimate.joint_covariance.shape == (9, 9)


def test_fixed_pose_ablation_recovers_noiseless_target() -> None:
    target, world_body, body_array, body_camera, camera, rf, visual = _observations()
    estimate = estimate_position_fixed_pose(
        rf,
        world_body.compose(body_array),
        camera,
        world_body.compose(body_camera),
        visual,
    )
    assert np.allclose(estimate.position_world_m, target, atol=1e-8)
    assert np.linalg.eigvalsh(estimate.position_covariance)[0] > 0.0


def test_range_azimuth_plus_vision_ablation_is_identifiable() -> None:
    target, world_body, body_array, body_camera, camera, rf, visual = _observations()
    pose_covariance = np.diag([0.2**2] * 3 + [np.deg2rad(0.5) ** 2] * 3)
    estimate = estimate_position_map(
        rf,
        world_body,
        body_array,
        pose_covariance,
        camera,
        body_camera,
        visual,
        rf_component_indices=(0, 1),
    )
    assert np.allclose(estimate.position_world_m, target, atol=1e-8)
    assert np.linalg.eigvalsh(estimate.position_covariance)[0] > 0.0

import inspect

import numpy as np

from uav_ican_3d.geometry import perturb_world_body
from uav_ican_3d.localization import (
    localize_online,
    predict_rf_observation,
)
from uav_ican_3d.simulation import complementarity_scene
from uav_ican_3d.types import RFObservation


def test_online_f7_api_has_no_ground_truth_inputs() -> None:
    parameters = set(inspect.signature(localize_online).parameters)
    forbidden = {
        "user_gt_position",
        "served_target_gt_id",
        "path_delay",
        "path_aoa",
        "sionna_paths",
    }
    assert not forbidden & parameters


def test_online_f7_noiseless_methods_recover_target() -> None:
    import yaml

    phase = yaml.safe_load(open("configs/quick.yaml", encoding="utf-8"))["phase3"]
    target, actual_body, body_array, body_camera, camera = complementarity_scene(
        phase, 60.0, 60.0
    )
    pose_covariance = np.diag([0.2**2] * 3 + [np.deg2rad(0.2) ** 2] * 3)
    nominal_body = perturb_world_body(actual_body, np.zeros(6))
    rf_covariance = np.diag([1.0, np.deg2rad(1.0) ** 2, np.deg2rad(1.0) ** 2])
    rf = RFObservation(
        *predict_rf_observation(target, actual_body.compose(body_array)), rf_covariance
    )
    pixel = camera.project_world(target, actual_body.compose(body_camera))
    candidates = np.vstack((pixel, pixel + np.array([100.0, 80.0])))
    result = localize_online(
        rf,
        nominal_body,
        body_array,
        pose_covariance,
        camera,
        body_camera,
        candidates,
        np.eye(2) * 4.0,
        0.99,
    )
    assert result.primary_selected_index == 0
    assert result.reduced_array_selected_index == 0
    for estimate in result.estimates.values():
        assert np.allclose(estimate.position_world_m, target, atol=1e-7)

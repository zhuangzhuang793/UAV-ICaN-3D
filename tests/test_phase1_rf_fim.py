import numpy as np
import pytest

from uav_ican_3d.geometry import RigidTransform
from uav_ican_3d.localization import (
    SingularGeometryError,
    fisher_information,
    position_bounds,
    predict_rf_observation,
    rf_position_jacobian,
)


def test_rf_observation_for_three_four_twelve_triangle() -> None:
    observation = predict_rf_observation([3.0, 4.0, 12.0], RigidTransform.identity())
    assert np.allclose(observation, [13.0, np.arctan2(4.0, 3.0), np.arctan2(12.0, 5.0)])


def test_range_jacobian_row() -> None:
    jacobian = rf_position_jacobian([3.0, 4.0, 12.0], RigidTransform.identity())
    assert np.allclose(jacobian[0], [3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0])


def test_world_to_array_rotation_is_applied() -> None:
    rotation_world_array = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform_world_array = RigidTransform(rotation_world_array, np.array([10.0, 20.0, 30.0]))
    position_world = transform_world_array.apply_point([3.0, 4.0, 12.0])
    observation = predict_rf_observation(position_world, transform_world_array)
    assert np.allclose(observation, [13.0, np.arctan2(4.0, 3.0), np.arctan2(12.0, 5.0)])


def test_full_measurement_covariance_is_supported() -> None:
    jacobian = rf_position_jacobian([30.0, 40.0, -10.0], RigidTransform.identity())
    covariance = np.array([[0.25, 0.001, 0.0], [0.001, 0.01, 0.002], [0.0, 0.002, 0.02]])
    information = fisher_information(jacobian, covariance)
    bounds = position_bounds(information)
    assert np.all(np.linalg.eigvalsh(bounds.covariance) > 0.0)
    assert bounds.peb_3d_m > 0.0


def test_singular_array_axis_is_explicit() -> None:
    with pytest.raises(SingularGeometryError, match="azimuth"):
        predict_rf_observation([0.0, 0.0, 10.0], RigidTransform.identity())


def test_singular_fim_is_explicit_and_not_pseudoinverted() -> None:
    information = np.diag([1.0, 1.0, 0.0])
    with pytest.raises(SingularGeometryError, match="singular position FIM"):
        position_bounds(information)


def test_invalid_covariance_is_rejected() -> None:
    jacobian = np.eye(3)
    with pytest.raises(ValueError, match="positive definite"):
        fisher_information(jacobian, np.diag([1.0, 1.0, 0.0]))


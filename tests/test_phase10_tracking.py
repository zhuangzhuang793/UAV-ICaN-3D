import numpy as np

from uav_ican_3d.prediction import update_position_belief
from uav_ican_3d.types import BeliefState


def test_position_belief_update_is_finite_and_positive_semidefinite() -> None:
    prior = BeliefState(np.zeros(6), np.eye(6), 0.0)
    posterior = update_position_belief(prior, [1.0, 2.0, 0.0], np.eye(3) * 0.2, 0.5)
    assert np.all(np.isfinite(posterior.mean))
    assert np.linalg.eigvalsh(posterior.covariance)[0] >= -1e-12
    assert posterior.timestamp_s == 0.5

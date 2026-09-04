import numpy as np

from uav_ican_3d.control import (
    AngularBelief,
    propagate_position_to_angles,
    robust_mpc_step,
    sample_trajectory_distribution,
    select_robust_beam,
)
from uav_ican_3d.geometry import RigidTransform


def test_robust_mpc_respects_speed_and_altitude_constraints() -> None:
    rng = np.random.default_rng(1)
    mean = np.array([[20.0 + step, 5.0, 0.0] for step in range(1, 6)])
    samples = sample_trajectory_distribution(mean, np.ones_like(mean), 32, rng)
    decision = robust_mpc_step(
        np.array([0.0, 0.0, 50.0]), samples, 0.5, 5.0, (30.0, 100.0)
    )
    assert np.linalg.norm(decision.velocity_world_mps) <= 5.0 + 1e-12
    assert np.all((decision.planned_positions_world_m[:, 2] >= 30.0))
    assert np.all((decision.planned_positions_world_m[:, 2] <= 100.0))
    assert np.isfinite(decision.score)


def test_position_covariance_propagates_to_angular_covariance() -> None:
    world_array = RigidTransform(
        np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 50.0])
    )
    low = propagate_position_to_angles([20.0, 5.0, 0.0], np.eye(3) * 0.1, world_array)
    high = propagate_position_to_angles([20.0, 5.0, 0.0], np.eye(3) * 2.0, world_array)
    assert np.trace(high.covariance_rad2) > np.trace(low.covariance_rad2)


def test_beam_selection_is_finite() -> None:
    belief = AngularBelief(np.array([0.1, 0.5]), np.diag([0.01, 0.02]))
    selection = select_robust_beam(
        belief,
        np.deg2rad(np.arange(-60, 61, 15)),
        np.deg2rad(np.arange(15, 76, 15)),
        np.random.default_rng(2),
        sample_count=64,
    )
    assert selection.aperture_size in (2, 4)
    assert np.isfinite(selection.robust_gain)

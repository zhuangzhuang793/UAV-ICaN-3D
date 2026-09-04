import torch

from uav_ican_3d.prediction import (
    BeliefTrajectoryPredictor,
    diagonal_gaussian_nll,
    generate_synthetic_beliefs,
)


def test_synthetic_dataset_contains_straight_and_turning_motion() -> None:
    dataset = generate_synthetic_beliefs(8, 4, 3, 0.5, 7)
    assert dataset.means.shape == (8, 4, 6)
    assert dataset.covariances.shape == (8, 4, 6, 6)
    assert dataset.future_positions.shape == (8, 3, 3)
    assert set(dataset.turning.tolist()) == {0, 1}


def test_covariance_path_changes_predicted_distribution() -> None:
    torch.manual_seed(3)
    model = BeliefTrajectoryPredictor(4, 3, hidden_size=16, use_covariance=True)
    means = torch.zeros(1, 4, 6)
    low = torch.eye(6).reshape(1, 1, 6, 6).repeat(1, 4, 1, 1) * 0.01
    high = low * 100.0
    _, low_std = model(means, low)
    _, high_std = model(means, high)
    assert torch.mean(high_std) > torch.mean(low_std)


def test_gaussian_nll_rewards_accurate_mean() -> None:
    target = torch.zeros(2, 3, 3)
    standard_deviation = torch.ones_like(target)
    accurate = diagonal_gaussian_nll(target, target, standard_deviation)
    inaccurate = diagonal_gaussian_nll(target, target + 2.0, standard_deviation)
    assert accurate < inaccurate

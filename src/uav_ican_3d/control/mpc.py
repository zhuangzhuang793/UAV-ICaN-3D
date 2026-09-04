"""Sampling-based robust trajectory control for the QUICK closed loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MPCDecision:
    velocity_world_mps: FloatArray
    planned_positions_world_m: FloatArray
    score: float
    expected_rate: float
    localization_quality: float


def sample_trajectory_distribution(
    mean_positions: ArrayLike,
    standard_deviations: ArrayLike,
    sample_count: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw complete future trajectories from a predicted diagonal joint Gaussian."""

    mean = np.asarray(mean_positions, dtype=float)
    standard_deviation = np.asarray(standard_deviations, dtype=float)
    if mean.ndim != 2 or mean.shape[1] != 3 or standard_deviation.shape != mean.shape:
        raise ValueError("trajectory mean/std must both have shape (horizon, 3)")
    if sample_count < 1 or np.any(standard_deviation <= 0.0):
        raise ValueError("sample_count and standard deviations must be positive")
    return mean + rng.normal(size=(sample_count, *mean.shape)) * standard_deviation


def robust_mpc_step(
    uav_position_world_m: ArrayLike,
    trajectory_samples_world_m: ArrayLike,
    dt_s: float,
    maximum_speed_mps: float,
    altitude_bounds_m: tuple[float, float],
    communication_weight: float = 1.0,
    localization_weight: float = 150.0,
    risk_weight: float = 0.25,
    effort_weight: float = 0.01,
) -> MPCDecision:
    """Choose a constant short-horizon velocity from a finite robust action set."""

    uav = np.asarray(uav_position_world_m, dtype=float)
    samples = np.asarray(trajectory_samples_world_m, dtype=float)
    if uav.shape != (3,) or samples.ndim != 3 or samples.shape[2] != 3:
        raise ValueError("invalid UAV position or trajectory samples")
    if dt_s <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("time step and speed limit must be positive")
    headings = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    actions = [np.zeros(3)]
    actions.extend(
        maximum_speed_mps * np.array([np.cos(angle), np.sin(angle), 0.0])
        for angle in headings
    )
    actions.extend(
        [
            np.array([0.0, 0.0, maximum_speed_mps * 0.5]),
            np.array([0.0, 0.0, -maximum_speed_mps * 0.5]),
        ]
    )
    times = dt_s * np.arange(1, samples.shape[1] + 1, dtype=float)
    best: MPCDecision | None = None
    for velocity in actions:
        planned = uav + times[:, None] * velocity
        if np.any(planned[:, 2] < altitude_bounds_m[0]) or np.any(
            planned[:, 2] > altitude_bounds_m[1]
        ):
            continue
        relative = samples - planned[None, :, :]
        ranges = np.linalg.norm(relative, axis=2)
        horizontal_squared = np.sum(relative[:, :, :2] ** 2, axis=2)
        rates = np.log2(1.0 + 1.0e5 / np.maximum(ranges**2, 1.0))
        episode_rates = np.mean(rates, axis=1)
        expected_rate = float(np.mean(episode_rates))
        localization_quality = float(
            np.mean(horizontal_squared / np.maximum(ranges**4, 1.0))
        )
        score = (
            communication_weight * expected_rate
            - risk_weight * float(np.std(episode_rates))
            + localization_weight * localization_quality
            - effort_weight * float(np.dot(velocity, velocity))
        )
        decision = MPCDecision(
            velocity.copy(), planned, score, expected_rate, localization_quality
        )
        if best is None or decision.score > best.score:
            best = decision
    if best is None:
        raise RuntimeError("no feasible MPC action")
    return best

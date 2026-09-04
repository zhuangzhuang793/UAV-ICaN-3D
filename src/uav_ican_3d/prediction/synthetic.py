"""Small straight/turning synthetic belief histories for Quick Gate 7."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticBeliefDataset:
    means: Tensor
    covariances: Tensor
    future_positions: Tensor
    turning: Tensor


def generate_synthetic_beliefs(
    sample_count: int,
    history_steps: int,
    horizon_steps: int,
    dt_s: float,
    seed: int,
) -> SyntheticBeliefDataset:
    """Generate noisy Gaussian beliefs for both straight and constant-turn vehicle motion."""

    if sample_count < 1 or history_steps < 2 or horizon_steps < 1 or dt_s <= 0.0:
        raise ValueError("invalid synthetic dataset dimensions")
    rng = np.random.default_rng(seed)
    means = np.empty((sample_count, history_steps, 6), dtype=np.float32)
    covariances = np.zeros((sample_count, history_steps, 6, 6), dtype=np.float32)
    futures = np.empty((sample_count, horizon_steps, 3), dtype=np.float32)
    turning = np.empty(sample_count, dtype=np.int64)
    total_steps = history_steps + horizon_steps

    for sample in range(sample_count):
        is_turning = bool(sample % 2)
        turning[sample] = int(is_turning)
        position = np.array(
            [rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0), 0.0]
        )
        speed = rng.uniform(3.0, 10.0)
        heading = rng.uniform(-np.pi, np.pi)
        turn_rate = rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.24) if is_turning else 0.0
        states = []
        for _ in range(total_steps):
            velocity = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0])
            states.append(np.concatenate((position.copy(), velocity)))
            position = position + velocity * dt_s
            heading += turn_rate * dt_s
        states_array = np.asarray(states)
        position_std = float(np.exp(rng.uniform(np.log(0.08), np.log(1.8))))
        velocity_std = float(np.exp(rng.uniform(np.log(0.03), np.log(0.7))))
        for history_index in range(history_steps):
            time_factor = 0.85 + 0.3 * history_index / (history_steps - 1)
            variances = np.array(
                [position_std**2] * 3 + [velocity_std**2] * 3
            ) * time_factor
            covariance = np.diag(variances)
            covariances[sample, history_index] = covariance
            means[sample, history_index] = rng.multivariate_normal(
                states_array[history_index], covariance
            )
        futures[sample] = states_array[history_steps : history_steps + horizon_steps, :3]
    return SyntheticBeliefDataset(
        torch.from_numpy(means),
        torch.from_numpy(covariances),
        torch.from_numpy(futures),
        torch.from_numpy(turning),
    )
